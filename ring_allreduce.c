/*
 * ring_allreduce.c -- the simplest ring Reduce-Scatter / All-Gather / All-Reduce
 * over InfiniBand Verbs. Single threaded, no tricks.
 *
 * Build:  gcc -O2 -Wall ring_allreduce.c -libverbs -o test
 * Run:    ./test -myindex 1 -list mlx-stud-01 mlx-stud-02 mlx-stud-03 mlx-stud-04   (one per node, myindex 1..4)
 *
 * Ring:  every rank has two QPs
 *          qp_right : connected to the right neighbour, we SEND DATA through it
 *          qp_left  : connected to the left neighbour, we RECEIVE DATA through it
 *        An RC QP is two-way, so the same QPs also carry small control messages
 *        the other way (used by rendezvous).
 *
 * Two protocols (chosen per collective from the block size):
 *   EAGER      : copy block to a bounce buffer, SEND it. The receiver copies it
 *                out of its pre-posted slot. No handshake, one extra copy.
 *   RENDEZVOUS : (sent in NPIECE pieces = pipelining)
 *                the receiver first tells the sender "write here" (address+rkey).
 *                The sender then does RDMA_WRITE_WITH_IMM straight into that
 *                memory: zero copy. The IMM completion tells the receiver the
 *                data has arrived.
 *
 * Every ring step is "send one block to the right, receive one from the left",
 * and we finish a step completely before starting the next one.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <infiniband/verbs.h>
#include "pg.h"

#define ESZ        4            /* element size in bytes (int32 and float) */
#define IB_PORT    1
#define SLOT_SIZE  65536        /* size of one eager receive slot */
#define NSLOT      16           /* number of eager receive slots */
#define NCTRL      4            /* number of control receive slots */

int pg_eager_max = 16384;
static int g_rank, g_size, g_port = 18515;

void pg_set_ring(int rank, int size, int base_port)
{
    g_rank = rank; g_size = size; g_port = base_port;
}

/* Rendezvous control message: "write the data to this address with this rkey" */
struct ctrl { uint64_t addr; uint32_t rkey; };

/* Handle of the process group (what pg_handle points to). */
struct pg {
    int rank, size, rdvz;                  /* rdvz = protocol of current collective */
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;                     /* one CQ for everything */
    struct ibv_port_attr port;
    struct ibv_qp *qp_right, *qp_left;

    char *slots;      struct ibv_mr *slots_mr;      /* eager receive slots   */
    char *send_buf;   struct ibv_mr *send_mr;       /* eager send bounce buf */
    struct ctrl *ctrl_recv, *ctrl_send;             /* rendezvous messages   */
    struct ibv_mr *ctrl_recv_mr, *ctrl_send_mr;

    /* what the CQ poll has found so far but nobody consumed yet */
    int send_done;                                  /* finished sends        */
    int dq[NSLOT], dhead, dtail;                    /* arrived data (slot)   */
    int cqq[NCTRL], chead, ctail;                   /* arrived ctrl (slot)   */
};

static void die(const char *msg) { perror(msg); exit(1); }
#define CHECK(x) do { if (x) die(#x); } while (0)

/* ------------------------------------------------------------------------ */
/* Small verbs helpers                                                      */
/* ------------------------------------------------------------------------ */

static struct ibv_mr *reg(struct pg *pg, void *p, size_t n)
{
    struct ibv_mr *mr = ibv_reg_mr(pg->pd, p, n,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) die("ibv_reg_mr");
    return mr;
}

static void post_recv(struct ibv_qp *qp, void *buf, int len, uint32_t lkey, uint64_t id)
{
    struct ibv_sge sge = { (uintptr_t)buf, len, lkey };
    struct ibv_recv_wr wr = { .wr_id = id, .sg_list = &sge, .num_sge = 1 }, *bad;
    CHECK(ibv_post_recv(qp, &wr, &bad));
}

/* op is IBV_WR_SEND or IBV_WR_RDMA_WRITE_WITH_IMM (raddr/rkey used by the latter) */
static void post_send(struct ibv_qp *qp, enum ibv_wr_opcode op, void *buf, int len,
                      uint32_t lkey, uint64_t raddr, uint32_t rkey)
{
    struct ibv_sge sge = { (uintptr_t)buf, len, lkey };
    struct ibv_send_wr wr = { .sg_list = &sge, .num_sge = 1, .opcode = op,
                              .send_flags = IBV_SEND_SIGNALED }, *bad;
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = rkey;
    CHECK(ibv_post_send(qp, &wr, &bad));
}

/* Poll the CQ until ONE completion arrives and file it in the right place.
 * wr_id of a receive: data slot i -> i, control slot j -> NSLOT + j. */
static void progress(struct pg *pg)
{
    struct ibv_wc wc;
    while (ibv_poll_cq(pg->cq, 1, &wc) == 0) ;
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "rank %d: completion error: %s\n", pg->rank, ibv_wc_status_str(wc.status));
        exit(1);
    }
    if (wc.opcode < IBV_WC_RECV)                        /* a send or write finished */
        pg->send_done++;
    else if (wc.wr_id < NSLOT)                          /* data arrived */
        pg->dq[pg->dtail++ % NSLOT] = (int)wc.wr_id;
    else                                                /* control message arrived */
        pg->cqq[pg->ctail++ % NCTRL] = (int)wc.wr_id - NSLOT;
}

static void wait_send(struct pg *pg)
{
    while (pg->send_done == 0) progress(pg);
    pg->send_done--;
}
static int wait_data(struct pg *pg)     /* returns the slot that got data */
{
    while (pg->dhead == pg->dtail) progress(pg);
    return pg->dq[pg->dhead++ % NSLOT];
}
static int wait_ctrl(struct pg *pg)     /* returns the control slot */
{
    while (pg->chead == pg->ctail) progress(pg);
    return pg->cqq[pg->chead++ % NCTRL];
}

/* ------------------------------------------------------------------------ */
/* Connecting the ring                                                      */
/* ------------------------------------------------------------------------ */

struct conn { uint32_t lid, qpn; };     /* what a peer needs to know about our QP */

static int tcp_listen(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port),
                             .sin_addr.s_addr = INADDR_ANY };
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    CHECK(bind(fd, (struct sockaddr *)&a, sizeof a));
    CHECK(listen(fd, 4));
    return fd;
}

/* Neighbours start at slightly different times, so retry until it listens. */
static int tcp_connect(char *host, int port)
{
    char p[16];
    snprintf(p, sizeof p, "%d", port);
    for (int tries = 0; tries < 600; tries++) {
        struct addrinfo hints = { .ai_family = AF_INET, .ai_socktype = SOCK_STREAM }, *ai;
        if (getaddrinfo(host, p, &hints, &ai) == 0) {
            int fd = socket(AF_INET, SOCK_STREAM, 0);
            if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) { freeaddrinfo(ai); return fd; }
            close(fd);
            freeaddrinfo(ai);
        }
        usleep(100000);
    }
    die("cannot connect to right neighbour");
    return -1;
}

static struct ibv_qp *make_qp(struct pg *pg)
{
    struct ibv_qp_init_attr init = {
        .send_cq = pg->cq, .recv_cq = pg->cq, .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 32, .max_recv_wr = 64, .max_send_sge = 1, .max_recv_sge = 1 } };
    struct ibv_qp *qp = ibv_create_qp(pg->pd, &init);
    if (!qp) die("ibv_create_qp");
    struct ibv_qp_attr a = { .qp_state = IBV_QPS_INIT, .pkey_index = 0, .port_num = IB_PORT,
                             .qp_access_flags = IBV_ACCESS_REMOTE_WRITE };
    CHECK(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
    return qp;
}

/* INIT -> RTR -> RTS, connected to the remote QP described by r */
static void qp_connect(struct pg *pg, struct ibv_qp *qp, struct conn *r)
{
    struct ibv_qp_attr rtr = { .qp_state = IBV_QPS_RTR, .path_mtu = pg->port.active_mtu,
        .dest_qp_num = r->qpn, .rq_psn = 0, .max_dest_rd_atomic = 1, .min_rnr_timer = 12,
        .ah_attr = { .dlid = r->lid, .sl = 0, .src_path_bits = 0, .port_num = IB_PORT } };
    CHECK(ibv_modify_qp(qp, &rtr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
    struct ibv_qp_attr rts = { .qp_state = IBV_QPS_RTS, .timeout = 14, .retry_cnt = 7,
                               .rnr_retry = 7, .sq_psn = 0, .max_rd_atomic = 1 };
    CHECK(ibv_modify_qp(qp, &rts, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
}

int connect_process_group(char *servername, void **pg_handle)
{
    struct pg *pg = calloc(1, sizeof *pg);
    pg->rank = g_rank;
    pg->size = g_size;

    /* 1. verbs objects */
    struct ibv_device **dev = ibv_get_device_list(NULL);
    if (!dev || !dev[0]) die("no InfiniBand device");
    pg->ctx = ibv_open_device(dev[0]);
    pg->pd  = ibv_alloc_pd(pg->ctx);
    pg->cq  = ibv_create_cq(pg->ctx, 256, NULL, NULL, 0);
    CHECK(ibv_query_port(pg->ctx, IB_PORT, &pg->port));
    pg->qp_right = make_qp(pg);
    pg->qp_left  = make_qp(pg);

    /* 2. registered buffers: eager slots, eager send buffer, control messages */
    pg->slots     = malloc(NSLOT * SLOT_SIZE);
    pg->send_buf  = malloc(SLOT_SIZE);
    pg->ctrl_recv = malloc(NCTRL * sizeof(struct ctrl));
    pg->ctrl_send = malloc(sizeof(struct ctrl));
    pg->slots_mr     = reg(pg, pg->slots, NSLOT * SLOT_SIZE);
    pg->send_mr      = reg(pg, pg->send_buf, SLOT_SIZE);
    pg->ctrl_recv_mr = reg(pg, pg->ctrl_recv, NCTRL * sizeof(struct ctrl));
    pg->ctrl_send_mr = reg(pg, pg->ctrl_send, sizeof(struct ctrl));

    /* 3. pre-post receives. Data from the left lands in the slots (an eager SEND
     *    and an RDMA write-with-imm both use up one receive); control messages
     *    from the right land in ctrl_recv. */
    for (int i = 0; i < NSLOT; i++)
        post_recv(pg->qp_left, pg->slots + i * SLOT_SIZE, SLOT_SIZE, pg->slots_mr->lkey, i);
    for (int j = 0; j < NCTRL; j++)
        post_recv(pg->qp_right, &pg->ctrl_recv[j], sizeof(struct ctrl), pg->ctrl_recv_mr->lkey, NSLOT + j);

    /* 4. exchange QP info over TCP: connect to the right neighbour (servername),
     *    accept the left one. Both sides SEND first and READ after, no deadlock. */
    int lfd   = tcp_listen(g_port + g_rank);
    int right = tcp_connect(servername, g_port + (g_rank + 1) % g_size);
    int left  = accept(lfd, NULL, NULL);
    CHECK(left < 0);
    struct conn mine_r = { pg->port.lid, pg->qp_right->qp_num };
    struct conn mine_l = { pg->port.lid, pg->qp_left->qp_num };
    struct conn from_r, from_l;
    CHECK(send(right, &mine_r, sizeof mine_r, 0) < 0);
    CHECK(send(left,  &mine_l, sizeof mine_l, 0) < 0);
    CHECK(recv(right, &from_r, sizeof from_r, MSG_WAITALL) != sizeof from_r);
    CHECK(recv(left,  &from_l, sizeof from_l, MSG_WAITALL) != sizeof from_l);
    qp_connect(pg, pg->qp_right, &from_r);      /* our right QP <-> its left QP  */
    qp_connect(pg, pg->qp_left,  &from_l);      /* our left QP  <-> its right QP */

    /* 5. one more round so nobody sends before the neighbours are ready */
    char b = 1;
    CHECK(send(right, &b, 1, 0) < 0);
    CHECK(send(left,  &b, 1, 0) < 0);
    CHECK(recv(right, &b, 1, MSG_WAITALL) != 1);
    CHECK(recv(left,  &b, 1, MSG_WAITALL) != 1);
    close(right); close(left); close(lfd);

    *pg_handle = pg;
    return 0;
}

/* ------------------------------------------------------------------------ */
/* One ring step                                                            */
/* ------------------------------------------------------------------------ */

static void reduce(char *dst, const char *src, int n, DATATYPE dt, OPERATION op)
{
    if (dt == PG_INT32) {
        int *d = (int *)dst; const int *s = (const int *)src;
        for (int i = 0; i < n; i++) d[i] = (op == PG_SUM) ? d[i] + s[i] : (d[i] > s[i] ? d[i] : s[i]);
    } else {
        float *d = (float *)dst; const float *s = (const float *)src;
        for (int i = 0; i < n; i++) d[i] = (op == PG_SUM) ? d[i] + s[i] : (d[i] > s[i] ? d[i] : s[i]);
    }
}

/* Pipelining: in rendezvous a block is sent as up to NPIECE separate RDMA writes.
 * The receiver can then work on piece k while pieces k+1... are still arriving.
 * Sender and receiver compute the same split from the (same) block length. */
#define NPIECE 8
static int split(int elems, int *piece_elems)
{
    if (elems == 0) { *piece_elems = 1; return 0; }              /* empty block: nothing to send */
    *piece_elems = (elems + NPIECE - 1) / NPIECE;
    return (elems + *piece_elems - 1) / *piece_elems;          /* number of pieces */
}

/* Send n_send bytes from src to the right neighbour and receive n_recv bytes
 * from the left neighbour into dst. If acc != NULL, the received data is also
 * added into acc (reduce-scatter), piece by piece while the rest still arrives.
 * src_mr / dst_mr are the registered regions containing src / dst (rendezvous). */
static void ring_step(struct pg *pg, char *src, int n_send, struct ibv_mr *src_mr,
                      char *dst, int n_recv, struct ibv_mr *dst_mr,
                      char *acc, DATATYPE dt, OPERATION op)
{
    if (!pg->rdvz) {
        /* EAGER: copy into the bounce buffer and SEND; the receiver copies out of
         * the slot. A block bigger than one slot goes as several fragments. */
        int nfs = (n_send + SLOT_SIZE - 1) / SLOT_SIZE;       /* fragments we send    */
        int nfr = (n_recv + SLOT_SIZE - 1) / SLOT_SIZE;       /* fragments we receive */
        for (int k = 0; k < nfs || k < nfr; k++) {
            int off = k * SLOT_SIZE;
            if (k < nfs) {
                int len = n_send - off < SLOT_SIZE ? n_send - off : SLOT_SIZE;
                memcpy(pg->send_buf, src + off, len);
                post_send(pg->qp_right, IBV_WR_SEND, pg->send_buf, len, pg->send_mr->lkey, 0, 0);
            }
            if (k < nfr) {
                int len = n_recv - off < SLOT_SIZE ? n_recv - off : SLOT_SIZE;
                int i = wait_data(pg);
                memcpy(dst + off, pg->slots + i * SLOT_SIZE, len);
                post_recv(pg->qp_left, pg->slots + i * SLOT_SIZE, SLOT_SIZE, pg->slots_mr->lkey, i);
                if (acc) reduce(acc + off, dst + off, len / ESZ, dt, op);
            }
            if (k < nfs) wait_send(pg);
        }
    } else {
        /* RENDEZVOUS 1: tell the left neighbour where it should write. */
        pg->ctrl_send->addr = (uintptr_t)dst;
        pg->ctrl_send->rkey = dst_mr->rkey;
        post_send(pg->qp_left, IBV_WR_SEND, pg->ctrl_send, sizeof(struct ctrl),
                  pg->ctrl_send_mr->lkey, 0, 0);

        /* 2: wait until the right neighbour tells us where to write, then write
         *    our block in pieces straight from our buffer (no copy). All the
         *    writes are posted at once; the NIC sends them one after another. */
        int j = wait_ctrl(pg);
        struct ctrl ready = pg->ctrl_recv[j];
        post_recv(pg->qp_right, &pg->ctrl_recv[j], sizeof(struct ctrl),
                  pg->ctrl_recv_mr->lkey, NSLOT + j);

        int pe, np = split(n_send / ESZ, &pe);
        for (int k = 0; k < np; k++) {
            int from = k * pe, to = (k + 1) * pe;
            if (to > n_send / ESZ) to = n_send / ESZ;
            post_send(pg->qp_right, IBV_WR_RDMA_WRITE_WITH_IMM, src + (size_t)from * ESZ,
                      (to - from) * ESZ, src_mr->lkey, ready.addr + (size_t)from * ESZ, ready.rkey);
        }

        /* 3: each IMM completes one receive = "piece k is in dst". RC keeps order,
         *    so the k-th completion is piece k. Reduce it right away. */
        int qe, nq = split(n_recv / ESZ, &qe);
        for (int k = 0; k < nq; k++) {
            int i = wait_data(pg);
            post_recv(pg->qp_left, pg->slots + i * SLOT_SIZE, SLOT_SIZE, pg->slots_mr->lkey, i);
            int from = k * qe, to = (k + 1) * qe;
            if (to > n_recv / ESZ) to = n_recv / ESZ;
            if (acc) reduce(acc + (size_t)from * ESZ, dst + (size_t)from * ESZ, to - from, dt, op);
        }

        for (int k = 0; k < np + 1; k++) wait_send(pg);     /* ctrl message + all writes */
    }
}

/* ------------------------------------------------------------------------ */
/* The collectives                                                          */
/* ------------------------------------------------------------------------ */

/* Block b of a vector of `count` elements: the first count%P blocks have one extra. */
static void block(int count, int P, int b, int *off, int *len)
{
    int base = count / P, rem = count % P;
    *off = b * base + (b < rem ? b : rem);
    *len = base + (b < rem ? 1 : 0);
}

static int mod(int a, int P) { return ((a % P) + P) % P; }

/* Rendezvous when the biggest block is larger than the eager limit. All ranks
 * compute the same answer because count is the same everywhere. */
static int use_rendezvous(int count, int P)
{
    return ((count / P) + (count % P ? 1 : 0)) * ESZ > pg_eager_max;
}

/* Reduce-scatter phase, done IN PLACE on `work` (which holds this rank's vector).
 * Step s: send block r-1-s, receive block r-2-s and add it to our copy of it.
 * After P-1 steps block r of `work` is the full reduction. */
static void rs_phase(struct pg *pg, char *work, int count, DATATYPE dt, OPERATION op)
{
    int P = pg->size, r = pg->rank;
    int maxbytes = ((count / P) + (count % P ? 1 : 0)) * ESZ;
    char *staging = malloc(maxbytes + 1);               /* where the incoming block lands */
    struct ibv_mr *work_mr = NULL, *stage_mr = NULL;

    pg->rdvz = use_rendezvous(count, P);
    if (pg->rdvz) {
        work_mr  = reg(pg, work, (size_t)count * ESZ);
        stage_mr = reg(pg, staging, maxbytes);
    }
    for (int s = 0; s < P - 1; s++) {
        int so, sl, ro, rl;
        block(count, P, mod(r - 1 - s, P), &so, &sl);
        block(count, P, mod(r - 2 - s, P), &ro, &rl);
        ring_step(pg, work + (size_t)so * ESZ, sl * ESZ, work_mr, staging, rl * ESZ, stage_mr,
                  work + (size_t)ro * ESZ, dt, op);     /* received block is added into work */
    }
    if (pg->rdvz) { ibv_dereg_mr(work_mr); ibv_dereg_mr(stage_mr); }
    free(staging);
}

/* All-gather phase, IN PLACE on `buf`: block r of buf is already filled.
 * Step s: send block r-s, receive block r-1-s directly into its final place. */
static void ag_phase(struct pg *pg, char *buf, int count)
{
    int P = pg->size, r = pg->rank;
    struct ibv_mr *mr = NULL;

    pg->rdvz = use_rendezvous(count, P);
    if (pg->rdvz) mr = reg(pg, buf, (size_t)count * ESZ);   /* zero-copy: write into buf */
    for (int s = 0; s < P - 1; s++) {
        int so, sl, ro, rl;
        block(count, P, mod(r - s, P), &so, &sl);
        block(count, P, mod(r - 1 - s, P), &ro, &rl);
        ring_step(pg, buf + (size_t)so * ESZ, sl * ESZ, mr, buf + (size_t)ro * ESZ, rl * ESZ, mr,
                  NULL, PG_INT32, PG_SUM);              /* nothing to reduce */
    }
    if (pg->rdvz) ibv_dereg_mr(mr);
}

/* recvbuf gets block `rank` of the reduction (its length is the block's length). */
int pg_reduce_scatter(void *sendbuf, void *recvbuf, int count,
                      DATATYPE dt, OPERATION op, void *handle)
{
    struct pg *pg = handle;
    int off, len;
    char *work = malloc((size_t)count * ESZ + 1);       /* don't destroy sendbuf */
    memcpy(work, sendbuf, (size_t)count * ESZ);
    rs_phase(pg, work, count, dt, op);
    block(count, pg->size, pg->rank, &off, &len);
    memcpy(recvbuf, work + (size_t)off * ESZ, (size_t)len * ESZ);
    free(work);
    return 0;
}

/* sendbuf = this rank's block, recvbuf = the whole vector (count elements). */
int pg_all_gather(void *sendbuf, void *recvbuf, int count, DATATYPE dt, void *handle)
{
    struct pg *pg = handle;
    int off, len;
    (void)dt;
    block(count, pg->size, pg->rank, &off, &len);
    char *mine = (char *)recvbuf + (size_t)off * ESZ;
    if (sendbuf != mine) memcpy(mine, sendbuf, (size_t)len * ESZ);   /* may be in place */
    ag_phase(pg, recvbuf, count);
    return 0;
}

/* All-reduce = reduce-scatter followed by all-gather, both in place on recvbuf. */
int pg_all_reduce(void *sendbuf, void *recvbuf, int count,
                  DATATYPE dt, OPERATION op, void *handle)
{
    struct pg *pg = handle;
    if (sendbuf != recvbuf) memcpy(recvbuf, sendbuf, (size_t)count * ESZ);
    rs_phase(pg, recvbuf, count, dt, op);
    ag_phase(pg, recvbuf, count);
    return 0;
}

int pg_close(void *handle)
{
    struct pg *pg = handle;
    int x = 1, y;
    pg_all_reduce(&x, &y, 1, PG_INT32, PG_SUM, pg);     /* barrier: everybody is done */
    ibv_destroy_qp(pg->qp_right);
    ibv_destroy_qp(pg->qp_left);
    ibv_dereg_mr(pg->slots_mr);     ibv_dereg_mr(pg->send_mr);
    ibv_dereg_mr(pg->ctrl_recv_mr); ibv_dereg_mr(pg->ctrl_send_mr);
    ibv_destroy_cq(pg->cq);
    ibv_dealloc_pd(pg->pd);
    ibv_close_device(pg->ctx);
    free(pg->slots); free(pg->send_buf); free(pg->ctrl_recv); free(pg->ctrl_send);
    free(pg);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Test program:  test -myindex N -list host1 host2 ...                     */
/* One run checks correctness and then measures EAGER and RENDEZVOUS for    */
/* all three collectives, 4 B .. 32 MB. (-threshold is accepted but ignored)*/
/* ------------------------------------------------------------------------ */

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Rank r holds x[i] = r*1000 + i. Check all three collectives for this count. */
static int check(void *pg, int rank, int P, int count)
{
    int bad = 0, off, len;
    int *x = malloc(count * sizeof(int)), *y = malloc(count * sizeof(int));
    for (int i = 0; i < count; i++) x[i] = rank * 1000 + i;
    int sum0 = 1000 * P * (P - 1) / 2;                  /* sum of r*1000 over all ranks */

    pg_all_reduce(x, y, count, PG_INT32, PG_SUM, pg);
    for (int i = 0; i < count; i++) bad += (y[i] != sum0 + P * i);

    pg_all_reduce(x, y, count, PG_INT32, PG_MAX, pg);
    for (int i = 0; i < count; i++) bad += (y[i] != (P - 1) * 1000 + i);

    block(count, P, rank, &off, &len);
    pg_reduce_scatter(x, y, count, PG_INT32, PG_SUM, pg);
    for (int j = 0; j < len; j++) bad += (y[j] != sum0 + P * (off + j));

    pg_all_gather(x + off, y, count, PG_INT32, pg);     /* x+off = my own block */
    for (int b = 0; b < P; b++) {
        int bo, bl;
        block(count, P, b, &bo, &bl);
        for (int j = 0; j < bl; j++) bad += (y[bo + j] != b * 1000 + bo + j);
    }
    free(x); free(y);
    return bad;
}

static const char *op_name[] = { "reduce_scatter", "all_gather", "all_reduce" };

static void run_op(int op, void *pg, int *x, int *y, int count)
{
    if (op == 0)      pg_reduce_scatter(x, y, count, PG_INT32, PG_SUM, pg);
    else if (op == 1) pg_all_gather(x, y, count, PG_INT32, pg);   /* sends the first block-length of x */
    else              pg_all_reduce(x, y, count, PG_INT32, PG_SUM, pg);
}

/* average time of one collective in microseconds (all ranks run the same loop) */
static double time_op(int op, void *pg, int *x, int *y, int count)
{
    int bytes = count * 4;
    int iters = bytes <= 65536 ? 200 : (bytes <= (4 << 20) ? 30 : 10);
    run_op(op, pg, x, y, count);                        /* warm-up, also syncs the ranks */
    double t = now();
    for (int it = 0; it < iters; it++) run_op(op, pg, x, y, count);
    return (now() - t) / iters * 1e6;
}

int main(int argc, char **argv)
{
    int rank = 0, P = 0;
    char *hosts[64];
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-myindex") && i + 1 < argc) rank = atoi(argv[++i]) - 1;
        else if (!strcmp(argv[i], "-threshold") && i + 1 < argc) i++;   /* ignored: we run both */
        else if (!strcmp(argv[i], "-list"))
            while (i + 1 < argc && argv[i + 1][0] != '-') hosts[P++] = argv[++i];
    }
    if (P < 2 || rank < 0 || rank >= P) {
        fprintf(stderr, "usage: %s -myindex N -list host1 host2 ...\n", argv[0]);
        return 1;
    }

    void *pg;
    pg_set_ring(rank, P, 18515);
    connect_process_group(hosts[(rank + 1) % P], &pg);      /* right neighbour */

    /* pg_eager_max = 1<<30 -> every block eager,  0 -> every (non-empty) block rendezvous */
    int mode_threshold[2] = { 1 << 30, 0 };
    const char *mode_name[2] = { "eager", "rendezvous" };

    /* 1. correctness of both protocols (the big count needs several eager fragments) */
    int counts[] = { 1, 5, 1000, 100003 };
    for (int m = 0; m < 2; m++) {
        pg_eager_max = mode_threshold[m];
        for (int k = 0; k < 4; k++) {
            int bad = check(pg, rank, P, counts[k]);
            if (rank == 0) printf("check %-10s count %-7d %s\n", mode_name[m], counts[k], bad ? "WRONG" : "ok");
        }
    }

    /* 2. eager vs rendezvous, every collective, vector sizes 4 B .. 32 MB */
    for (int op = 0; op < 3; op++) {
        if (rank == 0)
            printf("\n== %s, P=%d ==\n%10s %12s %12s %10s %12s %12s\n", op_name[op], P,
                   "bytes", "eager_us", "rdvz_us", "rdvz/eager", "eager_Gbps", "rdvz_Gbps");
        for (int count = 1; count <= (8 << 20); count *= 2) {
            int *x = calloc(count, sizeof(int)), *y = calloc(count, sizeof(int));
            double t[2];
            for (int m = 0; m < 2; m++) {
                pg_eager_max = mode_threshold[m];
                t[m] = time_op(op, pg, x, y, count);
            }
            /* bus bandwidth = bytes each rank puts on the wire / time */
            double wire = (op == 2 ? 2.0 : 1.0) * (P - 1) / P * count * 4;
            if (rank == 0)
                printf("%10d %12.1f %12.1f %10.2f %12.2f %12.2f\n", count * 4, t[0], t[1],
                       t[1] / t[0], wire * 8 / t[0] / 1e3, wire * 8 / t[1] / 1e3);
            free(x); free(y);
        }
    }

    pg_close(pg);
    return 0;
}
