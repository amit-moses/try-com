/* pg.h -- API of the simple ring collectives (implemented in ring_allreduce.c) */
#ifndef PG_H
#define PG_H

typedef enum { PG_INT32, PG_FLOAT } DATATYPE;   /* both are 4 bytes */
typedef enum { PG_SUM, PG_MAX } OPERATION;

/* Largest block (in bytes) that is sent with the eager protocol.
 * Bigger blocks use rendezvous. Default 16 KB (the test program overrides it). */
extern int pg_eager_max;

/* Tell the library who we are: rank (0..size-1), ring size, first TCP port.
 * Call this before connect_process_group(). */
void pg_set_ring(int rank, int size, int base_port);

/* servername = host name of the RIGHT neighbour (rank+1 mod size). */
int connect_process_group(char *servername, void **pg_handle);

/* count is always the number of elements of the WHOLE vector. */
int pg_reduce_scatter(void *sendbuf, void *recvbuf, int count,
                      DATATYPE datatype, OPERATION op, void *pg_handle);
int pg_all_gather(void *sendbuf, void *recvbuf, int count,
                  DATATYPE datatype, void *pg_handle);
int pg_all_reduce(void *sendbuf, void *recvbuf, int count,
                  DATATYPE datatype, OPERATION op, void *pg_handle);

int pg_close(void *pg_handle);

#endif
