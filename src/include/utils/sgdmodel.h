#ifndef SGDMODEL_H
#define SGDMODEL_H

#define DEFAULT_IO_BIG_BLOCK_SIZE  80 // 1 page = 8K, default 10 pages
#define DEFAULT_BUFFER_SIZE  800 // default 100 pages = 800KB
#define DEFAULT_BUFFER_TUPLE_NUM 1000
#define DEFAULT_BUFFER_BLOCK_NUM 1.0


#define DEFAULT_BATCH_SIZE  512
#define DEFAULT_ITER_NUM  10
#define DEFAULT_LEARNING_RATE	0.1

typedef struct Model {
    double total_loss;
	double* w;
    int batch_size;
    double learning_rate;
    double n_features;
    int iter_num;
    int tuple_num;
} Model;


// guc variables
// can be set via "SET VAR = XX" in the psql console
extern int set_io_big_block_size;
extern int set_buffer_size;
extern int set_buffer_tuple_num;
extern double set_buffer_block_num;

extern int set_batch_size;
extern int set_iter_num;
extern double set_learning_rate;




#endif   