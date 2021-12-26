#ifndef SGDMODEL_H
#define SGDMODEL_H

#define DEFAULT_BLOCK_PAGE_NUM 8 // 1280 // 1024 // 512 //256 // 256 * 8KB = 2MB, 256 for higgs
#define DEFAULT_BUFFER_TUPLE_NUM 325000  //50000 // 9000000

#define DEFAULT_BATCH_SIZE  1
#define DEFAULT_ITER_NUM  20
#define DEFAULT_LEARNING_RATE	0.1 // 0.1 for higgs
#define DEFAULT_MODEL_NAME "LR"
#define DEFAULT_TABLE_NAME "yfcc_clustered" // "criteo" // "criteo_clustered" //"sample_criteo_clustered" //"splicesite_clustered" // "sample_splice_clustered"
#define DEFAULT_DECAY 0.95
#define DEFAULT_MU 0.00001 // 0.01 for higgs

#define DEFAULT_BLOCK_SHUFFLE 0 // 0 (w/o block-shuffle), 1 (with block-shuffle)
#define DEFAULT_TUPLE_SHUFFLE 0 // 0 (w/o tuple-shuffle), 1 (with tuple-shuffle using one buffer), 2 (with tuple-shuffle using two buffers)
#define DEFAULT_USE_MALLOC false

#define DEFAULT_IO_BIG_BLOCK_SIZE  800 // 1 page = 8K, default 10 pages
#define DEFAULT_BUFFER_SIZE  800 // default 100 pages = 800KB
#define DEFAULT_BUFFER_BLOCK_NUM 1.0

typedef struct Model {
    char* model_name;
    double total_loss;
	double* w;
    int batch_size;
    double learning_rate;
	double decay;
	double mu; // for regularization
    double n_features;
    int iter_num;
    int tuple_num;
	double accuracy;
	
	// for mini-batch SGD
	int current_batch_num;
    double* current_batch_gradient;
	
	// for mini-batch sparse data
	// double* w_old;
	int* feature_k_non_zeros;
	int feature_k_index;

} Model;

typedef struct SGDTupleDesc
{ 
	// e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
	// Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
	int k_col; // 1 // just for sparse dataset, if dense, only v_col is used.
	int v_col; // 2
	int label_col; // 3
	int n_features;  // 8

	int attr_num; // 3 for forest
	bool dense;
	int max_sparse_count;
} SGDTupleDesc;


typedef struct
{
	bool	 isnull;
	int*	 features_k;		/* features of a tuple, n_dim */	
	int		 k_len;
	double*  features_v;
    int		 class_label;	/* the class label of a tuple, -1 if there is not any label */

	int 	 sparse_array_len;
	// the following variable are not used
	//void	   *tuple;			/* the tuple proper */
	// can be changed to feature/label Datum
	//Datum		datum1;			/* value of first key column */
	//bool		isnull1;		/* is first key column NULL? */
	//int			tupindex;		/* see notes above */
	void* 	 v_array;
	void* 	 k_array;
	// for debug
	// int 		did;
} SortTuple;

// guc variables
// can be set via "SET VAR = XX" in the psql console
extern int set_io_big_block_size;
extern int set_block_page_num;
extern int set_buffer_size;
extern int set_buffer_tuple_num;
extern double set_buffer_block_num;

extern int set_batch_size;
extern int set_iter_num;
extern double set_learning_rate;
extern double set_decay;
extern double set_mu;

extern char* set_model_name;
extern char* set_table_name;

extern int table_page_number;

extern bool set_run_test;
extern bool set_use_malloc;

extern int set_block_shuffle;
extern int set_tuple_shuffle;

extern SGDTupleDesc* sgd_tupledesc;

extern bool is_training;










#endif   