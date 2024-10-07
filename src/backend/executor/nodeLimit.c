/*-------------------------------------------------------------------------
 *
 * nodeLimit.c
 *	  Routines to handle limiting of query results where appropriate
 *
 * Portions Copyright (c) 1996-2012, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/backend/executor/nodeLimit.c
 *
 *-------------------------------------------------------------------------
 */
/*
 * INTERFACE ROUTINES
 *		ExecLimit		- extract a limited range of tuples
 *		ExecInitLimit	- initialize node and subnodes..
 *		ExecEndLimit	- shutdown node and subnodes
 */

#include "postgres.h"

#include "executor/execdebug.h"
#include "executor/executor.h"
#include "executor/nodeLimit.h"
#include "nodes/nodeFuncs.h"
#include "utils/guc.h"
#include "access/tuptoaster.h"

#include "utils/array.h"
#include <sys/time.h>
#include "time.h"
#include "math.h"


char timeBuf[256];
// guc variables
// can be set via "SET VAR = XX" in the psql console
int set_batch_size = DEFAULT_BATCH_SIZE;
int set_iter_num = DEFAULT_ITER_NUM;
double set_learning_rate = DEFAULT_LEARNING_RATE;
double set_decay = DEFAULT_DECAY;
double set_mu = DEFAULT_MU;
char* set_model_name = DEFAULT_MODEL_NAME;
int table_page_number = 0;
int set_class_num = DEFAULT_CLASS_NUM;

// char* table_name = "dflife";
// char* set_table_name = "forest";
char* set_table_name = DEFAULT_TABLE_NAME;

bool set_run_test = false;
int set_block_shuffle = DEFAULT_BLOCK_SHUFFLE;
int set_tuple_shuffle = DEFAULT_TUPLE_SHUFFLE;
bool is_training = true;

bool set_use_malloc = DEFAULT_USE_MALLOC;

char* set_algo_name = DEFAULT_ALGO_NAME;
double* after_training_model = NULL;

// int set_use_test_buffer_num = DEFAULT_USE_TEST_BUFFER_NUM;

SGDTupleDesc* sgd_tupledesc; // also used in nodeSort.c for parsing tuple_slot to double* features

static Model* init_model(int n_features, int max_sparse_count);
static void ExecFreeModel(Model* model);
// static SGDBatchState* init_SGDBatchState(int n_features);
// static SGDTuple* init_SGDTuple(int n_features);
static SortTuple* init_SortTuple(int n_features);
static SGDTupleDesc* init_SGDTupleDesc(int n_features, bool dense, int max_sparse_count);
// static void clear_SGDBatchState(SGDBatchState* batchstate, int n_features);
// static void free_SGDBatchState(SGDBatchState* batchstate);
//static void free_SGDTuple(SGDTuple* sgd_tuple);
static void free_SortTuple(SortTuple* sort_tuple);
static void free_SGDTupleDesc(SGDTupleDesc* sgd_tupledesc);
static char* get_current_time();
// static void compute_tuple_gradient_loss_LR(SGDTuple* tp, Model* model, SGDBatchState* batchstate);
// static void compute_tuple_gradient_loss_SVM(SGDTuple* tp, Model* model, SGDBatchState* batchstate);

// static void compute_tuple_accuracy(Model* model, SGDTuple* tp, TestState* test_state);
// static void update_model(Model* model, SGDBatchState* batchstate);
// static void perform_SGD(Model *model, SGDTuple* sgd_tuple, SGDBatchState* batchstate, int i);
// static void transfer_slot_to_sgd_tuple(TupleTableSlot* slot, SGDTuple* sgd_tuple, SGDTupleDesc* sgd_tupledesc);
//static void fast_transfer_slot_to_sgd_tuple(TupleTableSlot* slot, SGDTuple* sgd_tuple, SGDTupleDesc* sgd_tupledesc);
// static void transfer_slot_to_sgd_tuple_getattr(TupleTableSlot* slot, SGDTuple* sgd_tuple, SGDTupleDesc* sgd_tupledesc);

// static int my_parse_array_no_copy(struct varlena* input, int typesize, char** output);


// static void compute_tuple_loss_LR(SortTuple* tp, Model* model, SGDBatchState* batchstate);
// static void compute_tuple_gradient_LR(SortTuple* tp, Model* model, SGDBatchState* batchstate);

// for selecting the gradient and loss computation function
void (*compute_tuple_gradient)(SortTuple* stup, Model* model) = NULL;
void (*compute_tuple_loss)(SortTuple* stup, Model* model) = NULL;


static char* get_current_time()
{
    time_t t = time(0);
    strftime(timeBuf, 256, "%Y-%m-%d %H:%M:%S", localtime(&t)); //format date and time.
    return timeBuf;
}

double diff_timeofday_seconds(struct timeval* start, struct timeval* end)
{
    double time_use = (end->tv_sec - start->tv_sec) * 1000 + (end->tv_usec - start->tv_usec) / 1000; //微秒
    return time_use / 1000; //seconds
}


// static char* get_current_time() {
// 	time_t t;
//     time(&t);
//     ctime_r(&t, timeBuf);
// 	return timeBuf;
// }

// from bismarck
double
dot(const double* x, const double* y, const int size)
{
    double ret = 0.0;
    int i;
    for (i = size - 1; i >= 0; i--)
    {
        ret += x[i] * y[i];
    }
    return ret;
}


// for softmax regression
double
softmax_dot(const double* w, const int j, const double* x, const int n, const int K)
{
    double ret = 0.0;
    int i;
    for (i = 0; i < n; i++)
    {
        double wji = w[K * i + j];
        ret += wji * x[i];
    }
    return ret;
}

// from bismarck
double
dot_dss(const double* x, const int* k, const double* v, const int sparseSize)
{
    double ret = 0.0;
    int i;
    for (i = sparseSize - 1; i >= 0; i--)
    {
        ret += x[k[i]] * v[i];
    }
    return ret;
}

// from bismarck
void
add_and_scale(double* x, const int size, const double* y, const double c)
{
    int i;
    for (i = size - 1; i >= 0; i--)
    {
        x[i] += y[i] * c;
    }
}

// for softmax regression
void
softmax_add_and_scale(double* w, const int j, const int n, const double* x, const double c, const int K)
{
    int i;
    for (i = 0; i < n; i++)
    {
        // wjx += x[i] * c
        int index = K * i + j;
        w[index] += x[i] * c;
    }
}

void
batch_softmax_add_and_scale(double* w, const int n, const double* batch_w, const double c, const int K)
{
    int i;
    for (i = 0; i < n * K; i++)
    {
        w[i] += batch_w[i] * c;
    }
}

void
add_c_dss(double* x, const int* k, const int sparseSize, const double c)
{
    int i;
    for (i = sparseSize - 1; i >= 0; i--)
    {
        x[k[i]] += c;
    }
}

// from bismarck
void
add_and_scale_dss(double* x, const int* k, const double* v, const int sparseSize, const double c)
{
    int i;
    for (i = sparseSize - 1; i >= 0; i--)
    {
        x[k[i]] += v[i] * c;
    }
}

// from bismarck
void
scale_i(double* x, const int size, const double c)
{
    int i;
    for (i = size - 1; i >= 0; i--)
    {
        x[i] *= c;
    }
}

// from bismarck
double
norm(const double* x, const int size)
{
    double norm = 0;
    int i;
    for (i = size - 1; i >= 0; i--)
    {
        norm += x[i] * x[i];
    }
    return norm;
}

// from bismarck
double
sigma(const double v)
{
    if (v > 30) { return 1.0 / (1.0 + exp(-v)); }
    else { return exp(v) / (1.0 + exp(v)); }
}

// from bismarck
void
l1_shrink_mask(double* x, const double u, const int* k, const int sparseSize)
{
    int i;
    for (i = sparseSize - 1; i >= 0; i--)
    {
        if (x[k[i]] > u) { x[k[i]] -= u; }
        else if (x[k[i]] < -u) { x[k[i]] += u; }
        else { x[k[i]] = 0; }
    }
}

void
l2_shrink_mask_d(double* x, const double u, const int size)
{
    int i;
    for (i = size - 1; i >= 0; i--)
    {
        if (x[i] == 0.0) { continue; }
        x[i] /= 1 + u;
    }
}

// from bismarck
void
l1_shrink_mask_d(double* x, const double u, const int size)
{
    int i;
    double xi = 0.0;
    for (i = size - 1; i >= 0; i--)
    {
        xi = x[i];
        if (xi > u) { x[i] -= u; }
        else if (xi < -u) { x[i] += u; }
        else { x[i] = 0.0; }
    }
}


// from bismarck
double
log_sum(const double a, const double b)
{
    return a + log(1.0 + exp(b - a));
}


void
my_sparse_add_and_scale_dss(double* w, double* current_batch_gradient, const double c, Model* model)
{
    // e.g., model->feature_k_non_zeros = [1, 2, 3; 2, 3, 4; 1, 4, 6], model->feature_k_index = 9
    int i;
    for (i = 0; i < model->feature_k_index; i++)
    {
        int index = model->feature_k_non_zeros[i];
        w[index] += current_batch_gradient[index] * c;
        current_batch_gradient[index] = 0;
        model->feature_k_non_zeros[i] = 0;
    }

    model->feature_k_index = 0;
}


Model* init_model(int n_features, int max_sparse_count)
{
    Model* model = (Model*)palloc(sizeof(Model));

    model->model_name = set_model_name;
    model->total_loss = 0;
    model->batch_size = set_batch_size;
    model->iter_num = set_iter_num;
    model->learning_rate = set_learning_rate;
    model->tuple_num = 0;
    model->n_features = n_features;
    model->decay = set_decay;
    model->mu = set_mu;
    model->class_num = set_class_num;

    model->accuracy = 0;
    model->tp = 0;
    model->fp = 0;
    model->fn = 0;

    model->current_batch_num = 0;
    // use memorycontext later
    if (model->class_num > 2)
    {
        model->w = (double*)palloc0(sizeof(double) * n_features * model->class_num);
        model->current_batch_gradient = (double*)palloc0(sizeof(double) * n_features * model->class_num);
    }
    else
    {
        model->w = (double*)palloc0(sizeof(double) * n_features);
        if (set_batch_size > 1)
            model->current_batch_gradient = (double*)palloc0(sizeof(double) * n_features);
        if (strcmp(set_algo_name, "ada") == 0 || strcmp(set_algo_name, "ADA") == 0)
            model->adagrad = (double*)palloc0(sizeof(double) * n_features);
    }
    //memset(model->w, 0, sizeof(double) * n_features);


    // for mini-batch on sparse data
    // model->w_old = (double *) palloc0(sizeof(double) * n_features);
    if (set_batch_size > 1)
        model->feature_k_non_zeros = (int*)palloc0(sizeof(int) * max_sparse_count * set_batch_size);
    model->feature_k_index = 0;

    return model;
}

void ExecFreeModel(Model* model)
{
    // free(model->gradient);
    pfree(model->w);
    //pfree(model->w_old);
    if (set_batch_size > 1)
    {
        pfree(model->current_batch_gradient);
        pfree(model->feature_k_non_zeros);
    }
    if (strcmp(set_algo_name, "ada") == 0 || strcmp(set_algo_name, "ADA") == 0)
        pfree(model->adagrad);
    pfree(model);
}

static SortTuple* init_SortTuple(int n_features)
{
    SortTuple* sgd_tuple = (SortTuple*)palloc(sizeof(SortTuple));
    //sgd_tuple->features = (double *) palloc0(sizeof(double) * n_features);
    return sgd_tuple;
}

static void free_SortTuple(SortTuple* sort_tuple)
{
    pfree(sort_tuple);
}

static SGDTupleDesc* init_SGDTupleDesc(int n_features, bool dense, int max_sparse_count)
{
    SGDTupleDesc* sgd_tupledesc = (SGDTupleDesc*)palloc(sizeof(SGDTupleDesc));

    // sgd_tupledesc->values = (Datum *) palloc0(sizeof(Datum) * col_num);
    // sgd_tupledesc->isnulls = (bool *) palloc0(sizeof(bool) * col_num);

    // just for dblife:
    /*
    CREATE TABLE dblife (
    did serial,
    k integer[],
    v double precision[],
    label integer);
    */
    if (dense == false)
    {
        /* for dblife */
        sgd_tupledesc->k_col = 1;
        sgd_tupledesc->v_col = 2;
        sgd_tupledesc->label_col = 3;
        sgd_tupledesc->attr_num = 4;
        sgd_tupledesc->max_sparse_count = max_sparse_count;
    }
    else
    {
        /* for forest */
        sgd_tupledesc->k_col = -1; // from 0
        sgd_tupledesc->v_col = 1;
        sgd_tupledesc->label_col = 2;
        sgd_tupledesc->attr_num = 3;
    }

    sgd_tupledesc->n_features = n_features;
    sgd_tupledesc->dense = dense;
    return sgd_tupledesc;
}


static void free_SGDTupleDesc(SGDTupleDesc* sgd_tupledesc)
{
    // pfree(sgd_tupledesc->values);
    // pfree(sgd_tupledesc->isnulls);
    pfree(sgd_tupledesc);
}

/*
static void free_TestState(TestState* test_state) {
    pfree(test_state);
}
*/

void
compute_dense_tuple_gradient_LR(SortTuple* tp, Model* model)
{
    int y = tp->class_label;
    double* x = tp->features_v;

    int n = model->n_features;

    // compute gradients of the incoming tuple
    double wx = dot(model->w, x, n);
    double sig = sigma(-wx * y);

    double c = model->learning_rate * y * sig; // scale factor
    add_and_scale(model->w, n, x, c);

    // regularization
    double u = model->mu * model->learning_rate;
    // l2_shrink_mask_d(model->w, u, n);
    l1_shrink_mask_d(model->w, u, n);
}


void
batch_compute_dense_tuple_gradient_LR(SortTuple* tp, Model* model)
{
    int n = model->n_features;
    if (tp == NULL)
    {
        if (model->current_batch_num > 0)
            add_and_scale(model->w, n, model->current_batch_gradient, 1.0 / model->current_batch_num);
        memset(model->current_batch_gradient, 0, sizeof(double) * n);
        model->current_batch_num = 0;
        return;
    }

    int y = tp->class_label;
    double* x = tp->features_v;

    // compute gradients of the incoming tuple
    double wx = dot(model->w, x, n);
    double sig = sigma(-wx * y);

    double c = model->learning_rate * sig * y; // scale factor
    // add_and_scale(model->w, n, x, c);
    add_and_scale(model->current_batch_gradient, n, x, c);

    model->current_batch_num += 1;

    if (model->current_batch_num == model->batch_size)
    {
        add_and_scale(model->w, n, model->current_batch_gradient, 1.0 / model->current_batch_num);
        memset(model->current_batch_gradient, 0, sizeof(double) * n);
        model->current_batch_num = 0;
    }

    // regularization
    double u = model->mu * model->learning_rate;
    // l2_shrink_mask_d(model->w, u, n);
    l1_shrink_mask_d(model->w, u, n);
}

void
compute_dense_tuple_adagrad_LR(SortTuple* tp, Model* model)
{
    int y = tp->class_label;
    double* x = tp->features_v;

    int n = model->n_features;

    // compute gradients of the incoming tuple
    double wx = dot(model->w, x, n);
    double sig = sigma(-wx * y);

    double c = model->learning_rate * y * sig; // scale factor
    for (int i = 0; i < n; ++i)
    {
        double grad = x[i] * c; // 计算当前维度的梯度
        model->w[i] += grad / sqrt(model->adagrad[i] + 1e-8); // 更新权重向量
        model->adagrad[i] += grad * grad; // 累积梯度平方
    }

    // regularization
    double u = model->mu * model->learning_rate;
    // l2_shrink_mask_d(model->w, u, n);
    l1_shrink_mask_d(model->w, u, n);
}

void
batch_compute_dense_tuple_adagrad_LR(SortTuple* tp, Model* model)
{
    int n = model->n_features;
    if (tp == NULL)
    {
        if (model->current_batch_num > 0)
        {
            for (int i = 0; i < n; ++i)
            {
                double grad = model->current_batch_gradient[i] / model->current_batch_num;
                model->w[i] += grad / sqrt(model->adagrad[i] + 1e-8);
                model->adagrad[i] += grad * grad;
            }
        }
        memset(model->current_batch_gradient, 0, sizeof(double) * n);
        model->current_batch_num = 0;
        return;
    }

    int y = tp->class_label;
    double* x = tp->features_v;

    // compute gradients of the incoming tuple
    double wx = dot(model->w, x, n);
    double sig = sigma(-wx * y);

    double c = model->learning_rate * sig * y; // scale factor
    add_and_scale(model->current_batch_gradient, n, x, c);

    model->current_batch_num += 1;

    if (model->current_batch_num == model->batch_size)
    {
        for (int i = 0; i < n; ++i)
        {
            double grad = model->current_batch_gradient[i] / model->current_batch_num;
            model->w[i] += grad / sqrt(model->adagrad[i] + 1e-8);
            model->adagrad[i] += grad * grad;
        }
        memset(model->current_batch_gradient, 0, sizeof(double) * n);
        model->current_batch_num = 0;
    }

    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask_d(model->w, u, n);
}

void
compute_sparse_tuple_gradient_LR(SortTuple* tp, Model* model)
{
    int y = tp->class_label;
    int* k = tp->features_k;
    int k_len = tp->k_len;
    double* v = tp->features_v;
    double* w = model->w;

    // grad
    double wx = dot_dss(w, k, v, k_len);
    double sig = sigma(-wx * y);
    double c = model->learning_rate * y * sig; // scale factor
    add_and_scale_dss(w, k, v, k_len, c);
    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask(w, u, k, k_len);
}

void
batch_compute_sparse_tuple_gradient_LR(SortTuple* tp, Model* model)
{
    int n = model->n_features;
    if (tp == NULL)
    {
        if (model->current_batch_num > 0)
            my_sparse_add_and_scale_dss(model->w, model->current_batch_gradient,
                                        model->learning_rate / model->current_batch_num, model);
        model->current_batch_num = 0;
        return;
    }

    int y = tp->class_label;
    int* k = tp->features_k;
    int k_len = tp->k_len;
    double* v = tp->features_v;
    double* w = model->w;

    // grad
    double wx = dot_dss(w, k, v, k_len);
    double sig = sigma(-wx * y);
    double c = sig * y; // scale factor

    add_and_scale_dss(model->current_batch_gradient, k, v, k_len, c);

    // e.g., model->feature_k_non_zeros = [1, 2, 3; 2, 3, 4; 1, 4, 6]
    int i;
    for (i = 0; i < k_len; i++)
    {
        model->feature_k_non_zeros[model->feature_k_index] = k[i];
        model->feature_k_index += 1;
    }

    model->current_batch_num += 1;

    if (model->current_batch_num == model->batch_size)
    {
        my_sparse_add_and_scale_dss(model->w, model->current_batch_gradient, model->learning_rate / model->batch_size,
                                    model);
        model->current_batch_num = 0;
    }

    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask(w, u, k, k_len);
}

void
compute_sparse_tuple_adagrad_LR(SortTuple* tp, Model* model)
{
    int y = tp->class_label;
    int* k = tp->features_k;
    int k_len = tp->k_len;
    double* v = tp->features_v;
    double* w = model->w;

    // grad
    double wx = dot_dss(w, k, v, k_len);
    double sig = sigma(-wx * y);
    double c = model->learning_rate * y * sig; // scale factor
    for (int idx = k_len - 1; idx >= 0; --idx)
    {
        int i = k[idx];
        double grad = v[idx] * c; // 计算当前维度的梯度
        model->w[i] += grad / sqrt(model->adagrad[i] + 1e-8); // 更新权重向量
        model->adagrad[i] += grad * grad; // 累积梯度平方
    }

    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask(w, u, k, k_len);
}

void
batch_compute_sparse_tuple_adagrad_LR(SortTuple* tp, Model* model)
{
    int n = model->n_features;
    if (tp == NULL)
    {
        if (model->current_batch_num > 0)
        {
            for (int idx = model->feature_k_index; idx >= 0; --idx)
            {
                int i = model->feature_k_non_zeros[idx];
                double grad = model->current_batch_gradient[i] / model->current_batch_num;
                model->w[i] += grad / sqrt(model->adagrad[i] + 1e-8);
                model->adagrad[i] += grad * grad;
            }
        }
        memset(model->current_batch_gradient, 0, sizeof(double) * n);
        model->current_batch_num = 0;
        model->feature_k_index = 0;
        return;
    }

    int y = tp->class_label;
    int* k = tp->features_k;
    int k_len = tp->k_len;
    double* v = tp->features_v;
    double* w = model->w;

    // grad
    double wx = dot_dss(w, k, v, k_len);
    double sig = sigma(-wx * y);
    double c = sig * y; // scale factor

    add_and_scale_dss(model->current_batch_gradient, k, v, k_len, c);

    // e.g., model->feature_k_non_zeros = [1, 2, 3; 2, 3, 4; 1, 4, 6]
    int i;
    for (i = 0; i < k_len; i++)
    {
        model->feature_k_non_zeros[model->feature_k_index] = k[i];
        model->feature_k_index += 1;
    }

    model->current_batch_num += 1;

    if (model->current_batch_num == model->batch_size)
    {
        for (int idx = model->feature_k_index; idx >= 0; --idx)
        {
            int i = model->feature_k_non_zeros[idx];
            double grad = model->current_batch_gradient[i] / model->current_batch_num;
            model->w[i] += grad / sqrt(model->adagrad[i] + 1e-8);
            model->adagrad[i] += grad * grad;
        }
        memset(model->current_batch_gradient, 0, sizeof(double) * n);
        model->current_batch_num = 0;
        model->feature_k_index = 0;
    }

    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask(w, u, k, k_len);
}

void
compute_dense_tuple_loss_LR(SortTuple* tp, Model* model)
{
    // double* x = tp->features_v;
    int y = tp->class_label;

    double wx;
    if (set_run_test)
        wx = dot(after_training_model, tp->features_v, model->n_features);
    else
        wx = dot(model->w, tp->features_v, model->n_features);;

    double _ywx = -y * wx;
    if (_ywx >= 35)
        model->total_loss += _ywx;
    else
        model->total_loss += log(1 + exp(_ywx));


    // By default, if f(wx) > 0.5, the outcome is positive, or negative otherwise
    double f_wx = sigma(wx);
    if (f_wx >= 0.5 && y == 1)
    {
        model->accuracy += 1;
        model->tp += 1;
    }
    else if (f_wx < 0.5 && y == -1)
    {
        model->accuracy += 1;
    }
    else if (f_wx >= 0.5 && y == -1)
    {
        model->fp += 1;
    }
    else
    {
        model->fn += 1;
    }
}


void
compute_sparse_tuple_loss_LR(SortTuple* tp, Model* model)
{
    int y = tp->class_label;
    double wx;
    if (set_run_test)
        wx = dot_dss(after_training_model, tp->features_k, tp->features_v, tp->k_len);
    else
        wx = dot_dss(model->w, tp->features_k, tp->features_v, tp->k_len);

    double _ywx = -y * wx;
    if (_ywx >= 35)
        model->total_loss += _ywx;
    else
        model->total_loss += log(1 + exp(_ywx));

    // By default, if f(wx) > 0.5, the outcome is positive, or negative otherwise
    double f_wx = sigma(wx);
    if (f_wx >= 0.5 && y == 1)
    {
        model->accuracy += 1;
        model->tp += 1;
    }
    else if (f_wx < 0.5 && y == -1)
    {
        model->accuracy += 1;
    }
    else if (f_wx >= 0.5 && y == -1)
    {
        model->fp += 1;
    }
    else
    {
        model->fn += 1;
    }
}


void
compute_dense_tuple_gradient_SVM(SortTuple* tp, Model* model)
{
    int y = tp->class_label;
    double* x = tp->features_v;
    int n = model->n_features;

    double wx = dot(model->w, x, n);
    double c = model->learning_rate * y;
    // writes
    if (1 - y * wx > 0)
    {
        add_and_scale(model->w, n, x, c);
    }
    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask_d(model->w, u, n);
}

void
batch_compute_dense_tuple_gradient_SVM(SortTuple* tp, Model* model)
{
    int n = model->n_features;
    if (tp == NULL)
    {
        if (model->current_batch_num > 0)
            add_and_scale(model->w, n, model->current_batch_gradient, 1.0 / model->current_batch_num);
        memset(model->current_batch_gradient, 0, sizeof(double) * n);
        model->current_batch_num = 0;
        return;
    }

    int y = tp->class_label;
    double* x = tp->features_v;


    double wx = dot(model->w, x, n);
    double c = model->learning_rate * y;
    // writes
    if (1 - y * wx > 0)
    {
        add_and_scale(model->current_batch_gradient, n, x, c);
    }

    model->current_batch_num += 1;

    if (model->current_batch_num == model->batch_size)
    {
        add_and_scale(model->w, n, model->current_batch_gradient, 1.0 / model->current_batch_num);
        memset(model->current_batch_gradient, 0, sizeof(double) * n);
        model->current_batch_num = 0;
    }

    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask_d(model->w, u, n);
}


void
compute_sparse_tuple_gradient_SVM(SortTuple* tp, Model* model)
{
    int y = tp->class_label;
    int* k = tp->features_k;
    int k_len = tp->k_len;
    double* v = tp->features_v;

    // read and prepare
    double wx = dot_dss(model->w, k, v, k_len);
    double c = model->learning_rate * y;
    // writes
    if (1 - y * wx > 0)
    {
        add_and_scale_dss(model->w, k, v, k_len, c);
    }
    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask(model->w, u, k, k_len);
}


void
batch_compute_sparse_tuple_gradient_SVM(SortTuple* tp, Model* model)
{
    int n = model->n_features;
    if (tp == NULL)
    {
        if (model->current_batch_num > 0)
            my_sparse_add_and_scale_dss(model->w, model->current_batch_gradient,
                                        model->learning_rate / model->current_batch_num, model);
        model->current_batch_num = 0;
        return;
    }

    int y = tp->class_label;
    int* k = tp->features_k;
    int k_len = tp->k_len;
    double* v = tp->features_v;

    // read and prepare
    double wx = dot_dss(model->w, k, v, k_len);
    double c = y;
    // writes
    if (1 - y * wx > 0)
    {
        add_and_scale_dss(model->current_batch_gradient, k, v, k_len, c);

        // e.g., model->feature_k_non_zeros = [1, 2, 3; 2, 3, 4; 1, 4, 6]
        int i;
        for (i = 0; i < k_len; i++)
        {
            model->feature_k_non_zeros[model->feature_k_index] = k[i];
            model->feature_k_index += 1;
        }
    }

    model->current_batch_num += 1;

    if (model->current_batch_num == model->batch_size)
    {
        my_sparse_add_and_scale_dss(model->w, model->current_batch_gradient, model->learning_rate / model->batch_size,
                                    model);
        model->current_batch_num = 0;
    }

    // regularization
    double u = model->mu * model->learning_rate;
    l1_shrink_mask(model->w, u, k, k_len);
}


void
compute_dense_tuple_loss_SVM(SortTuple* tp, Model* model)
{
    int y = tp->class_label;

    double wx = dot(model->w, tp->features_v, model->n_features);
    double loss = 1 - y * wx;
    model->total_loss += (loss > 0) ? loss : 0;

    //  if wx >= 0 then the outcome is positive, and negative otherwise.
    if (wx >= 0 && y == 1)
    {
        model->accuracy += 1;
    }
    else if (wx < 0 && y == -1)
    {
        model->accuracy += 1;
    }
}

void
compute_sparse_tuple_loss_SVM(SortTuple* tp, Model* model)
{
    int y = tp->class_label;
    double wx = dot_dss(model->w, tp->features_k, tp->features_v, tp->k_len);
    double loss = 1 - y * wx;
    model->total_loss += (loss > 0) ? loss : 0;

    //  if wx >= 0 then the outcome is positive, and negative otherwise.
    if (wx >= 0 && y == 1)
    {
        model->accuracy += 1;
    }
    else if (wx < 0 && y == -1)
    {
        model->accuracy += 1;
    }
}


// for softmax regression

void
compute_dense_tuple_gradient_Softmax(SortTuple* tp, Model* model)
{
    int y = tp->class_label; // 0， 1， 2， 3 (from 0), here K = 4
    double* x = tp->features_v;

    int n = model->n_features;
    int K = model->class_num;

    double gradient[K];
    double sum = 0.0;

    // wx_K_1 = np.exp(np.transpose(self.w).dot(x))
    int j;
    for (j = 0; j < K; j++)
    {
        double wjx = softmax_dot(model->w, j, x, n, K);
        gradient[j] = exp(wjx);
        sum += gradient[j];
    }

    for (j = 0; j < K; j++)
    {
        double c;
        if (j == y)
            c = model->learning_rate * (1.0 - gradient[j] / sum);
        else
            c = -1.0 * model->learning_rate * gradient[j] / sum;
        softmax_add_and_scale(model->w, j, n, x, c, K);

        // regularization
        // double u = model->mu * model->learning_rate;
        // // l2_shrink_mask_d(model->w, u, n);
        // l1_shrink_mask_d(model->w, u, n);
    }
}


void
batch_compute_dense_tuple_gradient_Softmax(SortTuple* tp, Model* model)
{
    int n = model->n_features;
    int K = model->class_num;

    if (tp == NULL)
    {
        if (model->current_batch_num > 0)
        {
            int j;
            for (j = 0; j < K; j++)
                batch_softmax_add_and_scale(model->w, n, model->current_batch_gradient, 1.0 / model->current_batch_num,
                                            K);
        }

        memset(model->current_batch_gradient, 0, sizeof(double) * n * K);
        model->current_batch_num = 0;
        return;
    }

    int y = tp->class_label; // 0， 1， 2， 3 (from 0), here K = 4
    double* x = tp->features_v;


    double gradient[K];
    double sum = 0.0;

    // wx_K_1 = np.exp(np.transpose(self.w).dot(x))
    int j;
    for (j = 0; j < K; j++)
    {
        double wjx = softmax_dot(model->w, j, x, n, K);
        gradient[j] = exp(wjx);
        sum += gradient[j];
    }

    for (j = 0; j < K; j++)
    {
        double c;
        if (j == y)
            c = model->learning_rate * (1.0 - gradient[j] / sum);
        else
            c = -1.0 * model->learning_rate * gradient[j] / sum;
        softmax_add_and_scale(model->current_batch_gradient, j, n, x, c, K);

        // regularization
        // double u = model->mu * model->learning_rate;
        // // l2_shrink_mask_d(model->w, u, n);
        // l1_shrink_mask_d(model->w, u, n);
    }

    model->current_batch_num += 1;

    if (model->current_batch_num == model->batch_size)
    {
        for (j = 0; j < K; j++)
            batch_softmax_add_and_scale(model->w, n, model->current_batch_gradient, 1.0 / model->batch_size, K);
        memset(model->current_batch_gradient, 0, sizeof(double) * n * K);
        model->current_batch_num = 0;
    }
}


void
compute_dense_tuple_loss_Softmax(SortTuple* tp, Model* model)
{
    // double* x = tp->features_v;
    int y = tp->class_label;
    double* x = tp->features_v;

    int n = model->n_features;
    int K = model->class_num;

    double gradient[K];
    double sum = 0.0;

    double top1 = 0.0;
    int top1_label = 0;

    // wx_K_1 = np.exp(np.transpose(self.w).dot(x))
    int j;
    for (j = 0; j < K; j++)
    {
        double wjx = softmax_dot(model->w, j, x, n, K);
        gradient[j] = exp(wjx);
        sum += gradient[j];

        if (gradient[j] > top1)
        {
            top1_label = j;
            top1 = gradient[j];
        }
    }

    if (top1_label == y)
        model->accuracy += 1;

    double tuple_loss = -1.0 * log(gradient[y] / sum);
    model->total_loss += tuple_loss;
}

void
fast_transfer_slot_to_sgd_tuple(
    TupleTableSlot* slot,
    SortTuple* sgd_tuple)
{
    // store the values of slot to values/isnulls arrays
    int k_col = sgd_tupledesc->k_col;
    int v_col = sgd_tupledesc->v_col;
    int label_col = sgd_tupledesc->label_col;

    // int attnum = HeapTupleHeaderGetNatts(slot->tts_tuple->t_data);
    slot_deform_tuple(slot, sgd_tupledesc->attr_num);


    // e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
    // Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
    Datum v_dat = slot->tts_values[v_col];
    ArrayType* v_array = DatumGetArrayTypeP(v_dat); // Datum{0.1, 0.2, 0.3}

    int v_num = ArrayGetNItems(ARR_NDIM(v_array), ARR_DIMS(v_array));
    double* v = (double*)ARR_DATA_PTR(v_array);

    // double *v;
    // int v_num = my_parse_array_no_copy((struct varlena*) v_array,
    //         sizeof(float8), (char **) &v);

    Datum label_dat = slot->tts_values[label_col];
    sgd_tuple->class_label = DatumGetInt32(label_dat);
    sgd_tuple->features_v = v;

    sgd_tuple->k_array = NULL;
    sgd_tuple->v_array = NULL;
    // double* v => double* features
    //int n_features = sgd_tupledesc->n_features;
    // if sparse dataset
    if (k_col >= 0)
    {
        // k Datum array => int* k
        Datum k_dat = slot->tts_values[k_col];
        ArrayType* k_array = DatumGetArrayTypeP(k_dat);

        int k_num = ArrayGetNItems(ARR_NDIM(k_array), ARR_DIMS(k_array));
        int* k = (double*)ARR_DATA_PTR(k_array);

        // int *k;
        // int k_num = my_parse_array_no_copy((struct varlena*) k_array,
        //     	sizeof(int), (char **) &k);
        sgd_tuple->features_k = k;
        sgd_tuple->k_len = k_num;

        if VARATT_IS_EXTENDED((struct varlena *) DatumGetPointer(k_dat))
            sgd_tuple->k_array = (void*)k_array;
    }

    if VARATT_IS_EXTENDED((struct varlena *) DatumGetPointer(v_dat))
        sgd_tuple->v_array = (void*)v_array;
}


/* ----------------------------------------------------------------
 *		ExecLimit
 *
 *		This is a very simple node which just performs LIMIT/OFFSET
 *		filtering on the stream of tuples returned by a subplan.
 * ----------------------------------------------------------------
 */
inline
void train_with_shuffled_buffer(PlanState* outerNode, Model* model, int iter)
{
    bool end_of_reach = false;

    while (true)
    {
        // get a tuple from ShuffleSortNode
        TupleTableSlot* slot = ExecProcNode(outerNode);

        SortTuple* read_buffer = slot->read_buffer;
        int buffer_size = slot->read_buffer_size;
        int* read_buf_indexes = slot->read_buf_indexes;

        int j;

        for (j = 0; j < buffer_size; ++j)
        {
            if (read_buffer[read_buf_indexes[j]].isnull)
            {
                if (model->batch_size > 1)
                    compute_tuple_gradient(NULL, model);

                if (iter == 1)
                {
                    double avg_page_tuple_num = (double)model->tuple_num / table_page_number;
                    elog(INFO, "[%s] [ d Param] table_tuple_num = %d, buffer_block_num = %.2f",
                         get_current_time(), model->tuple_num,
                         (double)set_buffer_tuple_num / (set_block_page_num * avg_page_tuple_num));
                }

                end_of_reach = true;
                ExecReScan(outerNode);
                break;
            }

            compute_tuple_gradient(&read_buffer[read_buf_indexes[j]], model);

            if (iter == 1)
                model->tuple_num += 1;
        }

        if (end_of_reach)
            break;
    }
}

inline
void train_with_unshuffled_buffer(PlanState* outerNode, Model* model, int iter)
{
    bool end_of_reach = false;

    while (true)
    {
        // get a tuple from ShuffleSortNode
        TupleTableSlot* slot = ExecProcNode(outerNode);

        SortTuple* read_buffer = slot->read_buffer;
        int buffer_size = slot->read_buffer_size;

        int j;
        for (j = 0; j < buffer_size; ++j)
        {
            if (read_buffer[j].isnull)
            {
                if (model->batch_size > 1)
                {
                    compute_tuple_gradient(NULL, model);
                }

                if (iter == 1)
                {
                    double avg_page_tuple_num = (double)model->tuple_num / table_page_number;
                    elog(INFO, "[%s] [Computed Param] table_tuple_num = %d, buffer_block_num = %.2f",
                         get_current_time(), model->tuple_num,
                         (double)set_buffer_tuple_num / (set_block_page_num * avg_page_tuple_num));
                }

                end_of_reach = true;
                ExecReScan(outerNode);
                break;
            }

            compute_tuple_gradient(&read_buffer[j], model);


            if (iter == 1)
                model->tuple_num += 1;
        }

        if (end_of_reach)
            break;
    }
}

inline
void test_with_unshuffled_buffer(PlanState* outerNode, Model* model, int iter,
                                 int total_iter_num)
{
    bool end_of_reach = false;

    while (true)
    {
        TupleTableSlot* slot = ExecProcNode(outerNode);

        SortTuple* read_buffer = slot->read_buffer;
        int buffer_size = slot->read_buffer_size;

        int j;
        for (j = 0; j < buffer_size; ++j)
        {
            if (read_buffer[j].isnull)
            {
                /*
                clock_t iter_finish = clock();
                double comp_grad_time = (double)(train_finish - iter_start) / CLOCKS_PER_SEC;
                double iter_exec_time = (double)(iter_finish - iter_start) / CLOCKS_PER_SEC;
                double comp_loss_time = iter_exec_time - comp_grad_time;

                elog(INFO, "[%s] [Iter %2d] Loss = %.2f, exec_t = %.2fs, grad_t = %.2fs, loss_t = %.2fs",
                    get_current_time(), iter, model->total_loss, iter_exec_time,
                    comp_grad_time, comp_loss_time);
                */

                // model->total_loss = 0;
                end_of_reach = true;
                if (iter < total_iter_num) // finish
                    ExecReScan(outerNode);

                break;
            }

            compute_tuple_loss(&read_buffer[j], model);
        }

        if (end_of_reach)
            break;
    }
}

inline
void train_without_buffer(PlanState* outerNode, Model* model, int iter, SortTuple* sort_tuple)
{
    while (true)
    {
        TupleTableSlot* slot = ExecProcNode(outerNode);

        if (TupIsNull(slot))
        {
            if (model->batch_size > 1)
            {
                compute_tuple_gradient(NULL, model);
            }

            if (iter == 1)
            {
                double avg_page_tuple_num = (double)model->tuple_num / table_page_number;
                elog(INFO, "[%s] [Computed Param] table_tuple_num = %d, buffer_block_num = %.2f",
                     get_current_time(), model->tuple_num,
                     (double)set_buffer_tuple_num / (set_block_page_num * avg_page_tuple_num));
            }

            ExecReScan(outerNode);
            break;
        }

        fast_transfer_slot_to_sgd_tuple(slot, sort_tuple);
        compute_tuple_gradient(sort_tuple, model);

        if (sort_tuple->v_array != NULL)
            pfree((ArrayType*)(sort_tuple->v_array));
        if (sort_tuple->k_array != NULL)
            pfree((ArrayType*)(sort_tuple->k_array));

        if (iter == 1)
            model->tuple_num += 1;
    }
}

inline
void test_without_buffer(PlanState* outerNode, Model* model, int iter,
                         int total_iter_num,
                         SortTuple* sort_tuple)
{
    bool end_of_reach = false;

    while (true)
    {
        TupleTableSlot* slot = ExecProcNode(outerNode);

        if (TupIsNull(slot))
        {
            /*
            clock_t iter_finish = clock();
            double comp_grad_time = (double)(train_finish - iter_start) / CLOCKS_PER_SEC;
            double iter_exec_time = (double)(iter_finish - iter_start) / CLOCKS_PER_SEC;
            double comp_loss_time = iter_exec_time - comp_grad_time;

            elog(INFO, "[%s] [Iter %2d] Loss = %.2f, exec_t = %.2fs, grad_t = %.2fs, loss_t = %.2fs",
                get_current_time(), iter, model->total_loss, iter_exec_time,
                comp_grad_time, comp_loss_time);
            model->total_loss = 0;
            */


            end_of_reach = true;
            if (iter < total_iter_num) // finish
                ExecReScan(outerNode);

            break;
        }

        fast_transfer_slot_to_sgd_tuple(slot, sort_tuple);
        compute_tuple_loss(sort_tuple, model);

        if (sort_tuple->v_array != NULL)
            pfree((ArrayType*)(sort_tuple->v_array));
        if (sort_tuple->k_array != NULL)
            pfree((ArrayType*)(sort_tuple->k_array));

        if (set_run_test && iter == 1)
            model->tuple_num += 1;
    }
}


TupleTableSlot*
ExecLimit(LimitState* node)
{
    EState* estate;
    TupleTableSlot* slot;
    Model* model = node->model;

    SO1_printf("ExecSGD: %s\n", "entering routine");

    estate = node->ps.state;

    /*
     * If first time through, read all tuples from outer plan and pass them to
     * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
     */
    // SGDBatchState* batchstate = init_SGDBatchState(model->n_features);
    //SGDTuple* sgd_tuple = init_SGDTuple(model->n_features);
    SortTuple* sort_tuple = init_SortTuple(model->n_features);

    // TestState* test_state = init_TestState(set_run_test);

    PlanState* outerNode;
    TupleDesc tupDesc;

    SO1_printf("ExecSGD: %s\n", "SGD iteration ");

    estate->es_direction = ForwardScanDirection;

    // outerNode = ShuffleSortNode
    outerNode = outerPlanState(node);
    // tupDesc is the TupleDesc of the previous node
    tupDesc = ExecGetResultType(outerNode);
    // int col_num = tupDesc->natts;

    // node->tupleShuffleSortState = (void *) tupleShuffleSortState;
    int iter_num = model->iter_num;
    // int batch_size = node->model->batch_size;

    // for counting execution time for each iteration
    // clock_t iter_start, train_finish; //, iter_finish;

    struct timeval iter_start;
    struct timeval grad_finish;
    struct timeval loss_finish;

    double avg_iter_exec_time = 0;
    double avg_comp_grad_time = 0;
    double avg_comp_loss_time = 0;

    double first_iter_exec_time = 0;
    double first_comp_grad_time = 0;
    double first_comp_loss_time = 0;

    double second_iter_exec_time = 0;
    double second_comp_grad_time = 0;
    double second_comp_loss_time = 0;

    double max_accuracy = 0;


    int i;
    for (i = 1; i <= iter_num; i++)
    {
        // iter_start = clock();
        gettimeofday(&iter_start, NULL);

        if (!set_run_test)
        {
            // train
            is_training = true;
            /**
            shuffle_mode

            Block-Only Shuffle (block-shuffle = 0, tuple-shuffle = 0, i.e., 0 buffer, 1 thread)
            CorgiPile (block-shuffle = 1, tuple-shuffle = 2, i.e., 2 buffers for tuple shuffle, 2 threads)
            CorgiPile-Single-Thread (block-shuffle = 1, tuple-shuffle = 1, i.e., 1 buffer, 1 thread)
            No Shuffle (block-shuffle = 0, tuple-shuffle = 0, i.e, 0 buffer, 1 thread)

            block-shuffle = 1
            block-shuffle = 0

            tuple-shuffle = 2 (2 buffers, 2 threads based shuffle)
            tuple-shuffle = 1 (1 buffer, 1 thread based shuffle)
            tuple-shuffle = 0 (0 buffer, 1 thread, no shuffle)
            */

            if (set_tuple_shuffle > 0)
                train_with_shuffled_buffer(outerNode, model, i);
            else
                train_without_buffer(outerNode, model, i, sort_tuple);

            // train_finish = clock();
            // update learning_rate
            model->learning_rate = model->learning_rate * model->decay;
        }
        gettimeofday(&grad_finish, NULL);


        // test
        is_training = false;
        test_without_buffer(outerNode, model, i, iter_num, sort_tuple);

        gettimeofday(&loss_finish, NULL);

        double exec_t = diff_timeofday_seconds(&iter_start, &loss_finish);
        double grad_t = diff_timeofday_seconds(&iter_start, &grad_finish);
        double loss_t = exec_t - grad_t;

        avg_iter_exec_time += exec_t;
        avg_comp_grad_time += grad_t;
        avg_comp_loss_time += loss_t;

        if (i == 1)
        {
            first_iter_exec_time = exec_t;
            first_comp_grad_time = grad_t;
            first_comp_loss_time = loss_t;
        }
        else if (i == 2)
        {
            second_iter_exec_time = exec_t;
            second_comp_grad_time = grad_t;
            second_comp_loss_time = loss_t;
        }


        model->accuracy = model->accuracy / model->tuple_num * 100;
        model->fp = model->tp / (model->tp + model->fp) * 100;
        model->fn = model->tp / (model->tp + model->fn) * 100;
        model->tp = (2 * model->fp * model->fn) / (model->fp + model->fn);

        elog(INFO, "[%s] [Iter %2d] Loss = %.2f, acc = %.2f, exec_t = %.2fs, grad_t = %.2fs, loss_t = %.2fs\n "
                   "precision = %.2f, recall = %.2f, f1-score = %.2f",
             get_current_time(), i, model->total_loss, model->accuracy, exec_t,
             grad_t, loss_t, model->fp, model->fn, model->tp);

        if (model->accuracy > max_accuracy)
            max_accuracy = model->accuracy;
        model->total_loss = 0;
        model->accuracy = 0;
        model->tp = 0;
        model->fp = 0;
        model->fn = 0;
        if (set_run_test)
            break;
    }

    // clear states
    free_SortTuple(sort_tuple);
    free_SGDTupleDesc(sgd_tupledesc);

    // for debug
    //fclose(fp);

    node->sgd_done = true;
    SO1_printf("ExecSGD: %s\n", "Performing SGD done");

    // Get the first or next tuple from tuplesort. Returns NULL if no more tuples.

    // node->ps.ps_ResultTupleSlot; // = Model->w, => ExecStoreMinimalTuple();
    // slot = node->ps.ps_ResultTupleSlot;
    // elog(INFO, "[Model total loss %f]", model->total_loss);

    // slot = output_model_record(node->ps.ps_ResultTupleSlot, model);
    elog(INFO, "[%s] [Finish] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs",
         get_current_time(), avg_iter_exec_time / iter_num,
         avg_comp_grad_time / iter_num, avg_comp_loss_time / iter_num);

    if (set_run_test)
    {
        elog(INFO, "[%s] [MaxAcc] max_acc = %.2fs", get_current_time(), max_accuracy);
    }

    if (iter_num > 2)
    {
        avg_iter_exec_time -= first_iter_exec_time;
        avg_comp_grad_time -= first_comp_grad_time;
        avg_comp_loss_time -= first_comp_loss_time;
        elog(INFO, "[%s] [-first] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs",
             get_current_time(), avg_iter_exec_time / (iter_num - 1),
             avg_comp_grad_time / (iter_num - 1), avg_comp_loss_time / (iter_num - 1));

        avg_iter_exec_time -= second_iter_exec_time;
        avg_comp_grad_time -= second_comp_grad_time;
        avg_comp_loss_time -= second_comp_loss_time;

        elog(INFO, "[%s] [-1 & 2] avg_exec_t = %.2fs, avg_grad_t = %.2fs, avg_loss_t = %.2fs",
             get_current_time(), avg_iter_exec_time / (iter_num - 2),
             avg_comp_grad_time / (iter_num - 2), avg_comp_loss_time / (iter_num - 2));

        elog(INFO, "[%s] [MaxAcc] max_acc = %.2fs", get_current_time(), max_accuracy);
    }

    slot = NULL;
    return slot;
}

bool is_prefix(char* table_name, char* prefix)
{
    while (*table_name && *prefix)
    {
        if (*table_name != *prefix)
            return false;
        table_name++;
        prefix++;
    }

    return true;
}

/* ----------------------------------------------------------------
 *		ExecInitLimit
 *
 *		This initializes the limit node state structures and
 *		the node's subplan.
 * ----------------------------------------------------------------
 */
LimitState*
ExecInitLimit(Limit* node, EState* estate, int eflags)
{
    LimitState* sgdstate;

    SO1_printf("ExecInitSGD: %s\n",
               "initializing SGD node");

    //
    const char* work_mem_str = GetConfigOption("work_mem", false, false);
    elog(INFO, "[%s]", get_current_time());
    elog(INFO, "============== Begin Training on %s Using %s Model WITH %s==============", set_table_name,
         set_model_name, set_algo_name);
    elog(INFO, "[Param] model_name = %s", set_model_name);
    elog(INFO, "[Param] algo_name = %s", set_algo_name);
    elog(INFO, "[Param] use_malloc = %d", set_use_malloc);
    elog(INFO, "[Param] class_num = %d", set_class_num);
    elog(INFO, "[Param] block_shuffle = %d", set_block_shuffle);
    elog(INFO, "[Param] tuple_shuffle = %d", set_tuple_shuffle);
    elog(INFO, "[Param] work_mem = %s KB", work_mem_str);
    elog(INFO, "[Param] block_page_num = %d pages", set_block_page_num);
    // elog(INFO, "[Param] io_block_size = %d pages", set_io_big_block_size);
    elog(INFO, "[Param] buffer_tuple_num = %d tuples", set_buffer_tuple_num);
    elog(INFO, "[Param] batch_size = %d", set_batch_size);
    elog(INFO, "[Param] iter_num = %d", set_iter_num);
    elog(INFO, "[Param] learning_rate = %f", set_learning_rate);
    elog(INFO, "[Param] decay = %f", set_decay);
    elog(INFO, "[Param] mu = %f", set_mu);

    /*
     * create state structure
     */
    sgdstate = makeNode(LimitState);
    sgdstate->ps.plan = (Plan*)node;
    sgdstate->ps.state = estate;
    sgdstate->sgd_done = false;

    bool dense = true;
    int n_features;
    int max_sparse_count = 100;

    if (is_prefix(set_table_name, "dblife"))
    {
        n_features = 41270;
        dense = false;
    }
    else if (is_prefix(set_table_name, "splicesite"))
    {
        n_features = 11725480;
        dense = false;
        max_sparse_count = 3387;
    }
    else if (is_prefix(set_table_name, "sample_splice"))
    {
        n_features = 11725480;
        dense = false;
        max_sparse_count = 3387;
    }
    else if (is_prefix(set_table_name, "kdda"))
    {
        n_features = 20216830;
        dense = false;
        max_sparse_count = 85;
    }
    else if (is_prefix(set_table_name, "kdd2012"))
    {
        n_features = 54686452;
        dense = false;
    }
    else if (is_prefix(set_table_name, "avazu"))
    {
        n_features = 1000000;
        dense = false;
        max_sparse_count = 15;
    }
    else if (is_prefix(set_table_name, "url"))
    {
        n_features = 3231961;
        dense = false;
        max_sparse_count = 414;
    }
    else if (is_prefix(set_table_name, "criteo"))
    {
        n_features = 1000000;
        dense = false;
        max_sparse_count = 39;
    }
    else if (is_prefix(set_table_name, "sample_criteo"))
    {
        n_features = 1000000;
        dense = false;
        max_sparse_count = 39;
    }

    else if (is_prefix(set_table_name, "forest"))
        n_features = 54;
    else if (is_prefix(set_table_name, "susy"))
        n_features = 18;
    else if (is_prefix(set_table_name, "higgs"))
        n_features = 28;
    else if (is_prefix(set_table_name, "epsilon"))
        n_features = 2000;
    else if (is_prefix(set_table_name, "yfcc"))
        n_features = 4096;

    else if (is_prefix(set_table_name, "iris"))
        n_features = 4;

    else if (is_prefix(set_table_name, "mnist"))
        n_features = 780;

    else if (is_prefix(set_table_name, "mnist8m"))
        n_features = 784;


    sgdstate->model = init_model(n_features, max_sparse_count);
    sgd_tupledesc = init_SGDTupleDesc(n_features, dense, max_sparse_count);

    if (strcmp(set_model_name, "LR") == 0 || strcmp(set_model_name, "lr") == 0)
    {
        if (strcmp(set_algo_name, "ada") == 0 || strcmp(set_algo_name, "ADA") == 0)
        {
            if (dense)
            {
                if (sgdstate->model->batch_size > 1)
                    compute_tuple_gradient = batch_compute_dense_tuple_adagrad_LR;
                else
                    compute_tuple_gradient = compute_dense_tuple_adagrad_LR;
                compute_tuple_loss = compute_dense_tuple_loss_LR;
            }
            else
            {
                if (sgdstate->model->batch_size > 1)
                    compute_tuple_gradient = batch_compute_sparse_tuple_adagrad_LR;
                else
                    compute_tuple_gradient = compute_sparse_tuple_adagrad_LR;
                compute_tuple_loss = compute_sparse_tuple_loss_LR;
            }
        }
        else if (dense)
        {
            if (sgdstate->model->batch_size > 1)
                compute_tuple_gradient = batch_compute_dense_tuple_gradient_LR;
            else
                compute_tuple_gradient = compute_dense_tuple_gradient_LR;

            compute_tuple_loss = compute_dense_tuple_loss_LR;
        }
        else
        {
            if (sgdstate->model->batch_size > 1)
                compute_tuple_gradient = batch_compute_sparse_tuple_gradient_LR;
            else
                compute_tuple_gradient = compute_sparse_tuple_gradient_LR;
            compute_tuple_loss = compute_sparse_tuple_loss_LR;
        }
    }
    else if (strcmp(set_model_name, "SVM") == 0 || strcmp(set_model_name, "svm") == 0)
    {
        if (dense)
        {
            if (sgdstate->model->batch_size > 1)
                compute_tuple_gradient = batch_compute_dense_tuple_gradient_SVM;
            else
                compute_tuple_gradient = compute_dense_tuple_gradient_SVM;

            compute_tuple_loss = compute_dense_tuple_loss_SVM;
        }
        else
        {
            if (sgdstate->model->batch_size > 1)
                compute_tuple_gradient = batch_compute_sparse_tuple_gradient_SVM;
            else
                compute_tuple_gradient = compute_sparse_tuple_gradient_SVM;
            compute_tuple_loss = compute_sparse_tuple_loss_SVM;
        }
    }
    else if (strcmp(set_model_name, "Softmax") == 0 || strcmp(set_model_name, "softmax") == 0)
    {
        if (dense)
        {
            if (sgdstate->model->batch_size > 1)
                compute_tuple_gradient = batch_compute_dense_tuple_gradient_Softmax;
            else
                compute_tuple_gradient = compute_dense_tuple_gradient_Softmax;

            compute_tuple_loss = compute_dense_tuple_loss_Softmax;
        }
    }

    elog(INFO, "[Model] Initialize %s model", sgdstate->model->model_name);
    // elog(INFO, "[SVM] loss = 0, p1 = 0, p2 = 0, gradient = 0, batch_size = 10, learning_rate = 0.1");

    /*
     * tuple table initialization
     *
     * sort nodes only return scan tuples from their sorted relation.
     */
    ExecInitResultTupleSlot(estate, &sgdstate->ps);

    /*
     * initialize child nodes
     *
     * We shield the child node from the need to support REWIND, BACKWARD, or
     * MARK/RESTORE.
     */
    eflags &= ~(EXEC_FLAG_REWIND | EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK);

    outerPlanState(sgdstate) = ExecInitNode(outerPlan(node), estate, eflags);

    /*
     * initialize tuple type.  no need to initialize projection info because
     * this node doesn't do projections.
     */
    ExecAssignResultTypeFromTL(&sgdstate->ps);
    // ExecAssignScanTypeFromOuterPlan(&sortstate->ss);
    // sortstate->ss.ps.ps_ProjInfo = NULL;

    SO1_printf("ExecInitSGD: %s\n",
               "SGD node initialized");

    return sgdstate;
}

/* ----------------------------------------------------------------
 *		ExecEndLimit
 *
 *		This shuts down the subplan and frees resources allocated
 *		to this node.
 * ----------------------------------------------------------------
 */
void
ExecEndLimit(LimitState* node)
{
    /*
     * Free the exprcontext
     */
    ExecFreeExprContext(&node->ps);

    /*
     * clean out the tuple table
     */
    ExecClearTuple(node->ps.ps_ResultTupleSlot);

    /*
     * sotre the training model, use it in last test run
     */
    if (!set_run_test)
    {
        int n_features = node->model->n_features;
        after_training_model = (double*)malloc(sizeof(double) * n_features);
        for (int i = 0; i < n_features; i++)
        {
            after_training_model[i] = node->model->w[i];
        }
    }
    if (set_run_test && after_training_model != NULL)
    {
        free(after_training_model);
        after_training_model = NULL;
    }

    ExecFreeModel(node->model);

    ExecEndNode(outerPlanState(node));

    SO1_printf("ExecEndSGD: %s\n",
               "SGD node shutdown");
}


void
ExecReScanLimit(LimitState* node)
{
    // /*
    //  * Recompute limit/offset in case parameters changed, and reset the state
    //  * machine.  We must do this before rescanning our child node, in case
    //  * it's a Sort that we are passing the parameters down to.
    //  */
    // recompute_limits(node);

    // /*
    //  * if chgParam of subnode is not null then plan will be re-scanned by
    //  * first ExecProcNode.
    //  */
    // if (node->ps.lefttree->chgParam == NULL)
    // 	ExecReScan(node->ps.lefttree);
}
