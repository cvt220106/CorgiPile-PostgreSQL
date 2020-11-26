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
#include "time.h"
#include "math.h"

#define SHARED_MEM_SIZE (1 << 30)
#define ARRAY_HEAD_SIZE (20)
// guc variables
// can be set via "SET VAR = XX" in the psql console
int set_batch_size = DEFAULT_BATCH_SIZE;
int set_iter_num = DEFAULT_ITER_NUM;
double set_learning_rate = DEFAULT_LEARNING_RATE;
char* set_model_name = DEFAULT_MODEL_NAME;
int table_page_number = 0;

// char* table_name = "dflife";
char* set_table_name = "forest";

bool set_run_test = false;

static Model* init_model(int n_features);
static void ExecClearModel(Model* model);
static SGDBatchState* init_SGDBatchState(int n_features);
static SGDTuple* init_SGDTuple(int n_features);
static SGDTupleDesc* init_SGDTupleDesc(int col_num, int n_features);
static void clear_SGDBatchState(SGDBatchState* batchstate, int n_features);
static void free_SGDBatchState(SGDBatchState* batchstate);
static void free_SGDTuple(SGDTuple* sgd_tuple);
static void free_SGDTupleDesc(SGDTupleDesc* sgd_tupledesc);
static void compute_tuple_gradient_loss_LR(SGDTuple* tp, Model* model, SGDBatchState* batchstate);
static void compute_tuple_gradient_loss_SVM(SGDTuple* tp, Model* model, SGDBatchState* batchstate);
static void compute_tuple_accuracy(Model* model, SGDTuple* tp, TestState* test_state);
static void update_model(Model* model, SGDBatchState* batchstate);
static void perform_SGD(Model *model, SGDTuple* sgd_tuple, SGDBatchState* batchstate, int i);
// static void transfer_slot_to_sgd_tuple(TupleTableSlot* slot, SGDTuple* sgd_tuple, SGDTupleDesc* sgd_tupledesc);
static void fast_transfer_slot_to_sgd_tuple(TupleTableSlot* slot, SGDTuple* sgd_tuple, SGDTupleDesc* sgd_tupledesc);
static void transfer_slot_to_sgd_tuple_getattr(TupleTableSlot* slot, SGDTuple* sgd_tuple, SGDTupleDesc* sgd_tupledesc);

static int my_parse_array_no_copy(struct varlena* input, int typesize, char** output);



Model* init_model(int n_features) {
    Model* model = (Model *) palloc0(sizeof(Model));

	model->model_name = set_model_name;
	model->total_loss = 0;
	model->batch_size = set_batch_size;
	model->iter_num = set_iter_num;
	model->learning_rate = set_learning_rate;
	model->tuple_num = 0;
	model->n_features = n_features;

    // use memorycontext later
	model->w = (double *) palloc0(sizeof(double) * n_features);

	memset(model->w, 0, sizeof(double) * n_features);

    return model;
}

void ExecClearModel(Model* model) {
    // free(model->gradient);
	pfree(model->w);
    pfree(model);
}

static SGDBatchState* init_SGDBatchState(int n_features) {
    SGDBatchState* batchstate = (SGDBatchState *) palloc0(sizeof(SGDBatchState));
    batchstate->gradients = (double *) palloc0(sizeof(double) * n_features);
	for (int i = 0; i < n_features; i++)
		batchstate->gradients[i] = 0;
    batchstate->loss = 0;
	batchstate->tuple_num = 0;
    return batchstate;
}


static TestState* init_TestState(bool run_test) {
	TestState* test_state = NULL;
	if (run_test) {
		test_state = (TestState *) palloc0(sizeof(TestState));
   		test_state->test_total_loss = 0;
		test_state->test_accuracy = 0;
		test_state->right_count = 0;
	}
    return test_state;
}

static SGDTuple* init_SGDTuple(int n_features) {
    SGDTuple* sgd_tuple = (SGDTuple *) palloc0(sizeof(SGDTuple));
    sgd_tuple->features = (double *) palloc0(sizeof(double) * n_features);
    return sgd_tuple;
}

static SGDTupleDesc* init_SGDTupleDesc(int col_num, int n_features) {
    SGDTupleDesc* sgd_tupledesc = (SGDTupleDesc *) palloc0(sizeof(SGDTupleDesc));

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
	if (strcmp(set_table_name, "dblife") == 0) {
		/* for dblife */
		sgd_tupledesc->k_col = 1; 
		sgd_tupledesc->v_col = 2;
		sgd_tupledesc->label_col = 3;
	}
	else if (strcmp(set_table_name, "forest") == 0) {
		/* for forest */
		sgd_tupledesc->k_col = -1; // from 0
		sgd_tupledesc->v_col = 1;
		sgd_tupledesc->label_col = 2;
	}
	
	sgd_tupledesc->n_features = n_features;
    return sgd_tupledesc;
}

static void clear_SGDBatchState(SGDBatchState* batchstate, int n_features) {
	for (int i = 0; i < n_features; i++)
		batchstate->gradients[i] = 0;
    batchstate->loss = 0;
	batchstate->tuple_num = 0;
}

static void clear_TestState(TestState* test_state) {
	test_state->right_count = 0;
	test_state->test_accuracy = 0;
	test_state->test_total_loss = 0;
}

static void free_SGDBatchState(SGDBatchState* batchstate) {
    pfree(batchstate->gradients);
    pfree(batchstate);
}

static void free_SGDTuple(SGDTuple* sgd_tuple) {
    pfree(sgd_tuple->features);
    pfree(sgd_tuple);
}

static void free_SGDTupleDesc(SGDTupleDesc* sgd_tupledesc) {
    // pfree(sgd_tupledesc->values);
    // pfree(sgd_tupledesc->isnulls);
	pfree(sgd_tupledesc);
}

static void free_TestState(TestState* test_state) {
    pfree(test_state);
}

static void
compute_tuple_gradient_loss_LR(SGDTuple* tp, Model* model, SGDBatchState* batchstate)
{
    double y = tp->class_label;
    double* x = tp->features;

    int n = model->n_features;

    // compute gradients of the incoming tuple
    double wx = 0;
    for (int i = 0; i < n; i++)
        wx = wx + model->w[i] * x[i];
    double ywx = y * wx;

	double tuple_loss = log(1 + exp(-ywx));

	double g_base = -y * (1 - 1 / (1 + exp(-ywx)));

    // Add this tuple's gradient to the previous gradients in this batch
    for (int i = 0; i < n; i++) 
        batchstate->gradients[i] += g_base * x[i];

    // compute the loss of the incoming tuple
    batchstate->loss += tuple_loss;
	batchstate->tuple_num += 1;
}


static void
compute_tuple_gradient_loss_SVM(SGDTuple* tp, Model* model, SGDBatchState* batchstate)
{
    double y = tp->class_label;
    double* x = tp->features;

    int n = model->n_features;

    // double loss = 0;
    // double grad[n];

    // compute gradients of the incoming tuple
    double wx = 0;
    for (int i = 0; i < n; i++)
        wx = wx + model->w[i] * x[i];
    double ywx = y * wx;

    if (1 - ywx > 0) {
        for (int i = 0; i < n; i++)
			batchstate->gradients[i] = batchstate->gradients[i] - y * x[i];
    }

    // compute the loss of the incoming tuple
    double tuple_loss = 1 - ywx;
    if (tuple_loss < 0)
        tuple_loss = 0;
	
    batchstate->loss = batchstate->loss + tuple_loss;
	batchstate->tuple_num += 1;
}

static void update_model(Model* model, SGDBatchState* batchstate) {
	if (batchstate->tuple_num > 0) {
		 // add graidents to the model and clear the batch gradients
		for (int i = 0; i < model->n_features; i++) {
			model->w[i] = model->w[i] - model->learning_rate * batchstate->gradients[i] / batchstate->tuple_num;
			// model->w[i] = model->w[i] - model->learning_rate * 
			// 			 (batchstate->gradients[i] / batchstate->tuple_num + 0.01 * model->w[i]);
			batchstate->gradients[i] = 0;
		}

		model->total_loss = model->total_loss + batchstate->loss;
		 
		batchstate->loss = 0;
		batchstate->tuple_num = 0;
	}
   
}


static void perform_SGD(Model *model, SGDTuple* sgd_tuple, SGDBatchState* batchstate, int i) {
    if (sgd_tuple == NULL) /* slot == NULL means the end of the table. */
        update_model(model, batchstate);
    else {
		if (strcmp(model->model_name, "SVM") == 0)
        // add the batch's gradients to the model, and reset the batch's gradients.
        	compute_tuple_gradient_loss_SVM(sgd_tuple, model, batchstate);
		else if (strcmp(model->model_name, "LR") == 0)
			compute_tuple_gradient_loss_LR(sgd_tuple, model, batchstate);
		
		else {
			elog(ERROR, "The model name %s cannot be recognized!", model->model_name);
			exit(1);
		}
		
		
        if (i == model->batch_size - 1)
            update_model(model, batchstate);
        
    }   
}


// Extract features and class label from Tuple
static void
transfer_slot_to_sgd_tuple_getattr (
	TupleTableSlot* slot, 
	SGDTuple* sgd_tuple, 
	SGDTupleDesc* sgd_tupledesc) {

	// store the values of slot to values/isnulls arrays
	//heap_deform_tuple(slot->tts_tuple, slot->tts_tupleDescriptor, sgd_tupledesc->values, sgd_tupledesc->isnulls);

	int k_col = sgd_tupledesc->k_col;
	int v_col = sgd_tupledesc->v_col;
	int label_col = sgd_tupledesc->label_col;

	bool isnull;
	slot_getallattrs(slot);
	// Datum v_dat = slot_getattr(slot, v_col + 1, &isnull);
	Datum v_dat = slot->tts_values[v_col];
	//
	// e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
	// Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
	ArrayType  *v_array = DatumGetArrayTypeP(v_dat); // Datum{0.1, 0.2, 0.3}
	
	double *v;
    int v_num = my_parse_array_no_copy((struct varlena*) v_array, 
            sizeof(float8), (char **) &v);

	/* label dataum => int class_label */
	// Datum label_dat = heap_getattr(slot->tts_tuple, label_col + 1, slot->tts_tupleDescriptor, &isnull);
	// Datum label_dat = slot_getattr(slot, label_col + 1, &isnull);

	Datum label_dat = slot->tts_values[label_col];
	int label = DatumGetInt32(label_dat);
	sgd_tuple->class_label = label;


	/* double* v => double* features */
	double* features = sgd_tuple->features;
	int n_features = sgd_tupledesc->n_features;
	// if sparse dataset
	if (k_col >= 0) {
		/* k Datum array => int* k */
		// Datum k_dat = heap_getattr(slot->tts_tuple, k_col + 1, slot->tts_tupleDescriptor, &isnull); // Datum{0, 2, 5}
		Datum k_dat = slot->tts_values[k_col];
		ArrayType  *k_array = DatumGetArrayTypeP(k_dat);
		int *k;
    	int k_num = my_parse_array_no_copy((struct varlena*) k_array, 
            	sizeof(int), (char **) &k);

		memset(features, 0, sizeof(double) * n_features);

		
		for (int i = 0; i < k_num; i++) {
			int f_index = k[i]; // {0, 2, 5}, k[1] = 2
			features[f_index] = v[i]; // {0.1, 0.2, 0.3}, features[2] = 0.2
		}
	}
	
	else {
		memcpy(features, v, v_num * sizeof(double));
	}
	
}

/*
static void
transfer_slot_to_sgd_tuple(
	TupleTableSlot* slot, 
	SGDTuple* sgd_tuple, 
	SGDTupleDesc* sgd_tupledesc) {

	// store the values of slot to values/isnulls arrays
	// slot => Datum values/isnulls 
	heap_deform_tuple(slot->tts_tuple, slot->tts_tupleDescriptor, sgd_tupledesc->values, sgd_tupledesc->isnulls);
	// DatumGetInt32
	// tupleDesc->attrs[0]->atttypid

	int k_col = sgd_tupledesc->k_col;
	int v_col = sgd_tupledesc->v_col;
	int label_col = sgd_tupledesc->label_col;

	// Datum => double/int 
	// e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
	// Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
	Datum v_dat = sgd_tupledesc->values[v_col]; // Datum{0.1, 0.2, 0.3}
	Datum label_dat = sgd_tupledesc->values[label_col]; // Datum{-1}


	// feature datum arrary => double* v 
	ArrayType  *v_array = DatumGetArrayTypeP(v_dat);
	//Assert(ARR_ELEMTYPE(array) == FLOAT4OID);
	//int	v_num = ArrayGetNItems(ARR_NDIM(v_array), ARR_DIMS(v_array));
	// int	v_num = ARR_DIMS(v_array)[0];
	// double *v = (double *) ARR_DATA_PTR(v_array);
	double *v;
    int v_num = my_parse_array_no_copy((struct varlena*) v_array, 
            sizeof(float8), (char **) &v);


	// label dataum => int class_label
	int label = DatumGetInt32(label_dat);
	sgd_tuple->class_label = label;


	// double* v => double* features 
	double* features = sgd_tuple->features;
	int n_features = sgd_tupledesc->n_features;
	// if sparse dataset
	if (k_col >= 0) {
		// k Datum array => int* k 
		Datum k_dat = sgd_tupledesc->values[k_col]; // Datum{0, 2, 5}

		ArrayType  *k_array = DatumGetArrayTypeP(k_dat);
		// int	k_num = ArrayGetNItems(ARR_NDIM(k_array), ARR_DIMS(k_array));
		// int *k = (int *) ARR_DATA_PTR(k_array);
		int *k;
    	int k_num = my_parse_array_no_copy((struct varlena*) k_array, 
            	sizeof(int), (char **) &k);

		memset(features, 0, sizeof(double) * n_features);

		for (int i = 0; i < k_num; i++) {
			int f_index = k[i]; // {0, 2, 5}, k[1] = 2
			features[f_index] = v[i]; // {0.1, 0.2, 0.3}, features[2] = 0.2
		}
	}
	else {
		// Assert(n_features == v_num);
		// for (int i = 0; i < v_num; i++) {
		// 	features[i] = v[i];
		// }
		memcpy(features, v, v_num * sizeof(double));
	}
	
}
*/

static void
fast_transfer_slot_to_sgd_tuple (
	TupleTableSlot* slot, 
	SGDTuple* sgd_tuple, 
	SGDTupleDesc* sgd_tupledesc) {

	// store the values of slot to values/isnulls arrays
	int k_col = sgd_tupledesc->k_col;
	int v_col = sgd_tupledesc->v_col;
	int label_col = sgd_tupledesc->label_col;

	int attnum = HeapTupleHeaderGetNatts(slot->tts_tuple->t_data);
	slot_deform_tuple(slot, attnum);
	
	
	// e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
	// Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
	Datum v_dat = slot->tts_values[v_col];
	ArrayType  *v_array = DatumGetArrayTypeP(v_dat); // Datum{0.1, 0.2, 0.3}
	
	double *v;
    int v_num = my_parse_array_no_copy((struct varlena*) v_array, 
            sizeof(float8), (char **) &v);


	Datum label_dat = slot->tts_values[label_col];
	int label = DatumGetInt32(label_dat);
	sgd_tuple->class_label = label;


	/* double* v => double* features */
	double* features = sgd_tuple->features;
	int n_features = sgd_tupledesc->n_features;
	// if sparse dataset
	if (k_col >= 0) {
		/* k Datum array => int* k */
		Datum k_dat = slot->tts_values[k_col];
		ArrayType  *k_array = DatumGetArrayTypeP(k_dat);
		int *k;
    	int k_num = my_parse_array_no_copy((struct varlena*) k_array, 
            	sizeof(int), (char **) &k);

		memset(features, 0, sizeof(double) * n_features);

		for (int i = 0; i < k_num; i++) {
			int f_index = k[i]; // {0, 2, 5}, k[1] = 2
			features[f_index] = v[i]; // {0.1, 0.2, 0.3}, features[2] = 0.2
		}
	}
	
	else {
		//sgd_tuple->features = v;
		memcpy(features, v, v_num * sizeof(double));
	}
	
}

/**
 * From bismarck
 * parse the array by NO PALLOC? 
 *
 * args:
 *   input struct varlena*, variable length struct pointer
 *   typesize int, size of element type
 *   output (void*)*, start pointer of the array elements
 * return:
 *   int, length of the array, # of elements
 */
static int 
my_parse_array_no_copy(struct varlena* input, int typesize, char** output) {
	// elog(WARNING, "Inside loss(), for v, ISEXTERNAL %d, ISCOMPR %d, ISHORT %d, varsize_short %d", VARATT_IS_EXTERNAL(v2) ? 1 : 0, VARATT_IS_COMPRESSED(v2)  ? 1 : 0, VARATT_IS_SHORT(v2)  ? 1 : 0, VARSIZE_SHORT(v2));
	// elog(WARNING, "Inside loss(), for v, varlena = %x", input);
	
	if (VARATT_IS_EXTERNAL(input) || VARATT_IS_COMPRESSED(input)) {
		// if compressed, palloc is necessary
		input = heap_tuple_untoast_attr(input);
        *output = VARDATA(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE(input) - VARHDRSZ - ARRAY_HEAD_SIZE) / typesize;
	} else if (VARATT_IS_SHORT(input)) {
        *output = VARDATA_SHORT(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE_SHORT(input) - VARHDRSZ_SHORT - ARRAY_HEAD_SIZE) / typesize;
    } else {
        *output = VARDATA(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE(input) - VARHDRSZ - ARRAY_HEAD_SIZE) / typesize;
    }
}
/* ----------------------------------------------------------------
 *		ExecLimit
 *
 *		This is a very simple node which just performs LIMIT/OFFSET
 *		filtering on the stream of tuples returned by a subplan.
 * ----------------------------------------------------------------
 */
TupleTableSlot *				/* return: a tuple or NULL */
ExecLimit(LimitState *node)
{
	EState	   *estate;
	TupleTableSlot *slot;
	Model* model = node->model;

	/*
	 * get state info from node
	 */
	SO1_printf("ExecSGD: %s\n", "entering routine");

	estate = node->ps.state;
	
	// tupleSGDState = (TupleSGDState *) node->tupleSGDState;

	/*
	 * If first time through, read all tuples from outer plan and pass them to
	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
	 */
	SGDBatchState* batchstate = init_SGDBatchState(model->n_features);
	SGDTuple* sgd_tuple = init_SGDTuple(model->n_features);
	TestState* test_state = init_TestState(set_run_test);

	//Datum values[model->n_features + model-> slot->]
	// ShuffleSort	   *plannode = (ShuffleSort *) node->ss.ps.plan;
	PlanState  *outerNode;
	TupleDesc	tupDesc;

	SO1_printf("ExecSGD: %s\n", "SGD iteration ");

	estate->es_direction = ForwardScanDirection;

	// outerNode = ShuffleSortNode
	outerNode = outerPlanState(node);
	// tupDesc is the TupleDesc of the previous node
	tupDesc = ExecGetResultType(outerNode);
	int col_num = tupDesc->natts;
                                              
	// node->tupleShuffleSortState = (void *) tupleShuffleSortState;

	int iter_num = model->iter_num;
    int batch_size = node->model->batch_size;

	SGDTupleDesc* sgd_tupledesc = init_SGDTupleDesc(col_num, model->n_features);


	// for counting execution time for each iteration
	clock_t iter_start, iter_finish;
	double iter_exec_time;

	// for counting data parsing time
	clock_t parse_start, parse_finish;
	double parse_time = 0;

	// for counting the computation time
	clock_t comp_start, comp_finish;
	double comp_time = 0;

	// iterations
	for (int i = 1; i <= iter_num; i++) {
		iter_start = clock();

		int ith_tuple = 0;
		while(true) {
			// get a tuple from ShuffleSortNode
			slot = ExecProcNode(outerNode);

			if (TupIsNull(slot)) {
				perform_SGD(node->model, NULL, batchstate, ith_tuple);

				if (i == 1) {
					double avg_page_tuple_num = (double) model->tuple_num / table_page_number;
					elog(LOG, "[Computed Param] table_tuple_num = %d, buffer_block_num = %.2f", 
						model->tuple_num, (double) set_buffer_tuple_num / (set_block_page_num * avg_page_tuple_num));
				}

				iter_finish = clock();
				iter_exec_time = (double)(iter_finish - iter_start) / CLOCKS_PER_SEC; 
				double read_time = iter_exec_time - parse_time - comp_time;
				elog(LOG, "[Iter %2d] Loss = %.2f, exec_t = %.2fs, read_t = %.2fs, parse_t = %.2fs, comp_t = %.2fs", 
							i, model->total_loss, iter_exec_time, read_time, parse_time, comp_time);

				if (i == iter_num) { // finish
					if (set_run_test == false) {
						free_SGDBatchState(batchstate);
						free_SGDTuple(sgd_tuple);
						free_SGDTupleDesc(sgd_tupledesc);
					}  	
					else {
						ExecReScan(outerNode);
					}
					break;	
				}
				else { // for the next iteration
					model->total_loss = 0;
					parse_time = 0;
					comp_time = 0;
					clear_SGDBatchState(batchstate, model->n_features);
					ExecReScan(outerNode);	
					break;
				}
			}

			parse_start = clock();
			fast_transfer_slot_to_sgd_tuple(slot, sgd_tuple, sgd_tupledesc);
			parse_finish = clock();
			parse_time += (double)(parse_finish - parse_start) / CLOCKS_PER_SEC;    

			comp_start = clock();
			perform_SGD(node->model, sgd_tuple, batchstate, ith_tuple);
			comp_finish = clock();
			comp_time += (double)(comp_finish - comp_start) / CLOCKS_PER_SEC;

            ith_tuple = (ith_tuple + 1) % batch_size;

			if (i == 1)
				model->tuple_num = model->tuple_num + 1;
		}

		// decay the learning rate with 0.95^iter_num
		model->learning_rate = model->learning_rate * 0.95;

		if (set_run_test) {
			while(true) {
				slot = ExecProcNode(outerNode);
				
				if (TupIsNull(slot)) {
					test_state->test_accuracy = (double) test_state->right_count / model->tuple_num;
					
					elog(LOG, "[Iter %2d][Test] test_total_loss = %.2f, test_accuracy = %.2f", 
							i, test_state->test_total_loss, test_state->test_accuracy);

					if (i == iter_num) { // finish
						free_SGDBatchState(batchstate);
						free_SGDTuple(sgd_tuple);
						free_SGDTupleDesc(sgd_tupledesc);
						free_TestState(test_state);
						break;	
					}
					else { // for the next iteration
						clear_TestState(test_state);
						ExecReScan(outerNode);	
						break;
					}
				}
				fast_transfer_slot_to_sgd_tuple(slot, sgd_tuple, sgd_tupledesc);
				compute_tuple_accuracy(node->model, sgd_tuple, test_state);
			}

		}
	
	}
		
	node->sgd_done = true;
	SO1_printf("ExecSGD: %s\n", "Performing SGD done");

	// Get the first or next tuple from tuplesort. Returns NULL if no more tuples.

    // node->ps.ps_ResultTupleSlot; // = Model->w, => ExecStoreMinimalTuple();
	// slot = node->ps.ps_ResultTupleSlot;
	// elog(LOG, "[Model total loss %f]", model->total_loss);

	// slot = output_model_record(node->ps.ps_ResultTupleSlot, model);

	return slot;
}

void compute_tuple_accuracy(Model* model, SGDTuple* tp, TestState* test_state) {
	double y = tp->class_label;
    double* x = tp->features;
	int class_label = tp->class_label;

    int n = model->n_features;
	double tuple_loss = 0;
	


    // compute loss of the incoming tuple
	if (strcmp(model->model_name, "LR") == 0) {
		double wx = 0;
    	for (int i = 0; i < n; i++)
        	wx = wx + model->w[i] * x[i];
    	double ywx = y * wx;
		tuple_loss = log(1 + exp(-ywx));
		
		// By default, if f(wx) > 0.5, the outcome is positive, or negative otherwise
		double f_wx = 1 / (1 + exp(-wx));
		if (f_wx >= 0.5 && class_label == 1) {
			test_state->right_count += 1;
		}
		else if (f_wx < 0.5 && class_label == -1) {
			test_state->right_count += 1;
		}
			


	}
	else if (strcmp(model->model_name, "SVM") == 0) {
		double wx = 0;
		for (int i = 0; i < n; i++)
			wx = wx + model->w[i] * x[i];
		double ywx = y * wx;
		// compute the loss of the incoming tuple
		tuple_loss = 1 - ywx;
		
		if (tuple_loss < 0)
        	tuple_loss = 0;

		//  if wx >= 0 then the outcome is positive, and negative otherwise.
		if (wx >= 0 && class_label == 1) {
			test_state->right_count += 1;
		} 
		else if (wx < 0 && class_label == -1) {
			test_state->right_count += 1;
		}
	}

	test_state->test_total_loss += tuple_loss;
}

/* ----------------------------------------------------------------
 *		ExecInitLimit
 *
 *		This initializes the limit node state structures and
 *		the node's subplan.
 * ----------------------------------------------------------------
 */
LimitState *
ExecInitLimit(Limit *node, EState *estate, int eflags)
{
	LimitState  *sgdstate;

	SO1_printf("ExecInitSGD: %s\n",
			   "initializing SGD node");

	//
	const char* work_mem_str = GetConfigOption("work_mem", false, false);
	elog(LOG, "============== Begin Training on %s Using %s Model ==============", set_table_name, set_model_name);
	// elog(LOG, "[Param] model_name = %s", set_model_name);
	elog(LOG, "[Param] run_test = %d", set_run_test);
	elog(LOG, "[Param] work_mem = %s KB", work_mem_str);
	elog(LOG, "[Param] block_page_num = %d pages", set_block_page_num);
	// elog(LOG, "[Param] io_block_size = %d pages", set_io_big_block_size);
	elog(LOG, "[Param] buffer_tuple_num = %d tuples", set_buffer_tuple_num);
	elog(LOG, "[Param] batch_size = %d", set_batch_size);
	elog(LOG, "[Param] iter_num = %d", set_iter_num);
	elog(LOG, "[Param] learning_rate = %f", set_learning_rate);


	/*
	 * create state structure
	 */
	sgdstate = makeNode(LimitState);
	sgdstate->ps.plan = (Plan *) node;
	sgdstate->ps.state = estate;
    sgdstate->sgd_done = false;


	int n_features;
	if (strcmp(set_table_name, "dblife") == 0)
		// for dblife
		n_features = 41270; 
	else if (strcmp(set_table_name, "forest") == 0)
		// for forest
   	 	n_features = 54;
    
	

    sgdstate->model = init_model(n_features);
	

	elog(LOG, "[Model] Initialize %s model", sgdstate->model->model_name);
    // elog(LOG, "[SVM] loss = 0, p1 = 0, p2 = 0, gradient = 0, batch_size = 10, learning_rate = 0.1");

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
ExecEndLimit(LimitState *node)
{
	/*
	 * Free the exprcontext
	 */
	ExecFreeExprContext(&node->ps);

	/*
	 * clean out the tuple table
	 */
	ExecClearTuple(node->ps.ps_ResultTupleSlot);

	ExecClearModel(node->model);

	ExecEndNode(outerPlanState(node));

	SO1_printf("ExecEndSGD: %s\n",
			   "SGD node shutdown");
}


void
ExecReScanLimit(LimitState *node)
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
