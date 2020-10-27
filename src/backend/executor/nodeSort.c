/*-------------------------------------------------------------------------
 *
 * nodeSort.c
 *	  Routines to handle sorting of relations.
 *
 * Portions Copyright (c) 1996-2012, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/backend/executor/nodeSort.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "executor/execdebug.h"
#include "executor/nodeSort.h"
#include "miscadmin.h"
#include "utils/tuplesort.h"
#include "utils/array.h"
#include "time.h"
#include "math.h"




Model* init_model(int n_features) {
    Model* model = (Model *) palloc0(sizeof(Model));

	/* for dblife 
	model->total_loss = 0;
    model->batch_size = 500;
    model->learning_rate = 0.5;
    model->n_features = n_features;
	model->tuple_num = 0;
	model->iter_num = 10; // to change
	*/

	/* for forest */
	model->total_loss = 0;
    model->batch_size = 1000;
    model->learning_rate = 0.5;
    model->n_features = n_features;
	model->tuple_num = 0;
	model->iter_num = 50; // to change
    
	model->w = (double *) palloc0(sizeof(double) * n_features);

    for (int i = 0; i < n_features; i++) {
        // model->gradient[i] = 0;
        model->w[i] = 0;
    }

    return model;
}

void ExecClearModel(Model* model) {
    // free(model->gradient);
	pfree(model->w);
    pfree(model);
}

SGDBatchState* init_SGDBatchState(int n_features) {
    SGDBatchState* batchstate = (SGDBatchState *) palloc0(sizeof(SGDBatchState));
    batchstate->gradients = (double *) palloc0(sizeof(double) * n_features);
	for (int i = 0; i < n_features; i++)
		batchstate->gradients[i] = 0;
    batchstate->loss = 0;
    return batchstate;
}

SGDTuple* init_SGDTuple(int n_features) {
    SGDTuple* sgd_tuple = (SGDTuple *) palloc0(sizeof(SGDTuple));
    sgd_tuple->features = (double *) palloc0(sizeof(double) * n_features);
    return sgd_tuple;
}

SGDTupleDesc* init_SGDTupleDesc(int col_num, int n_features) {
    SGDTupleDesc* sgd_tupledesc = (SGDTupleDesc *) palloc0(sizeof(SGDTupleDesc));

    sgd_tupledesc->values = (Datum *) palloc0(sizeof(Datum) * col_num);
	sgd_tupledesc->isnulls = (bool *) palloc0(sizeof(bool) * col_num);

	// just for dblife: 
	/*
	CREATE TABLE dblife (
	did serial,
	k integer[],
	v double precision[],
	label integer);
	*/
	/* for dblife 
	sgd_tupledesc->k_col = 1; 
	sgd_tupledesc->v_col = 2;
	sgd_tupledesc->label_col = 3;
	sgd_tupledesc->n_features = n_features;
	*/

	/* for forest */ 
	sgd_tupledesc->k_col = -1; // from 0
	sgd_tupledesc->v_col = 1;
	sgd_tupledesc->label_col = 2;
	sgd_tupledesc->n_features = n_features;


    return sgd_tupledesc;
}

void clear_SGDBatchState(SGDBatchState* batchstate, int n_features) {
	for (int i = 0; i < n_features; i++)
		batchstate->gradients[i] = 0;
    batchstate->loss = 0;
}

void free_SGDBatchState(SGDBatchState* batchstate) {
    pfree(batchstate->gradients);
    pfree(batchstate);
}

void free_SGDTuple(SGDTuple* sgd_tuple) {
    pfree(sgd_tuple->features);
    pfree(sgd_tuple);
}

void free_SGDTupleDesc(SGDTupleDesc* sgd_tupledesc) {
    pfree(sgd_tupledesc->values);
    pfree(sgd_tupledesc->isnulls);
	pfree(sgd_tupledesc);
}

void
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
        batchstate->gradients[i] = batchstate->gradients[i] + g_base * x[i];

    // compute the loss of the incoming tuple
    batchstate->loss = batchstate->loss + tuple_loss;
}

void
compute_tuple_gradient_loss_SVM(SGDTuple* tp, Model* model, SGDBatchState* batchstate)
{
    double y = tp->class_label;
    double* x = tp->features;

    int n = model->n_features;

    // double loss = 0;
    double grad[n];

    // compute gradients of the incoming tuple
    double wx = 0;
    for (int i = 0; i < n; i++)
        wx = wx + model->w[i] * x[i];
    double ywx = y * wx;
    if (1 - ywx > 0) {
        for (int i = 0; i < n; i++)
            grad[i] = -y * x[i];
    }
    else {
        for (int i = 0; i < n; i++) 
            grad[i] = 0;
    }

    // Add this tuple's gradient to the previous gradients in this batch
    for (int i = 0; i < n; i++) 
        batchstate->gradients[i] = batchstate->gradients[i] + grad[i];

    // compute the loss of the incoming tuple
    double tuple_loss = 1 - ywx;
    if (tuple_loss < 0)
        tuple_loss = 0;
    batchstate->loss = batchstate->loss + tuple_loss;
}

void update_model(Model* model, SGDBatchState* batchstate) {

    // add graidents to the model and clear the batch gradients
    for (int i = 0; i < model->n_features; i++) {
        model->w[i] = model->w[i] - model->learning_rate * batchstate->gradients[i];
        batchstate->gradients[i] = 0;
    }

    model->total_loss = model->total_loss + batchstate->loss;
    batchstate->loss = 0;
}


void perform_SGD(Model *model, SGDTuple* sgd_tuple, SGDBatchState* batchstate, int i) {
    if (sgd_tuple == NULL) /* slot == NULL means the end of the table. */
        update_model(model, batchstate);
    else {
        // add the batch's gradients to the model, and reset the batch's gradients.
        compute_tuple_gradient_loss_LR(sgd_tuple, model, batchstate);
        if (i == model->batch_size - 1) 
            update_model(model, batchstate);
        
    }   
}


// Extract features and class label from Tuple
void
transfer_slot_to_sgd_tuple(
	TupleTableSlot* slot, 
	SGDTuple* sgd_tuple, 
	SGDTupleDesc* sgd_tupledesc) {

	// store the values of slot to values/isnulls arrays
	/* slot => Datum values/isnulls */
	heap_deform_tuple(slot->tts_tuple, slot->tts_tupleDescriptor, sgd_tupledesc->values, sgd_tupledesc->isnulls);
	// DatumGetInt32
	// tupleDesc->attrs[0]->atttypid

	int k_col = sgd_tupledesc->k_col;
	int v_col = sgd_tupledesc->v_col;
	int label_col = sgd_tupledesc->label_col;

	/* Datum => double/int */
	// e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
	// Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
	Datum v_dat = sgd_tupledesc->values[v_col]; // Datum{0.1, 0.2, 0.3}
	Datum label_dat = sgd_tupledesc->values[label_col]; // Datum{-1}


	/* feature datum arrary => double* v */
	ArrayType  *v_array = DatumGetArrayTypeP(v_dat);
	//Assert(ARR_ELEMTYPE(array) == FLOAT4OID);
	int	v_num = ArrayGetNItems(ARR_NDIM(v_array), ARR_DIMS(v_array));
	double *v = (double *) ARR_DATA_PTR(v_array);
	// for (int i = 0; i < v_num; i++)
	// 	elog(LOG, "%f, ", v[i]);


	/* label dataum => int class_label */
	int label = DatumGetInt32(label_dat);
	sgd_tuple->class_label = label;


	/* double* v => double* features */
	double* features = sgd_tuple->features;
	int n_features = sgd_tupledesc->n_features;
	// if sparse dataset
	if (k_col >= 0) {
		/* k Datum array => int* k */
		Datum k_dat = sgd_tupledesc->values[k_col]; // Datum{0, 2, 5}
		ArrayType  *k_array = DatumGetArrayTypeP(k_dat);
		int	k_num = ArrayGetNItems(ARR_NDIM(k_array), ARR_DIMS(k_array));
		int *k = (int *) ARR_DATA_PTR(k_array);

		// TODO: change to memset()
		// for (int i = 0; i < n_features; i++) {
		// 	features[i] = 0;
		// }
		memset(features, 0, sizeof(double) * n_features);

		for (int i = 0; i < k_num; i++) {
			int f_index = k[i]; // {0, 2, 5}, k[1] = 2
			features[f_index] = v[i]; // {0.1, 0.2, 0.3}, features[2] = 0.2
		}
	}
	else {
		Assert(n_features == v_num);
		for (int i = 0; i < v_num; i++) {
			features[i] = v[i];
		}
	}
	
}

/* ----------------------------------------------------------------
 *		ExecSort
 *
 *		Sorts tuples from the outer subtree of the node using tuplesort,
 *		which saves the results in a temporary file or memory. After the
 *		initial call, returns a tuple from the file with each call.
 *
 *		Conditions:
 *		  -- none.
 *
 *		Initial States:
 *		  -- the outer child is prepared to return the first tuple.
 * ----------------------------------------------------------------
 */
TupleTableSlot *
ExecSort(SortState *node)
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

	// for counting data parsing time
	clock_t parse_start, parse_finish;
	double parse_time = 0;

	// for counting the computation time
	clock_t comp_start, comp_finish;
	double comp_time = 0;

	// iterations
	for (int i = 1; i <= iter_num; i++) {
		int ith_tuple = 0;
		while(true) {
			// get a tuple from ShuffleSortNode
			slot = ExecProcNode(outerNode);

			if (TupIsNull(slot)) {
				if (i == iter_num) {
					// elog(LOG, "[Iteartion %d] Finalize the model.", i);
					perform_SGD(node->model, NULL, batchstate, ith_tuple);
					elog(LOG, "[Finish iteartion %d] Loss = %f, parse_t = %fs, comp_t = %fs", 
							i, model->total_loss, parse_time, comp_time);
                	// can also free_SGDBatchState in ExecEndSGD
                	free_SGDBatchState(batchstate);
					free_SGDTuple(sgd_tuple);
					free_SGDTupleDesc(sgd_tupledesc);
					break;	
				}
				else {
					// Current iteration ends, update model and print metrics
					perform_SGD(node->model, NULL, batchstate, ith_tuple);
					elog(LOG, "[Finish iteartion %d] Loss = %f, parse_t = %fs, comp_t = %fs", 
							i, model->total_loss, parse_time, comp_time);
					model->total_loss = 0;
					parse_time = 0;
					comp_time = 0;
					clear_SGDBatchState(batchstate, model->n_features);
					ExecReScan(outerNode);	
					break;
				}
			}

			parse_start = clock();
			transfer_slot_to_sgd_tuple(slot, sgd_tuple, sgd_tupledesc);
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


// TupleTableSlot *
// ExecSort(SortState *node)
// {
// 	EState	   *estate;
// 	TupleTableSlot *slot;
// 	Model* model = node->model;

// 	/*
// 	 * get state info from node
// 	 */
// 	SO1_printf("ExecSGD: %s\n", "entering routine");

// 	estate = node->ps.state;
	
// 	// tupleSGDState = (TupleSGDState *) node->tupleSGDState;

// 	/*
// 	 * If first time through, read all tuples from outer plan and pass them to
// 	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
// 	 */
// 	if (!node->sgd_done)
// 	{
// 		SGDBatchState* batchstate = init_SGDBatchState(model->n_features);
// 		SGDTuple* sgd_tuple = init_SGDTuple(model->n_features);
// 		//Datum values[model->n_features + model-> slot->]
// 		// ShuffleSort	   *plannode = (ShuffleSort *) node->ss.ps.plan;
// 		PlanState  *outerNode;
// 		TupleDesc	tupDesc;

// 		SO1_printf("ExecSGD: %s\n", "SGD iteration ");

// 		estate->es_direction = ForwardScanDirection;

// 		// outerNode = ShuffleSortNode
// 		outerNode = outerPlanState(node);
// 		// tupDesc is the TupleDesc of the previous node
// 		tupDesc = ExecGetResultType(outerNode);
// 		int col_num = tupDesc->natts;
                                              
// 		// node->tupleShuffleSortState = (void *) tupleShuffleSortState;

// 		int iter_num = model->iter_num;
//         int batch_size = node->model->batch_size;

// 		SGDTupleDesc* sgd_tupledesc = init_SGDTupleDesc(col_num, model->n_features);

// 		// iterations
// 		for (int i = 1; i <= iter_num; i++) {
// 			int ith_tuple = 0;
// 			while(true) {
// 				// get a tuple from ShuffleSortNode
// 				slot = ExecProcNode(outerNode);

// 				if (TupIsNull(slot)) {
// 					if (i == iter_num) {
// 						elog(LOG, "[Iteartion %d] Finalize the model.", i);
// 						perform_SGD(node->model, NULL, batchstate, ith_tuple);
//                 		// can also free_SGDBatchState in ExecEndSGD
//                 		free_SGDBatchState(batchstate);
// 						free_SGDTuple(sgd_tuple);
// 						free_SGDTupleDesc(sgd_tupledesc);
// 						break;	
// 					}
// 					else {
// 						elog(LOG, "[Iteartion %d] Finish current iteration.", i);
// 						perform_SGD(node->model, NULL, batchstate, ith_tuple);
// 						clear_SGDBatchState(batchstate, model->n_features);
// 						ExecReScan(outerNode);	
// 						break;
// 					}
// 				}

// 				transfer_slot_to_sgd_tuple(slot, sgd_tuple, sgd_tupledesc);
// 				perform_SGD(node->model, sgd_tuple, batchstate, ith_tuple);
//             	ith_tuple = (ith_tuple + 1) % batch_size;

// 				if (i == 1)
// 					model->tuple_num = model->tuple_num + 1;
// 			}
// 		}
		
// 		node->sgd_done = true;
// 		SO1_printf("ExecSGD: %s\n", "Performing SGD done");
// 	}

// 	// Get the first or next tuple from tuplesort. Returns NULL if no more tuples.

//     // node->ps.ps_ResultTupleSlot; // = Model->w, => ExecStoreMinimalTuple();
// 	// slot = node->ps.ps_ResultTupleSlot;

// 	SO1_printf("Model total loss: %s\n", model->total_loss);

// 	// slot = output_model_record(node->ps.ps_ResultTupleSlot, model);

// 	return slot;
// }

/* ----------------------------------------------------------------
 *		ExecInitSort
 *
 *		Creates the run-time state information for the sort node
 *		produced by the planner and initializes its outer subtree.
 * ----------------------------------------------------------------
 */
SortState *
ExecInitSort(Sort *node, EState *estate, int eflags)
{
	SortState  *sgdstate;

	SO1_printf("ExecInitSGD: %s\n",
			   "initializing SGD node");

	/*
	 * create state structure
	 */
	sgdstate = makeNode(SortState);
	sgdstate->ps.plan = (Plan *) node;
	sgdstate->ps.state = estate;
    sgdstate->sgd_done = false;

	// for forest
    int n_features = 54;
    
	// for dblife
	// int n_features = 41270; 

    sgdstate->model = init_model(n_features);
	
	elog(LOG, "[SVM] Initialize SVM model");
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
 *		ExecEndSort(node)
 * ----------------------------------------------------------------
 */
void
ExecEndSort(SortState *node)
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

/* ----------------------------------------------------------------
 *		ExecSortMarkPos
 *
 *		Calls tuplesort to save the current position in the sorted file.
 * ----------------------------------------------------------------
 */
void
ExecSortMarkPos(SortState *node)
{
	/*
	 * if we haven't sorted yet, just return
	 */
	// if (!node->sort_Done)
	// 	return;

	// tuplesort_markpos((Tuplesortstate *) node->tuplesortstate);
}

/* ----------------------------------------------------------------
 *		ExecSortRestrPos
 *
 *		Calls tuplesort to restore the last saved sort file position.
 * ----------------------------------------------------------------
 */
void
ExecSortRestrPos(SortState *node)
{
	// /*
	//  * if we haven't sorted yet, just return.
	//  */
	// if (!node->sort_Done)
	// 	return;

	// /*
	//  * restore the scan to the previously marked position
	//  */
	// tuplesort_restorepos((Tuplesortstate *) node->tuplesortstate);
}

void
ExecReScanSort(SortState *node)
{

}
