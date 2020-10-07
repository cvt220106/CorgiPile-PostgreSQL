
#include "postgres.h"

#include "executor/execdebug.h"
#include "executor/nodeSGD.h"
#include "include/catalog/pg_type.h"
#include "include/utils/array.h"
//#include "utils/rel.h"

/* transfering TupleTableSlot to SGDTuple */

// typedef struct
// {
// 	void	    *tuple;			/* the tuple proper */
// 	Datum		feature_x1;			/* value of first key column */
//     Datum		feature_x2;			
//     Datum		feature_x3;			
//     Datum		feature_x4;			
//     Datum		label_y;			
// 	// bool		isnull1;		/* is first key column NULL? */
// 	int			tupindex;		/* see notes above */
// } SGDTuple;


typedef struct SGDBatchState
{
	double*		gradients;	  /* sum the gradient of each tuple in a batch */		
    double		loss;	  /* sum the loss of each tuple in a batch */		
} SGDBatchState;


typedef struct SGDTuple
{
	double*		features;		/* features of a tuple */	
    double		label;			/* the label of a tuple */
	int			tupindex;		/* see notes above */
} SGDTuple;


Model* init_model(int n_features) {
    Model* model = (Model *) palloc0(sizeof(Model));

	model->total_loss = 0;
    model->batch_size = 10;
    model->learning_rate = 0.1;
    model->n_features = n_features;
	model->tuple_num = 0;

    
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

void clear_SGDBatchState(SGDBatchState* batchstate, int n_features) {
	for (int i = 0; i < n_features; i++)
		batchstate->gradients[i] = 0;
    batchstate->loss = 0;
}

void free_SGDBatchState(SGDBatchState* batchstate) {
    pfree(batchstate->gradients);
    pfree(batchstate);
}

void
compute_tuple_gradient_loss(SGDTuple* tp, Model* model, SGDBatchState* batchstate)
{
    double y = tp->label;
    double* x = tp->features;

    int n = model->n_features;

    double loss = 0;
    double grad[n];

    // compute gradients of the incoming tuple
    double wx = 0;
    for (int i = 0; i < n; i++)
        wx = wx + model->w[i] * x[i];
    double ywx = y * wx;
    if (ywx < 1) {
        for (int i = 0; i < n; i++)
            grad[i] = -y * x[i];
    }
    else {
        for (int i = 0; i < n; i++) 
            grad[i] = 0;
    }

    // Add this tuple's gradient to the previous gradients in this batch
    for (int i = 0; i < n; i++) 
        batchstate->gradients[i] += grad[i];

    // compute the loss of the incoming tuple
    double tuple_loss = 1 - ywx;
    if (tuple_loss < 0)
        tuple_loss = 0;
    batchstate->loss += tuple_loss;
}

/* ----------------------------------------------------------------
 *		ExecInitSGD
 * ----------------------------------------------------------------
 */
SGDState *
ExecInitSGD(SGD *node, EState *estate, int eflags)
{
	SGDState  *sgdstate;

	SO1_printf("ExecInitSGD: %s\n",
			   "initializing SGD node");

	/*
	 * create state structure
	 */
	sgdstate = makeNode(SGDState);
	sgdstate->ps.plan = (Plan *) node;
	sgdstate->ps.state = estate;
    sgdstate->sgd_done = false;

    int n_features = 4;
    
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

void update_model(Model* model, SGDBatchState* batchstate) {

    // add graidents to the model and clear the batch gradients
    for (int i = 0; i < model->n_features; i++) {
        model->w[i] = model->w[i] + model->learning_rate * batchstate->gradients[i];
        batchstate->gradients[i] = 0;
    }

    model->total_loss = model->total_loss + batchstate->loss;
    batchstate->loss = 0;
}


void perform_SGD(Model *model, TupleTableSlot *slot, SGDBatchState* batchstate, int i) {
    if (slot == NULL) /* slot == NULL means the end of the table. */
        update_model(model, batchstate);
    else {
        // add the batch's gradients to the model, and reset the batch's gradients.
        compute_tuple_gradient_loss(model, slot, batchstate);
        if (i == model->batch_size - 1) 
            update_model(model, batchstate);
        
    }   
}

/*
TupleTableSlot *
ExecSGD(SGDState *node)
{

    // model <- 1 update (fetch all the tuples)

	EState	   *estate;
	TupleTableSlot *slot;
	Model* model = node->model;

	//get state info from node
	SO1_printf("ExecSGD: %s\n",
			   "entering routine");

	estate = node->ps.state;
	
	// tupleSGDState = (TupleSGDState *) node->tupleSGDState;
    SGDBatchState* batchstate = init_SGDBatchState(model->n_features);
	// If first time through, read all tuples from outer plan and pass them to
	// tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
  
	if (!node->sgd_done)
	{
		// ShuffleSort	   *plannode = (ShuffleSort *) node->ss.ps.plan;
		PlanState  *outerNode;
		TupleDesc	tupDesc;

		SO1_printf("ExecSGD: %s\n",
				   "SGD subplan");

		estate->es_direction = ForwardScanDirection;

		// outerNode = ShuffleScanNode
		outerNode = outerPlanState(node);
		tupDesc = ExecGetResultType(outerNode);
                                              
		// node->tupleShuffleSortState = (void *) tupleShuffleSortState;

		int iter_num = model->iter_num;
		int ith_iter = 1;
        int ith_tuple = 0;

		for (;;)
		{
			// Lijie: read a tuple from the previous node (e.g., ShuffleSort)
			slot = ExecProcNode(outerNode);

			if (TupIsNull(slot)) {
				if (ith_iter == iter_num) {
					elog(LOG, "[SVM] Finalize the model.");
					perform_SGD(node->model, NULL, batchstate, ith_tuple);
                	// can also free_SGDBatchState in ExecEndSGD
                	free_SGDBatchState(batchstate);
					break;
				}
				else {
					elog(LOG, "[SVM] Finish Iteration %d.", ith_iter);
					perform_SGD(node->model, NULL, batchstate, ith_tuple);
					free_SGDBatchState(batchstate);
					ExecReScan(outerNode);
					ith_tuple = 0;
				}
				++ith_iter;
			}

            
            perform_SGD(node->model, slot, batchstate, ith_tuple);
            ith_tuple = (ith_tuple + 1) % node->model->batch_size;
			 	
		}

        estate->es_direction = ForwardScanDirection;
		
		node->sgd_done = true;
		SO1_printf("ExecSGD: %s\n", "Performing SGD done");
	}

	// Get the first or next tuple from tuplesort. Returns NULL if no more tuples.

    // TODO: using ExecStoreMinimalTuple to genreate the result tuple
    node->ps.ps_ResultTupleSlot; // = Model->w, => ExecStoreMinimalTuple();
	slot = node->ps.ps_ResultTupleSlot;

	// (void) tupleshufflesort_gettupleslot(tupleShuffleSortState,
	// 							  ScanDirectionIsForward(dir),
	// 							  slot);
	return slot;
}
*/

TupleTableSlot *
ExecSGD(SGDState *node)
{

    // model <- 1 update (fetch all the tuples)

	EState	   *estate;
	TupleTableSlot *slot;
	Model* model = node->model;

	/*
	 * get state info from node
	 */
	SO1_printf("ExecSGD: %s\n",
			   "entering routine");

	estate = node->ps.state;
	
	// tupleSGDState = (TupleSGDState *) node->tupleSGDState;
    SGDBatchState* batchstate = init_SGDBatchState(model->n_features);
	/*
	 * If first time through, read all tuples from outer plan and pass them to
	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
	 */
    //for (n )
	if (!node->sgd_done)
	{
		// ShuffleSort	   *plannode = (ShuffleSort *) node->ss.ps.plan;
		PlanState  *outerNode;
		TupleDesc	tupDesc;

		SO1_printf("ExecSGD: %s\n",
				   "SGD subplan");

		estate->es_direction = ForwardScanDirection;

		// outerNode = ShuffleScanNode
		outerNode = outerPlanState(node);
		tupDesc = ExecGetResultType(outerNode);
                                              
		// node->tupleShuffleSortState = (void *) tupleShuffleSortState;

		int iter_num = model->iter_num;
        
		for (int i = 1; i <= iter_num; i++) {
			int ith_tuple = 0;
			while(true) {
				slot = ExecProcNode(outerNode);

				if (TupIsNull(slot)) {
					if (i == iter_num) {
						elog(LOG, "[Iteartion %d] Finalize the model.", i);
						perform_SGD(node->model, NULL, batchstate, ith_tuple);
                		// can also free_SGDBatchState in ExecEndSGD
                		free_SGDBatchState(batchstate);
						break;	
					}
					else {
						elog(LOG, "[Iteartion %d] Finish current iteration.", i);
						perform_SGD(node->model, NULL, batchstate, ith_tuple);
						clear_SGDBatchState(batchstate, model->n_features);
						ExecReScan(outerNode);	
						break;
					}
				}
				perform_SGD(node->model, slot, batchstate, ith_tuple);
            	ith_tuple = (ith_tuple + 1) % node->model->batch_size;

				if (i == 1)
					model->tuple_num = model->tuple_num + 1;
			}
		}
		
		node->sgd_done = true;
		SO1_printf("ExecSGD: %s\n", "Performing SGD done");
	}

	// Get the first or next tuple from tuplesort. Returns NULL if no more tuples.

    // TODO: using ExecStoreMinimalTuple to genreate the result tuple
    // node->ps.ps_ResultTupleSlot; // = Model->w, => ExecStoreMinimalTuple();
	// slot = node->ps.ps_ResultTupleSlot;

	slot = output_model_record(node->ps.ps_ResultTupleSlot, model);

	
	// (void) tupleshufflesort_gettupleslot(tupleShuffleSortState,
	// 							  ScanDirectionIsForward(dir),
	// 							  slot);
	return slot;
}

/*
 * This utility function takes a C array of Oids, and returns a Datum
 * pointing to a one-dimensional Postgres array of regtypes. An empty
 * array is returned as a zero-element array, not NULL.
 */
static Datum
build_float_array(Model* model)
{
	Datum	   *tmp_ary;
	ArrayType  *result;
	int			i;
	int num_params = model->n_features;

	tmp_ary = (Datum *) palloc(num_params * sizeof(Datum));

	for (i = 0; i < num_params; i++)
		tmp_ary[i] = Float4GetDatum(model->w[i]);

	result = construct_array(tmp_ary, num_params, FLOAT4OID, 4, true, 'i');
	return PointerGetDatum(result);
}

/*
SVM output record from MADLib
-[ RECORD 1 ]------+--------------------------------------------------------------------------------
coef               | {0.103994021495116,-0.00288252192097756,0.0540748706580464,0.00131729978010033}
loss               | 0.928463796644648
norm_of_gradient   | 7849.34910604307
num_iterations     | 100
num_rows_processed | 15
num_rows_skipped   | 0
dep_var_mapping    | {f,t}
*/
TupleTableSlot* output_model_record(TupleTableSlot* slot, Model* model) {

	int columns = 5;
	TupleDesc tupdesc = CreateTemplateTupleDesc(columns, false);
	TupleDescInitEntry(tupdesc, (AttrNumber) 1, "coef",
					   FLOAT4ARRAYOID, -1, 0);
	TupleDescInitEntry(tupdesc, (AttrNumber) 2, "loss",
					   FLOAT4OID, -1, 0);
	TupleDescInitEntry(tupdesc, (AttrNumber) 3, "gradient",
					   FLOAT4OID, -1, 0);
	TupleDescInitEntry(tupdesc, (AttrNumber) 4, "num_iterations",
					   INT4OID, -1, 0);
	TupleDescInitEntry(tupdesc, (AttrNumber) 5, "num_rows_processed",
					   INT4OID, -1, 0);

	Datum		values[columns];
	bool		nulls[columns];

	// coef: i.e., model-w
	values[0] = build_float_array(model);
	nulls[0] = false;

	// loss
	values[1] = Float4GetDatum(model->total_loss);
	nulls[1] = false;
	// norm of gradient
	values[2] = Float4GetDatum(0);
	nulls[2] = false;
	// num_iterationss
	values[3] = Int32GetDatum(model->iter_num);
	nulls[3] = false;
	// num_rows_processed
	values[4] = Int32GetDatum(model->tuple_num);
	nulls[4] = false;

	MinimalTuple mtuple = heap_form_minimal_tuple(tupdesc, values, nulls);
	bool should_free = true;
	slot = ExecStoreMinimalTuple(mtuple, slot, should_free);
	return slot;
}




/* ----------------------------------------------------------------
 *		ExecEndSeqScan
 *
 *		frees any storage allocated through C routines.
 * ----------------------------------------------------------------
 */
void
ExecEndSGD(SGDState *node)
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
 *						Join Support
 * ----------------------------------------------------------------
 */

/* ----------------------------------------------------------------
 *		ExecReScanSeqScan
 *
 *		Rescans the relation.
 * ----------------------------------------------------------------
 */
void
ExecReScanSGD(SGDState *node)
{
	ExecScanReScan((ScanState *) node);
}

/* ----------------------------------------------------------------
 *		ExecSeqMarkPos(node)
 *
 *		Marks scan position.
 * ----------------------------------------------------------------
 */
void
ExecSeqMarkPos(SeqScanState *node)
{
	HeapScanDesc scan = node->ss_currentScanDesc;

	heap_markpos(scan);
}

/* ----------------------------------------------------------------
 *		ExecSeqRestrPos
 *
 *		Restores scan position.
 * ----------------------------------------------------------------
 */
void
ExecSeqRestrPos(SeqScanState *node)
{
	HeapScanDesc scan = node->ss_currentScanDesc;

	/*
	 * Clear any reference to the previously returned tuple.  This is needed
	 * because the slot is simply pointing at scan->rs_cbuf, which
	 * heap_restrpos will change; we'd have an internally inconsistent slot if
	 * we didn't do this.
	 */
	ExecClearTuple(node->ss_ScanTupleSlot);

	heap_restrpos(scan);
}


/* ----------------------------------------------------------------
 *		ExecResultMarkPos
 * ----------------------------------------------------------------
 */
void
ExecResultMarkPos(ResultState *node)
{
	PlanState  *outerPlan = outerPlanState(node);

	if (outerPlan != NULL)
		ExecMarkPos(outerPlan);
	else
		elog(DEBUG2, "Result nodes do not support mark/restore");
}

/* ----------------------------------------------------------------
 *		ExecResultRestrPos
 * ----------------------------------------------------------------
 */
void
ExecResultRestrPos(ResultState *node)
{
	PlanState  *outerPlan = outerPlanState(node);

	if (outerPlan != NULL)
		ExecRestrPos(outerPlan);
	else
		elog(ERROR, "Result nodes do not support mark/restore");
}

// int
// compute_loss_and_update_model(TupleShuffleSortState* state, Model* model,
// 							  int ith_tuple, int batch_size, bool last_tuple) 
// {
	
// 	ShuffleSortTuple* tuples = state->memtuples;
// 	int last_updated = 0;
// 	int i = 0;

// 	for (ShuffleSortTuple* p = tuples; p < tuples + n; p++) {
// 		double tuple_loss = compute_loss(p, model);
// 		model->loss = model->loss + tuple_loss;
// 		elog(LOG, "[SVM][Tuple %d] >>> Add %.2f loss to model.", ith_tuple, tuple_loss);
// 		ith_tuple = (ith_tuple + 1) % batch_size;
		
// 		// going to update model
// 		if (ith_tuple == 0) {
// 			// update model
// 			model->p1 += 1;
// 			model->p2 += 1;
// 			elog(LOG, "[SVM] >>> Update model (p1 = %d, p2 = %d, loss = %.2f).", model->p1, model->p2, model->loss);
// 			last_updated = i;
// 		}
// 		++i;

// 	}

// 	if (last_tuple) {
// 		if (n > 0 && last_updated < n - 1) {
// 			model->p1 += 1;
// 			model->p2 += 1;
// 			elog(LOG, "[SVM] >>> Last: Update model (p1 = %d, p2 = %d, loss = %.2f).", model->p1, model->p2, model->loss);
// 		}
// 		else {
// 			elog(LOG, "[SVM] >>> Has updated the model.");
// 		}
// 	}

// 	return ith_tuple;
// }