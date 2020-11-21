/*-------------------------------------------------------------------------
 *
 * nodeLimit.h
 *
 *
 *
 * Portions Copyright (c) 1996-2012, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * src/include/executor/nodeLimit.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef NODELIMIT_H
#define NODELIMIT_H

#include "nodes/execnodes.h"
#include "utils/sgdmodel.h"

extern LimitState *ExecInitLimit(Limit *node, EState *estate, int eflags);
extern TupleTableSlot *ExecLimit(LimitState *node);
extern void ExecEndLimit(LimitState *node);
extern void ExecReScanLimit(LimitState *node);


// for sgd operator

typedef struct SGDBatchState
{
	double*		gradients;	  /* sum the gradient of each tuple in a batch, n_dim */		
    double		loss;	      /* sum the loss of each tuple in a batch */
	int         tuple_num;		
} SGDBatchState;


typedef struct SGDTuple
{
	double*	 features;		/* features of a tuple, n_dim */	
    int		 class_label;	/* the class label of a tuple, -1 if there is not any label */
	// int			tupindex;		/* the ith-tuple */
} SGDTuple;

typedef struct TestState
{
	double test_total_loss;
	double test_accuracy;
	int right_count; // rightly classified
} TestState;

typedef struct SGDTupleDesc
{ 
	// e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
	// Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
	int k_col; // 1 // just for sparse dataset, if dense, only v_col is used.
	int v_col; // 2
	int label_col; // 3
	int n_features;  // 8
	
	Datum* values;
	bool* isnulls;
} SGDTupleDesc;


// extern Model* init_model(int n_features);
// extern void ExecClearModel(Model* model);
// extern SGDBatchState* init_SGDBatchState(int n_features);
// extern SGDTuple* init_SGDTuple(int n_features);
// extern SGDTupleDesc* init_SGDTupleDesc(int col_num, int n_features);
// extern void clear_SGDBatchState(SGDBatchState* batchstate, int n_features);
// extern void free_SGDBatchState(SGDBatchState* batchstate);
// extern void free_SGDTuple(SGDTuple* sgd_tuple);
// extern void free_SGDTupleDesc(SGDTupleDesc* sgd_tupledesc);
// extern void compute_tuple_gradient_loss(SGDTuple* tp, Model* model, SGDBatchState* batchstate);
// extern void update_model(Model* model, SGDBatchState* batchstate);
// extern void perform_SGD(Model *model, SGDTuple* sgd_tuple, SGDBatchState* batchstate, int i);
// extern void transfer_slot_to_sgd_tuple(TupleTableSlot* slot, SGDTuple* sgd_tuple, SGDTupleDesc* sgd_tupledesc);



#endif   /* NODELIMIT_H */
