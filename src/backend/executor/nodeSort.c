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

#include "catalog/pg_type.h"
#include "utils/array.h"


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

static Datum
build_float_array()
{
	Datum	   *tmp_ary;
	ArrayType  *result;
	int			i;
	int num_params = 4;

	tmp_ary = (Datum *) palloc(num_params * sizeof(Datum));

	for (i = 0; i < num_params; i++)
		tmp_ary[i] = Float4GetDatum(i);

	result = construct_array(tmp_ary, num_params, FLOAT4OID, 4, true, 'i');
	return PointerGetDatum(result);
}


TupleTableSlot* output_model_record(TupleTableSlot* slot) {

    int columns = 5;
    TupleDesc tupdesc = CreateTemplateTupleDesc(columns, false);
    TupleDescInitEntry(tupdesc, (AttrNumber) 1, "coef",
                       FLOAT4ARRAYOID, -1, 0);
	// TupleDescInitEntry(tupdesc, (AttrNumber) 1, "coef",
    //                    FLOAT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber) 2, "loss",
                       FLOAT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber) 3, "gradient",
                       FLOAT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber) 4, "num_iterations",
                       INT4OID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber) 5, "num_rows_processed",
                       INT4OID, -1, 0);

    Datum       values[columns];
    bool        nulls[columns];

    // coef: i.e., model-w
    values[0] = build_float_array();
	// values[0] = Float4GetDatum(0.000001);
    nulls[0] = false;

    // loss
    values[1] = Float4GetDatum(0.000005);
    nulls[1] = false;
    // norm of gradient
    values[2] = Float4GetDatum(0);
    nulls[2] = false;
    // num_iterationss
    values[3] = Int32GetDatum(100);
    nulls[3] = false;
    // num_rows_processed
    values[4] = Int32GetDatum(20);
    nulls[4] = false;

    // MinimalTuple mtuple = (MinimalTuple) heap_form_tuple(tupdesc, values, nulls);
    // bool should_free = true;
    // slot = ExecStoreMinimalTuple(mtuple, slot, should_free);
    // return slot;

	// HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    // bool should_free = true;
	// slot->tts_tupleDescriptor = tupdesc;
	// slot = ExecStoreTuple(tuple, slot, InvalidBuffer, should_free);
    
	/* make sure the slot is clear */
	
	ExecSetSlotDescriptor(slot, tupdesc);
	/* insert data */
	memcpy(slot->tts_values, values, columns * sizeof(Datum));
	memcpy(slot->tts_isnull, nulls, columns * sizeof(bool));
	
	/* mark slot as containing a virtual tuple */
	ExecStoreVirtualTuple(slot);

    return slot;
}


TupleTableSlot *
ExecSort(SortState *node)
{
	EState	   *estate;
	ScanDirection dir;
	Tuplesortstate *tuplesortstate;
	TupleTableSlot *slot;

	/*
	 * get state info from node
	 */
	SO1_printf("ExecSort: %s\n",
			   "entering routine");

	estate = node->ss.ps.state;
	dir = estate->es_direction;
	tuplesortstate = (Tuplesortstate *) node->tuplesortstate;

	/*
	 * If first time through, read all tuples from outer plan and pass them to
	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
	 */

	if (node->sort_Done)
		return NULL;

	if (!node->sort_Done)
	{
		Sort	   *plannode = (Sort *) node->ss.ps.plan;
		PlanState  *outerNode;
		TupleDesc	tupDesc;

		SO1_printf("ExecSort: %s\n",
				   "sorting subplan");

		/*
		 * Want to scan subplan in the forward direction while creating the
		 * sorted data.
		 */
		estate->es_direction = ForwardScanDirection;

		/*
		 * Initialize tuplesort module.
		 */
		SO1_printf("ExecSort: %s\n",
				   "calling tuplesort_begin");

		outerNode = outerPlanState(node);
		tupDesc = ExecGetResultType(outerNode);

		tuplesortstate = tuplesort_begin_heap(tupDesc,
											  plannode->numCols,
											  plannode->sortColIdx,
											  plannode->sortOperators,
											  plannode->collations,
											  plannode->nullsFirst,
											  work_mem,
											  node->randomAccess);
		if (node->bounded)
			tuplesort_set_bound(tuplesortstate, node->bound);
		node->tuplesortstate = (void *) tuplesortstate;

		/*
		 * Scan the subplan and feed all the tuples to tuplesort.
		 */
		// added by Lijie
		int i = 0;
		// added end
		for (;;)
		{
			slot = ExecProcNode(outerNode);

			if (TupIsNull(slot))
				break;
				
			else
				tuplesort_puttupleslot(tuplesortstate, slot);
		}

		/*
		 * Complete the sort.
		 */
		tuplesort_performsort(tuplesortstate);

		/*
		 * restore to user specified direction
		 */
		estate->es_direction = dir;

		/*
		 * finally set the sorted flag to true
		 */
		node->sort_Done = true;
		node->bounded_Done = node->bounded;
		node->bound_Done = node->bound;
		SO1_printf("ExecSort: %s\n", "sorting done");
	}

	SO1_printf("ExecSort: %s\n",
			   "retrieving tuple from tuplesort");

	/*
	 * Get the first or next tuple from tuplesort. Returns NULL if no more
	 * tuples.
	 */
	slot = node->ss.ps.ps_ResultTupleSlot;
	/* original code
	(void) tuplesort_gettupleslot(tuplesortstate,
								  ScanDirectionIsForward(dir),
								  slot);
	*/

	slot = output_model_record(slot);

	return slot;
}

// Original code:
// TupleTableSlot *
// ExecSort(SortState *node)
// {
// 	EState	   *estate;
// 	ScanDirection dir;
// 	Tuplesortstate *tuplesortstate;
// 	TupleTableSlot *slot;

// 	/*
// 	 * get state info from node
// 	 */
// 	SO1_printf("ExecSort: %s\n",
// 			   "entering routine");

// 	estate = node->ss.ps.state;
// 	dir = estate->es_direction;
// 	tuplesortstate = (Tuplesortstate *) node->tuplesortstate;

// 	/*
// 	 * If first time through, read all tuples from outer plan and pass them to
// 	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
// 	 */

// 	if (!node->sort_Done)
// 	{
// 		Sort	   *plannode = (Sort *) node->ss.ps.plan;
// 		PlanState  *outerNode;
// 		TupleDesc	tupDesc;

// 		SO1_printf("ExecSort: %s\n",
// 				   "sorting subplan");

// 		/*
// 		 * Want to scan subplan in the forward direction while creating the
// 		 * sorted data.
// 		 */
// 		estate->es_direction = ForwardScanDirection;

// 		/*
// 		 * Initialize tuplesort module.
// 		 */
// 		SO1_printf("ExecSort: %s\n",
// 				   "calling tuplesort_begin");

// 		outerNode = outerPlanState(node);
// 		tupDesc = ExecGetResultType(outerNode);

// 		tuplesortstate = tuplesort_begin_heap(tupDesc,
// 											  plannode->numCols,
// 											  plannode->sortColIdx,
// 											  plannode->sortOperators,
// 											  plannode->collations,
// 											  plannode->nullsFirst,
// 											  work_mem,
// 											  node->randomAccess);
// 		if (node->bounded)
// 			tuplesort_set_bound(tuplesortstate, node->bound);
// 		node->tuplesortstate = (void *) tuplesortstate;

// 		/*
// 		 * Scan the subplan and feed all the tuples to tuplesort.
// 		 */

// 		for (;;)
// 		{
// 			slot = ExecProcNode(outerNode);

// 			if (TupIsNull(slot))
// 				break;

// 			tuplesort_puttupleslot(tuplesortstate, slot);
// 		}

// 		/*
// 		 * Complete the sort.
// 		 */
// 		tuplesort_performsort(tuplesortstate);

// 		/*
// 		 * restore to user specified direction
// 		 */
// 		estate->es_direction = dir;

// 		/*
// 		 * finally set the sorted flag to true
// 		 */
// 		node->sort_Done = true;
// 		node->bounded_Done = node->bounded;
// 		node->bound_Done = node->bound;
// 		SO1_printf("ExecSort: %s\n", "sorting done");
// 	}

// 	SO1_printf("ExecSort: %s\n",
// 			   "retrieving tuple from tuplesort");

// 	/*
// 	 * Get the first or next tuple from tuplesort. Returns NULL if no more
// 	 * tuples.
// 	 */
// 	slot = node->ss.ps.ps_ResultTupleSlot;
// 	(void) tuplesort_gettupleslot(tuplesortstate,
// 								  ScanDirectionIsForward(dir),
// 								  slot);
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
	SortState  *sortstate;

	SO1_printf("ExecInitSort: %s\n",
			   "initializing sort node");

	/*
	 * create state structure
	 */
	sortstate = makeNode(SortState);
	sortstate->ss.ps.plan = (Plan *) node;
	sortstate->ss.ps.state = estate;

	/*
	 * We must have random access to the sort output to do backward scan or
	 * mark/restore.  We also prefer to materialize the sort output if we
	 * might be called on to rewind and replay it many times.
	 */
	sortstate->randomAccess = (eflags & (EXEC_FLAG_REWIND |
										 EXEC_FLAG_BACKWARD |
										 EXEC_FLAG_MARK)) != 0;

	sortstate->bounded = false;
	sortstate->sort_Done = false;
	sortstate->tuplesortstate = NULL;

	/*
	 * Miscellaneous initialization
	 *
	 * Sort nodes don't initialize their ExprContexts because they never call
	 * ExecQual or ExecProject.
	 */

	/*
	 * tuple table initialization
	 *
	 * sort nodes only return scan tuples from their sorted relation.
	 */
	ExecInitResultTupleSlot(estate, &sortstate->ss.ps);
	ExecInitScanTupleSlot(estate, &sortstate->ss);

	/*
	 * initialize child nodes
	 *
	 * We shield the child node from the need to support REWIND, BACKWARD, or
	 * MARK/RESTORE.
	 */
	eflags &= ~(EXEC_FLAG_REWIND | EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK);

	outerPlanState(sortstate) = ExecInitNode(outerPlan(node), estate, eflags);

	/*
	 * initialize tuple type.  no need to initialize projection info because
	 * this node doesn't do projections.
	 */
	ExecAssignResultTypeFromTL(&sortstate->ss.ps);
	ExecAssignScanTypeFromOuterPlan(&sortstate->ss);
	sortstate->ss.ps.ps_ProjInfo = NULL;

	SO1_printf("ExecInitSort: %s\n",
			   "sort node initialized");

	return sortstate;
}

/* ----------------------------------------------------------------
 *		ExecEndSort(node)
 * ----------------------------------------------------------------
 */
void
ExecEndSort(SortState *node)
{
	SO1_printf("ExecEndSort: %s\n",
			   "shutting down sort node");

	/*
	 * clean out the tuple table
	 */
	ExecClearTuple(node->ss.ss_ScanTupleSlot);
	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);

	/*
	 * Release tuplesort resources
	 */
	if (node->tuplesortstate != NULL)
		tuplesort_end((Tuplesortstate *) node->tuplesortstate);
	node->tuplesortstate = NULL;

	/*
	 * shut down the subplan
	 */
	ExecEndNode(outerPlanState(node));

	SO1_printf("ExecEndSort: %s\n",
			   "sort node shutdown");
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
	if (!node->sort_Done)
		return;

	tuplesort_markpos((Tuplesortstate *) node->tuplesortstate);
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
	/*
	 * if we haven't sorted yet, just return.
	 */
	if (!node->sort_Done)
		return;

	/*
	 * restore the scan to the previously marked position
	 */
	tuplesort_restorepos((Tuplesortstate *) node->tuplesortstate);
}

void
ExecReScanSort(SortState *node)
{
	/*
	 * If we haven't sorted yet, just return. If outerplan's chgParam is not
	 * NULL then it will be re-scanned by ExecProcNode, else no reason to
	 * re-scan it at all.
	 */
	if (!node->sort_Done)
		return;

	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);

	/*
	 * If subnode is to be rescanned then we forget previous sort results; we
	 * have to re-read the subplan and re-sort.  Also must re-sort if the
	 * bounded-sort parameters changed or we didn't select randomAccess.
	 *
	 * Otherwise we can just rewind and rescan the sorted output.
	 */
	if (node->ss.ps.lefttree->chgParam != NULL ||
		node->bounded != node->bounded_Done ||
		node->bound != node->bound_Done ||
		!node->randomAccess)
	{
		node->sort_Done = false;
		tuplesort_end((Tuplesortstate *) node->tuplesortstate);
		node->tuplesortstate = NULL;

		/*
		 * if chgParam of subnode is not null then plan will be re-scanned by
		 * first ExecProcNode.
		 */
		if (node->ss.ps.lefttree->chgParam == NULL)
			ExecReScan(node->ss.ps.lefttree);
	}
	else
		tuplesort_rescan((Tuplesortstate *) node->tuplesortstate);
}
