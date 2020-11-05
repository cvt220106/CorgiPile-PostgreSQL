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


void clear_buffer(Tuplesortstate* tuplesortstate)
{
	// We may do some other clearing jobs
	// e.g., need to delete state->memtuples to avoid memory leak
	clear_tupleshufflesort_state(tuplesortstate);

}


void init_Tuplesortstate(SortState *node) {
	// node denotes the SortState
	Sort  *plannode = (Sort *) node->ss.ps.plan;
	PlanState  *outerNode = outerPlanState(node);

	TupleDesc	tupDesc = ExecGetResultType(outerNode);

	// estate->es_direction = ShuffleScanDirection;
	// init buffer
	// work_mem = 1024 default
	Tuplesortstate *state = tupleshufflesort_begin_heap(tupDesc, work_mem);
                                              
	node->tuplesortstate = (void *) state;
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
	EState	   *estate = node->ss.ps.state;
	ScanDirection dir = estate->es_direction; // going to be set to ShuffleScanDirection
	Tuplesortstate *state = (Tuplesortstate *) node->tuplesortstate;
	TupleTableSlot *slot;

	/*
	 * If first time through, read all tuples from outer plan and pass them to
	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
	 */
	if (state == NULL) {
		init_Tuplesortstate(node);

		node->shuffle_sort_Done = false;
		node->buffer_empty = true;
		node->eof_reach = false;
		node->rescan_count = 0;

		state = (Tuplesortstate *) node->tuplesortstate;
	}


	if (node->buffer_empty) {
		if (node->eof_reach) {
			if (node->rescan_count++ < 1) {
				ExecReScanSort(node);
				init_Tuplesortstate(node);

				node->shuffle_sort_Done = false;
				node->buffer_empty = true;
				node->eof_reach = false;

				state = (Tuplesortstate *) node->tuplesortstate;
			}
			else
				return NULL;
		}
			

		bool buffer_full = false;
		PlanState  *outerNode = outerPlanState(node);

		while(true) {
      		// fetch a tuple from the ShuffleScanNode 
      		slot = ExecProcNode(outerNode);
      		// put_tuple_into_buffer(tuple, buffer);

			if (!TupIsNull(slot)) { // a non-empty slot, put it into the buffer
				buffer_full = tupleshufflesort_puttupleslot(state, slot);
			} 
			else {
				node->eof_reach = true;
				// buffer = [     ], tuple = null (eof_reach)
				if (is_shuffle_buffer_emtpy(state)) {
					return NULL; // return slot = null;
				}
			}
			
			if (buffer_full || node->eof_reach) {
				tupleshufflesort_performshuffle(state);
				node->buffer_empty = false;
				break;
			}	
   		}
	}
	
	slot = node->ss.ps.ps_ResultTupleSlot;

	// buffer_full or buffer_is_halfly_filled in the end
	bool tuple_left = tupleshufflesort_gettupleslot(state, slot);

	// all the tuples are extracted from buffer
	if (tuple_left == false) {
		node->buffer_empty = true;
	}

	return slot;
}

// can compile
// TupleTableSlot *
// ExecSort(SortState *node)
// {
// 	// case 1: buffer_empty, going to put tuples into it.
// 	// case 2: buffer is filled with tuples.
// 	// case 3: buffer is half-filled with tuples. The last tuple cannot fill the buffer.
// 	// case 4: buffer_empty, last tuple
// 	EState	   *estate = node->ss.ps.state;
// 	ScanDirection dir = estate->es_direction; // going to be set to ShuffleScanDirection
// 	Tuplesortstate *state = (Tuplesortstate *) node->tuplesortstate;
// 	TupleTableSlot *slot;

// 	/*
// 	 * get state info from node
// 	 */
// 	// SO1_printf("ExecSort: %s\n",
// 	// 		   "entering routine");

// 	/*
// 	 * If first time through, read all tuples from outer plan and pass them to
// 	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
// 	 */
// 	if (state == NULL) {
// 		init_Tuplesortstate(node);

// 		node->shuffle_sort_Done = false;
// 		node->buffer_empty = true;
// 		node->eof_reach = false;
// 		node->rescan_count = 2;

// 		state = (Tuplesortstate *) node->tuplesortstate;
// 	}


// 	if (node->buffer_empty) {
// 		if (node->eof_reach) 
// 			return NULL;

// 		bool buffer_full = false;
// 		PlanState  *outerNode = outerPlanState(node);

// 		while(true) {
//       		// fetch a tuple from the ShuffleScanNode 
//       		slot = ExecProcNode(outerNode);
//       		// put_tuple_into_buffer(tuple, buffer);

// 			if (!TupIsNull(slot)) { // a non-empty slot, put it into the buffer
// 				buffer_full = tupleshufflesort_puttupleslot(state, slot);
// 			} 
// 			else {
// 				node->eof_reach = true;
// 				// buffer = [     ], tuple = null (eof_reach)
// 				if (is_shuffle_buffer_emtpy(state)) {
// 					return NULL; // return slot = null;
// 				}
// 			}
			
// 			if (buffer_full || node->eof_reach) {
// 				tupleshufflesort_performshuffle(state);
// 				node->buffer_empty = false;
// 				break;
// 			}	
//    		}
// 	}
	
// 	slot = node->ss.ps.ps_ResultTupleSlot;

// 	// buffer_full or buffer_is_halfly_filled in the end
// 	bool tuple_left = tupleshufflesort_gettupleslot(state, slot);

// 	// all the tuples are extracted from buffer
// 	if (tuple_left == false) {
// 		node->buffer_empty = true;
// 	}

// 	return slot;
// }

// TupleTableSlot *
// ExecSort(SortState *node)
// {

// 	// case 1: buffer_empty, going to put tuples into it.
// 	// case 2: buffer is filled with tuples.
// 	// case 3: buffer is half-filled with tuples. The last tuple cannot fill the buffer.
// 	// case 4: buffer_empty, last tuple
// 	EState	   *estate;
// 	ScanDirection dir;
// 	Tuplesortstate *state;
// 	TupleTableSlot *slot;

// 	/*
// 	 * get state info from node
// 	 */
// 	// SO1_printf("ExecSort: %s\n",
// 	// 		   "entering routine");

// 	estate = node->ss.ps.state;
// 	dir = estate->es_direction;
// 	state = (Tuplesortstate *) node->tuplesortstate;

// 	PlanState  *outerNode;
// 	/*
// 	 * If first time through, read all tuples from outer plan and pass them to
// 	 * tuplesort.c. Subsequent calls just fetch tuples from tuplesort.
// 	 */

// 	if (state == NULL) // Going to init // can be put into init() function
// 	{
// 		Sort	   *plannode = (Sort *) node->ss.ps.plan;
// 		TupleDesc	tupDesc;

// 		// SO1_printf("ExecShuffleSort: %s\n",
// 		// 		   "shuffle_sorting subplan");

// 		/*
// 		 * Want to scan subplan in the forward direction while creating the
// 		 * sorted data.
// 		 */
// 		// estate->es_direction = ShuffleScanDirection;

// 		/*
// 		 * Initialize tuplesort module.
// 		 */

// 		// outerNode = ShuffleScanNode
// 		outerNode = outerPlanState(node);
// 		tupDesc = ExecGetResultType(outerNode);

// 		// init buffer
// 		// work_mem = 1024 default
// 		state = tupleshufflesort_begin_heap(tupDesc, work_mem);
                                              
// 		node->tuplesortstate = (void *) state;

// 	}

// 	if (node->buffer_empty) {
// 		if (node->eof_reach)
// 			return NULL;

// 		bool buffer_full = false;
// 		outerNode = outerPlanState(node);

// 		while(true) {
//       		// fetch a tuple from the ShuffleScanNode 
//       		slot = ExecProcNode(outerNode);
//       		// put_tuple_into_buffer(tuple, buffer);

// 			if (!TupIsNull(slot)) {
// 				buffer_full = tupleshufflesort_puttupleslot(state, slot);
// 			} 
// 			else {
// 				node->eof_reach = true;
// 				if (is_shuffle_buffer_emtpy(state)) {
// 					return NULL; // return slot = null;
// 				}
// 				//tupleshufflesort_set_end(state);
// 			}
			
// 			if (buffer_full || TupIsNull(slot)) {
// 				tupleshufflesort_performshuffle(state);
// 				node->buffer_empty = false;
// 				break;
// 			}	
//    		}
// 	}
	

// 	slot = node->ss.ps.ps_ResultTupleSlot;

// 	// buffer_full or TupIsNull
// 	bool tuple_left = tupleshufflesort_gettupleslot(state, slot);
// 	if (tuple_left == false) {
// 		node->buffer_empty = true;
// 		// clear_buffer(); set current_count = 0
// 		// if (node->eof_reach)
// 		// 	slot = NULL;
// 	}

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

	// SO1_printf("ExecInitSort: %s\n",
	// 		   "initializing sort node");

	/*
	 * create state structure
	 */
	sortstate = makeNode(SortState);
	sortstate->ss.ps.plan = (Plan *) node; // change to node->plan
	sortstate->ss.ps.state = estate;

	/*
	 * We must have random access to the sort output to do backward scan or
	 * mark/restore.  We also prefer to materialize the sort output if we
	 * might be called on to rewind and replay it many times.
	 */
	// sortstate->randomAccess = (eflags & (EXEC_FLAG_REWIND |
	// 									 EXEC_FLAG_BACKWARD |
	// 									 EXEC_FLAG_MARK)) != 0;
	
	// init_Tuplesortstate(sortstate);
	
	
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

	clear_buffer(node->tuplesortstate);
	/*
	 * Release tuplesort resources
	 */
	if (node->tuplesortstate != NULL)
		tupleshufflesort_end((Tuplesortstate *) node->tuplesortstate);
	// pfree(node->tuplesortstate);

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
	if (!node->shuffle_sort_Done)
		return;

	tupleshufflesort_markpos((Tuplesortstate *) node->tuplesortstate);
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
	if (!node->shuffle_sort_Done)
		return;

	/*
	 * restore the scan to the previously marked position
	 */
	tupleshufflesort_restorepos((Tuplesortstate *) node->tuplesortstate);
}

void
ExecReScanSort(SortState *node)
{
	/*
	 * If we haven't sorted yet, just return. If outerplan's chgParam is not
	 * NULL then it will be re-scanned by ExecProcNode, else no reason to
	 * re-scan it at all.
	 */
	// if (!node->sort_Done)
	// 	return;

	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);

	/*
	 * If subnode is to be rescanned then we forget previous sort results; we
	 * have to re-read the subplan and re-sort.  Also must re-sort if the
	 * bounded-sort parameters changed or we didn't select randomAccess.
	 *
	 * Otherwise we can just rewind and rescan the sorted output.
	 */
	if (node->ss.ps.lefttree->chgParam != NULL)
	{
		node->shuffle_sort_Done = false;
		tupleshufflesort_end((Tuplesortstate *) node->tuplesortstate);
		node->tuplesortstate = NULL;

		/*
		 * if chgParam of subnode is not null then plan will be re-scanned by
		 * first ExecProcNode.
		 */
		if (node->ss.ps.lefttree->chgParam == NULL)
			ExecReScan(node->ss.ps.lefttree);
	}
	else
		tupleshufflesort_rescan((Tuplesortstate *) node->tuplesortstate);

}
