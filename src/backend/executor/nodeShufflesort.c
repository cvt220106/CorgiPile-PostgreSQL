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
 *	  src/backend/executor/nodeShuffleSort.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "executor/execdebug.h"
#include "executor/nodeShuffleSort.h"
#include "miscadmin.h"
#include "utils/tupleShuffleSort.h"





void clear_buffer(TupleShuffleSortState* tupleShuffleSortState)
{
	// We may do some other clearing jobs
	// e.g., need to delete state->memtuples to avoid memory leak
	clear_tupleshufflesort_state(tupleShuffleSortState);

}

// Lijie: add end

/* ----------------------------------------------------------------
 *		ExecSort
 *
 *		Sorts tuples from the outer subtree of the node using tupleshufflesort,
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
ExecShuffleSort(ShuffleSortState *node)
{
	EState	   *estate;
	ScanDirection dir;
	TupleShuffleSortState *state;
	TupleTableSlot *slot;

	estate = node->ss.ps.state;
	dir = estate->es_direction;
	state = (TupleShuffleSortState *) node->tupleShuffleSortState;

	PlanState  *outerNode;
	
	if (state == NULL) // Going to init
	{
		ShuffleSort	   *plannode = (ShuffleSort *) node->ss.ps.plan;
		TupleDesc	tupDesc;

		SO1_printf("ExecShuffleSort: %s\n",
				   "shuffle_sorting subplan");

		/*
		 * Want to scan subplan in the forward direction while creating the
		 * sorted data.
		 */
		estate->es_direction = ShuffleScanDirection;

		/*
		 * Initialize tuplesort module.
		 */
		SO1_printf("ExecShuffleSort: %s\n",
				   "calling tupleshufflesort_begin");

		// outerNode = ShuffleScanNode
		outerNode = outerPlanState(node);
		tupDesc = ExecGetResultType(outerNode);

		// init buffer
		state = tupleshufflesort_begin_heap(tupDesc, work_mem);
                                              
		node->tupleShuffleSortState = (void *) state;

	}

	if (node->buffer_empty) {
		bool buffer_full = false;

		while(true) {
      		// fetch a tuple from the ShuffleScanNode 
      		slot = ExecProcNode(outerNode);
      		// put_tuple_into_buffer(tuple, buffer);

			if (TupIsNull(slot) == false) {
				buffer_full = tupleshufflesort_puttupleslot(state, slot);
			} 
			else {
				node->eof_reach = true;
				//tupleshufflesort_set_end(state);
			}
			
			if (buffer_full || TupIsNull(slot)) {
				tupleshufflesort_performshuffle(state);
				node->buffer_empty = false;
				break;
			}	
   		}
	}
	

	slot = node->ss.ps.ps_ResultTupleSlot;
	bool tuple_left = tupleshufflesort_gettupleslot(state, slot, false);
	if (!tuple_left) {
		node->buffer_empty = true;
		// clear_buffer(); set current_count = 0

		if (node->eof_reach) {
			tupleshufflesort_gettupleslot(state, slot, true);
		}
	}

	return slot;
	
}	


// 			// Lijie: read a tuple from the previous node (e.g., SeqScan)
// 			slot = ExecProcNode(outerNode);

// 			// Lijie: we finalize the model when finishing reading all the tuples
// 			if (TupIsNull(slot) || buffer_is_full) {
// 				// True means the last tuple, so that we need to force shuffling the buffered tuples
// 				last_tuple = true;
// 				bool is_buffer_empty = tupleshufflesort_puttupleslot(tupleShuffleSortState, slot, last_tuple);
				
				
// 				// Lijie: add end
// 				break;
// 			}

// 			tupleshufflesort_puttupleslot(tupleShuffleSortState, slot, last_tuple);
// 			// Lijie: put a tuple into the buffer and perform shuffling when the buffer is full
// 			bool buffer_full_and_shuffled = tupleshufflesort_puttupleslot(tupleShuffleSortState, slot, last_tuple);
// 			if (buffer_full_and_shuffled) {
// 				// perform SGD on the buffered tuples, update the model
// 				ith_tuple = perform_SGD(tupleShuffleSortState, svm_model, ith_tuple, batch_size, last_tuple);
// 				// and then clear the buffer for further reading
// 				clear_buffer(tupleShuffleSortState);
// 			}

// 			shuffle_sort();
			 	
// 		}

// 		/*
// 		 * Complete the sort.
// 		 */
// 		// tuplesort_performsort(tuplesortstate);
		
// 		/*
// 		 * restore to user specified direction
// 		 */
// 		estate->es_direction = dir;

// 		/*
// 		 * finally set the sorted flag to true
// 		 */
// 		node->shuffle_sort_Done = true;
// 		node->bounded_Done = node->bounded;
// 		node->bound_Done = node->bound;
// 		SO1_printf("ExecSort: %s\n", "sorting done");
// 	}

// 	SO1_printf("ExecShuffleSort: %s\n",
// 			   "retrieving tuple from tupleshufflesort");

// 	/*
// 	 * Get the first or next tuple from tuplesort. Returns NULL if no more
// 	 * tuples.
// 	 */
// 	slot = node->ss.ps.ps_ResultTupleSlot;
// 	(void) tupleshufflesort_gettupleslot(tupleShuffleSortState,
// 								  ScanDirectionIsForward(dir),
// 								  slot);
// 	return slot;
// }

// 	while(true) {
//     	if (node->buffer_full == false) {
//       		// fetch a tuple from the ShuffleScanNode 
//       		tuple = ExecProcNode(ShuffleScanNode);
//       		put_tuple_into_buffer(tuple, buffer);
//    		}
//     	else {
//       		if (buffer_unshuffled)
//         	shuffle_buffer();
//       		++index;
//       		return buffer[index];
//     	}
//   	}

// 	if (buffer_is_not_full)
// 		fetch_
// 		put_tuple_into_buffer()
	


/* ----------------------------------------------------------------
 *		ExecInitSort
 *
 *		Creates the run-time state information for the sort node
 *		produced by the planner and initializes its outer subtree.
 * ----------------------------------------------------------------
 */
ShuffleSortState *
ExecInitShuffleSort(ShuffleSort *node, EState *estate, int eflags)
{
	ShuffleSortState  *shuffleSortState;

	SO1_printf("ExecInitShuffleSort: %s\n",
			   "initializing shuffle sort node");

	/*
	 * create state structure
	 */
	shuffleSortState = makeNode(ShuffleSortState);
	shuffleSortState->ss.ps.plan = (Plan *) node;
	shuffleSortState->ss.ps.state = estate;

	/*
	 * We must have random access to the sort output to do backward scan or
	 * mark/restore.  We also prefer to materialize the sort output if we
	 * might be called on to rewind and replay it many times.
	 */
	
	shuffleSortState->shuffle_sort_Done = false;
	shuffleSortState->tupleShuffleSortState = NULL;

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
	ExecInitResultTupleSlot(estate, &shuffleSortState->ss.ps);
	ExecInitScanTupleSlot(estate, &shuffleSortState->ss);

	/*
	 * initialize child nodes
	 *
	 * We shield the child node from the need to support REWIND, BACKWARD, or
	 * MARK/RESTORE.
	 */
	eflags &= ~(EXEC_FLAG_REWIND | EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK);

	outerPlanState(shuffleSortState) = ExecInitNode(outerPlan(node), estate, eflags);

	/*
	 * initialize tuple type.  no need to initialize projection info because
	 * this node doesn't do projections.
	 */
	ExecAssignResultTypeFromTL(&shuffleSortState->ss.ps);
	ExecAssignScanTypeFromOuterPlan(&shuffleSortState->ss);
	shuffleSortState->ss.ps.ps_ProjInfo = NULL;

	SO1_printf("ExecInitShuffleSort: %s\n",
			   "shuffle sort node initialized");

	return shuffleSortState;
}

/* ----------------------------------------------------------------
 *		ExecEndSort(node)
 * ----------------------------------------------------------------
 */
void
ExecEndShuffleSort(ShuffleSortState *node)
{
	SO1_printf("ExecEndShuffleSort: %s\n",
			   "shutting down sort node");

	/*
	 * clean out the tuple table
	 */
	ExecClearTuple(node->ss.ss_ScanTupleSlot);
	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);

	clear_buffer(tupleShuffleSortState);

	/*
	 * Release tuplesort resources
	 */
	if (node->tupleShuffleSortState != NULL)
		tupleshufflesort_end((TupleShuffleSortState *) node->tupleShuffleSortState);
	node->tupleShuffleSortState = NULL;

	/*
	 * shut down the subplan
	 */
	ExecEndNode(outerPlanState(node));

	SO1_printf("ExecEndShuffleSort: %s\n",
			   "sort node shutdown");
}

/* ----------------------------------------------------------------
 *		ExecSortMarkPos
 *
 *		Calls tuplesort to save the current position in the sorted file.
 * ----------------------------------------------------------------
 */
void
ExecShuffleSortMarkPos(ShuffleSortState *node)
{
	/*
	 * if we haven't sorted yet, just return
	 */
	if (!node->shuffle_sort_Done)
		return;

	tupleshufflesort_markpos((TupleShuffleSortState *) node->tupleShuffleSortState);
}

/* ----------------------------------------------------------------
 *		ExecSortRestrPos
 *
 *		Calls tuplesort to restore the last saved sort file position.
 * ----------------------------------------------------------------
 */
void
ExecShuffleSortRestrPos(ShuffleSortState *node)
{
	/*
	 * if we haven't sorted yet, just return.
	 */
	if (!node->shuffle_sort_Done)
		return;

	/*
	 * restore the scan to the previously marked position
	 */
	tupleshufflesort_restorepos((TupleShuffleSortState *) node->tupleShuffleSortState);
}

void
ExecReScanShuffleSort(ShuffleSortState *node)
{
	/*
	 * If we haven't sorted yet, just return. If outerplan's chgParam is not
	 * NULL then it will be re-scanned by ExecProcNode, else no reason to
	 * re-scan it at all.
	 */
	// if (!node->shuffle_sort_Done)
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
		tupleshufflesort_end((TupleShuffleSortState *) node->tupleShuffleSortState);
		node->tupleShuffleSortState = NULL;

		/*
		 * if chgParam of subnode is not null then plan will be re-scanned by
		 * first ExecProcNode.
		 */
		if (node->ss.ps.lefttree->chgParam == NULL)
			ExecReShuffleScan(node->ss.ps.lefttree);
	}
	else
		tupleshufflesort_rescan((TupleShuffleSortState *) node->tupleShuffleSortState);
}
