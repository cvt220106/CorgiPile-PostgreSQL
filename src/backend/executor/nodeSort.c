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
#include "pthread.h"


static void free_buffer(Tuplesortstate* tuplesortstate);
static void init_Tuplesortstate(SortState *node);
static void reset_sort_state(SortState *node);
static void wait_buffer_full(SortState *node);
static void signal_buffer_full(SortState *node);
static void wait_swap_finished(SortState *node);
static void signal_swap_finished(SortState *node);
static void* write_thread_run(SortState *node);
static void start_write_thread(SortState *node);

pthread_t write_thread;


pthread_mutex_t buffer_mutex;
pthread_mutex_t swap_mutex;
pthread_cond_t buffer_full_cond;
pthread_cond_t swap_finished_cond;


// pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;
// pthread_mutex_t swap_mutex = PTHREAD_MUTEX_INITIALIZER;
// pthread_cond_t buffer_full_cond = PTHREAD_COND_INITIALIZER;
// pthread_cond_t swap_finished_cond = PTHREAD_COND_INITIALIZER;



void wait_buffer_full(SortState *node) {
	//clock_t start = clock(); 

    pthread_mutex_lock(&buffer_mutex);
    while (!node->buffer_full_signal) {
        //printf("\n[read thread] wait_buffer_full()\n");
        pthread_cond_wait(&buffer_full_cond, &buffer_mutex);
    }
    node->buffer_full_signal = false;
    pthread_mutex_unlock(&buffer_mutex);

	//clock_t finish = clock();    
   	//double duration = (double)(finish - start) / CLOCKS_PER_SEC;    
   	//elog(INFO, "[read][wait_buffer_full] %f seconds, is_training = %d", duration, (int)is_training);  
}

void signal_buffer_full(SortState *node) {
    pthread_mutex_lock(&buffer_mutex);
    node->buffer_full_signal = true;
    //printf("\n[write thread] signal_buffer_full()\n");
    pthread_cond_signal(&buffer_full_cond);
    pthread_mutex_unlock(&buffer_mutex);
}

void wait_swap_finished(SortState *node) {
    //clock_t start = clock(); 
    pthread_mutex_lock(&swap_mutex);
    while (!node->swap_finished_signal) {
        //printf("\n[write thread] wait_swap_finished()\n");
        pthread_cond_wait(&swap_finished_cond, &swap_mutex);
    }
    node->swap_finished_signal = false;
    pthread_mutex_unlock(&swap_mutex);

	// clock_t finish = clock();    
   	// double duration = (double)(finish - start) / CLOCKS_PER_SEC;    
   	// elog(INFO, "[write][wait_swap_finished] %f seconds, is_training = %d", duration, (int)is_training);  
}

void signal_swap_finished(SortState *node) {
    pthread_mutex_lock(&swap_mutex);
    node->swap_finished_signal = true;
    //printf("\n[read thread] signal_swap_finished()\n");
    pthread_cond_signal(&swap_finished_cond);
    pthread_mutex_unlock(&swap_mutex);
}

void free_buffer(Tuplesortstate* tuplesortstate)
{
	// We may do some other clearing jobs
	// e.g., need to delete state->memtuples to avoid memory leak
	if (set_use_malloc == false)
		free_tupleshufflesort_state(tuplesortstate);
	else
		plain_free_tupleshufflesort_state(tuplesortstate);

}

void reset_sort_state(SortState *node) {

	// The node state (SortState) cares about the thread-related state,
	// while tupleshufflesort state cares about the buffer-related state.
	node->shuffle_sort_Done = false;
	node->eof_reach = false;

	// for double buffer
	node->write_thread_not_started = true;
	node->buffer_full_signal = false;
    node->swap_finished_signal = false;

	tupleshufflesort_reset_state(node->tuplesortstate);

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
                                    
	node->tuplesortstate = state;
}


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

/** 
 * =====================  Write thread ===================== 
 */

void start_write_thread(SortState *node) {
    pthread_create(&write_thread, NULL, (void *)write_thread_run, node);
}

void* write_thread_run(SortState *node) {

	PlanState  *outerNode = outerPlanState(node);
	Tuplesortstate* state = node->tuplesortstate;

	// clock_t start = clock(); 

    while(true) {
		
        TupleTableSlot* tuple_slot = ExecProcNode(outerNode);
       
        // still put the tuple into the buffer when tuple == null
        bool write_buffer_full = tupleshufflesort_puttupleslot(state, tuple_slot);

        if (write_buffer_full || TupIsNull(tuple_slot)) {
			tupleshufflesort_performshuffle(state); // the last tuple can be null
            // clock_t finish = clock();    
   			// double duration = (double)(finish - start) / CLOCKS_PER_SEC;    
   			// elog(INFO, "[write_full] %f seconds", duration);  
			// start = finish;

			signal_buffer_full(node);
			//elog(INFO, "[write thread] Finish signal_buffer_full(node);");
            if (!TupIsNull(tuple_slot))
                wait_swap_finished(node);
           
        }

        if (TupIsNull(tuple_slot))
            break;
    }

    return NULL;
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
/*
TupleTableSlot *
ExecSort(SortState *node)
{
	// EState	   *estate = node->ss.ps.state;
	// ScanDirection dir = estate->es_direction; // going to be set to ShuffleScanDirection
	Tuplesortstate *state = node->tuplesortstate;
	TupleTableSlot *slot;

	if (state == NULL) {
		// build a new tuplesortstate with new buffer
		init_Tuplesortstate(node);
		// reset the tuplesortstate index, etc.
		reset_sort_state(node);
		// node->rescan_count = 0;
		state = node->tuplesortstate;
		// elog(INFO, "state == null, reset_sort_state");

		Assert(pthread_mutex_init(&buffer_mutex, NULL) == 0);
		Assert(pthread_mutex_init(&swap_mutex, NULL) == 0);
		Assert(pthread_cond_init(&buffer_full_cond, NULL) == 0);
		Assert(pthread_cond_init(&swap_finished_cond, NULL) == 0);
	}

	if (node->write_thread_not_started) {
        start_write_thread(node);
        node->write_thread_not_started = false;
    }

	// if read_buffer == NULL
	if (tupleshufflesort_is_read_buffer_null(state)) {     
        wait_buffer_full(node);
		// write_buffer = buffer2;
        // read_buffer = buffer1;
		tupleshufflesort_init_buffer(state);
        signal_swap_finished(node);
    }

	slot = node->ss.ps.ps_ResultTupleSlot;

	if (tupleshufflesort_has_tuple_in_buffer(state)) {
		// slot can be null
		// elog(INFO, "[Read thread] Finish tupleshufflesort_has_tuple_in_buffer(state);");	
		tupleshufflesort_gettupleslot(state, slot);
		// elog(INFO, "[Read thread] tupleshufflesort_gettupleslot (when read_buffer = null), slot = %x", slot);
		// slot can be empty, so TupleIsNull(slot) == true
		return slot;
	}
	
    else {
		//elog(INFO, "[Read thread] Begin wait_buffer_full(node);");
        wait_buffer_full(node);
		// swap(&read_buffer, &write_buffer) and reset fetch_index = 0;
		tupleshufflesort_swapbuffer(state);
        signal_swap_finished(node);

		//elog(INFO, "[Read thread] Begin tupleshufflesort_gettupleslot(state, slot) when read_buffer != null;");
		tupleshufflesort_gettupleslot(state, slot);
		//elog(INFO, "[Read thread] Finish tupleshufflesort_gettupleslot(state, slot) when read_buffer != null;");
		return slot;
    }  
	
}
*/


TupleTableSlot *
ExecSort(SortState *node)
{
	// EState	   *estate = node->ss.ps.state;
	// ScanDirection dir = estate->es_direction; // going to be set to ShuffleScanDirection
	Tuplesortstate *state = node->tuplesortstate;
	TupleTableSlot *slot;

	if (state == NULL) {
		// build a new tuplesortstate with new buffer
		init_Tuplesortstate(node);
		// reset the tuplesortstate index, etc.
		reset_sort_state(node);
		// node->rescan_count = 0;
		state = node->tuplesortstate;
		// elog(INFO, "state == null, reset_sort_state");

		Assert(pthread_mutex_init(&buffer_mutex, NULL) == 0);
		Assert(pthread_mutex_init(&swap_mutex, NULL) == 0);
		Assert(pthread_cond_init(&buffer_full_cond, NULL) == 0);
		Assert(pthread_cond_init(&swap_finished_cond, NULL) == 0);
	}

	if (node->write_thread_not_started) {
        start_write_thread(node);
        node->write_thread_not_started = false;
    }

	wait_buffer_full(node);

	if (tupleshufflesort_is_read_buffer_null(state))
		tupleshufflesort_init_buffer(state);
	else
		tupleshufflesort_swapbuffer(state);

	signal_swap_finished(node);

	slot = node->ss.ps.ps_ResultTupleSlot;

	slot->read_buffer = tupleshufflesort_getreadbuffer(state);
	slot->read_buf_indexes = tupleshufflesort_getbufferindexes(state);
	slot->tts_isempty = false;
	slot->tts_shouldFree = false;
	slot->read_buffer_size = tupleshufflesort_getbuffersize(state);


	return slot;
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
	//elog(LOG, "end ExecClearTuple ss_ScanTupleSlot.");
	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);
	//elog(LOG, "end ExecClearTuple ps_ResultTupleSlot.");

	//elog(INFO, "begin free_buffer.");
	free_buffer(node->tuplesortstate);
	//elog(INFO, "end free_buffer.");
	/*
	 * Release tuplesort resources
	 */
	if (node->tuplesortstate != NULL)
		tupleshufflesort_end(node->tuplesortstate);
	// pfree(node->tuplesortstate);
	// elog(LOG, "end tupleshufflesort_end.");
	/*
	 * shut down the subplan
	 */
	ExecEndNode(outerPlanState(node));

	// elog(LOG, "end execendsort.");
	SO1_printf("ExecEndSort: %s\n",
			   "sort node shutdown");

	Assert(pthread_mutex_destroy(&buffer_mutex) == 0);
    Assert(pthread_mutex_destroy(&swap_mutex) == 0);
    Assert(pthread_cond_destroy(&buffer_full_cond) == 0);
    Assert(pthread_cond_destroy(&swap_finished_cond) == 0);
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

	tupleshufflesort_markpos(node->tuplesortstate);
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
	tupleshufflesort_restorepos(node->tuplesortstate);
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
	reset_sort_state(node);

	// Assert(pthread_mutex_destroy(&buffer_mutex) == 0);
    // Assert(pthread_mutex_destroy(&swap_mutex) == 0);
    // Assert(pthread_cond_destroy(&buffer_full_cond) == 0);
    // Assert(pthread_cond_destroy(&swap_finished_cond) == 0);

	
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
		tupleshufflesort_end(node->tuplesortstate);
		node->tuplesortstate = NULL;

		/*
		 * if chgParam of subnode is not null then plan will be re-scanned by
		 * first ExecProcNode.
		 */
		if (node->ss.ps.lefttree->chgParam == NULL)
			ExecReScan(node->ss.ps.lefttree);
	}

	// we just clear the buffer and rescan lefttree (shufflescan)
	else {
		// node->tuplesortstate->rescaned = true;
		// tupleshufflesort_rescan(node->tuplesortstate);
		if (node->ss.ps.lefttree != NULL)
			ExecReScan(node->ss.ps.lefttree);

	}

	
	
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


/*
TupleTableSlot *
ExecSort(SortState *node)
{
	EState	   *estate = node->ss.ps.state;
	ScanDirection dir = estate->es_direction; // going to be set to ShuffleScanDirection
	Tuplesortstate *state = node->tuplesortstate;
	TupleTableSlot *slot;

	if (state == NULL) {
		// build a new tuplesortstate with new buffer
		init_Tuplesortstate(node);
		// reset the tuplesortstate index, etc.
		reset_sort_state(node);
		// node->rescan_count = 0;
		state = node->tuplesortstate;
		
	}


	if (node->buffer_empty) {
		if (node->eof_reach) {
			// if (node->rescan_count++ < 2) {
			// 	ExecReScanSort(node);
			// 	state = node->tuplesortstate;
			// }
			// else
			// 	return NULL;
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
*/