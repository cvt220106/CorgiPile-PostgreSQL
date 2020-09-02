
// Lijie: add begin
void init_model(Model* model) {
	model->loss = 0;
	model->p1 = 0;
	model->p2 = 0;
}

double
compute_loss(ShuffleSortTuple* p, Model* model)
{
	double loss = (double) p->datum1;
	return loss;
}


int
compute_loss_and_update_model(TupleShuffleSortState* state, Model* model,
							  int ith_tuple, int batch_size, bool last_tuple) 
{
	int n = state->memtupcount;
	ShuffleSortTuple* tuples = state->memtuples;
	int last_updated = 0;
	int i = 0;

	for (ShuffleSortTuple* p = tuples; p < tuples + n; p++) {
		double tuple_loss = compute_loss(p, model);
		model->loss = model->loss + tuple_loss;
		elog(LOG, "[SVM][Tuple %d] >>> Add %.2f loss to model.", ith_tuple, tuple_loss);
		ith_tuple = (ith_tuple + 1) % batch_size;
		
		// going to update model
		if (ith_tuple == 0) {
			// update model
			model->p1 += 1;
			model->p2 += 1;
			elog(LOG, "[SVM] >>> Update model (p1 = %d, p2 = %d, loss = %.2f).", model->p1, model->p2, model->loss);
			last_updated = i;
		}
		++i;

	}

	if (last_tuple) {
		if (n > 0 && last_updated < n - 1) {
			model->p1 += 1;
			model->p2 += 1;
			elog(LOG, "[SVM] >>> Last: Update model (p1 = %d, p2 = %d, loss = %.2f).", model->p1, model->p2, model->loss);
		}
		else {
			elog(LOG, "[SVM] >>> Has updated the model.");
		}
	}

	return ith_tuple;
}

// Lijie: add begin 
		// =================== Model initialization =========================
		// We may put init_model() to ExecInitSort
		Model svm;
		Model* svm_model = &svm;
		init_model(svm_model);
		elog(LOG, "[SVM] Initialize SVM model (loss = 0, p1 = 0, p2 = 0)");


#include "postgres.h"

#include "executor/execdebug.h"
#include "executor/nodeSGD.h"
//#include "utils/rel.h"

static void InitScanRelation(SeqScanState *node, EState *estate);
static TupleTableSlot *SeqNext(SeqScanState *node);

/* ----------------------------------------------------------------
 *						Scan Support
 * ----------------------------------------------------------------
 */

/* ----------------------------------------------------------------
 *		SeqNext
 *
 *		This is a workhorse for ExecSeqScan
 * ----------------------------------------------------------------
 */
static TupleTableSlot *
SeqNext(SeqScanState *node)
{
	HeapTuple	tuple;
	HeapScanDesc scandesc;
	EState	   *estate;
	ScanDirection direction;
	TupleTableSlot *slot;

	/*
	 * get information from the estate and scan state
	 */
	scandesc = node->ss_currentScanDesc;
	estate = node->ps.state;
	direction = estate->es_direction;
	slot = node->ss_ScanTupleSlot;

	/*
	 * get the next tuple from the table
	 */
	tuple = heap_getnext(scandesc, direction);

	/*
	 * save the tuple and the buffer returned to us by the access methods in
	 * our scan tuple slot and return the slot.  Note: we pass 'false' because
	 * tuples returned by heap_getnext() are pointers onto disk pages and were
	 * not created with palloc() and so should not be pfree()'d.  Note also
	 * that ExecStoreTuple will increment the refcount of the buffer; the
	 * refcount will not be dropped until the tuple table slot is cleared.
	 */
	if (tuple)
		ExecStoreTuple(tuple,	/* tuple to store */
					   slot,	/* slot to store in */
					   scandesc->rs_cbuf,		/* buffer associated with this
												 * tuple */
					   false);	/* don't pfree this pointer */
	else
		ExecClearTuple(slot);

	return slot;
}

/*
 * SeqRecheck -- access method routine to recheck a tuple in EvalPlanQual
 */
static bool
SeqRecheck(SeqScanState *node, TupleTableSlot *slot)
{
	/*
	 * Note that unlike IndexScan, SeqScan never use keys in heap_beginscan
	 * (and this is very bad) - so, here we do not check are keys ok or not.
	 */
	return true;
}

/* ----------------------------------------------------------------
 *		ExecSeqScan(node)
 *
 *		Scans the relation sequentially and returns the next qualifying
 *		tuple.
 *		We call the ExecScan() routine and pass it the appropriate
 *		access method functions.
 * ----------------------------------------------------------------
 */
TupleTableSlot *
ExecSGD(SGDState *node)
{
	return ExecSGD((ScanState *) node,
					(ExecScanAccessMtd) SeqNext,
					(ExecScanRecheckMtd) SeqRecheck);
}

/* ----------------------------------------------------------------
 *		InitScanRelation
 *
 *		This does the initialization for scan relations and
 *		subplans of scans.
 * ----------------------------------------------------------------
 */
static void
InitScanRelation(SeqScanState *node, EState *estate)
{
	Relation	currentRelation;
	HeapScanDesc currentScanDesc;

	/*
	 * get the relation object id from the relid'th entry in the range table,
	 * open that relation and acquire appropriate lock on it.
	 */
	currentRelation = ExecOpenScanRelation(estate,
									 ((SeqScan *) node->ps.plan)->scanrelid);

	currentScanDesc = heap_beginscan(currentRelation,
									 estate->es_snapshot,
									 0,
									 NULL);

	node->ss_currentRelation = currentRelation;
	node->ss_currentScanDesc = currentScanDesc;

	ExecAssignScanType(node, RelationGetDescr(currentRelation));
}


/* ----------------------------------------------------------------
 *		ExecInitSeqScan
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

	/*
	 * We must have random access to the sort output to do backward scan or
	 * mark/restore.  We also prefer to materialize the sort output if we
	 * might be called on to rewind and replay it many times.
	 */
	
	// sortstate->bounded = false;
	// sortstate->sort_Done = false;
	// sortstate->tuplesortstate = NULL;

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
	// ExecInitScanTupleSlot(estate, &sortstate->ss);

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
	ExecAssignResultTypeFromTL(&sortstate->ss.ps);
	ExecAssignScanTypeFromOuterPlan(&sortstate->ss);
	// sortstate->ss.ps.ps_ProjInfo = NULL;

	SO1_printf("ExecInitSort: %s\n",
			   "sort node initialized");

	return sortstate;
}

/* ----------------------------------------------------------------
 *		ExecEndSeqScan
 *
 *		frees any storage allocated through C routines.
 * ----------------------------------------------------------------
 */
void
ExecEndSGD(SeqScanState *node)
{
	Relation	relation;
	HeapScanDesc scanDesc;

	/*
	 * get information from node
	 */
	relation = node->ss_currentRelation;
	scanDesc = node->ss_currentScanDesc;

	/*
	 * Free the exprcontext
	 */
	ExecFreeExprContext(&node->ps);

	/*
	 * clean out the tuple table
	 */
	ExecClearTuple(node->ps.ps_ResultTupleSlot);
	ExecClearTuple(node->ss_ScanTupleSlot);

	/*
	 * close heap scan
	 */
	heap_endscan(scanDesc);

	/*
	 * close the heap relation.
	 */
	ExecCloseScanRelation(relation);
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
ExecReScanSeqScan(SeqScanState *node)
{
	HeapScanDesc scan;

	scan = node->ss_currentScanDesc;

	heap_rescan(scan,			/* scan desc */
				NULL);			/* new scan keys */

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
