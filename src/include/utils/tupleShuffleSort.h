/*-------------------------------------------------------------------------
 *
 * tuplesort.h
 *	  Generalized tuple sorting routines.
 *
 * This module handles sorting of heap tuples, index tuples, or single
 * Datums (and could easily support other kinds of sortable objects,
 * if necessary).  It works efficiently for both small and large amounts
 * of data.  Small amounts are sorted in-memory using qsort().  Large
 * amounts are sorted using temporary files and a standard external sort
 * algorithm.
 *
 * Portions Copyright (c) 1996-2012, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * src/include/utils/tuplesort.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef TUPLESHUFFLESORT_H
#define TUPLESHUFFLESORT_H

#include "access/itup.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "utils/relcache.h"


/* Tuplesortstate is an opaque type whose details are not known outside
 * tuplesort.c.
 */
typedef struct TupleShuffleSortState TupleShuffleSortState;

/*
 * We provide multiple interfaces to what is essentially the same code,
 * since different callers have different data to be sorted and want to
 * specify the sort key information differently.  There are two APIs for
 * sorting HeapTuples and two more for sorting IndexTuples.  Yet another
 * API supports sorting bare Datums.
 *
 * The "heap" API actually stores/sorts MinimalTuples, which means it doesn't
 * preserve the system columns (tuple identity and transaction visibility
 * info).  The sort keys are specified by column numbers within the tuples
 * and sort operator OIDs.  We save some cycles by passing and returning the
 * tuples in TupleTableSlots, rather than forming actual HeapTuples (which'd
 * have to be converted to MinimalTuples).  This API works well for sorts
 * executed as parts of plan trees.
 *
 * The "cluster" API stores/sorts full HeapTuples including all visibility
 * info. The sort keys are specified by reference to a btree index that is
 * defined on the relation to be sorted.  Note that putheaptuple/getheaptuple
 * go with this API, not the "begin_heap" one!
 *
 * The "index_btree" API stores/sorts IndexTuples (preserving all their
 * header fields).  The sort keys are specified by a btree index definition.
 *
 * The "index_hash" API is similar to index_btree, but the tuples are
 * actually sorted by their hash codes not the raw data.
 */
extern void tupleshufflesort_set_end(TupleShuffleSortState *state);

extern TupleShuffleSortState *tupleshufflesort_begin_heap(TupleDesc tupDesc, int workMem);
extern TupleShuffleSortState *tupleshufflesort_begin_cluster(TupleDesc tupDesc,
						Relation indexRel,
						int workMem, bool randomAccess);
extern TupleShuffleSortState *tupleshufflesort_begin_index_btree(Relation indexRel,
							bool enforceUnique,
							int workMem, bool randomAccess);
extern TupleShuffleSortState *tupleshufflesort_begin_index_hash(Relation indexRel,
						   uint32 hash_mask,
						   int workMem, bool randomAccess);
extern TupleShuffleSortState *tupleshufflesort_begin_datum(Oid datumType,
					  Oid shuffleSortOperator, Oid shuffleSortCollation,
					  bool nullsFirstFlag,
					  int workMem, bool randomAccess);

extern void tupleshufflesort_set_bound(TupleShuffleSortState *state, int64 bound);

extern void tupleshufflesort_putheaptuple(TupleShuffleSortState *state, HeapTuple tup);
extern void tupleshufflesort_putindextuple(TupleShuffleSortState *state, IndexTuple tuple);
extern void tupleshufflesort_putdatum(TupleShuffleSortState *state, Datum val,
				   bool isNull);

extern void tupleshufflesort_performsort(TupleShuffleSortState *state);

// lijie: begin
extern void tupleshufflesort_performshuffle(TupleShuffleSortState *state);

extern int compute_loss_and_update_model(TupleShuffleSortState* state, Model* model, 
			int ith_tuple, int batch_size, bool last_tuple);

extern void clear_tupleshufflesort_state(TupleShuffleSortState* tuplesortstate);

extern bool tupleshufflesort_puttupleslot(TupleShuffleSortState *state, TupleTableSlot *slot);
// lijie: end

extern bool tupleshufflesort_gettupleslot(TupleShuffleSortState *state, 
					   TupleTableSlot *slot, bool eof_reach);
extern HeapTuple tupleshufflesort_getheaptuple(TupleShuffleSortState *state, bool forward,
					   bool *should_free);
extern IndexTuple tupleshufflesort_getindextuple(TupleShuffleSortState *state, bool forward,
						bool *should_free);
extern bool tupleshufflesort_getdatum(TupleShuffleSortState *state, bool forward,
				   Datum *val, bool *isNull);

extern void tupleshufflesort_end(TupleShuffleSortState *state);

extern void tupleshufflesort_get_stats(TupleShuffleSortState *state,
					const char **sortMethod,
					const char **spaceType,
					long *spaceUsed);

extern int	tupleshufflesort_merge_order(long allowedMem);

/*
 * These routines may only be called if randomAccess was specified 'true'.
 * Likewise, backwards scan in gettuple/getdatum is only allowed if
 * randomAccess was specified.
 */

extern void tupleshufflesort_rescan(TupleShuffleSortState *state);
extern void tupleshufflesort_markpos(TupleShuffleSortState *state);
extern void tupleshufflesort_restorepos(TupleShuffleSortState *state);

#endif   /* TUPLESHUFFLESORT_H */
