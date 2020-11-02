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
#ifndef TUPLESORT_H
#define TUPLESORT_H

#include "access/itup.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "utils/relcache.h"


/* Tuplesortstate is an opaque type whose details are not known outside
 * tuplesort.c.
 */
typedef struct Tuplesortstate Tuplesortstate;

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
extern void tupleshufflesort_set_end(Tuplesortstate *state);

extern Tuplesortstate *tupleshufflesort_begin_heap(TupleDesc tupDesc, int workMem);
extern Tuplesortstate *tupleshufflesort_begin_cluster(TupleDesc tupDesc,
						Relation indexRel,
						int workMem, bool randomAccess);
extern Tuplesortstate *tupleshufflesort_begin_index_btree(Relation indexRel,
							bool enforceUnique,
							int workMem, bool randomAccess);
extern Tuplesortstate *tupleshufflesort_begin_index_hash(Relation indexRel,
						   uint32 hash_mask,
						   int workMem, bool randomAccess);
extern Tuplesortstate *tupleshufflesort_begin_datum(Oid datumType,
					  Oid shuffleSortOperator, Oid shuffleSortCollation,
					  bool nullsFirstFlag,
					  int workMem, bool randomAccess);

extern void tupleshufflesort_set_bound(Tuplesortstate *state, int64 bound);

extern void tupleshufflesort_putheaptuple(Tuplesortstate *state, HeapTuple tup);
extern void tupleshufflesort_putindextuple(Tuplesortstate *state, IndexTuple tuple);
extern void tupleshufflesort_putdatum(Tuplesortstate *state, Datum val,
				   bool isNull);

extern void tupleshufflesort_performsort(Tuplesortstate *state);

// lijie: begin
extern void tupleshufflesort_performshuffle(Tuplesortstate *state);


extern void clear_tupleshufflesort_state(Tuplesortstate* tuplesortstate);

extern bool tupleshufflesort_puttupleslot(Tuplesortstate *state, TupleTableSlot *slot);

extern bool is_shuffle_buffer_emtpy(Tuplesortstate *state);

// lijie: end

extern bool tupleshufflesort_gettupleslot(Tuplesortstate *state, 
					   TupleTableSlot *slot);
extern HeapTuple tupleshufflesort_getheaptuple(Tuplesortstate *state, bool forward,
					   bool *should_free);
extern IndexTuple tupleshufflesort_getindextuple(Tuplesortstate *state, bool forward,
						bool *should_free);
extern bool tupleshufflesort_getdatum(Tuplesortstate *state, bool forward,
				   Datum *val, bool *isNull);

extern void tupleshufflesort_end(Tuplesortstate *state);

extern void tupleshufflesort_get_stats(Tuplesortstate *state,
					const char **sortMethod,
					const char **spaceType,
					long *spaceUsed);

extern int	tupleshufflesort_merge_order(long allowedMem);

/*
 * These routines may only be called if randomAccess was specified 'true'.
 * Likewise, backwards scan in gettuple/getdatum is only allowed if
 * randomAccess was specified.
 */

extern void tupleshufflesort_rescan(Tuplesortstate *state);
// extern void tupleshufflesort_markpos(Tuplesortstate *state);
// extern void tupleshufflesort_restorepos(Tuplesortstate *state);

#endif   /* TUPLESHUFFLESORT_H */