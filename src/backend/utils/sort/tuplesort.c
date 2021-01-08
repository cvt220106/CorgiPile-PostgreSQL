/*-------------------------------------------------------------------------
 *
 * tuplesort.c
 *	  Generalized tuple sorting routines.
 *
 * This module handles sorting of heap tuples, index tuples, or single
 * Datums (and could easily support other kinds of sortable objects,
 * if necessary).  It works efficiently for both small and large amounts
 * of data.  Small amounts are sorted in-memory using qsort().  Large
 * amounts are sorted using temporary files and a standard external sort
 * algorithm.
 *
 * See Knuth, volume 3, for more than you want to know about the external
 * sorting algorithm.  We divide the input into sorted runs using replacement
 * selection, in the form of a priority tree implemented as a heap
 * (essentially his Algorithm 5.2.3H), then merge the runs using polyphase
 * merge, Knuth's Algorithm 5.4.2D.  The logical "tapes" used by Algorithm D
 * are implemented by logtape.c, which avoids space wastage by recycling
 * disk space as soon as each block is read from its "tape".
 *
 * We do not form the initial runs using Knuth's recommended replacement
 * selection data structure (Algorithm 5.4.1R), because it uses a fixed
 * number of records in memory at all times.  Since we are dealing with
 * tuples that may vary considerably in size, we want to be able to vary
 * the number of records kept in memory to ensure full utilization of the
 * allowed sort memory space.  So, we keep the tuples in a variable-size
 * heap, with the next record to go out at the top of the heap.  Like
 * Algorithm 5.4.1R, each record is stored with the run number that it
 * must go into, and we use (run number, key) as the ordering key for the
 * heap.  When the run number at the top of the heap changes, we know that
 * no more records of the prior run are left in the heap.
 *
 * The approximate amount of memory allowed for any one sort operation
 * is specified in kilobytes by the caller (most pass work_mem).  Initially,
 * we absorb tuples and simply store them in an unsorted array as long as
 * we haven't exceeded workMem.  If we reach the end of the input without
 * exceeding workMem, we sort the array using qsort() and subsequently return
 * tuples just by scanning the tuple array sequentially.  If we do exceed
 * workMem, we construct a heap using Algorithm H and begin to emit tuples
 * into sorted runs in temporary tapes, emitting just enough tuples at each
 * step to get back within the workMem limit.  Whenever the run number at
 * the top of the heap changes, we begin a new run with a new output tape
 * (selected per Algorithm D).  After the end of the input is reached,
 * we dump out remaining tuples in memory into a final run (or two),
 * then merge the runs using Algorithm D.
 *
 * When merging runs, we use a heap containing just the frontmost tuple from
 * each source run; we repeatedly output the smallest tuple and insert the
 * next tuple from its source tape (if any).  When the heap empties, the merge
 * is complete.  The basic merge algorithm thus needs very little memory ---
 * only M tuples for an M-way merge, and M is constrained to a small number.
 * However, we can still make good use of our full workMem allocation by
 * pre-reading additional tuples from each source tape.  Without prereading,
 * our access pattern to the temporary file would be very erratic; on average
 * we'd read one block from each of M source tapes during the same time that
 * we're writing M blocks to the output tape, so there is no sequentiality of
 * access at all, defeating the read-ahead methods used by most Unix kernels.
 * Worse, the output tape gets written into a very random sequence of blocks
 * of the temp file, ensuring that things will be even worse when it comes
 * time to read that tape.  A straightforward merge pass thus ends up doing a
 * lot of waiting for disk seeks.  We can improve matters by prereading from
 * each source tape sequentially, loading about workMem/M bytes from each tape
 * in turn.  Then we run the merge algorithm, writing but not reading until
 * one of the preloaded tuple series runs out.  Then we switch back to preread
 * mode, fill memory again, and repeat.  This approach helps to localize both
 * read and write accesses.
 *
 * When the caller requests random access to the sort result, we form
 * the final sorted run on a logical tape which is then "frozen", so
 * that we can access it randomly.  When the caller does not need random
 * access, we return from tuplesort_performsort() as soon as we are down
 * to one run per logical tape.  The final merge is then performed
 * on-the-fly as the caller repeatedly calls tuplesort_getXXX; this
 * saves one cycle of writing all the data out to disk and reading it in.
 *
 * Before Postgres 8.2, we always used a seven-tape polyphase merge, on the
 * grounds that 7 is the "sweet spot" on the tapes-to-passes curve according
 * to Knuth's figure 70 (section 5.4.2).  However, Knuth is assuming that
 * tape drives are expensive beasts, and in particular that there will always
 * be many more runs than tape drives.  In our implementation a "tape drive"
 * doesn't cost much more than a few Kb of memory buffers, so we can afford
 * to have lots of them.  In particular, if we can have as many tape drives
 * as sorted runs, we can eliminate any repeated I/O at all.  In the current
 * code we determine the number of tapes M on the basis of workMem: we want
 * workMem/M to be large enough that we read a fair amount of data each time
 * we preread from a tape, so as to maintain the locality of access described
 * above.  Nonetheless, with large workMem we can have many tapes.
 *
 *
 * Portions Copyright (c) 1996-2012, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * IDENTIFICATION
 *	  src/backend/utils/sort/tuplesort.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include <limits.h>

#include "access/nbtree.h"
#include "catalog/index.h"
#include "commands/tablespace.h"
#include "executor/executor.h"
#include "miscadmin.h"
#include "pg_trace.h"
#include "utils/datum.h"
#include "utils/logtape.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/pg_rusage.h"
#include "utils/rel.h"
#include "utils/sortsupport.h"
#include "utils/tuplesort.h"

#include "access/tuptoaster.h"
#include "utils/array.h"


#include <time.h>
#include "utils/sgdmodel.h"


// /* sort-type codes for sort__start probes */
// #define HEAP_SORT		0
// #define INDEX_SORT		1
// #define DATUM_SORT		2
// #define CLUSTER_SORT	3

/* GUC variables */
#ifdef TRACE_SORT
bool		trace_sort = false;
#endif

#define SHARED_MEM_SIZE (1 << 30)
#define ARRAY_HEAD_SIZE (20)

int set_buffer_size = DEFAULT_BUFFER_SIZE; // 800 (KB)
int set_buffer_tuple_num = DEFAULT_BUFFER_TUPLE_NUM;
double set_buffer_block_num = DEFAULT_BUFFER_BLOCK_NUM;
/*
 * The objects we actually sort are SortTuple structs.  These contain
 * a pointer to the tuple proper (might be a MinimalTuple or IndexTuple),
 * which is a separate palloc chunk --- we assume it is just one chunk and
 * can be freed by a simple pfree().  SortTuples also contain the tuple's
 * first key column in Datum/nullflag format, and an index integer.
 *
 * Storing the first key column lets us save heap_getattr or index_getattr
 * calls during tuple comparisons.  We could extract and save all the key
 * columns not just the first, but this would increase code complexity and
 * overhead, and wouldn't actually save any comparison cycles in the common
 * case where the first key determines the comparison result.  Note that
 * for a pass-by-reference datatype, datum1 points into the "tuple" storage.
 *
 * When sorting single Datums, the data value is represented directly by
 * datum1/isnull1.  If the datatype is pass-by-reference and isnull1 is false,
 * then datum1 points to a separately palloc'd data value that is also pointed
 * to by the "tuple" pointer; otherwise "tuple" is NULL.
 *
 * While building initial runs, tupindex holds the tuple's run number.  During
 * merge passes, we re-use it to hold the input tape number that each tuple in
 * the heap was read from, or to hold the index of the next tuple pre-read
 * from the same tape in the case of pre-read entries.  tupindex goes unused
 * if the sort occurs entirely in memory.
 */



/*
 * Possible states of a Tuplesort object.  These denote the states that
 * persist between calls of Tuplesort routines.
 */
typedef enum
{
	TSS_INITIAL,				/* Loading tuples; still within memory limit */
	// TSS_BOUNDED,				/* Loading tuples into bounded-size heap */
	// TSS_BUILDRUNS,				/* Loading tuples; writing to tape */
	TSS_BUFFER_FULL,			/* Sort completed entirely in memory */
	// TSS_SORTEDONTAPE,			/* Sort completed, final run is on tape */
	// TSS_FINALMERGE				/* Performing final merge on-the-fly */
} TupShuffleSortStatus;

/*
 * Parameters for calculation of number of tapes to use --- see inittapes()
 * and tuplesort_merge_order().
 *
 * In this calculation we assume that each tape will cost us about 3 blocks
 * worth of buffer space (which is an underestimate for very large data
 * volumes, but it's probably close enough --- see logtape.c).
 *
 * MERGE_BUFFER_SIZE is how much data we'd like to read from each input
 * tape during a preread cycle (see discussion at top of file).
 */
// #define MINORDER		6		/* minimum merge order */
// #define TAPE_BUFFER_OVERHEAD		(BLCKSZ * 3)
// #define MERGE_BUFFER_SIZE			(BLCKSZ * 32)

// typedef int (*ShuffleSortTupleComparator) (const ShuffleSortTuple *a, const ShuffleSortTuple *b,
// 												TupleShuffleSortState *state);

/*
 * Private state of a Tuplesort operation.
 */
struct Tuplesortstate
{
	// TupShuffleSortStatus status;		/* enumerated value as shown above */
	// int			nKeys;			/* number of columns in sort key */
	// bool		randomAccess;	/* did caller request random access? */
	// bool		bounded;		/* did caller specify a maximum number of
	// 							 * tuples to return? */
	// bool		boundUsed;		/* true if we made use of a bounded heap */
	// int			bound;			/* if bounded, the maximum number of tuples */
	long		availMem;		/* remaining memory available, in bytes */
	long		allowedMem;		/* total memory allowed, in bytes */
	// int			maxTapes;		/* number of tapes (Knuth's T) */
	// int			tapeRange;		/* maxTapes-1 (Knuth's P) */
	MemoryContext shufflesortcontext;	/* memory context holding all sort data */
	// LogicalTapeSet *tapeset;	/* logtape.c object for tapes in a temp file */

	/*
	 * These function pointers decouple the routines that must know what kind
	 * of tuple we are sorting from the routines that don't need to know it.
	 * They are set up by the tuplesort_begin_xxx routines.
	 *
	 * Function to compare two tuples; result is per qsort() convention, ie:
	 * <0, 0, >0 according as a<b, a=b, a>b.  The API must match
	 * qsort_arg_comparator.
	 */
	// ShuffleSortTupleComparator comparetup;

	/*
	 * Function to copy a supplied input tuple into palloc'd space and set up
	 * its SortTuple representation (ie, set tuple/datum1/isnull1).  Also,
	 * state->availMem must be decreased by the amount of space used for the
	 * tuple copy (note the SortTuple struct itself is not counted).
	 */
	void		(*copytup) (Tuplesortstate *state, SortTuple *stup, void *tup);

	/*
	 * Function to write a stored tuple onto tape.  The representation of the
	 * tuple on tape need not be the same as it is in memory; requirements on
	 * the tape representation are given below.  After writing the tuple,
	 * pfree() the out-of-line data (not the SortTuple struct!), and increase
	 * state->availMem by the amount of memory space thereby released.
	 */
	// void		(*writetup) (TupleShuffleSortState *state, int tapenum,
	// 									 ShuffleSortTuple *stup);

	/*
	 * Function to read a stored tuple from tape back into memory. 'len' is
	 * the already-read length of the stored tuple.  Create a palloc'd copy,
	 * initialize tuple/datum1/isnull1 in the target SortTuple struct, and
	 * decrease state->availMem by the amount of memory space consumed.
	 */
	// void		(*readtup) (TupleShuffleSortState *state, ShuffleSortTuple *stup,
	// 									int tapenum, unsigned int len);

	/*
	 * Function to reverse the sort direction from its current state. (We
	 * could dispense with this if we wanted to enforce that all variants
	 * represent the sort key information alike.)
	 */
	// void		(*reversedirection) (TupleShuffleSortState *state);

	/*
	 * This array holds the tuples now in sort memory.  If we are in state
	 * INITIAL, the tuples are in no particular order; if we are in state
	 * SORTEDINMEM, the tuples are in final sorted order; in states BUILDRUNS
	 * and FINALMERGE, the tuples are organized in "heap" order per Algorithm
	 * H.  (Note that memtupcount only counts the tuples that are part of the
	 * heap --- during merge passes, memtuples[] entries beyond tapeRange are
	 * never in the heap and are used to hold pre-read tuples.)  In state
	 * SORTEDONTAPE, the array is not used.
	 */
	SortTuple  *memtuples_buffer_1;		/* array of ShuffleSortTuple structs */
	SortTuple  *memtuples_buffer_2;		/* array of ShuffleSortTuple structs */


	SortTuple	*write_buffer;
	SortTuple	*read_buffer;

	int			*read_buf_indexes;


	int			write_buf_count;	/* number of tuples currently present */
	int			read_buf_count;
	int			memtupsize;		/* allocated length of memtuples array */

	int			put_index;
	int			fetch_index;	
	/*
	 * While building initial runs, this is the current output run number
	 * (starting at 0).  Afterwards, it is the number of initial runs we made.
	 */
	int			currentRun;

	/*
	 * Unless otherwise noted, all pointer variables below are pointers to
	 * arrays of length maxTapes, holding per-tape data.
	 */

	/*
	 * These variables are only used during merge passes.  mergeactive[i] is
	 * true if we are reading an input run from (actual) tape number i and
	 * have not yet exhausted that run.  mergenext[i] is the memtuples index
	 * of the next pre-read tuple (next to be loaded into the heap) for tape
	 * i, or 0 if we are out of pre-read tuples.  mergelast[i] similarly
	 * points to the last pre-read tuple from each tape.  mergeavailslots[i]
	 * is the number of unused memtuples[] slots reserved for tape i, and
	 * mergeavailmem[i] is the amount of unused space allocated for tape i.
	 * mergefreelist and mergefirstfree keep track of unused locations in the
	 * memtuples[] array.  The memtuples[].tupindex fields link together
	 * pre-read tuples for each tape as well as recycled locations in
	 * mergefreelist. It is OK to use 0 as a null link in these lists, because
	 * memtuples[0] is part of the merge heap and is never a pre-read tuple.
	 */
	// bool	   *mergeactive;	/* active input run source? */
	// int		   *mergenext;		/* first preread tuple for each source */
	// int		   *mergelast;		/* last preread tuple for each source */
	// int		   *mergeavailslots;	/* slots left for prereading each tape */
	// long	   *mergeavailmem;	/* availMem for prereading each tape */
	// int			mergefreelist;	/* head of freelist of recycled slots */
	// int			mergefirstfree; /* first slot never used in this merge */

	/*
	 * Variables for Algorithm D.  Note that destTape is a "logical" tape
	 * number, ie, an index into the tp_xxx[] arrays.  Be careful to keep
	 * "logical" and "actual" tape numbers straight!
	 */
	// int			Level;			/* Knuth's l */
	// int			destTape;		/* current output tape (Knuth's j, less 1) */
	// int		   *tp_fib;			/* Target Fibonacci run counts (A[]) */
	// int		   *tp_runs;		/* # of real runs on each tape */
	// int		   *tp_dummy;		/* # of dummy runs for each tape (D[]) */
	// int		   *tp_tapenum;		/* Actual tape numbers (TAPE[]) */
	// int			activeTapes;	/* # of active input tapes in merge pass */

	/*
	 * These variables are used after completion of sorting to keep track of
	 * the next tuple to return.  (In the tape case, the tape's current read
	 * position is also critical state.)
	 */
	// int			result_tape;	/* actual tape number of finished output */
	// int			fetchindex;		/* array index (only used if SORTEDINMEM) */
	bool		eof_reached;	/* reached EOF (needed for cursors) */

	/* markpos_xxx holds marked position for mark and restore */
	// long		markpos_block;	/* tape block# (only used if SORTEDONTAPE) */
	int			markpos_offset; /* saved "current", or offset in tape block */
	bool		markpos_eof;	/* saved "eof_reached" */

	/*
	 * These variables are specific to the MinimalTuple case; they are set by
	 * tuplesort_begin_heap and used only by the MinimalTuple routines.
	 */
	TupleDesc	tupDesc;
	// SortSupport sortKeys;		/* array of length nKeys */

	/*
	 * This variable is shared by the single-key MinimalTuple case and the
	 * Datum case (which both use qsort_ssup()).  Otherwise it's NULL.
	 */
	// SortSupport onlyKey;

	/*
	 * These variables are specific to the CLUSTER case; they are set by
	 * tuplesort_begin_cluster.  Note CLUSTER also uses tupDesc and
	 * indexScanKey.
	 */
	IndexInfo  *indexInfo;		/* info about index being used for reference */
	EState	   *estate;			/* for evaluating index expressions */

	/*
	 * These variables are specific to the IndexTuple case; they are set by
	 * tuplesort_begin_index_xxx and used only by the IndexTuple routines.
	 */
	Relation	indexRel;		/* index being built */

	/* These are specific to the index_btree subcase: */
	ScanKey		indexScanKey;
	bool		enforceUnique;	/* complain if we find duplicate tuples */

	/* These are specific to the index_hash subcase: */
	uint32		hash_mask;		/* mask for sortable part of hash code */

	/*
	 * These variables are specific to the Datum case; they are set by
	 * tuplesort_begin_datum and used only by the DatumTuple routines.
	 */
	Oid			datumType;
	/* we need typelen and byval in order to know how to copy the Datums. */
	int			datumTypeLen;
	bool		datumTypeByVal;

	/*
	 * Resource snapshot for time of sort start.
	 */
#ifdef TRACE_SORT
	PGRUsage	ru_start;
#endif
};

// #define COMPARETUP(state,a,b)	((*(state)->comparetup) (a, b, state))
#define COPYTUP(state,stup,tup) ((*(state)->copytup) (state, stup, tup))
// #define WRITETUP(state,tape,stup)	((*(state)->writetup) (state, tape, stup))
// #define READTUP(state,stup,tape,len) ((*(state)->readtup) (state, stup, tape, len))
// #define REVERSEDIRECTION(state) ((*(state)->reversedirection) (state))
#define LACKMEM(state)		((state)->availMem < 0)
#define USEMEM(state,amt)	((state)->availMem -= (amt))
#define FREEMEM(state,amt)	((state)->availMem += (amt))

/*
 * NOTES about on-tape representation of tuples:
 *
 * We require the first "unsigned int" of a stored tuple to be the total size
 * on-tape of the tuple, including itself (so it is never zero; an all-zero
 * unsigned int is used to delimit runs).  The remainder of the stored tuple
 * may or may not match the in-memory representation of the tuple ---
 * any conversion needed is the job of the writetup and readtup routines.
 *
 * If state->randomAccess is true, then the stored representation of the
 * tuple must be followed by another "unsigned int" that is a copy of the
 * length --- so the total tape space used is actually sizeof(unsigned int)
 * more than the stored length value.  This allows read-backwards.  When
 * randomAccess is not true, the write/read routines may omit the extra
 * length word.
 *
 * writetup is expected to write both length words as well as the tuple
 * data.  When readtup is called, the tape is positioned just after the
 * front length word; readtup must read the tuple data and advance past
 * the back length word (if present).
 *
 * The write/read routines can make use of the tuple description data
 * stored in the Tuplesortstate record, if needed.  They are also expected
 * to adjust state->availMem by the amount of memory space (not tape space!)
 * released or consumed.  There is no error return from either writetup
 * or readtup; they should ereport() on failure.
 *
 *
 * NOTES about memory consumption calculations:
 *
 * We count space allocated for tuples against the workMem limit, plus
 * the space used by the variable-size memtuples array.  Fixed-size space
 * is not counted; it's small enough to not be interesting.
 *
 * Note that we count actual space used (as shown by GetMemoryChunkSpace)
 * rather than the originally-requested size.  This is important since
 * palloc can add substantial overhead.  It's not a complete answer since
 * we won't count any wasted space in palloc allocation blocks, but it's
 * a lot better than what we were doing before 7.3.
 */


static void copytup_heap(Tuplesortstate *state, SortTuple *stup, void *tup);
static void shuffle_tuple(SortTuple *a, size_t n);
static bool puttuple_into_buffer(Tuplesortstate *state, SortTuple *tuple);
static Tuplesortstate * tupleshufflesort_begin_common(int workMem);
static SortTuple* tupleshufflesort_gettuple_common(Tuplesortstate *state);
static void copytup_heap_original(Tuplesortstate *state, SortTuple *stup, void *tup);

static bool puttupleslot_into_buffer(Tuplesortstate *state, TupleTableSlot *slot);
static int my_parse_array_no_copy(struct varlena* input, int typesize, char** output);
static void fast_transfer_slot_to_sgd_tuple(Tuplesortstate *state, TupleTableSlot* slot, SortTuple* sort_tuple);

// static SortTuple* tupleshufflesort_getreadbuffer(Tuplesortstate *state);
// static int tupleshufflesort_getbuffersize(Tuplesortstate *state);
// static void tupleshufflesort_gettupleslot(Tuplesortstate *state, TupleTableSlot *slot);
// static void shuffle_tuple(SortTuple *a, size_t n);
// static void clear_tupleshufflesort_state(Tuplesortstate* tuplesortstate);
// bool is_shuffle_buffer_emtpy(Tuplesortstate *state);
// bool puttuple_into_buffer(Tuplesortstate *state, SortTuple *tuple);
// static Tuplesortstate * tupleshufflesort_begin_common(int workMem);
// Tuplesortstate * tupleshufflesort_begin_heap(TupleDesc tupDesc, int workMem);


/* When using this macro, beware of double evaluation of len */
// #define LogicalTapeReadExact(tapeset, tapenum, ptr, len) \
// 	do { \
// 		if (LogicalTapeRead(tapeset, tapenum, ptr, len) != (size_t) (len)) \
// 			elog(ERROR, "unexpected end of data"); \
// 	} while(0)


// static Tuplesortstate *tupleshufflesort_begin_common(int workMem);
// static void puttuple_common(Tuplesortstate *state, SortTuple *tuple);

// static void tuplesort_heap_insert(Tuplesortstate *state, SortTuple *tuple,
// 					  int tupleindex, bool checkIndex);

// static unsigned int getlen(Tuplesortstate *state, int tapenum, bool eofOK);

// static void free_sort_tuple(Tuplesortstate *state, SortTuple *stup);

// Lijie: add begin
// static void shuffle_tuple(SortTuple *a, size_t n);
// Lijie: add end


// Lijie: add begin
// shuffle_tuple(state->memtuples, state->memtupcount, state);

inline void
shuffle_tuple(SortTuple *a, size_t n)
{
	srand(time(0) + rand());

	int i;

	// for (i = n - 1; i > 0; i--) {
	// 	SortTuple t = *(a + i);
	// 	if (t.did == 309482)
	// 	    elog(INFO, "Before shuffle_tuple(): %d, {%f, %f, %f, %f}, %d", t.did, t.features_v[0], 
	// 		t.features_v[1], t.features_v[2], t.features_v[3], t.class_label);
	// }

	for (i = n - 1; i > 0; --i) {
		int r = rand() % (i + 1);
		// swap(a + i, a + r);
		SortTuple t = *(a + i);
		*(a + i) = *(a + r);
		*(a + r) = t;
	}

}



void 
free_tupleshufflesort_state(Tuplesortstate* state)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);

	int i;
	for (i = 0; i < state->memtupsize; i++) {
		if (state->memtuples_buffer_1[i].features_v != NULL) {
			FREEMEM(state, GetMemoryChunkSpace(state->memtuples_buffer_1[i].features_v));
			pfree(state->memtuples_buffer_1[i].features_v);
			// elog(INFO, "pfree %d features_v", i);
		}
		if (state->memtuples_buffer_2[i].features_v != NULL) {
			FREEMEM(state, GetMemoryChunkSpace(state->memtuples_buffer_2[i].features_v));
			pfree(state->memtuples_buffer_2[i].features_v);
		}
	}

	FREEMEM(state, GetMemoryChunkSpace(state->memtuples_buffer_1));
	FREEMEM(state, GetMemoryChunkSpace(state->memtuples_buffer_2));

	FREEMEM(state, GetMemoryChunkSpace(state->read_buf_indexes));
	pfree(state->memtuples_buffer_1);
	pfree(state->memtuples_buffer_2);
	pfree(state->read_buf_indexes);

	MemoryContextSwitchTo(oldcontext);
}

/*
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

// Lijie: add end
*/
/*
bool 
is_shuffle_buffer_emtpy(Tuplesortstate *state) {
	return state->memtupcount == 0;
}
*/
int 
my_parse_array_no_copy(struct varlena* input, int typesize, char** output) {
	// elog(WARNING, "Inside loss(), for v, ISEXTERNAL %d, ISCOMPR %d, ISHORT %d, varsize_short %d", VARATT_IS_EXTERNAL(v2) ? 1 : 0, VARATT_IS_COMPRESSED(v2)  ? 1 : 0, VARATT_IS_SHORT(v2)  ? 1 : 0, VARSIZE_SHORT(v2));
	// elog(WARNING, "Inside loss(), for v, varlena = %x", input);
	
	if (VARATT_IS_EXTERNAL(input) || VARATT_IS_COMPRESSED(input)) {
		// if compressed, palloc is necessary
		input = heap_tuple_untoast_attr(input);
        *output = VARDATA(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE(input) - VARHDRSZ - ARRAY_HEAD_SIZE) / typesize;
	} else if (VARATT_IS_SHORT(input)) {
        *output = VARDATA_SHORT(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE_SHORT(input) - VARHDRSZ_SHORT - ARRAY_HEAD_SIZE) / typesize;
    } else {
        *output = VARDATA(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE(input) - VARHDRSZ - ARRAY_HEAD_SIZE) / typesize;
    }
}

void fast_transfer_slot_to_sgd_tuple (
	Tuplesortstate *state, 
	TupleTableSlot* slot, 
	SortTuple* sort_tuple) {

	//Assert(sort_tuple->features_v != NULL);
	// store the values of slot to values/isnulls arrays
	// int k_col = sgd_tupledesc->k_col;
	// int v_col = sgd_tupledesc->v_col;
	// int label_col = sgd_tupledesc->label_col;

	// int attnum = HeapTupleHeaderGetNatts(slot->tts_tuple->t_data);
	slot_deform_tuple(slot, 3);
	
	// e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
	// Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
	Datum v_dat = slot->tts_values[sgd_tupledesc->v_col];
	ArrayType  *v_array = DatumGetArrayTypeP(v_dat); // Datum{0.1, 0.2, 0.3}
	
	double *v;
    int v_num = my_parse_array_no_copy((struct varlena*) v_array, 
            sizeof(float8), (char **) &v);

	Datum label_dat = slot->tts_values[sgd_tupledesc->label_col];
	sort_tuple->class_label = DatumGetInt32(label_dat);

	int n_features = sgd_tupledesc->n_features;
	// if sparse dataset
	if (sgd_tupledesc->k_col >= 0) {
		// k Datum array => int* k 
		Datum k_dat = slot->tts_values[sgd_tupledesc->k_col];
		ArrayType  *k_array = DatumGetArrayTypeP(k_dat);
		int *k;
    	int k_num = my_parse_array_no_copy((struct varlena*) k_array, 
            	sizeof(int), (char **) &k);

		memset(sort_tuple->features_v, 0, sizeof(double) * n_features);

		int i;
		for (i = 0; i < k_num; i++) {
			int f_index = k[i]; // {0, 2, 5}, k[1] = 2
			sort_tuple->features_v[f_index] = v[i]; // {0.1, 0.2, 0.3}, features[2] = 0.2
		}
	}
	
	else {
		// sgd_tuple->features_v = v;
		if (sort_tuple->features_v == NULL) {
			MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
			sort_tuple->features_v = (double *)palloc(n_features * sizeof(double));
			USEMEM(state, GetMemoryChunkSpace(sort_tuple->features_v));
			MemoryContextSwitchTo(oldcontext);
		}
		// Assert(v_num == n_features);
		memcpy(sort_tuple->features_v, v, v_num * sizeof(double));
		//sort_tuple->features_v = v;
		
	}

	sort_tuple->isnull = false;



	// for debug
	// Datum did_dat = slot->tts_values[0];
	// sort_tuple->did = DatumGetInt32(did_dat);

	// if (sort_tuple->did == 309482) {
	// 	elog(INFO, "After parsing: %d, {%f, %f, %f, %f}, %d", sort_tuple->did, sort_tuple->features_v[0], 
	// 	sort_tuple->features_v[1], sort_tuple->features_v[2], sort_tuple->features_v[3], sort_tuple->class_label);
	// }

}

bool
puttupleslot_into_buffer(Tuplesortstate *state, TupleTableSlot *slot) {
	// Assert(state->memtupcount < state->memtupsize);
	bool write_buffer_full = false;

	if (!TupIsNull(slot)) {
		fast_transfer_slot_to_sgd_tuple(state, slot, &state->write_buffer[state->put_index]);


		// if (state->write_buffer[state->put_index].did == 309482) {
		// 	SortTuple t = state->write_buffer[state->put_index];
		// 	elog(INFO, "into_buffer: %d, {%f, %f, %f, %f}, %d", t.did, t.features_v[0], 
		// 	t.features_v[1], t.features_v[2], t.features_v[3], t.class_label);
		// }

		++state->write_buf_count; // only counts non-empty tuples
	}
	else {
		state->write_buffer[state->put_index].isnull = true;
	}

	state->put_index++; 

	if (state->put_index == state->memtupsize) {
		write_buffer_full = true;
		state->put_index = 0;
	}
		
	return write_buffer_full;
}

/*
bool
puttupleslot_into_buffer(Tuplesortstate *state, TupleTableSlot *slot) {

	

	// Assert(state->memtupcount < state->memtupsize);
	bool write_buffer_full = false;
	// TODO: there is a copy
	if (slot != NULL) {
		int attnum = HeapTupleHeaderGetNatts(slot->tts_tuple->t_data);
		slot_deform_tuple(slot, attnum);

		// store the values of slot to values/isnulls arrays
		int v_col = 1;
		int label_col = 2;

		
		// e.g., features = [0.1, 0, 0.2, 0, 0, 0.3, 0, 0], class_label = -1
		// Tuple = {10, {0, 2, 5}, {0.1, 0.2, 0.3}, -1}
		Datum v_dat = slot->tts_values[v_col];
		ArrayType  *v_array = DatumGetArrayTypeP(v_dat); // Datum{0.1, 0.2, 0.3}
		
		double *v;
		int v_num = my_parse_array_no_copy((struct varlena*) v_array, 
				sizeof(float8), (char **) &v);


		Datum label_dat = slot->tts_values[label_col];
		int label = DatumGetInt32(label_dat);
		// if (state->write_buffer[state->put_index].tts_values == NULL) 
		// 	state->write_buffer[state->put_index].tts_values = (Datum *)malloc(attnum * sizeof(Datum));
		memcpy(state->write_buffer[state->put_index].features, v, v_num * sizeof(double));
		state->write_buffer[state->put_index].class_label = label;
		
		// memcpy(state->write_buffer[state->put_index].tts_values, slot->tts_values, attnum * sizeof(Datum));
		state->write_buffer[state->put_index].isnull = false;

		//elog(INFO, "[Write thread] features[0] = %f", state->write_buffer[state->put_index].features[0]);
		state->put_index++; 
	}
	else {
		state->write_buffer[state->put_index].isnull = true;
		//elog(INFO, "Slot = NULL!!!");
		// create a null SortTuple and insert it into the buffer
		state->put_index++; 
	}

	

	++state->memtupcount;

	if (state->put_index == state->memtupsize) {
		write_buffer_full = true;
		state->put_index = 0;
	}
		
	return write_buffer_full;
}
*/

bool
puttuple_into_buffer(Tuplesortstate *state, SortTuple *tuple) {
	// Assert(state->memtupcount < state->memtupsize);
	bool write_buffer_full = false;
	// TODO: there is a copy
	if (tuple != NULL) {
		state->write_buffer[state->put_index++] = *tuple; 
	}
	else {
		// create a null SortTuple and insert it into the buffer
		SortTuple null_sorttuple;
		//null_sorttuple.tuple = NULL;
		state->write_buffer[state->put_index++] = null_sorttuple; 
	}

	++state->write_buf_count;

	if (state->put_index == state->memtupsize) {
		write_buffer_full = true;
		state->put_index = 0;
	}
		
	return write_buffer_full;
}

// void
// tuplesort_end(Tuplesortstate *state)
// {
// 	state->eof_reached = true;
// }

// Lijie add begin
/*
bool
puttuple_into_buffer(TupleShuffleSortState *state, ShuffleSortTuple *tuple, bool last_tuple) {
	switch (state->status)
	{
		case TSS_INITIAL:
			if (last_tuple) {
				// buffer is empty
				if (state->memtupcount == 0)
					return true;
				else {
					tupleshufflesort_performshuffle(state);
					return false;
				}
				
			}
			Assert(state->memtupcount < state->memtupsize);
			state->memtuples[state->memtupcount++] = *tuple;
			if (state->memtupcount == state->memtupsize) {
				tupleshufflesort_performshuffle(state);
				// buffer is full and we have shuffled the buffered tuples
				return true;
			}
			else
				return false;
		case TSS_SORTEDINMEM
		default:
			elog(ERROR, "invalid tuplesort state");
			return false;
	}
}
*/

// Lijie add end

/*
 * Special versions of qsort just for SortTuple objects.  qsort_tuple() sorts
 * any variant of SortTuples, using the appropriate comparetup function.
 * qsort_ssup() is specialized for the case where the comparetup function
 * reduces to ApplySortComparator(), that is single-key MinimalTuple sorts
 * and Datum sorts.
 */
// #include "qsort_tuple.c"

void tupleshufflesort_reset_state(Tuplesortstate *state) {
	state->write_buf_count = 0; // how many tuples are stored in the write_buffer right now
	state->read_buf_count = 0;
	state->fetch_index = 0;
	state->put_index = 0;

    state->write_buffer = state->memtuples_buffer_1;
	state->read_buffer = NULL;

	// Are not used
	state->markpos_offset = 0;
	state->markpos_eof = false;
	state->eof_reached = false;
}


bool tupleshufflesort_is_read_buffer_null(Tuplesortstate *state) {
	return state->read_buffer == NULL;
}


void tupleshufflesort_init_buffer(Tuplesortstate *state) {
	state->read_buffer = state->memtuples_buffer_1;
	state->write_buffer = state->memtuples_buffer_2;
}

bool tupleshufflesort_has_tuple_in_buffer(Tuplesortstate *state) {
	//TODO: change to state->fetch_index < state->memtupcount;
	//elog(INFO, "[Read thread] fetch_index = %d, memtupsize = %d", state->fetch_index, state->memtupsize);
	return state->fetch_index < state->memtupsize;
}
/*
 *		tuplesort_begin_xxx
 *
 * Initialize for a tuple sort operation.
 *
 * After calling tuplesort_begin, the caller should call tuplesort_putXXX
 * zero or more times, then call tuplesort_performsort when all the tuples
 * have been supplied.  After performsort, retrieve the tuples in sorted
 * order by calling tuplesort_getXXX until it returns false/NULL.  (If random
 * access was requested, rescan, markpos, and restorepos can also be called.)
 * Call tuplesort_end to terminate the operation and release memory/disk space.
 *
 * Each variant of tuplesort_begin has a workMem parameter specifying the
 * maximum number of kilobytes of RAM to use before spilling data to disk.
 * (The normal value of this parameter is work_mem, but some callers use
 * other values.)  Each variant also has a randomAccess parameter specifying
 * whether the caller needs non-sequential access to the sort result.
 */

Tuplesortstate *
tupleshufflesort_begin_common(int workMem)
{
	Tuplesortstate *state;
	MemoryContext shufflesortcontext;
	MemoryContext oldcontext;

	/*
	 * Create a working memory context for this sort operation. All data
	 * needed by the sort will live inside this context.
	 */

	// TODO: change ALLOCSET_DEFAULT_MAXSIZE to 1GB or larger.
	shufflesortcontext = AllocSetContextCreate(CurrentMemoryContext,
										"TupleSort",
										ALLOCSET_DEFAULT_MINSIZE,
										ALLOCSET_DEFAULT_INITSIZE,
										ALLOCSET_DEFAULT_MAXSIZE);

	/*
	 * Make the Tuplesortstate within the per-sort context.  This way, we
	 * don't need a separate pfree() operation for it at shutdown.
	 */
	oldcontext = MemoryContextSwitchTo(shufflesortcontext);

	state = (Tuplesortstate *) palloc0(sizeof(Tuplesortstate));

#ifdef TRACE_SORT
	if (trace_sort)
		pg_rusage_init(&state->ru_start);
#endif

	// state->status = TSS_INITIAL;
	// state->randomAccess = randomAccess;
	// state->bounded = false;
	// state->boundUsed = false;
	state->allowedMem = workMem * 1024L;
	state->availMem = state->allowedMem;
	state->shufflesortcontext = shufflesortcontext;
	// state->tapeset = NULL;

	

	/*
	 * Initial size of array must be more than ALLOCSET_SEPARATE_THRESHOLD;
	 * see comments in grow_memtuples().
	 */
	// state->memtupsize = Max(1024,
	// 					ALLOCSET_SEPARATE_THRESHOLD / sizeof(SortTuple) + 1);

	/* if using set_buffer_size (KB)
	Assert(state->availMem >= set_buffer_size * 1024L);
	state->memtupsize = set_buffer_size * 1024L  / sizeof(SortTuple) + 1;
	elog(LOG, "[buffer size] %d tuples, %d big blocks, %d pages", 
			state->memtupsize, set_buffer_size / set_io_big_block_size, set_buffer_size / 8);
	*/

	state->memtupsize = set_buffer_tuple_num;
	Assert(state->memtuples_buffer_1 == NULL);
	Assert(state->memtuples_buffer_2 == NULL);
	// elog(INFO, "Begin to allocate buffers !!!");
	state->memtuples_buffer_1 = (SortTuple *) palloc(state->memtupsize * sizeof(SortTuple));
	state->memtuples_buffer_2 = (SortTuple *) palloc(state->memtupsize * sizeof(SortTuple));

	state->read_buf_indexes = (int *) palloc(state->memtupsize * sizeof(int));
	int j;
	for (j = 0; j < state->memtupsize; ++j)
		state->read_buf_indexes[j] = j;

	/*
	for (int i = 0; i < state->memtupsize; i++) {
		SortTuple null_tuple1;
		//null_tuple.tts_values = (Datum *)palloc(3 * sizeof(Datum));
		null_tuple1.isnull = true;

		SortTuple null_tuple2;
		//null_tuple.tts_values = (Datum *)palloc(3 * sizeof(Datum));
		null_tuple2.isnull = true;
		state->memtuples_buffer_1[i] = null_tuple1;
		state->memtuples_buffer_2[i] = null_tuple2;
		state->memtuples_buffer_1[i].features = (double *)palloc(54 * sizeof(double));
		state->memtuples_buffer_2[i].features = (double *)palloc(54 * sizeof(double));
	}
	*/
	int i;
	for (i = 0; i < state->memtupsize; ++i) {
		SortTuple null_tuple;
		null_tuple.isnull = true;
		null_tuple.features_v = NULL;

		state->memtuples_buffer_1[i] = null_tuple;
		state->memtuples_buffer_2[i] = null_tuple;
	}

	USEMEM(state, GetMemoryChunkSpace(state->memtuples_buffer_1));
	USEMEM(state, GetMemoryChunkSpace(state->memtuples_buffer_2));

	USEMEM(state, GetMemoryChunkSpace(state->read_buf_indexes));
	// tupleshufflesort_reset_state(state);

	/* workMem must be large enough for the minimal memtuples array */
	if (LACKMEM(state))
		elog(ERROR, "insufficient memory allowed for shuffle sort");

	state->currentRun = 0;

	/*
	 * maxTapes, tapeRange, and Algorithm D variables will be initialized by
	 * inittapes(), if needed
	 */

	// state->result_tape = -1;	/* flag that result tape has not been formed */

	MemoryContextSwitchTo(oldcontext);

	return state;
}

// nkeys = number of sort-key columns
Tuplesortstate *
tupleshufflesort_begin_heap(TupleDesc tupDesc, int workMem)
{
	Tuplesortstate *state = tupleshufflesort_begin_common(workMem);
	//MemoryContext oldcontext;
	// int			i;

	//oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);

	// AssertArg(nkeys > 0);


	// state->nKeys = nkeys;
	// TRACE_POSTGRESQL_SORT_START(HEAP_SORT,
	// 							false,	/* no unique check */
	// 							workMem);
	// state->comparetup = comparetup_heap;
	state->copytup = copytup_heap;
	// state->writetup = writetup_heap;
	// state->readtup = readtup_heap;
	// state->reversedirection = reversedirection_heap;
	state->tupDesc = tupDesc;	/* assume we need not copy tupDesc */
	/* Prepare SortSupport data for each column */
	// state->sortKeys = (SortSupport) palloc0(nkeys * sizeof(SortSupportData));
	/*
	for (i = 0; i < nkeys; i++)
	{
		SortSupport sortKey = state->sortKeys + i;
		AssertArg(attNums[i] != 0);
		AssertArg(sortOperators[i] != 0);
		sortKey->ssup_cxt = CurrentMemoryContext;
		sortKey->ssup_collation = sortCollations[i];
		sortKey->ssup_nulls_first = nullsFirstFlags[i];
		sortKey->ssup_attno = attNums[i];
		PrepareSortSupportFromOrderingOp(sortOperators[i], sortKey);
	}
	if (nkeys == 1)
		state->onlyKey = state->sortKeys;
	*/
	//MemoryContextSwitchTo(oldcontext);
	return state;
}
/*
 * tuplesort_end
 *
 *	Release resources and clean up.
 *
 * NOTE: after calling this, any pointers returned by tuplesort_getXXX are
 * pointing to garbage.  Be careful not to attempt to use or free such
 * pointers afterwards!
 */
void
tupleshufflesort_end(Tuplesortstate *state)
{
	/* context swap probably not needed, but let's be safe */
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);

// #ifdef TRACE_SHUFFLE_SORT
// 	long		spaceUsed;
// 	if (state->tapeset)
// 		spaceUsed = LogicalTapeSetBlocks(state->tapeset);
// 	else
// 		spaceUsed = (state->allowedMem - state->availMem + 1023) / 1024;
// #endif
	/*
	 * Delete temporary "tape" files, if any.
	 *
	 * Note: want to include this in reported total cost of sort, hence need
	 * for two #ifdef TRACE_SORT sections.
	 */
	// if (state->tapeset)
	// 	LogicalTapeSetClose(state->tapeset);
// #ifdef TRACE_SHUFFLE_SORT
// 	if (trace_shuffle_sort)
// 	{
// 		if (state->tapeset)
// 			elog(LOG, "external sort ended, %ld disk blocks used: %s",
// 				 spaceUsed, pg_rusage_show(&state->ru_start));
// 		else
// 			elog(LOG, "internal sort ended, %ld KB used: %s",
// 				 spaceUsed, pg_rusage_show(&state->ru_start));
// 	}
// 	TRACE_POSTGRESQL_SORT_DONE(state->tapeset != NULL, spaceUsed);
// #else
// 	/*
// 	 * If you disabled TRACE_SORT, you can still probe sort__done, but you
// 	 * ain't getting space-used stats.
// 	 */
// 	TRACE_POSTGRESQL_SORT_DONE(state->tapeset != NULL, 0L);
// #endif
	/* Free any execution state created for CLUSTER case */
	if (state->estate != NULL)
	{
		ExprContext *econtext = GetPerTupleExprContext(state->estate);
		ExecDropSingleTupleTableSlot(econtext->ecxt_scantuple);
		FreeExecutorState(state->estate);
	}
	MemoryContextSwitchTo(oldcontext);
	/*
	 * Free the per-sort memory context, thereby releasing all working memory,
	 * including the Tuplesortstate struct itself.
	 */
	MemoryContextDelete(state->shufflesortcontext);
}


void tupleshufflesort_swapbuffer(Tuplesortstate *state) {
	SortTuple *temp = state->write_buffer;
	state->write_buffer = state->read_buffer;
	state->read_buffer = temp;

	state->fetch_index = 0;
}


/*
 * Lijie: add begin
 * 
 * Accept one tuple while collecting input data for sort.
 *
 * Note that the input data is always copied; the caller need not save it.
 */
// bool
// tupleshufflesort_puttupleslot(Tuplesortstate *state, TupleTableSlot *slot)
// {
// 	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
// 	SortTuple	stup;
// 	bool write_buffer_full = false;

// 	if (!TupIsNull(slot)) {
// 		/*
// 		* Copy the given tuple into memory we control, and decrease availMem.
// 		* Then call the common code.
// 		*/
// 		copytup_heap(state, &stup, (void *) slot); // HeapTuple slot => MinimalTuple stup
// 		write_buffer_full = puttuple_into_buffer(state, &stup);
// 		//elog(INFO, "[Write thread] >> Finish puttuple_into_buffer(state, &stup), put_index = %d, stup = %x", state->put_index, stup.tuple);
// 	}

// 	else {
// 		write_buffer_full = puttuple_into_buffer(state, NULL);
// 	}
	
// 	MemoryContextSwitchTo(oldcontext);
	
// 	return write_buffer_full;
// }

bool
tupleshufflesort_puttupleslot(Tuplesortstate *state, TupleTableSlot *slot)
{
	//MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
	// SortTuple	stup;
	bool write_buffer_full = false;

	write_buffer_full = puttupleslot_into_buffer(state, slot);
	
	//MemoryContextSwitchTo(oldcontext);
	
	return write_buffer_full;
}
// Lijie: add end

/*
 * Accept one Datum while collecting input data for sort.
 *
 * If the Datum is pass-by-ref type, the value will be copied.
 */
// void
// tupleshufflesort_putdatum(Tuplesortstate *state, Datum val, bool isNull)
// {
// 	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
// 	SortTuple	stup;
// 	/*
// 	 * If it's a pass-by-reference value, copy it into memory we control, and
// 	 * decrease availMem.  Then call the common code.
// 	 */
// 	if (isNull || state->datumTypeByVal)
// 	{
// 		stup.datum1 = val;
// 		stup.isnull1 = isNull;
// 		stup.tuple = NULL;		/* no separate storage */
// 	}
// 	else
// 	{
// 		stup.datum1 = datumCopy(val, false, state->datumTypeLen);
// 		stup.isnull1 = false;
// 		stup.tuple = DatumGetPointer(stup.datum1);
// 		USEMEM(state, GetMemoryChunkSpace(stup.tuple));
// 	}
// 	puttuple_common(state, &stup);
// 	MemoryContextSwitchTo(oldcontext);
// }


void
tupleshufflesort_performshuffle(Tuplesortstate *state) 
{
	// clock_t start, finish;    
	// start = clock(); 
	// int n = state->memtupcount;
	//elog(INFO, "[Write thread] perform_shuffle: state->memtupcount = %d", state->memtupcount);
	/*
	if (state->memtupcount > 1) {
		if (is_training)
			shuffle_tuple_indexes(state->read_buf_indexes, state->memtupcount);
		state->memtupcount = 0;
	}
	*/

	state->read_buf_count = state->write_buf_count;
	state->write_buf_count = 0;

	if (is_training) {

		if (state->read_buf_count < state->memtupsize) {
			int i;
			for (i = 0; i < state->memtupsize; ++i) 
				state->read_buf_indexes[i] = i;
		} 
		// shuffle read_buf_indexes
		int* a = state->read_buf_indexes;
		int i;
		for (i = state->read_buf_count - 1; i > 0; --i) {
			int r = rand() % (i + 1);
			int t = *(a + i);
			*(a + i) = *(a + r);
			*(a + r) = t;
		}
	}

	if (state->read_buf_count < state->memtupsize) {
		++state->read_buf_count;
	}

	// finish = clock();    
   	// double duration = (double)(finish - start) / CLOCKS_PER_SEC;    
   	// elog(INFO, "[shuffle %d tuples] %f seconds\n", n, duration);      
}


/*
void
tupleshufflesort_performshuffle(Tuplesortstate *state) 
{
	// clock_t start, finish;    
	// start = clock(); 
	// int n = state->memtupcount;
	//elog(INFO, "[Write thread] perform_shuffle: state->memtupcount = %d", state->memtupcount);
	if (state->memtupcount > 1) {
		if (is_training)
			shuffle_tuple(state->write_buffer, state->memtupcount);
		state->memtupcount = 0;
	}

	// finish = clock();    
   	// double duration = (double)(finish - start) / CLOCKS_PER_SEC;    
   	// elog(INFO, "[shuffle %d tuples] %f seconds\n", n, duration);      
}
*/
/*
void
tupleshufflesort_performshuffle(TupleShuffleSortState *state)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
	switch (state->status)
	{
		case TSS_INITIAL:
			
			if (state->memtupcount > 1)
			{
				
				
				if (state->onlyKey != NULL)
					qsort_ssup(state->memtuples, state->memtupcount,
							   state->onlyKey);
				else
					qsort_tuple(state->memtuples,
								state->memtupcount,
								state->comparetup,
								state);
			
				shuffle_tuple(state->memtuples, state->memtupcount, state);
			}
			
			break;
		case TSS_BOUNDED:
			sort_bounded_heap(state);
			state->current = 0;
			state->eof_reached = false;
			state->markpos_offset = 0;
			state->markpos_eof = false;
			state->status = TSS_SORTEDINMEM;
			break;
		case TSS_BUILDRUNS:
			dumptuples(state, true);
			mergeruns(state);
			state->eof_reached = false;
			state->markpos_block = 0L;
			state->markpos_offset = 0;
			state->markpos_eof = false;
			break;
		default:
			elog(ERROR, "invalid tuplesort state");
			break;
	}
	MemoryContextSwitchTo(oldcontext);
}
*/
/*
 * Internal routine to fetch the next tuple in either forward or back
 * direction into *stup.  Returns FALSE if no more tuples.
 * If *should_free is set, the caller must pfree stup.tuple when done with it.
 */
SortTuple*
tupleshufflesort_gettuple_common(Tuplesortstate *state)
{
	return &(state->read_buffer[state->fetch_index++]);
}



/*
static bool
tupleshufflesort_gettuple_common(Tuplesortstate *state, SortTuple *stup)
{
	bool tuple_left = true;
	// unsigned int tuplen;
	if (state->fetchindex < state->memtupcount)
		*stup = state->memtuples[state->fetchindex++];


	// if there is not any tuple left in the buffer, clear the buffer indexes
	if (state->fetchindex == state->memtupcount) {
		tuple_left = false;
		state->fetchindex = 0;
		state->memtupcount = 0;
	}
	return tuple_left;
}
*/
	
/*
static bool
tupleshufflesort_gettuple_common(TupleShuffleSortState *state, ShuffleSortTuple *stup)
{
	unsigned int tuplen;
	switch (state->status)
	{
		case TSS_SORTEDINMEM:
			Assert(forward || state->randomAccess);
			*should_free = false;
			if (forward)
			{
				if (state->current < state->memtupcount)
				{
					*stup = state->memtuples[state->current++];
					return true;
				}
				state->eof_reached = true;
				return false;
			}
			
			break;
		default:
			elog(ERROR, "invalid tuplesort state");
			return false;		
	}
}
*/
/*
 * Fetch the next tuple in either forward or back direction.
 * If successful, put tuple in slot and return TRUE; else, clear the slot
 * and return FALSE.
 */
/*
bool
tupleshufflesort_gettupleslot(Tuplesortstate *state, TupleTableSlot *slot)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
	SortTuple	stup;
	bool should_free = true;

	bool tuple_left = tupleshufflesort_gettuple_common(state, &stup);
	
	MemoryContextSwitchTo(oldcontext);
	if (stup.tuple)
	{
		ExecStoreMinimalTuple((MinimalTuple) stup.tuple, slot, should_free);
		return tuple_left;
	}
	else
	{
		ExecClearTuple(slot);
		return false;
	}
}
*/

/*
bool
tupleshufflesort_gettupleslot(Tuplesortstate *state, TupleTableSlot *slot)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
	SortTuple	stup;
	bool should_free = true;

	bool tuple_left = tupleshufflesort_gettuple_common(state, &stup);
	
	MemoryContextSwitchTo(oldcontext);
	if (stup.tuple)
	{
		ExecStoreMinimalTuple((MinimalTuple) stup.tuple, slot, should_free);
		return tuple_left;
	}
	else
	{
		ExecClearTuple(slot);
		return false;
	}
}
*/


SortTuple* tupleshufflesort_getreadbuffer(Tuplesortstate *state) {
	return state->read_buffer;
}

int* tupleshufflesort_getbufferindexes(Tuplesortstate *state) {
	return state->read_buf_indexes;
}


int tupleshufflesort_getbuffersize(Tuplesortstate *state) {
	// return state->memtupsize;
	return state->read_buf_count;
}

void
tupleshufflesort_gettupleslot(Tuplesortstate *state, TupleTableSlot *slot)
{
	//MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
	// bool should_free = false;
	
	SortTuple*	stup = tupleshufflesort_gettuple_common(state);
	// elog(INFO, "[Read thread] >> Finish tupleshufflesort_gettuple_common(state, &stup); stup = %x", stup);

	if (stup->isnull == false)
	{
		//slot->tts_tuple = stup.tuple;
		//slot->tts_values = stup.tts_values;
		slot->tts_isempty = false;
		slot->tts_shouldFree = false;
	
		// slot->tts_nvalid = 0;
		slot->features_v = stup->features_v;
		slot->label = stup->class_label;

		// for debug
		// slot->did = stup.did;

		// if (slot->did == 309482) {
		// 	double a0 = slot->features_v[0] - 0.127257;
		// 	double a1 = slot->features_v[1] + 1.051316;
		// 	if (a0 < 0)
		// 		a0 = a0 * -1;
		// 	if (a1 < 0)
		// 		a1 = a1 * -1;

		// 	Assert(a0 <= 0.000002);
		// 	Assert(a1 <= 0.000002);
		// 	// elog(INFO, "%d, {%f, %f, %f, %f}, %d", slot->did, slot->features_v[0], 
		// 	// 	slot->features_v[1], slot->features_v[2], slot->features_v[3], slot->label);
		// }

		// if (stup.features == NULL) {
		// 	elog(INFO, "[Read thread] get stup.features == NULL, fetch_index = %d", state->fetch_index);
		// }
		//ExecStoreMinimalTuple((MinimalTuple) stup.tuple, slot, should_free);
		//elog(INFO, "[Read thread] >> Finish ExecStoreMinimalTuple(stup.tuple, slot, should_free); tuple = %x", stup.tuple);
	}
	else
	{
		//elog(INFO, "To clear the slot with ExecClearTuple(slot)");
		ExecClearTuple(slot);
	}

	//MemoryContextSwitchTo(oldcontext);
}


/*
bool
tupleshufflesort_gettupleslot(Tuplesortstate *state, 
					   TupleTableSlot *slot, bool eof_reach)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
	SortTuple	stup;
	bool should_free = false;
	bool tuple_left;
	if (eof_reach) {
		stup.tuple = NULL;
	}
	else {
		tuple_left = tupleshufflesort_gettuple_common(state, &stup);
	}
	
	MemoryContextSwitchTo(oldcontext);
	if (stup.tuple)
	{
		ExecStoreMinimalTuple((MinimalTuple) stup.tuple, slot, should_free);
		return tuple_left;
	}
	else
	{
		ExecClearTuple(slot);
		return false;
	}
}
*/

/*
bool
tupleshufflesort_gettupleslot(TupleShuffleSortState *state, bool forward,
					   TupleTableSlot *slot, )
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
	ShuffleSortTuple	stup;
	bool should_free = false;
	if (!tupleshufflesort_gettuple_common(state, &stup))   //, &should_free))
		stup.tuple = NULL;
	MemoryContextSwitchTo(oldcontext);
	if (stup.tuple)
	{
		ExecStoreMinimalTuple((MinimalTuple) stup.tuple, slot, should_free);
		return true;
	}
	else
	{
		ExecClearTuple(slot);
		return false;
	}
}
*/
/*
 * Fetch the next tuple in either forward or back direction.
 * Returns NULL if no more tuples.  If *should_free is set, the
 * caller must pfree the returned tuple when done with it.
 */
/*
HeapTuple
tupleshufflesort_getheaptuple(Tuplesortstate *state, bool forward, bool *should_free)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);
	SortTuple	stup;
	if (!tupleshufflesort_gettuple_common(state, &stup))
		stup.tuple = NULL;
	MemoryContextSwitchTo(oldcontext);
	return stup.tuple;
}
*/
/*
 * tuplesort_get_stats - extract summary statistics
 *
 * This can be called after tuplesort_performsort() finishes to obtain
 * printable summary information about how the sort was performed.
 * spaceUsed is measured in kilobytes.
 */
// void
// tupleshufflesort_get_stats(Tuplesortstate *state,
// 					const char **sortMethod,
// 					const char **spaceType,
// 					long *spaceUsed)
// {
	/*
	 * Note: it might seem we should provide both memory and disk usage for a
	 * disk-based sort.  However, the current code doesn't track memory space
	 * accurately once we have begun to return tuples to the caller (since we
	 * don't account for pfree's the caller is expected to do), so we cannot
	 * rely on availMem in a disk sort.  This does not seem worth the overhead
	 * to fix.  Is it worth creating an API for the memory context code to
	 * tell us how much is actually used in sortcontext?
	 */
	// if (state->tapeset)
	// {
	// 	*spaceType = "Disk";
	// 	*spaceUsed = LogicalTapeSetBlocks(state->tapeset) * (BLCKSZ / 1024);
	// }
	// else
	// // {
	// 	*spaceType = "Memory";
	// 	*spaceUsed = (state->allowedMem - state->availMem + 1023) / 1024;
	// // }
	// switch (state->status)
	// {
	// 	case TSS_SORTEDINMEM:
	// 		if (state->boundUsed)
	// 			*sortMethod = "top-N heapsort";
	// 		else
	// 			*sortMethod = "quicksort";
	// 		break;
	// 	case TSS_SORTEDONTAPE:
	// 		*sortMethod = "external sort";
	// 		break;
	// 	case TSS_FINALMERGE:
	// 		*sortMethod = "external merge";
	// 		break;
	// 	default:
	// 		*sortMethod = "still in progress";
	// 		break;
	// }
// }
/*
 * Heap manipulation routines, per Knuth's Algorithm 5.2.3H.
 *
 * Compare two SortTuples.  If checkIndex is true, use the tuple index
 * as the front of the sort key; otherwise, no.
 */
// #define HEAPCOMPARE(tup1,tup2) \
// 	(checkIndex && ((tup1)->tupindex != (tup2)->tupindex) ? \
// 	 ((tup1)->tupindex) - ((tup2)->tupindex) : \
// 	 COMPARETUP(state, tup1, tup2))
/*
 * Inline-able copy of FunctionCall2Coll() to save some cycles in sorting.
 */
// static inline Datum
// myFunctionCall2Coll(FmgrInfo *flinfo, Oid collation, Datum arg1, Datum arg2)
// {
// 	FunctionCallInfoData fcinfo;
// 	Datum		result;
// 	InitFunctionCallInfoData(fcinfo, flinfo, 2, collation, NULL, NULL);
// 	fcinfo.arg[0] = arg1;
// 	fcinfo.arg[1] = arg2;
// 	fcinfo.argnull[0] = false;
// 	fcinfo.argnull[1] = false;
// 	result = FunctionCallInvoke(&fcinfo);
// 	/* Check for null result, since caller is clearly not expecting one */
// 	if (fcinfo.isnull)
// 		elog(ERROR, "function %u returned NULL", fcinfo.flinfo->fn_oid);
// 	return result;
// }
/*
 * Convenience routine to free a tuple previously loaded into sort memory
 */
// static void
// free_sort_tuple(Tuplesortstate *state, SortTuple *stup)
// {
// 	FREEMEM(state, GetMemoryChunkSpace(stup->tuple));
// 	pfree(stup->tuple);
// }


void
copytup_heap_original(Tuplesortstate *state, SortTuple *stup, void *tup)
{
	/*
	 * We expect the passed "tup" to be a TupleTableSlot, and form a
	 * MinimalTuple using the exported interface for that.
	 */
	TupleTableSlot *slot = (TupleTableSlot *) tup;
	MinimalTuple tuple;
	// HeapTupleData htup;

	/* copy the tuple into sort storage */
	tuple = ExecCopySlotMinimalTuple(slot);
	//stup->tuple = (void *) tuple;
	USEMEM(state, GetMemoryChunkSpace(tuple));
	// /* set up first-column key value */
	// htup.t_len = tuple->t_len + MINIMAL_TUPLE_OFFSET;
	// htup.t_data = (HeapTupleHeader) ((char *) tuple - MINIMAL_TUPLE_OFFSET);
	// stup->datum1 = heap_getattr(&htup,
	// 							state->sortKeys[0].ssup_attno,
	// 							state->tupDesc,
	// 							&stup->isnull1);
}

void
copytup_heap(Tuplesortstate *state, SortTuple *stup, void *tup)
{
	/*
	 * We expect the passed "tup" to be a TupleTableSlot, and form a
	 * MinimalTuple using the exported interface for that.
	 */
	TupleTableSlot *slot = (TupleTableSlot *) tup;
	MinimalTuple tuple;
	// HeapTupleData htup;

	/* copy the tuple into sort storage */
	tuple = ExecCopySlotMinimalTuple(slot);
	//stup->tuple = (void *) tuple;
	USEMEM(state, GetMemoryChunkSpace(tuple));



	// /* set up first-column key value */
	// htup.t_len = tuple->t_len + MINIMAL_TUPLE_OFFSET;
	// htup.t_data = (HeapTupleHeader) ((char *) tuple - MINIMAL_TUPLE_OFFSET);
	// stup->datum1 = heap_getattr(&htup,
	// 							state->sortKeys[0].ssup_attno,
	// 							state->tupDesc,
	// 							&stup->isnull1);
}


/*
 * tuplesort_markpos	- saves current position in the merged sort file
 */
void
tupleshufflesort_markpos(Tuplesortstate *state)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);

	// Assert(state->randomAccess);

	
	state->markpos_offset = state->fetch_index;
	state->markpos_eof = state->eof_reached;

	MemoryContextSwitchTo(oldcontext);
}

/*
 * tuplesort_restorepos - restores current position in merged sort file to
 *						  last saved position
 */
void
tupleshufflesort_restorepos(Tuplesortstate *state)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);

	// Assert(state->randomAccess);
	state->fetch_index = state->markpos_offset;
	state->eof_reached = state->markpos_eof;
	
	MemoryContextSwitchTo(oldcontext);
}

void
tupleshufflesort_rescan(Tuplesortstate *state)
{
	//MemoryContext oldcontext = MemoryContextSwitchTo(state->shufflesortcontext);

	tupleshufflesort_reset_state(state);
	
	// MemoryContextSwitchTo(oldcontext);
}





// for covering the apis of tuplesort.c




Tuplesortstate *tuplesort_begin_heap(TupleDesc tupDesc,
					 int nkeys, AttrNumber *attNums,
					 Oid *sortOperators, Oid *sortCollations,
					 bool *nullsFirstFlags,
					 int workMem, bool randomAccess)
{
	return NULL;
}

Tuplesortstate *tuplesort_begin_cluster(TupleDesc tupDesc,
						Relation indexRel,
						int workMem, bool randomAccess) 
{
	return NULL;
}

Tuplesortstate *tuplesort_begin_index_btree(Relation indexRel,
							bool enforceUnique,
							int workMem, bool randomAccess)
{
	return NULL;
}

Tuplesortstate *tuplesort_begin_index_hash(Relation indexRel,
						   uint32 hash_mask,
						   int workMem, bool randomAccess)
{
	return NULL;
}

Tuplesortstate *tuplesort_begin_datum(Oid datumType,
					  Oid sortOperator, Oid sortCollation,
					  bool nullsFirstFlag,
					  int workMem, bool randomAccess)
{
	return NULL;
}

void tuplesort_set_bound(Tuplesortstate *state, int64 bound)
{
	
}

void tuplesort_puttupleslot(Tuplesortstate *state,
					   TupleTableSlot *slot)
{

}

void tuplesort_putheaptuple(Tuplesortstate *state, HeapTuple tup)
{

}

void tuplesort_putindextuple(Tuplesortstate *state, IndexTuple tuple)
{

}

void tuplesort_putdatum(Tuplesortstate *state, Datum val,
				   bool isNull)
{
	
}

void tuplesort_performsort(Tuplesortstate *state)
{
	
}

bool tuplesort_gettupleslot(Tuplesortstate *state, bool forward,
					   TupleTableSlot *slot)
{
	return false;
}

HeapTuple tuplesort_getheaptuple(Tuplesortstate *state, bool forward,
					   bool *should_free)
{
	return NULL;
}

IndexTuple tuplesort_getindextuple(Tuplesortstate *state, bool forward,
						bool *should_free)
{
	return NULL;
}

bool tuplesort_getdatum(Tuplesortstate *state, bool forward,
				   Datum *val, bool *isNull)
{
	return false;
}

void tuplesort_end(Tuplesortstate *state)
{
	
}

void tuplesort_get_stats(Tuplesortstate *state,
					const char **sortMethod,
					const char **spaceType,
					long *spaceUsed)
{

}

int	tuplesort_merge_order(long allowedMem) 
{
	return 0;
}

/*
 * These routines may only be called if randomAccess was specified 'true'.
 * Likewise, backwards scan in gettuple/getdatum is only allowed if
 * randomAccess was specified.
 */
void tuplesort_rescan(Tuplesortstate *state)
{
	tupleshufflesort_reset_state(state);
}


void tuplesort_markpos(Tuplesortstate *state)
{
	
}
void tuplesort_restorepos(Tuplesortstate *state)
{

}