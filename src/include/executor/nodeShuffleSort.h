/*-------------------------------------------------------------------------
 *
 * nodeSort.h
 *
 *
 *
 * Portions Copyright (c) 1996-2012, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * src/include/executor/nodeShuffleSort.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef NODESHUFFLESORT_H
#define NODESHUFLLESORT_H

#include "nodes/execnodes.h"

extern ShuffleSortState *ExecInitShuffleSort(ShuffleSort *node, EState *estate, int eflags);
extern TupleTableSlot *ExecShuffleSort(ShuffleSortState *node);
extern void ExecEndShuffleSort(ShuffleSortState *node);
extern void ExecShuffleSortMarkPos(ShuffleSortState *node);
extern void ExecShuffleSortRestrPos(ShuffleSortState *node);
extern void ExecReScanShuffleSort(ShuffleSortState *node);

#endif   /* NODESHUFFLESORT_H */
