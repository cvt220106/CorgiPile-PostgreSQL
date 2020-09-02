
#ifndef NODESGD_H
#define NODESGD_H

#include "nodes/execnodes.h"

extern SeqScanState *ExecInitSeqScan(SeqScan *node, EState *estate, int eflags);
extern TupleTableSlot *ExecSeqScan(SeqScanState *node);
extern void ExecEndSeqScan(SeqScanState *node);
extern void ExecSeqMarkPos(SeqScanState *node);
extern void ExecSeqRestrPos(SeqScanState *node);
extern void ExecReScanSeqScan(SeqScanState *node);

#endif   /* NODESEQSCAN_H */
