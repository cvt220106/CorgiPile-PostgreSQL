
#ifndef NODESGD_H
#define NODESGD_H

#include "nodes/execnodes.h"

typedef struct Model {
    double total_loss;
	double* w;
    int batch_size;
    double learning_rate;
    double n_features;
    int iter_num;
    int tuple_num;
} Model;

extern SGDState *ExecInitSGD(SGD *node, EState *estate, int eflags);
extern TupleTableSlot *ExecSGD(SGDState *node);
extern void ExecEndSGD(SGDState *node);
extern void ExecSGDMarkPos(SGDState *node);
extern void ExecSGDRestrPos(SGDState *node);
extern void ExecReScanSGD(SGDState *node);

#endif



