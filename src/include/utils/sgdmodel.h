#ifndef SGDMODEL_H
#define SGDMODEL_H

typedef struct Model {
    double total_loss;
	double* w;
    int batch_size;
    double learning_rate;
    double n_features;
    int iter_num;
    int tuple_num;
} Model;

#endif   