# In-Database Machine Learning with CorgiPile: Stochastic Gradient Descent without Full Data Shuffle


# Introduction
CorgiPile is a novel *two-level hierarchical* data shuffle mechanism for efficient SGD in DB systems. CorgiPile first samples and shuffles data at *block-level*, and then shuffles data at *tuple-level* within the sampled data blocks, i.e., firstly shuffling data blocks (a batch of table pages per block), and then merging sampled blocks in a buffer, and finally shuffling tuples in the buffer for SGD.
Compared with existing technologies, our approach can avoid the full shuffle while maintaining comparable convergence rate as if a full shuffle were performed.

We have implemented CorgiPile inside PostgreSQL, with three new inherent physical operators.
Extensive experimental results show that our CorgiPile in PostgreSQL
can achieve comparable convergence rate with the full-shuffle based SGD, and 2.0X-12.8X faster than state-of-the-art in-DB ML systems including MADlib and Bismarck, on both HDD and SSD.



# Implementation in PostgreSQL

The following figure illustrates the implementation of CorgiPile with new operators and double-buffer optimization, in PostgreSQL.

![Implementation](corgipile-docs/impl/Shuffle-free-SGD-implementation-2.png)

The query, control flow and iteration paradigm of CorgiPile in PostgreSQL is as follows.
```SQL
Query: SELECT * FROM table_name TRAIN BY model_name WITH params = args;
```
```c++
// Query plan: ExecSGD => ExecTupleShuffle => ExecBlockShuffle 

/************************
 *  SGD operator 
************************/
void ExecInit():
   initialize the ML model;
    
Model ExecSGD():
   for each epoch:
      /* training */
      while(true):
         pull a tuple i from TupleShuffle operator;
         if (tuple i is not null):
            compute the gradient g of tuple i;
            update model with g;
         else:
            break;
      ExecReScan();
   return trained model;
    
void ExecReScan():
   invoke TupleShuffle.ExecReScan();

/************************
 * TupleShuffle operator
************************/
void ExecInit(): 
   initialize buffer and I/O offsets;
   
Tuple ExecTupleShuffle():
   if (buffer is empty):
      pull tuples from previous BlockShuffle operator
           one by one until the buffer is filled;
      shuffle tuples in the buffer;
   else:
      return buffer[offset++];
      
void ExecReScan():
   clear the buffer and I/O offsets;
   invoke BlockShuffle.ExecReScan();

/************************
 * BlockShuffle operator
************************/
void ExecInit(): 
   compute the total block number;
   shuffle the block indexes;
   
Tuple ExecBlockShuffle():
   for each shuffled block i:
      for each page j in the block i:
         read a tuple k from the page j;
         return tuple k;
            
void ExecReScan():
   re-shuffle the block indexes;
   clear I/O offsets;

```

In our current implemention, the source code of the three operators are available at 
[SGD operator with SGD computation](src/backend/executor/nodeLimit.c),
[TupleShuffle operator](src/backend/executor/nodeSort.c) with its [TupleShuffle implementation](src/backend/utils/sort/tuplesort.c), and
[BlockShuffle operator](src/backend/executor/nodeSeqscan.c) with its [BlockShuffle implementation](src/backend/access/heap/heapam.c).



# CorgiPile Performance


## End-to-end exeuction time
The end-to-end execution time of SGD with different data shuffle strategies in PostgreSQL, for clustered datasets on HDD and SSD.

![Performance](corgipile-docs/performance/end_to_end_bismarck_madlib_ours.png)


## Convergence rate
![Convergence](corgipile-docs/performance/convergence-rate-all-datasets.png)


## Per-epoch exeuction time
![per-epoch-time](corgipile-docs/performance/per-iter-perf-on-clustered-data.png)
