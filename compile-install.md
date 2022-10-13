# The Compilation and installation guide for CorgiPile in PostgreSQL:

## 1. Install the original PostgreSQL and load necessary datasets into PostgreSQL

Due to some implementation issues that will be detailed later, I recommend that you first install the original official PostgreSQL (https://github.com/DS3Lab/CorgiPile-PostgreSQL/tree/original_9_2) with compilation configurations (https://github.com/DS3Lab/CorgiPile-PostgreSQL/blob/original_9_2/.vscode/tasks.json) in your machine. After that, you can setup the database with 'initdb' in this original PG and store the datasets into the DB directory (e.g., in /datadisk/data/pg9data directory).

For example, for the susy dataset (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.xz), you can first transform each row of susy dataset to `{features} label` format as follows. Then, you can create a dataset named `susy` in DB and upload the transformed tuples into it. If you would like to use the *clustered* datasets, you may perform additional 
```sql
create table susy_clustered as SELECT * FROM susy ORDER BY labeli
```
once you have uploaded the susy table into DB. Alternatively, you can cluster the susy dataset beforehand, and upload the clustered dataset as follows.
```sql
 
        CREATE TABLE susy_clustered (
                   did serial,
                   vec double precision[],
                   labeli integer
        );

 
COPY susy_clustered (vec, labeli) FROM STDIN;
{1.6679730415344238,0.06419061869382858,-1.225171446800232,0.5061022043228149,-0.33893898129463196,1.6725428104400635,3.475464344024658,-1.219136357307434,0.012954562902450562,3.7751736640930176,1.0459771156311035,0.568051278591156,0.48192843794822693,0.0,0.4484102725982666,0.20535576343536377,1.3218934535980225,0.3775840103626251} 1
 
{0.4448399245738983,-0.13429801166057587,-0.7099716067314148,0.45171892642974854,-1.6138712167739868,-0.7686609029769897,1.219918131828308,0.5040258169174194,1.8312475681304932,-0.4313853085041046,0.5262832045555115,0.9415140151977539,1.58753502368927,2.024308204650879,0.6034975647926331,1.5623739957809448,1.1354544162750244,0.18091000616550446} 1

... more tuples ...
```

Now, the datasets have been stored in the PG DB tables (e.g., in /datadisk/data/pg9data directory) and you should stop the original PG processes.

## 2. Install CorgiPile and perform CorgiPile on the data tables.

Compile and install CorgiPile (https://github.com/DS3Lab/CorgiPile-PostgreSQL/tree/corgipile) with configurations (https://github.com/DS3Lab/CorgiPile-PostgreSQL/blob/corgipile/.vscode/tasks.json) into a directly such as `/home/lijie/postgres-sgd`.

Then, start PG-CorgiPile as follows.
```shell
PG="postgres-sgd"
/home/lijie/$PG/bin/pg_ctl -D /datadisk/data/pg9data -l /datadisk/data/pg9data/logs/server.log start
```
Then, access your own database where the datasets are stored (here we have stored the datasets into the `mldb`) using psql:
```shell
/home/lijie/$PG/bin/psql mldb
```
After that, you can set some necessary hyperparameters and execute CorgiPile training in your psql as follows:
Here, `block_shuffle = 1` denotes using block shuffle, and `tuple_shuffle = 2` means shuffling tuples with two threads (two buffers).
You can refer to https://github.com/DS3Lab/CorgiPile-PostgreSQL/blob/corgipile/src/backend/executor/nodeLimit.c for more details about the parameters as follows.

```
shuffle_mode:
	1. Block-Only Shuffle (block-shuffle = 0, tuple-shuffle = 0, i.e., 0 buffer, 1 thread)
	2. CorgiPile (block-shuffle = 1, tuple-shuffle = 2, i.e., 2 buffers for tuple shuffle, 2 threads)
	3. CorgiPile-Single-Thread (block-shuffle = 1, tuple-shuffle = 1, i.e., 1 buffer, 1 thread)
	4. No Shuffle (block-shuffle = 0, tuple-shuffle = 0, i.e, 0 buffer, 1 thread)
```


For example, for training a LR model on susy_clustered dataset, you may perform the following `set` and `select` queries in PostgreSQL as follows.

```sql
/* Step 1: Set the model name and hyperparameters. */
set table_name = susy_clustered;
set model_name = LR;
set block_shuffle = 1;
set tuple_shuffle = 2;
set block_page_num = 1280;
set buffer_tuple_num = 450000;
set iter_num = 10;
set learning_rate = 0.001;
set mu = 0;
set batch_size = 1;
\timing
 
/* Step 2: Train LR model on susy dataset (4.5 millions of tuples) with 10 epochs. */
/* Note that our CorgiPile do not need to shuffle the dataset table before training. */
 
 
\timing
 
select * from susy_clustered order by did limit 10;
 
 
/* Our CorgiPile also supports mini-batch SGD. */
/* Now, let us run mini-batch SGD in CorgiPile. */

set batch_size = 128;
set learning_rate = 0.1;
select * from susy_clustered order by did limit 10;
```
 
 
 

Why do we use `SELECT * FROM table ORDER BY did Limit 10;` to train the model?


For simplicity, we now directly implement our *BlockShuffle*, *TupleShuffle*, and *SGD* operators by modifying the *Scan*, *Sort*, and *Limit* operators in PostgreSQL. Thus, we can leverage `SELECT * FROM table ORDER BY did Limit 10` query plan to invoke `BlockShuffle ->TupleShuffle -> SGD operators` to mimic the query of `SELECT * FROM table TRAIN BY model WITH params` as described in our paper.

The drawbacks are that this modified version only supports CorgiPile and some original SQL queries may suffer from unexpected results or errors. However, it is a just engineering problem to add *BlockShuffle*, *TupleShuffle*, and *SGD* operators as independent operators into the query plan of `SELECT * FROM table TRAIN BY model WITH params`, instead of modifying the *Scan*, *Sort*, and *Limit* operators. Actually, openGauss DB adopts our approach, implements independent operators for the `CREATE MODEL` query as described in https://opengauss.org/en/docs/2.1.0/docs/Developerguide/db4ai-query-for-model-training-and-prediction.html. If you would like to use openGauss-CorgiPile, you may refer to https://gitee.com/opengauss-db4ai/openGauss-server/tree/3.0.0/ or contact us.