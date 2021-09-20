# In-DB ML Datasets

The datasets used in our experiments:

| Name | Type | \#Tuples | \#Features | Size in DB |
| :------------- |:-------------:| :-----:|:-----:|:-----:|
| higgs | dense | 10.0/1.0M | 28 | 2.8 GB |
| susy | dense | 4.5/0.5M | 18 | 0.9 GB |
| epsilon | dense | 0.4/0.1M | 2,000 | 6.3 GB  |
| criteo | sparse | 92/6.0M | 1,000,000 | 50 GB |
| yfcc | dense | 3.3/0.3M | 4,096 | 55 GB  |
|  | |  | | |

The first three datasets can be downloaded from [LIBSVM dataset website](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). The criteo and yfcc datasets can be downloaded using the following guide.

## 1. criteo_clustered dataset

| Type   | Name | Size | Size after decompression | \#Tuples |
| :------------- |:-------------| :-----:|:-----:|:-----:|
| Train (negative examples) | [criteo_train_clustered.sql_-1.tar.gz](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/criteo_datasets/criteo_train_clustered.sql_-1.tar.gz) | 5.7 GB | 38 GB | 68 M |
| Train (postive exmaples) | [criteo_train_clustered.sql_1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/criteo_datasets/criteo_train_clustered.sql_1.bz2) |  1.2 GB | 13 GB  | 24 M |
| Test (negative + postive) | [criteo_test.sql.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/criteo_datasets/criteo_test.sql.bz2) | 301.1MB | 3.4 GB | 6 M |
|   | |  | | |


Training dataset includes two files:

* `criteo_train_clustered.sql_-1.tar.gz`. This file represents the negative tuples with "-1" labels, which is about 5.7 GB and will be 38 GB after decompression.

* `criteo_train_clustered.sql_1.bz2`. This file represents the positive tuples with "+1" labels, which about 1.2 GB and will be 13 GB after decompression.

Testing dataset includes only one file:

* `criteo_test.sql.bz2`. This file includes both the negative and positve tuples for testing, which is about 300 MB and will be 3.4 GB after decompression.


After decompression, each tuple has `{feature_k[], feature_v[]} label` schema as follows, where `feature_k[]` represents which feature dimensions have non-zero values, `feature_v[]` denotes the corresponding non-zero values, and `label` is -1 or 1.

```
{48838,78061,...,989080}	{0.16013,0.16013,...,0.16013}	-1
{54892,98079,...,837590}	{0.16013,0.16013,...,0.16013}	-1
```

There are several steps to import these data files into DB:

1. Decompress these dataset files

```
tar zxvf criteo_train_clustered.sql_-1.tar.gz
bunzip2 criteo_train_clustered.sql_1.bz2
bunzip2 criteo_test.sql.bz2

```

2. Create a table named as `criteo_clustered` in DB (like PostgreSQL)

```SQL
e.g., In PostgreSQL:

DROP TABLE IF EXISTS criteo_clustered CASCADE;

CREATE TABLE criteo_clustered (
	        did serial,
	        k integer[],
	        v double precision[],
	        label integer
        );

```

3. Import the negative tuples into DB

```SQL
// The following command be executed in DB, 
// because the first line of criteo_train_clustered.sql_-1 is a COPY command,
// as "COPY criteo_clustered (k, v, label) FROM STDIN;";

postgres/bin/psql -d YOUR_DB_NAME -f /datadisk/data/criteo/criteo_train_clustered.sql_-1
```

4. Import the positive tuples into DB

```SQL
postgres/bin/psql -d YOUR_DB_NAME -f /datadisk/data/criteo/criteo_train_clustered.sql_1

```

5. Check the table size

```SQL
mldb= select pg_size_pretty(pg_table_size('criteo_clustered'));
 pg_size_pretty 
----------------
 50 GB
(1 row)

```


6. Generate criteo_shuffled datasets

If you would like to use the shuffled criteo dataset, you can create a table named `criteo_shuffled` using the `order by random()` command as follows. 

```SQL
// e.g., in PostgreSQL

DROP TABLE IF EXISTS criteo_shuffled CASCADE;

CREATE TABLE criteo_shuffled AS 
SELECT * FROM criteo_clustered ORDER BY random();
```

## 2. yfcc_clustered dataset

| Type   | Name | Size | Size after decompression | \#Tuples |
| :------------- |:-------------| :-----:|:-----:|:-----:|
| Train (negative examples) (part 0) | [yfcc_train_0_clustered.sql_-1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_0_clustered.sql_-1.bz2) | 2.5 GB | 9 GB | 0.3 M |
| Train (postive examples) (part 0)| [yfcc_train_0_clustered.sql_1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_0_clustered.sql_1.bz2) | 1.8 GB | 6 GB | 0.2 M |
| Train (negative examples) (part 1) | [yfcc_train_1_clustered.sql_-1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_1_clustered.sql_-1.bz2) | 2.5 GB | 9 GB | 0.3 M |
| Train (postive examples) (part 1)| [yfcc_train_1_clustered.sql_1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_1_clustered.sql_1.bz2) | 1.8 GB | 6 GB | 0.2 M |
| Train (negative examples) (part 2)| [yfcc_train_2_clustered.sql_-1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_2_clustered.sql_-1.bz2) | 2.5 GB | 9 GB | 0.3 M |
| Train (postive examples) (part 2)| [yfcc_train_2_clustered.sql_1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_2_clustered.sql_1.bz2) | 1.8 GB | 6 GB | 0.2 M |
| Train (negative examples) (part 3)| [yfcc_train_3_clustered.sql_-1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_3_clustered.sql_-1.bz2) | 2.5 GB | 9 GB | 0.3 M |
| Train (postive examples) (part 3)| [yfcc_train_3_clustered.sql_1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_3_clustered.sql_1.bz2) | 1.8 GB | 6 GB | 0.2 M |
| Train (negative examples) (part 4)| [yfcc_train_4_clustered.sql_-1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_4_clustered.sql_-1.bz2) | 2.5 GB | 9 GB | 0.3 M |
| Train (postive examples) (part 4)| [yfcc_train_4_clustered.sql_1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_4_clustered.sql_1.bz2) | 1.8 GB | 6 GB | 0.2 M |
| Train/Test (negative examples) (part 5)| [yfcc_train_5_clustered.sql_-1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_5_clustered.sql_-1.bz2) | 2.5 GB | 9 GB | 0.3 M |
| Train/Test (postive examples) (part 5)| [yfcc_train_5_clustered.sql_1.bz2](https://in-db-ml-datasets.s3.eu-central-1.amazonaws.com/yfcc100m_datasets/yfcc_train_5_clustered.sql_1.bz2) | 1.8 GB | 6 GB | 0.2 M |
|  |||||


The dataset consists of 12 files as follows:

* `yfcc_train_ID_clustered.sql_-1.bz2`: this file represents the negative tuples (with "-1" labels). Each one is about 2.5 GB and will be 9 GB after decompression.

* `yfcc_train_ID_clustered.sql_1.bz2`: this file represents the positive tuples (with "+1" labels). Each one is about 1.8 GB and will be 6 GB after decompression.

Here, ID is in \[0-5\].

After decompression, each tuple has 4,096 features in `{}` and a label (+1/-1) at the end of each line as follows:

```
{-23.613,3.101,-28.141,7.419,-2.928,...,-27.605,-8.006}	1
{-4.415,5.488,-0.011,-4.227,-6.708,...,-7.041,13.451}	-1
```

There are several steps to import these data files into DB:

1. Decompress these dataset files
```
bunzip2 FileName.bz2

```

2. Create a table named as `yfcc_clustered` in DB (like PostgreSQL)

```SQL
// In PostgreSQL:

DROP TABLE IF EXISTS yfcc_clustered CASCADE;

        CREATE TABLE yfcc_clustered (
	        did serial,
	        vec double precision[],
	        labeli integer
        );

```

3. Import the negative tuples into DB


```SQL
// The following command be executed in DB, 
// because the first line of yfcc_train_{0-5}_clustered.sql_-1 is a COPY command,
// as "COPY yfcc_clustered (vec, labeli) FROM STDIN";

postgres/bin/psql -d YOUR_DB_NAME -f /datadisk/data/yfcc/yfcc_train_{0-5}_clustered.sql_-1
```

You can also leave one of the data files (e.g., ID = 5) or some tuples in a data file, as the test dataset.


4. Import the positive tuples into DB
```SQL
postgres/bin/psql -d YOUR_DB_NAME -f /datadisk/data/yfcc/yfcc_train_{0-5}_clustered.sql_1

```
You can also leave one of the data files (e.g., ID = 5) or some tuples in a data file, as the test dataset.

5. Check the table size

```SQL
mldb= select pg_size_pretty(pg_table_size('yfcc_clustered'));
 pg_size_pretty 
----------------
 55 GB
(1 row)
```


6. Generate yfcc_shuffled datasets

If you would like to use the shuffled yfcc dataset, you can create a table named `yfcc_shuffled` using the `order by random()` command as follows.

```SQL
// e.g., in PostgreSQL

DROP TABLE IF EXISTS yfcc_shuffled CASCADE;

CREATE TABLE yfcc_shuffled AS 
SELECT * FROM yfcc_clustered ORDER BY random();
```