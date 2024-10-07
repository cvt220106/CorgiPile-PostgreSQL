set table_name = epsilon_train;
set model_name = LR;
set block_shuffle = 1;
set tuple_shuffle = 2;
set block_page_num = 256;
set buffer_tuple_num = 40000;
set iter_num = 20;
set learning_rate = 0.001;
set mu = 0;
set batch_size = 1;

-- train
select * from epsilon_train order by did limit 10;
