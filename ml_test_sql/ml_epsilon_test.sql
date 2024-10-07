-- test
set table_name = epsilon_test;
set run_test = true;
set block_shuffle = 0;
set tuple_shuffle = 0;
select * from epsilon_test order by did limit 10;
set run_test = false;