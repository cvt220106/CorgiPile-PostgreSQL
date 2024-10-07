#!/usr/bin/zsh

echo "Begin the re-compile postgres!"

set -e
# Uninstall the current installation and clean the build
make uninstall
make clean

# ./configure --prefix=/home/jacky/Documents/postgres-sgd --enable-depend --enable-cassert --enable-debug --with-python CFLAGS='-O3 -Wall -lpthread'

./configure --prefix=/home/jacky/Documents/postgres-sgd --enable-depend --enable-cassert --enable-debug --enable-profiling --with-python CFLAGS='-ggdb -Og -g3 -Wall -lpthread -fno-omit-frame-pointer'

make -j8 all
make install

echo "The re-compile is done!"