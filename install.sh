#!/bin/bash
apt-get install -y sqlite3 libsqlite3-dev libfmt-dev

git clone https://github.com/jeremy-rifkin/cpptrace.git
cd cpptrace
git checkout v0.7.5
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../cpptrace_install
make -j
make install
cd ../..

make; make install
