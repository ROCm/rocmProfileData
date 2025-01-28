#!/bin/bash
apt-get install -y sqlite3 libsqlite3-dev libfmt-dev
apt-get install -y libzstd-dev

make; make install
