#!/bin/bash
apt-get install -y sqlite3 libsqlite3-dev libfmt-dev nlohmann-json3-dev xxd
apt-get install -y libzstd-dev

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cmake --install build
