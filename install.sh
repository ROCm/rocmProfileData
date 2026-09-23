#!/bin/bash
apt-get install -y sqlite3 libsqlite3-dev libfmt-dev nlohmann-json3-dev xxd

cmake -B build -S .
cmake --build build -j"$(nproc)"
cmake --install build
