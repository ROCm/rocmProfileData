#!/bin/bash

OUTPUT_FILE="trace.rpd"

if [ "$1" = "-o" ] ; then
  OUTPUT_FILE=$2
  shift
  shift
fi
 
if [ -e ${OUTPUT_FILE} ] ; then
  rm ${OUTPUT_FILE}
fi

python3 -m rocpd.schema --create ${OUTPUT_FILE}
if [ $? != 0 ] ; then
  echo "Error: Could not create rpd file. Please run 'python setup.py install' from the rocpd_python dir"
  exit
fi

export RPDT_FILENAME=${OUTPUT_FILE}
LD_PRELOAD=./rpd_tracer.so "$@"
