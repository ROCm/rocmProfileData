#!/bin/bash
################################################################################
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################
OUTPUT_FILE=$(rlog-config get rpd_tracer:filename 2>/dev/null || echo "trace.rpd")
RANK=""
MASTER=""
EXIT_DELAY=30
LOAD_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    -o)          OUTPUT_FILE="$2"; shift 2 ;;
    --rank)      RANK="$2"; shift 2 ;;
    --master)    MASTER="$2"; shift 2 ;;
    --exit-delay) EXIT_DELAY="$2"; shift 2 ;;
    --load)      LOAD_ONLY=1; shift ;;
    *)           break ;;
  esac
done

rm -f ${OUTPUT_FILE}
export RPDT_FILENAME=${OUTPUT_FILE}

if [ "$LOAD_ONLY" = "1" ]; then
  export RPDT_AUTOSTART=0
  export RPDT_DELAYINIT=1
fi

if [ -n "$RANK" ]; then
  export RPDT_CLOCKSYNC_RANK=${RANK}
  if [ -n "$MASTER" ]; then
    export RPDT_CLOCKSYNC_MASTER=${MASTER}
  fi
  DELAY_ARG=""
  if [ "$RANK" = "0" ]; then
    DELAY_ARG="--exit-delay ${EXIT_DELAY}"
  fi
  rpdrun ${DELAY_ARG} "$@"
else
  LD_PRELOAD=librpd_tracer.so "$@"
fi
