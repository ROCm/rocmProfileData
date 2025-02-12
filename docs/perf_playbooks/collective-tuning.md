## Performance contribution of collectives

### Preface
Collectives currently are serviced either through RCCL or - under the covers - by MSCCL. The queries for all collectives and to compute their total cost is:
```
sqlite> select * from top where Name like '%ccl%';
Name|TotalCalls|TotalDuration|Ave|Percentage
mscclKernel_Sum_half_Simple_false(ncclDevComm*, mscclAlgo*, mscclWork*)|52224|125822179|2409|18.8200893749884
ncclDevKernel_Generic(ncclDevComm*, channelMasks, ncclWork*)|26592|69872574|2627|10.4513218907667
mscclKernel_Sum_uint8_t_LL_false(ncclDevComm*, mscclAlgo*, mscclWork*)|8|3559|444|0.000532429084356098
sqlite> select sum(Percentage) from top where Name like '%ccl%';
sum(Percentage)
29.2719436948395
```

Note: Collective communications are inherent inter-device synchronization points. Hence, any variability in execution between devices will show up in collectives. To first order, this can be assessed by comparing execution times for individual calls across devices - if some devices take longer than others, the faster are waiting on the slower to get ready.

The decision to tune and to gauge what uplift can be expected requires 1) the collective operations, buffer sizes, and data types involved, 2) the roof line for a given operation and device configuration (e.g., 8 devices in one machine versus 2 machines with 4 devices each and a specific interconnect).

### Obtaining collective details from RCCL logs
The environment variables `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=COLL` enable logs of collectives (as of recently both NCCL and MSCCL serviced ones), `NCCL_DEBUG_FILE=` can be used to redirect this output to a file.

If a more detailed view of the collectives is necessary, `RCCL_KERNEL_COLL_TRACE_ENABLE=1` should additionally be set.

An example full log for an operation: 
```
banff-cyxtera-s78-4:52937:52937 [0] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f8bccf79600 recvbuff 0x7f8bccf79600 count 268435456 datatype 6 op 0 root 0 comm 0xaef8a00 [nranks=8] stream 0xac16b80 task 0 globalrank 0
banff-cyxtera-s78-4:52939:52939 [2] NCCL INFO AllReduce: opCount 2 sendbuff 0x7f2a8436e600 recvbuff 0x7f2a8436e600 count 1 datatype 8 op 0 root 0 comm 0xac3ea00 [nranks=8] stream 0xa78f810 task 0 globalrank 2
banff-cyxtera-s78-4:52944:53177 [0] NCCL INFO ## [170260.347287] [07:00] 000001 KL HWID 42300700 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId df000 nRanks 8
banff-cyxtera-s78-4:52942:52942 [5] NCCL INFO AllReduce: opCount 1 sendbuff 0x7f7e139fe000 recvbuff 0x7f7e139fe000 count 9 datatype 8 op 0 root 0 comm 0xb7f1460 [nranks=8] stream 0xba10e60 task 0 globalrank 5
banff-cyxtera-s78-4:52943:53178 [0] NCCL INFO ## [170260.348516] [06:00] 000001 KE busId bf000 nRanks 8
banff-cyxtera-s78-4:52943:53178 [0] NCCL INFO ## [170260.348587] [06:00] 000002 KL HWID 42300330 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId bf000 nRanks 8
banff-cyxtera-s78-4:52939:53175 [0] NCCL INFO ## [170260.348133] [02:00] 000001 KL HWID 42300630 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 38000 nRanks 8
banff-cyxtera-s78-4:52939:53175 [0] NCCL INFO ## [170260.348966] [02:00] 000001 KE busId 38000 nRanks 8
banff-cyxtera-s78-4:52939:53175 [0] NCCL INFO ## [170260.349068] [02:00] 000002 KL HWID 42300500 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 38000 nRanks 8
banff-cyxtera-s78-4:52942:52942 [5] NCCL INFO AllReduce: opCount 2 sendbuff 0x7f7e139fec00 recvbuff 0x7f7e139fec00 count 1 datatype 8 op 0 root 0 comm 0xb7f1460 [nranks=8] stream 0xba10e60 task 0 globalrank 5
banff-cyxtera-s78-4:52940:53171 [0] NCCL INFO ## [170260.348309] [03:00] 000001 KE busId 5c000 nRanks 8
banff-cyxtera-s78-4:52940:53171 [0] NCCL INFO ## [170260.348414] [03:00] 000002 KL HWID 42300010 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 5c000 nRanks 8
banff-cyxtera-s78-4:52940:53171 [0] NCCL INFO ## [170260.348527] [03:00] 000002 KE busId 5c000 nRanks 8
banff-cyxtera-s78-4:52942:53172 [0] NCCL INFO ## [170260.348483] [05:00] 000001 KL HWID 42300330 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId af000 nRanks 8
banff-cyxtera-s78-4:52942:53172 [0] NCCL INFO ## [170260.348504] [05:00] 000001 KE busId af000 nRanks 8
banff-cyxtera-s78-4:52941:53173 [0] NCCL INFO ## [170260.348640] [04:00] 000001 KE busId 9f000 nRanks 8
banff-cyxtera-s78-4:52942:53172 [0] NCCL INFO ## [170260.348703] [05:00] 000002 KL HWID 42300620 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId af000 nRanks 8
banff-cyxtera-s78-4:52942:53172 [0] NCCL INFO ## [170260.348722] [05:00] 000002 KE busId af000 nRanks 8
banff-cyxtera-s78-4:52941:53173 [0] NCCL INFO ## [170260.348739] [04:00] 000002 KL HWID 42300320 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 9f000 nRanks 8
banff-cyxtera-s78-4:52941:53173 [0] NCCL INFO ## [170260.348858] [04:00] 000002 KE busId 9f000 nRanks 8
banff-cyxtera-s78-4:52937:53174 [0] NCCL INFO ## [170260.349051] [00:00] 000001 KE busId c000 nRanks 8
banff-cyxtera-s78-4:52937:53174 [0] NCCL INFO ## [170260.349247] [00:00] 000002 KL HWID 42300520 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId c000 nRanks 8
banff-cyxtera-s78-4:52937:53174 [0] NCCL INFO ## [170260.349269] [00:00] 000002 KE busId c000 nRanks 8
```
It contains the operation type `AllReduce`, the number of elements `268435456` of a datatype `6` as well as the kernel choice `ncclDevFunc_AllReduce_RING_LL_Sum_double` - indicating the use of a low-latency ring protocol for the collective.

Subsequently, [rccl replayer](https://github.com/ROCm/rccl/blob/master/tools/rccl_replayer/README.md) can be used with the generated log file to both sanitize collectives as well as replay them and collect standalone performance data.

### Benchmarking current performance
Executing [rccl replayer](https://github.com/ROCm/rccl/blob/master/tools/rccl_replayer/README.md) on the previously generated log file will result in a log file and a `csv` table of performance:
```
...
Running Collective Call 581 of 49480
OpCount: 3
  - Rank 00: comm 1
  - Task 00:                        AllReduce inPlace=1 count=268435456 datatype=6 op=0 root=0
  - Rank 01: comm 1
  - Task 00:                        AllReduce inPlace=1 count=268435456 datatype=6 op=0 root=0
  - Rank 02: comm 1
  - Task 00:                        AllReduce inPlace=1 count=268435456 datatype=6 op=0 root=0
  - Rank 03: comm 1
  - Task 00:                        AllReduce inPlace=1 count=268435456 datatype=6 op=0 root=0
  - Rank 04: comm 1
  - Task 00:                        AllReduce inPlace=1 count=268435456 datatype=6 op=0 root=0
  - Rank 05: comm 1
  - Task 00:                        AllReduce inPlace=1 count=268435456 datatype=6 op=0 root=0
  - Rank 06: comm 1
  - Task 00:                        AllReduce inPlace=1 count=268435456 datatype=6 op=0 root=0
  - Rank 07: comm 1
  - Task 00:                        AllReduce inPlace=1 count=268435456 datatype=6 op=0 root=0
banff-cyxtera-s78-4:54466:54521 [0] NCCL INFO ## [171051.121743] [01:00] 000001 KL HWID 42306740 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 22000 nRanks 8
banff-cyxtera-s78-4:54466:54521 [0] NCCL INFO ## [171051.121768] [01:00] 000001 KE busId 22000 nRanks 8
banff-cyxtera-s78-4:54466:54521 [0] NCCL INFO ## [171051.122112] [01:00] 000002 KL HWID 42306760 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 22000 nRanks 8
banff-cyxtera-s78-4:54466:54521 [0] NCCL INFO ## [171051.122131] [01:00] 000002 KE busId 22000 nRanks 8
banff-cyxtera-s78-4:54466:54522 [0] NCCL INFO ## [171051.122293] [07:00] 000002 KL HWID 42302850 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId df000 nRanks 8
banff-cyxtera-s78-4:54466:54522 [0] NCCL INFO ## [171051.122339] [07:00] 000002 KE busId df000 nRanks 8
banff-cyxtera-s78-4:54466:54520 [0] NCCL INFO ## [171051.122848] [00:00] 000002 KL HWID 42302270 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId c000 nRanks 8 
banff-cyxtera-s78-4:54466:54520 [0] NCCL INFO ## [171051.122863] [00:00] 000002 KE busId c000 nRanks 8
banff-cyxtera-s78-4:54466:54524 [0] NCCL INFO ## [171051.122418] [04:00] 000002 KL HWID 42306570 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 9f000 nRanks 8
banff-cyxtera-s78-4:54466:54524 [0] NCCL INFO ## [171051.122451] [04:00] 000002 KE busId 9f000 nRanks 8
banff-cyxtera-s78-4:54466:54526 [0] NCCL INFO ## [171051.122093] [03:00] 000002 KL HWID 42302160 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 5c000 nRanks 8
banff-cyxtera-s78-4:54466:54526 [0] NCCL INFO ## [171051.122121] [03:00] 000002 KE busId 5c000 nRanks 8
banff-cyxtera-s78-4:54466:54527 [0] NCCL INFO ## [171051.122755] [02:00] 000002 KL HWID 42306770 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId 38000 nRanks 8
banff-cyxtera-s78-4:54466:54523 [0] NCCL INFO ## [171051.122278] [05:00] 000002 KL HWID 42306240 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId af000 nRanks 8
banff-cyxtera-s78-4:54466:54527 [0] NCCL INFO ## [171051.122778] [02:00] 000002 KE busId 38000 nRanks 8
banff-cyxtera-s78-4:54466:54525 [0] NCCL INFO ## [171051.122286] [06:00] 000002 KL HWID 42306260 ncclDevFunc_AllReduce_RING_LL_Sum_double nw 4 bi 0 nc 0 busId bf000 nRanks 8
banff-cyxtera-s78-4:54466:54525 [0] NCCL INFO ## [171051.122328] [06:00] 000002 KE busId bf000 nRanks 8
banff-cyxtera-s78-4:54466:54523 [0] NCCL INFO ## [171051.122316] [05:00] 000002 KE busId af000 nRanks 8
banff-cyxtera-s78-4:54466:54466 [7] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f96eb600000 recvbuff 0x7f96eb600000 count 268435456 datatype 6 op 0 root 0 comm 0x4fd1e80 [nranks=8] stream 0x4bdded0 task 0 globalrank 0
banff-cyxtera-s78-4:54466:54466 [7] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f96cb400000 recvbuff 0x7f96cb400000 count 268435456 datatype 6 op 0 root 0 comm 0x4c3ea60 [nranks=8] stream 0x4930d30 task 0 globalrank 1
banff-cyxtera-s78-4:54466:54466 [7] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f96ab200000 recvbuff 0x7f96ab200000 count 268435456 datatype 6 op 0 root 0 comm 0x4c54e80 [nranks=8] stream 0x4b107f0 task 0 globalrank 2
banff-cyxtera-s78-4:54466:54466 [7] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f968b000000 recvbuff 0x7f968b000000 count 268435456 datatype 6 op 0 root 0 comm 0x4b7a830 [nranks=8] stream 0x49d49f0 task 0 globalrank 3
banff-cyxtera-s78-4:54466:54466 [7] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f966ae00000 recvbuff 0x7f966ae00000 count 268435456 datatype 6 op 0 root 0 comm 0x4b63190 [nranks=8] stream 0x4785e50 task 0 globalrank 4
banff-cyxtera-s78-4:54466:54466 [7] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f964ac00000 recvbuff 0x7f964ac00000 count 268435456 datatype 6 op 0 root 0 comm 0x48b2b70 [nranks=8] stream 0x49877d0 task 0 globalrank 5
banff-cyxtera-s78-4:54466:54466 [7] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f962aa00000 recvbuff 0x7f962aa00000 count 268435456 datatype 6 op 0 root 0 comm 0x4a571c0 [nranks=8] stream 0x4684cc0 task 0 globalrank 6
banff-cyxtera-s78-4:54466:54466 [7] NCCL INFO AllReduce: opCount 3 sendbuff 0x7f960a800000 recvbuff 0x7f960a800000 count 268435456 datatype 6 op 0 root 0 comm 0x50534c0 [nranks=8] stream 0x4b77e60 task 0 globalrank 7
...
```
and
```
callNumber, functionName, inPlace, count(numElements), datatype, op, root, time(msec), groupCallBusBandwidth(GB/s)
...
3, AllReduce, 1, 268435456, Float16, Sum, 0, 3.25299, 330.079
...
6, AllReduce, 1, 268435456, Float16, Sum, 0, 3.22935, 332.494
...
9, AllReduce, 1, 268435456, Float16, Sum, 0, 3.20286, 335.244
...
```
Note that this replays the exact order of operations and hence identical configurations needs to be averaged over to obtain the current performance - this is possible by importing the csv into a spreadsheet.

