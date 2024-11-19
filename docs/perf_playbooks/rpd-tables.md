## rpd tables and stored data

### Preface
rpd databases are sqlite3 compatible. Hence, sql queries are a powerful tool to extract metrics of performance interest. Below follows documentation of a subset of tables/schema of particular relevance.

### Schema
The schema of the trace database can be queried from
```
sqlite> .schema
```
It documents tables and views present in the database.

### Tables
All tables in the database can be queried from
```
sqlite> .table
api                     rocpd_api               rocpd_monitor         
busy                    rocpd_api_ops           rocpd_op              
copy                    rocpd_barrierop         rocpd_op_inputSignals 
copyop                  rocpd_copyapi           rocpd_string          
kernel                  rocpd_kernelapi         top                   
ktop                    rocpd_kernelcodeobject
op                      rocpd_metadata 
```

In `rpd-101.md`, the `top` and `busy` tables were used for basic performance analysis, with `top` containing kernel-level percentages of GPU time, call counts, total time (in us), and average time per call (in us).
```
sqlite> select * from top;
Name|TotalCalls|TotalDuration|Ave|Percentage
mscclKernel_Sum_half_Simple_false(ncclDevComm*, mscclAlgo*, mscclWork*)|52224|125822179|2409|18.8200893749884
ncclDevKernel_Generic(ncclDevComm*, channelMasks, ncclWork*)|26592|69872574|2627|10.4513218907667
...
```


`busy` contains the per-device ratio of aggregate GPU times and CPU time between starting and stopping the profiler - either explicitly or implicitly through `runTracer.sh`.
```
sqlite> select * from busy;
gpuId|GpuTime|WallTime|Busy
2|81111298846|126018228685|0.643647349216032
3|74045687063|126018228685|0.587579176724404
4|84577265111|126018228685|0.671151038969232
5|85360254842|126018228685|0.677364344291569
6|85621957104|126018228685|0.67944104593014
7|85638792053|126018228685|0.679574637309544
8|85452706841|126018228685|0.678097984178153
9|86744546744|126018228685|0.688349198756237
```
The `GpuTime` is aggregate kernel and copy operation time in us, the `WallTime` is the time between profiler start and stop in us, and the `Busy` column is the ratio of the two - ideally as close to 1.0 as possible assuming targeted profiler start/stop for a given `gpuId`.

`rocpd_op` contains raw data on host operation starts and ends:
```
sqlite> select * from rocpd_op;
id|gpuId|queueId|sequenceId|completionSignal|start|end|description_id|opType_id
...
...
4294968319|2|3|0||78794216831989|78794216837883|4294967297|4294967369
4294968320|2|3|0||78794216867812|78794216873705|4294967297|4294967369
...
```
`gpuID`, `queueId`, `sequenceId`, `completionSignal`, `start` (in us), `end` (in us) are all directly usable. `id`, `description_id`, `opType_id` point to entries in the `rocpd_string` table and can be joined:

```
sqlite> select s1.string, o.gpuId, o.queueId, o.sequenceId, o.completionSignal, o.start, o.end, s2.string, s3.string from rocpd_op o inner join rocpd_string s1 on s1.id = o.id inner join rocpd_string s2 on s2.id = o.description_id inner join rocpd_string s3 on s3.id = o.opType_id;
string|gpuId|queueId|sequenceId|completionSignal|start|end|string|string
...
ptr=0x7f93c1200000 | size=0x1f800000|2|3|0||78794216831989|78794216837883||CopyHostToDevice
ptr=0x7f9356a00000 | size=0x6a600000|2|3|0||78794216867812|78794216873705||CopyHostToDevice
...
```

Similarly, `rocpd_kernelapi` contains raw data on kernels:
```
sqlite> select * from rocpd_kernelapi;
api_ptr_id|stream|gridX|gridY|gridZ|workgroupX|workgroupY|workgroupZ|groupSegmentSize|privateSegmentSize|kernelArgAddress|aquireFence|releaseFence|codeObject_id|kernelName_id
...
4294999902|0x0|256|3|6272|64|1|1|0|0|||||4294967414
4294999903|0x0|26656|1|1|128|1|1|0|0|||||4294967395
...
```
Which can again be joined if readability is wanted:
```
sqlite> select s1.string, k.stream, k.gridX, k.gridY, k.gridZ, k.workgroupX, k.workgroupY, k.workgroupZ, k.groupSegmentSize, k.privateSegmentSize, k.kernelArgAddress, k.aquireFence, k.releaseFence, k.codeObject_id, s2.string from rocpd_kernelapi k left join rocpd_string s1 on k.api_ptr_id = s1.id left join rocpd_string s2 on k.kernelName_id = s2.id;
...
|0x0|17077|1|1|512|1|1|0|0|||||void at::native::elementwise_kernel<512, 1, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1})
|0x0|256|3|6272|64|1|1|0|0|||||Cijk_Ailk_Bljk_HHS_BH_Bias_AS_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA8_GRVWB4_GSUAMB_IU1_K1_LBSPPA128_LBSPPB128_LBSPPM0_LPA16_LPB4_LPM0_LRVW4_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_TLDS1_USFGROn1_VWA1_VWB1_WSGRA0_WSGRB0_WG16_4_1_WGMXCC1
|0x0|26656|1|1|128|1|1|0|0|||||void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1})
...

```

The connection between host side APIs and kernels is made in `rocpd_api_ops`:
```
sqlite> select * from rocpd_api_ops;
id|api_id|op_id
1|8589934632|8589934593
2|8589936088|8589934594
```

`rocpd_copyapi` contains copy operations:

```
sqlite> select * from rocpd_copyapi;
api_ptr_id|stream|size|width|height|kind|dst|src|dstDevice|srcDevice|sync|pinned
4294967336|0x0|4|||1|0xaf52200|0x7f9f61400000|0|0|0|0
4294968799|0x7f9f3915ad70|8|||4|0x7f9f45b85008|0x7f9f16202000|0|0|0|0
4294968802|0x7f9f3915ad70|8|||4|0x7f9f45b85008|0x7f9f16202008|0|0|0|0
...
```

`rocpd_monitor` contains runtime GPU monitoring data (currently limited to `sclk` in MHz):
```
sqlite> select * from rocpd_monitor;
id|deviceType|deviceId|monitorType|start|end|value
...
1023|gpu|7|sclk|78791004915863|78791013550355|144
1024|gpu|1|sclk|78791012513674|78791020430870|139
...
```

### Summary
rpd databases contain tables which logically sort profiling data collected at runtime. Direct performance analysis requires understanding these tables, connecting their data, and formulating appropriate SQL queries as needed.
