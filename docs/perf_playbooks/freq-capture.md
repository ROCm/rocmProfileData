## Frequency capture and analysis during workload execution

### Preface
Analyzing the correlation of chip frequency to workload execution aides understanding of performance patterns in the workload. For this, an accurate sampling of the chip frequency and correlating it to kernel execution is required.

Currently, `rpd` polls `smi` at a 1 ms interval in the background for the chip frequency during workload execution. However, internal smi update frequency is lower than that so rpd is oversampling which minimizes edge effects. However, this limits the possible accuracy and fidelity of the overall analysis.


### Visualization
Careful visual inspection of the frequency and correlation of any changes to workload behavior is a valuable first step to identify any patterns.

For this, the `rpd` database can be converted to a `json` trace file to be visualized with `chrome://tracing` or `perfetto`:
```
python3 /path/to/rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json
``` 
The resulting visualization will contain a lane for GPU frequency in MHz.

Note that output files may easily exceed `chrome://tracing` or `perfetto` limits if not pruned before (or tracing was limited through manual use of starting/stopping the profiler). [Perfetto documentation](https://perfetto.dev/docs/visualization/large-traces) states the limit to be a OS, browser, architecture imposed limit around 2 GB which is related to but not exactly the file size. In practice, `json` sizes around 500 MB have worked in the past.

### rpd sqlite level
The `rocpd_monitor` table captures the monitoring values per device:
```
sqlite> select * from rocpd_monitor m;
id|deviceType|deviceId|monitorType|start|end|value
1|gpu|0|sclk|78788649569590|78788652595552|158
2|gpu|2|sclk|78788649757770|78788652729255|158
3|gpu|6|sclk|78788649994434|78788653016688|158
...
```
`deviceId` is the GPU ID, `value` is the clock frequency in MHz.

To estimate how long a device spends above a certain frequency threshold, first all recorded polls need to be counted:
```
sqlite> select count(*) from rocpd_monitor m where m.deviceId=7;
count(*)
55674
```
Subsequently, count the polls above the frequency threshold:
```
sqlite> select count(*) from rocpd_monitor m where CAST(m.value AS INTEGER)>=500 and m.deviceId=7;
count(*)
46893
sqlite> select count(*) from rocpd_monitor m where CAST(m.value AS INTEGER)>=1000 and m.deviceId=7;
46375
```
In this example, 46893 of 55674 polls on device 7 are above 500 MHz, or 84%, and 46375 of 55674 are above 1 GHz, or 83%. This indicates a mostly busy device. If a mostly idling device is encountered,it is most often caused by not starting and stopping data collection appropriately.

Alternatively, an appropriate window can be chosen after the fact either through the `start` and `end` time stamps or the running `id`:
```
sqlite> select count(*) from rocpd_monitor m where m.deviceId=7 and m.id>=200000 and m.id<=300000;
count(*)
12469
sqlite> select count(*) from rocpd_monitor m where m.deviceId=7 and m.id>=200000 and m.id<=300000 and CAST(m.value AS INTEGER)>=500;
count(*)
10525
```
This approach is particularly useful if a time window/area of interest has been identified through visualization.

Another possible analysis is to understand when frequency during workload execution drops to identify either thermal events or idling GPU.
```
sqlite> select m.id, m.start, m.value from rocpd_monitor m where m.deviceId=7 and m.id>=200000 and m.id<=300000 and CAST(m.value AS INTEGER)<=200;
id|start|value
215093|78917617925555|196
215101|78917625887555|192
215109|78917633930968|186
215117|78917641951000|182
215125|78917649971261|178
215133|78917657960611|175
215141|78917666077013|179
216474|78787410002905|137
216485|78787413472766|136
...

sqlite> select count(*) from rocpd_monitor m where m.deviceId=7 and m.id>=200000 and m.id<=300000 and CAST(m.value AS INTEGER)<=200;
count(*)
1566
```

Lastly, understanding what kernel executed under dropped frequencies is interesting. Assuming a time window with dropped frequencies between `78815798085930` and `78817982213660` was located
```
sqlite> select o.gpuId, o.queueId, o.start, o.end, s1.string, s2.string from rocpd_op o left join rocpd_string s1 on o.description_id = s1.id left join rocpd_string s2 on o.opType_id = s2.id where (o.start>=78815798085930 and o.start<=78817982213660)  limit 10;
gpuId|queueId|start|end|string|string
5|0|78817959684512|78817959795207|Cijk_Alik_Bljk_HHS_BH_Bias_AS_SAV_UserArgs_MT256x128x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA4_GRVWB4_GSUAMB_IU1_K1_LBSPPA512_LBSPPB128_LBSPPM0_LPA4_LPB4_LPM0_LRVW4_LWPMn1_MIAV0_MIWT8_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_TLDS1_USFGROn1_VWA8_VWB1_WSGRA0_WSGRB0_WG32_8_1_WGMXCC1|KernelExecution
5|0|78817959840912|78817959874991|void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::Half> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::Half> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::Half> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::Half> const&)::{lambda(int)#1})|KernelExecution
5|0|78817960126571|78817960170192|void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1})|KernelExecution
5|0|78817960235141|78817960339472|Cijk_Alik_Bljk_HHS_BH_Bias_AS_SAV_UserArgs_MT256x128x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA4_GRVWB4_GSUAMB_IU1_K1_LBSPPA512_LBSPPB128_LBSPPM0_LPA4_LPB4_LPM0_LRVW4_LWPMn1_MIAV0_MIWT8_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_TLDS1_USFGROn1_VWA8_VWB1_WSGRA0_WSGRB0_WG32_8_1_WGMXCC1|KernelExecution
5|0|78817960339472|78817960370864|void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::Half> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::Half> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::Half> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<c10::Half> const&)::{lambda(int)#1})|KernelExecution
5|0|78817960370864|78817960398408|void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#10}::operator()() const::{lambda(c10::Half)#1} const&)::{lambda(int)#1})|KernelExecution
...

sqlite> select count(*) from rocpd_op o where (o.start>=78815798085930 and o.start<=78817982213660);
count(*)
140
```
140 kernels were executed inside this window, starting with a matrix multipliation, followed by elementwise kernels and other matrix multiplications.

### Summary
Both visual inspection as well as analysis on the `rpd` database can be used (ideally in conjunction) to assess GPU frequency (and drops thereof) during workload runtime.
