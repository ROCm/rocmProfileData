## Stackframe analysis

### Preface
Stackframe analysis enables understanding what call chains ended in a given HIP API. To enable stackframe analysis, rpd must be compiled with support for this feature and the `cpptrace` submodule. Additionally, the environment variable `RPDT_STACKFRAMES=1` needs to be set when collecting the profile.

### Limitations
Stackframe analysis requires debug symbols to correctly resolve source files and line numbers. I.e., any relevant parts (the workload) should be compiled with debug symbols by setting `-g` for plain compilations, `RelWithDebInfo` as the cmake target, etc. If host libraries need to be resolved, installation of their debug packages is necessary. Without debug symbols present, only offsets into libraries/executables can be provided. E.g.,
```
in main_foo(int, char**) at /root/rocm-examples/Applications/bitonic_sort/main.hip:169:5
```
vs
```
0x00007f35a249d54b at /opt/rocm/lib/libamdhip64.so.6
```

### Basic usage with fully C++ code
After execution, the trace database will contain a `rocpd_stackframe` table:
```
sqlite> select * from rocpd_stackframe;
id|api_ptr_id|depth|name_id
1|2|0|5
2|2|1|6
3|2|2|7
4|2|3|8
5|2|4|9
6|2|5|10
7|3|0|12
8|3|1|13
9|3|2|14
10|3|3|15
11|3|4|16
12|4|0|18
13|4|1|19
14|4|2|14
15|4|3|15
16|4|4|16
17|5|0|18
18|5|1|20
19|5|2|14
20|5|3|15
```
Here, `id` is a running index, `api_ptr_id` maps to the HIP API call, `depth` is the stack depth with 0 as the highest frame (i.e., HIP API), and `name_id` maps to the frame string.

For easier readability, a temporary view can be created including the strings:
```
sqlite> create temporary view if not exists rocpd_stackstrings as select sf.id, s2.string as hip_api, s3.string as args, sf.depth, s1.string as frame from rocpd_stackframe sf join rocpd_string s1 on sf.name_id=s1.id join rocpd_api ap on sf.api_ptr_id=ap.id join rocpd_string s2 on ap.apiName_id=s2.id join rocpd_string s3 on ap.args_id=s3.id;
sqlite> select * from rocpd_stackstrings;
id|hip_api|args|depth|frame
1|hipMalloc|ptr=0x7f3491000000 | size=0x20000|0|#0 0x00007f35a249d54b in hipMalloc() at /opt/rocm/lib/libamdhip64.so.6
2|hipMalloc|ptr=0x7f3491000000 | size=0x20000|1|#1 (inlined)          in hipError_t hipMalloc<unsigned int>(unsigned int**, unsigned long) at /opt/rocm-6.3.0/lib/llvm/bin/../../../include/hip/hip_runtime_api.h:9360:12
3|hipMalloc|ptr=0x7f3491000000 | size=0x20000|2|#2 0x00000000002055ae in main_foo(int, char**) at /root/rocm-examples/Applications/bitonic_sort/main.hip:169:5
4|hipMalloc|ptr=0x7f3491000000 | size=0x20000|3|#3 0x00007f35a1d161c9 in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
5|hipMalloc|ptr=0x7f3491000000 | size=0x20000|4|#4 0x00007f35a1d1628a in __libc_start_main_impl at ./csu/../csu/libc-start.c:360:3
6|hipMalloc|ptr=0x7f3491000000 | size=0x20000|5|#5 0x0000000000204b04 at /root/rocm-examples/Applications/bitonic_sort/applications_bitonic_sort
7|hipMemcpy||0|#0 0x00007f35a24b4447 in hipMemcpy() at /opt/rocm/lib/libamdhip64.so.6
8|hipMemcpy||1|#1 0x00000000002055cf in main_foo(int, char**) at /root/rocm-examples/Applications/bitonic_sort/main.hip:170:5
9|hipMemcpy||2|#2 0x00007f35a1d161c9 in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
10|hipMemcpy||3|#3 0x00007f35a1d1628a in __libc_start_main_impl at ./csu/../csu/libc-start.c:360:3
...
```

Subsequently, it is easy to filter for what call chains ended up in a given API, e.g., to find what stacks called into `hipMemcpy`
```
sqlite> select * from rocpd_stackstrings where hip_api like 'hipMemcpy%';
id|hip_api|args|depth|frame
7|hipMemcpy||0|#0 0x00007f35a24b4447 in hipMemcpy() at /opt/rocm/lib/libamdhip64.so.6
8|hipMemcpy||1|#1 0x00000000002055cf in main_foo(int, char**) at /root/rocm-examples/Applications/bitonic_sort/main.hip:170:5
9|hipMemcpy||2|#2 0x00007f35a1d161c9 in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
10|hipMemcpy||3|#3 0x00007f35a1d1628a in __libc_start_main_impl at ./csu/../csu/libc-start.c:360:3
11|hipMemcpy||4|#4 0x0000000000204b04 at /root/rocm-examples/Applications/bitonic_sort/applications_bitonic_sort
1822|hipMemcpy||0|#0 0x00007f35a24b4447 in hipMemcpy() at /opt/rocm/lib/libamdhip64.so.6
1823|hipMemcpy||1|#1 0x0000000000205831 in main_foo(int, char**) at /root/rocm-examples/Applications/bitonic_sort/main.hip:217:5
1824|hipMemcpy||2|#2 0x00007f35a1d161c9 in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
1825|hipMemcpy||3|#3 0x00007f35a1d1628a in __libc_start_main_impl at ./csu/../csu/libc-start.c:360:3
1826|hipMemcpy||4|#4 0x0000000000204b04 at /root/rocm-examples/Applications/bitonic_sort/applications_bitonic_sort
```
In this instance, two stacks ended in `hipMemcpy` - one from `main.hip:170`, the other from `main.hip:217`.

More advanced usage modes filtering on, e.g., API arguments are possible. Similarily, more fields cna be exposed from the joined HIP API table and filtered over such as start times.

### Use with hybrid Python/C++ workloads
After creating the same temporary view as above and filtering for all `hipMalloc` invocations an exemplary output for a standalone Faiss benchmark would be:

```
...
65581120|hipMalloc|ptr=0x7b7534934000 | size=0x100|0|#0  0x00007b9a4550e1c in hipMalloc()1 at /opt/rocm/lib/libamdhip64.so.6
65581121|hipMalloc|ptr=0x7b7534934000 | size=0x100|1|#1  0x00007b98a97beaa9 in faiss::gpu::StandardGpuResourcesImpl::allocMemory(faiss::gpu::AllocRequest const&) at /root/faiss/faiss/gpu-rocm/StandardGpuResources.cpp:558:29
65581122|hipMalloc|ptr=0x7b7534934000 | size=0x100|2|#2  0x00007b98a97baed6 in faiss::gpu::GpuResources::allocMemoryHandle(faiss::gpu::AllocRequest const&) at /root/faiss/faiss/gpu-rocm/GpuResources.cpp:200:69
65581123|hipMalloc|ptr=0x7b7534934000 | size=0x100|3|#3  0x00007b98a97d7e74 in faiss::gpu::DeviceVector<unsigned char>::realloc_(unsigned long, ihipStream_t*) at /root/faiss/faiss/gpu-rocm/utils/DeviceVector.cuh:228:31
65581124|hipMalloc|ptr=0x7b7534934000 | size=0x100|4|#4  (inlined)          in faiss::gpu::DeviceVector<unsigned char>::reserve(unsigned long, ihipStream_t*) at /root/faiss/faiss/gpu-rocm/utils/DeviceVector.cuh:214:9
65581125|hipMalloc|ptr=0x7b7534934000 | size=0x100|5|#5  0x00007b98a97d271d in faiss::gpu::IVFBase::reserveMemory(long) at /root/faiss/faiss/gpu-rocm/impl/IVFBase.hip:88:20
65581126|hipMalloc|ptr=0x7b7534934000 | size=0x100|6|#6  0x00007b98a97b82e2 in faiss::gpu::GpuIndexIVFPQ::reserveMemory(unsigned long) at /root/faiss/faiss/gpu-rocm/GpuIndexIVFPQ.hip:224:17
65581127|hipMalloc|ptr=0x7b7534934000 | size=0x100|7|#7  0x00007b98a9785720 in faiss::gpu::ToGpuCloner::clone_Index(faiss::Index const*) at /root/faiss/faiss/gpu-rocm/GpuCloner.cpp:228:31
65581128|hipMalloc|ptr=0x7b7534934000 | size=0x100|8|#8  0x00007b98a97873cb in faiss::gpu::index_cpu_to_gpu_multiple(std::vector<faiss::gpu::GpuResourcesProvider*, std::allocator<faiss::gpu::GpuResourcesProvider*> >&, std::vector<int, std::allocator<int> >&, faiss::Index const*, faiss::gpu::GpuMultipleClonerOptions const*) at /root/faiss/faiss/gpu-rocm/GpuCloner.cpp:493:26
65581129|hipMalloc|ptr=0x7b7534934000 | size=0x100|9|#9  (inlined)          in _wrap_index_cpu_to_gpu_multiple__SWIG_0 at /root/faiss/build/faiss/python/CMakeFiles/swigfaiss.dir/swigfaissPYTHON_wrap.cxx:243527:69
65581130|hipMalloc|ptr=0x7b7534934000 | size=0x100|10|#10 0x00007b98bff3be13 in _wrap_index_cpu_to_gpu_multiple at /root/faiss/build/faiss/python/CMakeFiles/swigfaiss.dir/swigfaissPYTHON_wrap.cxx:244316:59
65581131|hipMalloc|ptr=0x7b7534934000 | size=0x100|11|#11 0x00005800df11f2a7 at /usr/bin/python3.10
65581132|hipMalloc|ptr=0x7b7534934000 | size=0x100|12|#12 0x00005800df115b4a at /usr/bin/python3.10
65581133|hipMalloc|ptr=0x7b7534934000 | size=0x100|13|#13 0x00005800df10f4c7 at /usr/bin/python3.10
65581134|hipMalloc|ptr=0x7b7534934000 | size=0x100|14|#14 0x00005800df11faeb at /usr/bin/python3.10
65581135|hipMalloc|ptr=0x7b7534934000 | size=0x100|15|#15 0x00005800df10eb79 at /usr/bin/python3.10
65581136|hipMalloc|ptr=0x7b7534934000 | size=0x100|16|#16 0x00005800df11faeb at /usr/bin/python3.10
65581137|hipMalloc|ptr=0x7b7534934000 | size=0x100|17|#17 0x00005800df1099a1 at /usr/bin/python3.10
65581138|hipMalloc|ptr=0x7b7534934000 | size=0x100|18|#18 0x00005800df11faeb at /usr/bin/python3.10
65581139|hipMalloc|ptr=0x7b7534934000 | size=0x100|19|#19 0x00005800df1099a1 at /usr/bin/python3.10
65581140|hipMalloc|ptr=0x7b7534934000 | size=0x100|20|#20 0x00005800df1eee55 at /usr/bin/python3.10
65581141|hipMalloc|ptr=0x7b7534934000 | size=0x100|21|#21 0x00005800df1eed25 at /usr/bin/python3.10
65581142|hipMalloc|ptr=0x7b7534934000 | size=0x100|22|#22 0x00005800df215ae7 at /usr/bin/python3.10
65581143|hipMalloc|ptr=0x7b7534934000 | size=0x100|23|#23 0x00005800df2102ee at /usr/bin/python3.10
65581144|hipMalloc|ptr=0x7b7534934000 | size=0x100|24|#24 0x00005800df215884 at /usr/bin/python3.10
65581145|hipMalloc|ptr=0x7b7534934000 | size=0x100|25|#25 0x00005800df214e67 at /usr/bin/python3.10
65581146|hipMalloc|ptr=0x7b7534934000 | size=0x100|26|#26 0x00005800df214b46 at /usr/bin/python3.10
65581147|hipMalloc|ptr=0x7b7534934000 | size=0x100|27|#27 0x00005800df20902d at /usr/bin/python3.10
65581148|hipMalloc|ptr=0x7b7534934000 | size=0x100|28|#28 0x00005800df1e2d6c at /usr/bin/python3.10
65581149|hipMalloc|ptr=0x7b7534934000 | size=0x100|29|#29 0x00007b9a46e48d8f in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
65581150|hipMalloc|ptr=0x7b7534934000 | size=0x100|30|#30 0x00007b9a46e48e3f in __libc_start_main_impl at ./csu/../csu/libc-start.c:392:3
65581151|hipMalloc|ptr=0x7b7534934000 | size=0x100|31|#31 0x00005800df1e2c64 at /usr/bin/python3.10
...
```
crossing the Python / C++ boundary. In this example, the call chain reaches through the C++ layer into the swig translation layer but detailed stacks vanish at the Python border.

Additionally adding the Python debug symbols via the operating system's package manger results in:
```
135208356|hipMalloc|ptr=0x7774fd504000 | size=0x200|0|#0  0x00007799178c71c in hipMalloc()1 at /opt/rocm/lib/libamdhip64.so.6
135208357|hipMalloc|ptr=0x7774fd504000 | size=0x200|1|#1  0x000077977b8f8aa9 in faiss::gpu::StandardGpuResourcesImpl::allocMemory(faiss::gpu::AllocRequest const&) at /root/faiss/faiss/gpu-rocm/StandardGpuResources.cpp:558:29
135208358|hipMalloc|ptr=0x7774fd504000 | size=0x200|2|#2  0x000077977b8f4ed6 in faiss::gpu::GpuResources::allocMemoryHandle(faiss::gpu::AllocRequest const&) at /root/faiss/faiss/gpu-rocm/GpuResources.cpp:200:69
135208359|hipMalloc|ptr=0x7774fd504000 | size=0x200|3|#3  0x000077977b911e74 in faiss::gpu::DeviceVector<unsigned char>::realloc_(unsigned long, ihipStream_t*) at /root/faiss/faiss/gpu-rocm/utils/DeviceVector.cuh:228:31
135208360|hipMalloc|ptr=0x7774fd504000 | size=0x200|4|#4  (inlined)          in faiss::gpu::DeviceVector<unsigned char>::reserve(unsigned long, ihipStream_t*) at /root/faiss/faiss/gpu-rocm/utils/DeviceVector.cuh:214:9
135208361|hipMalloc|ptr=0x7774fd504000 | size=0x200|5|#5  (inlined)          in faiss::gpu::DeviceVector<unsigned char>::resize(unsigned long, ihipStream_t*) at /root/faiss/faiss/gpu-rocm/utils/DeviceVector.cuh:136:19
135208362|hipMalloc|ptr=0x7774fd504000 | size=0x200|6|#6  0x000077977b9101ff in faiss::gpu::IVFBase::addVectors(faiss::Index*, faiss::gpu::Tensor<float, 2, true, long, faiss::gpu::traits::DefaultPtrTraits>&, faiss::gpu::Tensor<long, 1, true, long, faiss::gpu::traits::DefaultPtrTraits>&) at /root/faiss/faiss/gpu-rocm/impl/IVFBase.hip:744:25
135208363|hipMalloc|ptr=0x7774fd504000 | size=0x200|7|#7  0x000077977b8ee069 in faiss::gpu::GpuIndexIVF::addImpl_(long, float const*, long const*) at /root/faiss/faiss/gpu-rocm/GpuIndexIVF.hip:293:17
135208364|hipMalloc|ptr=0x7774fd504000 | size=0x200|8|#8  0x000077977b8e72d6 in faiss::gpu::GpuIndex::addPage_(long, float const*, long const*) at /root/faiss/faiss/gpu-rocm/GpuIndex.hip:190:9
135208365|hipMalloc|ptr=0x7774fd504000 | size=0x200|9|#9  0x000077977b8e6f93 in faiss::gpu::GpuIndex::add_with_ids(long, float const*, long const*) at /root/faiss/faiss/gpu-rocm/GpuIndex.hip:137:5
135208366|hipMalloc|ptr=0x7774fd504000 | size=0x200|10|#10 0x00007797bc4c4811 in _wrap_GpuIndex_add_with_ids at /root/faiss/build/faiss/python/CMakeFiles/swigfaiss.dir/swigfaissPYTHON_wrap.cxx:213456:27
135208367|hipMalloc|ptr=0x7774fd504000 | size=0x200|11|#11 0x000056918fd632a7 in cfunction_call at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Objects/methodobject.c:552:18
135208368|hipMalloc|ptr=0x7774fd504000 | size=0x200|12|#12 0x000056918fd59b4a in _PyObject_MakeTpCall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Objects/call.c:215:18
135208369|hipMalloc|ptr=0x7774fd504000 | size=0x200|13|#13 (inlined)          in _PyObject_VectorcallTstate at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:112:16
135208370|hipMalloc|ptr=0x7774fd504000 | size=0x200|14|#14 (inlined)          in _PyObject_VectorcallTstate at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:99:1
135208371|hipMalloc|ptr=0x7774fd504000 | size=0x200|15|#15 (inlined)          in PyObject_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:123:12
135208372|hipMalloc|ptr=0x7774fd504000 | size=0x200|16|#16 (inlined)          in call_function at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5893:13
135208373|hipMalloc|ptr=0x7774fd504000 | size=0x200|17|#17 0x000056918fd534c7 in _PyEval_EvalFrameDefault at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:4181:23
135208374|hipMalloc|ptr=0x7774fd504000 | size=0x200|18|#18 (inlined)          in _PyEval_EvalFrame at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/internal/pycore_ceval.h:46:12
135208375|hipMalloc|ptr=0x7774fd504000 | size=0x200|19|#19 (inlined)          in _PyEval_Vector at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5067:24
135208376|hipMalloc|ptr=0x7774fd504000 | size=0x200|20|#20 0x000056918fd63aeb in _PyFunction_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Objects/call.c:342:16
135208377|hipMalloc|ptr=0x7774fd504000 | size=0x200|21|#21 (inlined)          in _PyObject_VectorcallTstate at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:114:11
135208378|hipMalloc|ptr=0x7774fd504000 | size=0x200|22|#22 (inlined)          in PyObject_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:123:12
135208379|hipMalloc|ptr=0x7774fd504000 | size=0x200|23|#23 (inlined)          in call_function at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5893:13
135208380|hipMalloc|ptr=0x7774fd504000 | size=0x200|24|#24 0x000056918fd4dae7 in _PyEval_EvalFrameDefault at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:4198:23
135208381|hipMalloc|ptr=0x7774fd504000 | size=0x200|25|#25 (inlined)          in _PyEval_EvalFrame at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/internal/pycore_ceval.h:46:12
135208382|hipMalloc|ptr=0x7774fd504000 | size=0x200|26|#26 (inlined)          in _PyEval_Vector at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5067:24
135208383|hipMalloc|ptr=0x7774fd504000 | size=0x200|27|#27 0x000056918fd63aeb in _PyFunction_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Objects/call.c:342:16
135208384|hipMalloc|ptr=0x7774fd504000 | size=0x200|28|#28 (inlined)          in _PyObject_VectorcallTstate at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:114:11
135208385|hipMalloc|ptr=0x7774fd504000 | size=0x200|29|#29 (inlined)          in PyObject_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:123:12
135208386|hipMalloc|ptr=0x7774fd504000 | size=0x200|30|#30 (inlined)          in call_function at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5893:13
135208387|hipMalloc|ptr=0x7774fd504000 | size=0x200|31|#31 0x000056918fd4dae7 in _PyEval_EvalFrameDefault at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:4198:23
135208388|hipMalloc|ptr=0x7774fd504000 | size=0x200|32|#32 (inlined)          in _PyEval_EvalFrame at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/internal/pycore_ceval.h:46:12
135208389|hipMalloc|ptr=0x7774fd504000 | size=0x200|33|#33 (inlined)          in _PyEval_Vector at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5067:24
135208390|hipMalloc|ptr=0x7774fd504000 | size=0x200|34|#34 0x000056918fd63aeb in _PyFunction_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Objects/call.c:342:16
135208391|hipMalloc|ptr=0x7774fd504000 | size=0x200|35|#35 (inlined)          in _PyObject_VectorcallTstate at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:114:11
135208392|hipMalloc|ptr=0x7774fd504000 | size=0x200|36|#36 (inlined)          in PyObject_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:123:12
135208393|hipMalloc|ptr=0x7774fd504000 | size=0x200|37|#37 (inlined)          in call_function at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5893:13
135208394|hipMalloc|ptr=0x7774fd504000 | size=0x200|38|#38 0x000056918fd4d9a1 in _PyEval_EvalFrameDefault at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:4213:19
135208395|hipMalloc|ptr=0x7774fd504000 | size=0x200|39|#39 (inlined)          in _PyEval_EvalFrame at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/internal/pycore_ceval.h:46:12
135208396|hipMalloc|ptr=0x7774fd504000 | size=0x200|40|#40 (inlined)          in _PyEval_Vector at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5067:24
135208397|hipMalloc|ptr=0x7774fd504000 | size=0x200|41|#41 0x000056918fd63aeb in _PyFunction_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Objects/call.c:342:16
135208398|hipMalloc|ptr=0x7774fd504000 | size=0x200|42|#42 (inlined)          in _PyObject_VectorcallTstate at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:114:11
135208399|hipMalloc|ptr=0x7774fd504000 | size=0x200|43|#43 (inlined)          in PyObject_Vectorcall at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/cpython/abstract.h:123:12
135208400|hipMalloc|ptr=0x7774fd504000 | size=0x200|44|#44 (inlined)          in call_function at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5893:13
135208401|hipMalloc|ptr=0x7774fd504000 | size=0x200|45|#45 0x000056918fd4d9a1 in _PyEval_EvalFrameDefault at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:4213:19
135208402|hipMalloc|ptr=0x7774fd504000 | size=0x200|46|#46 (inlined)          in _PyEval_EvalFrame at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Include/internal/pycore_ceval.h:46:12
135208403|hipMalloc|ptr=0x7774fd504000 | size=0x200|47|#47 0x000056918fe32e55 in _PyEval_Vector at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:5067:24
135208404|hipMalloc|ptr=0x7774fd504000 | size=0x200|48|#48 0x000056918fe32d25 in PyEval_EvalCode at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/ceval.c:1134:12
135208405|hipMalloc|ptr=0x7774fd504000 | size=0x200|49|#49 0x000056918fe59ae7 in run_eval_code_obj at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/pythonrun.c:1291:9
135208406|hipMalloc|ptr=0x7774fd504000 | size=0x200|50|#50 0x000056918fe542ee in run_mod at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/pythonrun.c:1312:19
135208407|hipMalloc|ptr=0x7774fd504000 | size=0x200|51|#51 0x000056918fe59884 in pyrun_file at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/pythonrun.c:1208:15
135208408|hipMalloc|ptr=0x7774fd504000 | size=0x200|52|#52 0x000056918fe58e67 in _PyRun_SimpleFileObject at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/pythonrun.c:456:13
135208409|hipMalloc|ptr=0x7774fd504000 | size=0x200|53|#53 0x000056918fe58b46 in _PyRun_AnyFileObject at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Python/pythonrun.c:90:15
135208410|hipMalloc|ptr=0x7774fd504000 | size=0x200|54|#54 (inlined)          in pymain_run_file_obj at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Modules/main.c:353:15
135208411|hipMalloc|ptr=0x7774fd504000 | size=0x200|55|#55 (inlined)          in pymain_run_file at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Modules/main.c:372:15
135208412|hipMalloc|ptr=0x7774fd504000 | size=0x200|56|#56 (inlined)          in pymain_run_python at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Modules/main.c:587:21
135208413|hipMalloc|ptr=0x7774fd504000 | size=0x200|57|#57 0x000056918fe4d02d in Py_RunMain at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Modules/main.c:666:5
135208414|hipMalloc|ptr=0x7774fd504000 | size=0x200|58|#58 0x000056918fe26d6c in Py_BytesMain at /build/python3.10-TyQem4/python3.10-3.10.12/build-static/../Modules/main.c:720:12
135208415|hipMalloc|ptr=0x7774fd504000 | size=0x200|59|#59 0x0000779919201d8f in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
135208416|hipMalloc|ptr=0x7774fd504000 | size=0x200|60|#60 0x0000779919201e3f in __libc_start_main_impl at ./csu/../csu/libc-start.c:392:3
135208417|hipMalloc|ptr=0x7774fd504000 | size=0x200|61|#61 0x000056918fe26c64 at /usr/bin/python3.10
```
resolving the Python interpreter frames.


For a PyTorch example with debug symbols tracing on a specific `hipMalloc`:
```
6791|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|0|#0   0x00007fc0700431 in hipMalloc()c1 at /opt/rocm/lib/libamdhip64.so.6
6792|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|1|#1   0x00007f7f9e4c6b8e at /opt/rocm/lib/librocblas.so.4
6793|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|2|#2   0x00007f7f9e4c8b60 at /opt/rocm/lib/librocblas.so.4
6794|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|3|#3   0x00007f8029dce413 at /opt/rocm/lib/libMIOpen.so.1
6795|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|4|#4   0x00007f8028fdc64d at /opt/rocm/lib/libMIOpen.so.1
6796|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|5|#5   (inlined)          in createMIOpenHandle at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/miopen/Handle.cpp:10:15
6797|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|6|#6   (inlined)          in Handle at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/hip/detail/DeviceThreadHandles.h:36:26
6798|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|7|#7   (inlined)          in construct<at::cuda::(anonymous namespace)::DeviceThreadHandlePool<miopenHandle*, at::native::(anonymous namespace)::createMIOpenHandle, at::native::(anonymous namespace)::destroyMIOpenHandle>::Handle, bool> at /usr/include/c++/11/ext/new_allocator.h:162:4
6799|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|8|#8   (inlined)          in construct<at::cuda::(anonymous namespace)::DeviceThreadHandlePool<miopenHandle*, at::native::(anonymous namespace)::createMIOpenHandle, at::native::(anonymous namespace)::destroyMIOpenHandle>::Handle, bool> at /usr/include/c++/11/bits/alloc_traits.h:516:17
6800|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|9|#9   (inlined)          in _M_realloc_insert<bool> at /usr/include/c++/11/bits/vector.tcc:449:28
6801|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|10|#10  (inlined)          in emplace_back<bool> at /usr/include/c++/11/bits/vector.tcc:121:21
6802|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|11|#11  (inlined)          in reserve at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/hip/detail/DeviceThreadHandles.h:111:53
6803|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|12|#12  0x00007f803967bc39 in at::native::getMiopenHandle() at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/miopen/Handle.cpp:48:38
6804|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|13|#13  0x00007f80396600ed in at::native::raw_miopen_convolution_forward_out(at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/native/miopen/Conv_miopen.cpp:720:32
6805|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|14|#14  0x00007f8039660cdf in at::native::miopen_convolution_forward(char const*, at::TensorArg const&, at::TensorArg const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/native/miopen/Conv_miopen.cpp:790:37
6806|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|15|#15  0x00007f8039660f68 in at::native::miopen_convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/native/miopen/Conv_miopen.cpp:811:82
6807|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|16|#16  0x00007f80393c4b5b in wrapper_CUDA__miopen_convolution at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/RegisterCUDA_0.cpp:15548:225
6808|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|17|#17  (inlined)          in operator() at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17:68
6809|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|18|#18  0x00007f80393ef09e in call at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579:61
6810|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|19|#19  (inlined)          in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool>(void*, c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, c10::SymInt&&, bool&&, bool&&) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76:70
6811|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|20|#20  (inlined)          in at::Tensor c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool>(c10::OperatorHandle const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:129:38
6812|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|21|#21  (inlined)          in at::Tensor c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool)> const&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:808:54
6813|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|22|#22  (inlined)          in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool)>::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:605:43
6814|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|23|#23  0x00007f8041346931 in at::_ops::miopen_convolution::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/Operators_2.cpp:4406:99
6815|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|24|#24  (inlined)          in at::miopen_convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long, bool, bool) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/ops/miopen_convolution.h:27:204
6816|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|25|#25  0x00007f80406e2b46 in at::native::_convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long, bool, bool, bool, bool) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/native/Convolution.cpp:1572:38
6817|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|26|#26  0x00007f80419b4338 in wrapper_CompositeExplicitAutograd___convolution at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/RegisterCompositeExplicitAutograd_0.cpp:3240:300
6818|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|27|#27  (inlined)          in operator() at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17:68
6819|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|28|#28  0x00007f80419bb44a in call at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579:61
6820|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|29|#29  (inlined)          in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool>(void*, c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, bool&&, c10::ArrayRef<c10::SymInt>&&, c10::SymInt&&, bool&&, bool&&, bool&&, bool&&) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76:70
6821|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|30|#30  (inlined)          in at::Tensor c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool>(c10::OperatorHandle const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:129:38
6822|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|31|#31  (inlined)          in at::Tensor c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool)> const&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:808:54
6823|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|32|#32  (inlined)          in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool)>::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:605:43
6824|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|33|#33  0x00007f8040f9c5d3 in at::_ops::_convolution::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt, bool, bool, bool, bool) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/Operators_0.cpp:1928:155
6825|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|34|#34  (inlined)          in at::_convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long, bool, bool, bool, bool) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/ops/_convolution.h:27:280
6826|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|35|#35  0x00007f80406d6498 in at::native::convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/native/Convolution.cpp:1186:108
6827|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|36|#36  0x00007f80419b3e56 in wrapper_CompositeExplicitAutograd__convolution at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/RegisterCompositeExplicitAutograd_0.cpp:3052:246
6828|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|37|#37  (inlined)          in operator() at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17:68
6829|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|38|#38  0x00007f80419baf06 in call at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579:61
6830|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|39|#39  (inlined)          in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt>(void*, c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, bool&&, c10::ArrayRef<c10::SymInt>&&, c10::SymInt&&) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76:70
6831|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|40|#40  (inlined)          in at::Tensor c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt>(c10::OperatorHandle const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:129:38
6832|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|41|#41  (inlined)          in at::Tensor c10::Dispatcher::redispatch<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt)> const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:829:61
6833|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|42|#42  (inlined)          in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt)>::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:613:66
6834|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|43|#43  0x00007f8040f4dd68 in at::_ops::convolution::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/Operators_0.cpp:1893:124
6835|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|44|#44  (inlined)          in at::redispatch::convolution_symint(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/RedispatchFunctions.h:1897:148
6836|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|45|#45  (inlined)          in operator() at /gk_workspace/PyTDebugRPDStackTrace/pytorch/torch/csrc/autograd/generated/VariableType_0.cpp:8218:164
6837|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|46|#46  0x00007f8043691499 in convolution at /gk_workspace/PyTDebugRPDStackTrace/pytorch/torch/csrc/autograd/generated/VariableType_0.cpp:8219:6
6838|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|47|#47  (inlined)          in operator() at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17:68
6839|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|48|#48  0x00007f8043692118 in call at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:613:77
6840|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|49|#49  (inlined)          in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt>(void*, c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, bool&&, c10::ArrayRef<c10::SymInt>&&, c10::SymInt&&) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76:70
6841|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|50|#50  (inlined)          in at::Tensor c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt>(c10::OperatorHandle const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:129:38
6842|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|51|#51  (inlined)          in at::Tensor c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt)> const&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:808:54
6843|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|52|#52  (inlined)          in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt)>::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:605:43
6844|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|53|#53  0x00007f8040f9b495 in at::_ops::convolution::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/Operators_0.cpp:1886:102
6845|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|54|#54  (inlined)          in at::convolution_symint(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, bool, c10::ArrayRef<c10::SymInt>, c10::SymInt) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/ops/convolution.h:38:122
6846|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|55|#55  0x00007f80406daa9e in at::native::conv2d_symint(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/native/Convolution.cpp:960:36
6847|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|56|#56  (inlined)          in wrapper_CompositeImplicitAutograd__conv2d at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/RegisterCompositeImplicitAutograd_0.cpp:3621:90
6848|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|57|#57  (inlined)          in operator() at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:17:68
6849|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|58|#58  0x00007f8041aee8a1 in call at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:579:61
6850|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|59|#59  (inlined)          in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt>(void*, c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, c10::ArrayRef<c10::SymInt>&&, c10::SymInt&&) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:76:70
6851|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|60|#60  (inlined)          in at::Tensor c10::KernelFunction::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt>(c10::OperatorHandle const&, c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/boxing/KernelFunction_impl.h:129:38
6852|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|61|#61  (inlined)          in at::Tensor c10::Dispatcher::call<at::Tensor, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt>(c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt)> const&, at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:808:54
6853|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|62|#62  (inlined)          in c10::TypedOperatorHandle<at::Tensor (at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt)>::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt) const at /gk_workspace/PyTDebugRPDStackTrace/pytorch/aten/src/ATen/core/dispatch/Dispatcher.h:605:43
6854|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|63|#63  0x00007f804163142c in at::_ops::conv2d::call(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/Operators_4.cpp:1677:74
6855|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|64|#64  (inlined)          in at::conv2d_symint(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::SymInt) at /gk_workspace/PyTDebugRPDStackTrace/pytorch/build/aten/src/ATen/ops/conv2d.h:38:89
6856|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|65|#65  (inlined)          in operator() at /gk_workspace/PyTDebugRPDStackTrace/pytorch/torch/csrc/autograd/generated/python_torch_functions_1.cpp:2742:88
6857|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|66|#66  0x00007f8054914056 in THPVariable_conv2d at /gk_workspace/PyTDebugRPDStackTrace/pytorch/torch/csrc/autograd/generated/python_torch_functions_1.cpp:2744:34
6858|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|67|#67  0x00000000004fdcf6 in cfunction_call at /usr/local/src/conda/python-3.10.16/Objects/methodobject.c:543:19
6859|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|68|#68  0x00000000004f747a in _PyObject_MakeTpCall at /usr/local/src/conda/python-3.10.16/Objects/call.c:215:18
6860|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|69|#69  (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:112:16
6861|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|70|#70  (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:99:1
6862|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|71|#71  (inlined)          in PyObject_Vectorcall at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:123:12
6863|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|72|#72  (inlined)          in call_function at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5893:13
6864|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|73|#73  0x00000000004f3515 in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4181:23
6865|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|74|#74  (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6866|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|75|#75  (inlined)          in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6867|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|76|#76  (inlined)          in _PyFunction_Vectorcall at /usr/local/src/conda/python-3.10.16/Objects/call.c:342:16
6868|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|77|#77  (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:114:11
6869|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|78|#78  0x0000000000509a7d in method_vectorcall at /usr/local/src/conda/python-3.10.16/Objects/classobject.c:53:18
6870|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|79|#79  (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:114:11
6871|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|80|#80  (inlined)          in PyObject_Vectorcall at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:123:12
6872|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|81|#81  (inlined)          in call_function at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5893:13
6873|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|82|#82  0x00000000004f2c55 in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4181:23
6874|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|83|#83  (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6875|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|84|#84  (inlined)          in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6876|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|85|#85  (inlined)          in _PyFunction_Vectorcall at /usr/local/src/conda/python-3.10.16/Objects/call.c:342:16
6877|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|86|#86  (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:114:11
6878|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|87|#87  0x0000000000509bd5 in method_vectorcall at /usr/local/src/conda/python-3.10.16/Objects/classobject.c:83:18
6879|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|88|#88  (inlined)          in do_call_core at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5945:12
6880|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|89|#89  0x00000000004f0ca8 in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4277:22
6881|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|90|#90  (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6882|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|91|#91  (inlined)          in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6883|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|92|#92  (inlined)          in _PyFunction_Vectorcall at /usr/local/src/conda/python-3.10.16/Objects/call.c:342:16
6884|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|93|#93  (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:114:11
6885|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|94|#94  0x0000000000509bd5 in method_vectorcall at /usr/local/src/conda/python-3.10.16/Objects/classobject.c:83:18
6886|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|95|#95  (inlined)          in do_call_core at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5945:12
6887|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|96|#96  0x00000000004f0ca8 in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4277:22
6888|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|97|#97  (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6889|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|98|#98  (inlined)          in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6890|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|99|#99  (inlined)          in _PyFunction_Vectorcall at /usr/local/src/conda/python-3.10.16/Objects/call.c:342:16
6891|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|100|#100 0x00000000004f67cc in _PyObject_FastCallDictTstate at /usr/local/src/conda/python-3.10.16/Objects/call.c:142:15
6892|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|101|#101 0x0000000000507f65 in _PyObject_Call_Prepend at /usr/local/src/conda/python-3.10.16/Objects/call.c:431:24
6893|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|102|#102 0x00000000005d0152 in slot_tp_call at /usr/local/src/conda/python-3.10.16/Objects/typeobject.c:7494:15
6894|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|103|#103 0x00000000004f747a in _PyObject_MakeTpCall at /usr/local/src/conda/python-3.10.16/Objects/call.c:215:18
6895|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|104|#104 (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:112:16
6896|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|105|#105 (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:99:1
6897|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|106|#106 (inlined)          in PyObject_Vectorcall at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:123:12
6898|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|107|#107 (inlined)          in call_function at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5893:13
6899|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|108|#108 0x00000000004f3515 in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4181:23
6900|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|109|#109 (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6901|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|110|#110 (inlined)          in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6902|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|111|#111 (inlined)          in _PyFunction_Vectorcall at /usr/local/src/conda/python-3.10.16/Objects/call.c:342:16
6903|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|112|#112 (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:114:11
6904|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|113|#113 0x0000000000509bd5 in method_vectorcall at /usr/local/src/conda/python-3.10.16/Objects/classobject.c:83:18
6905|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|114|#114 (inlined)          in do_call_core at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5945:12
6906|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|115|#115 0x00000000004f0ca8 in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4277:22
6907|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|116|#116 (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6908|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|117|#117 (inlined)          in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6909|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|118|#118 (inlined)          in _PyFunction_Vectorcall at /usr/local/src/conda/python-3.10.16/Objects/call.c:342:16
6910|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|119|#119 (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:114:11
6911|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|120|#120 0x0000000000509bd5 in method_vectorcall at /usr/local/src/conda/python-3.10.16/Objects/classobject.c:83:18
6912|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|121|#121 (inlined)          in do_call_core at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5945:12
6913|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|122|#122 0x00000000004f0ca8 in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4277:22
6914|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|123|#123 (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6915|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|124|#124 (inlined)          in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6916|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|125|#125 (inlined)          in _PyFunction_Vectorcall at /usr/local/src/conda/python-3.10.16/Objects/call.c:342:16
6917|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|126|#126 0x00000000004f67cc in _PyObject_FastCallDictTstate at /usr/local/src/conda/python-3.10.16/Objects/call.c:142:15
6918|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|127|#127 0x0000000000507f65 in _PyObject_Call_Prepend at /usr/local/src/conda/python-3.10.16/Objects/call.c:431:24
6919|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|128|#128 0x00000000005d0152 in slot_tp_call at /usr/local/src/conda/python-3.10.16/Objects/typeobject.c:7494:15
6920|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|129|#129 0x00000000004f747a in _PyObject_MakeTpCall at /usr/local/src/conda/python-3.10.16/Objects/call.c:215:18
6921|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|130|#130 (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:112:16
6922|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|131|#131 (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:99:1
6923|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|132|#132 (inlined)          in PyObject_Vectorcall at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:123:12
6924|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|133|#133 (inlined)          in call_function at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5893:13
6925|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|134|#134 0x00000000004f2f0d in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4213:19
6926|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|135|#135 (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6927|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|136|#136 (inlined)          in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6928|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|137|#137 0x00000000004fe13e in _PyFunction_Vectorcall at /usr/local/src/conda/python-3.10.16/Objects/call.c:342:16
6929|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|138|#138 (inlined)          in _PyObject_VectorcallTstate at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:114:11
6930|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|139|#139 (inlined)          in PyObject_Vectorcall at /usr/local/src/conda/python-3.10.16/Include/cpython/abstract.h:123:12
6931|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|140|#140 (inlined)          in call_function at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5893:13
6932|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|141|#141 0x00000000004ee44e in _PyEval_EvalFrameDefault at /usr/local/src/conda/python-3.10.16/Python/ceval.c:4213:19
6933|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|142|#142 (inlined)          in _PyEval_EvalFrame at /usr/local/src/conda/python-3.10.16/Include/internal/pycore_ceval.h:46:12
6934|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|143|#143 0x00000000005953a1 in _PyEval_Vector at /usr/local/src/conda/python-3.10.16/Python/ceval.c:5067:24
6935|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|144|#144 0x00000000005952e6 in PyEval_EvalCode at /usr/local/src/conda/python-3.10.16/Python/ceval.c:1134:12
6936|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|145|#145 0x00000000005c6736 in run_eval_code_obj at /usr/local/src/conda/python-3.10.16/Python/pythonrun.c:1291:9
6937|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|146|#146 0x00000000005c186f in run_mod at /usr/local/src/conda/python-3.10.16/Python/pythonrun.c:1312:19
6938|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|147|#147 0x0000000000459838 in pyrun_file at /usr/local/src/conda/python-3.10.16/Python/pythonrun.c:1208:15
6939|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|148|#148 0x00000000005bbdfe in _PyRun_SimpleFileObject at /usr/local/src/conda/python-3.10.16/Python/pythonrun.c:456:13
6940|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|149|#149 0x00000000005bbb62 in _PyRun_AnyFileObject at /usr/local/src/conda/python-3.10.16/Python/pythonrun.c:90:15
6941|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|150|#150 (inlined)          in pymain_run_file_obj at /usr/local/src/conda/python-3.10.16/Modules/main.c:357:15
6942|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|151|#151 (inlined)          in pymain_run_file at /usr/local/src/conda/python-3.10.16/Modules/main.c:376:15
6943|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|152|#152 (inlined)          in pymain_run_python at /usr/local/src/conda/python-3.10.16/Modules/main.c:595:21
6944|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|153|#153 0x00000000005b891c in Py_RunMain at /usr/local/src/conda/python-3.10.16/Modules/main.c:674:5
6945|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|154|#154 0x00000000005885d8 in Py_BytesMain at /usr/local/src/conda/python-3.10.16/Modules/main.c:1094:12
6946|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|155|#155 0x00007fc07198fd8f in __libc_start_call_main at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
6947|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|156|#156 0x00007fc07198fe3f in __libc_start_main_impl at ./csu/../csu/libc-start.c:392:3
6948|hipMalloc|ptr=0x7f7aad600000 | size=0x8000000|157|#157 0x000000000058848d at /opt/conda/envs/py_3.10/bin/python3.10
```
showing a `hipMalloc` invocation from MIOpen setup during initialization tracking through PyTorch's native `aten` layer and generated code into the Python interpreter.


### Summary
Stackframe analysis helps to identify call chains in native code to the HIP API. With debug symbols present, full resolution through the native laye is possible.
