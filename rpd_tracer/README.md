# RPD_TRACER

This is a tracer that can attach to any process and record hip apis, ops, and roctx.


## Steps:
1) `cd` to the rocmProfileData root directory
2) Run `make; make install`
3) Run `runTracer.sh -o <output_file>.rpd <your_command_and_args>`.
4) `python tools/rpd2tracing <your_profile>.rpd <your_profile>.json` for Chrome tracing output.


<b>Manual Stuff:</b>
 - Use 'LD_PRELOAD=./librpd_tracer.so' to attach the profiler to any process
 - Default output file name is 'trace.rpd'
 - Override file name with env 'RPDT_FILENAME='
 - Create empty rpd file with python3 -m rocpd.schema --create ${OUTPUT_FILE}
 - Multiple processes can log to the same file concurrently
 - Files can be appended any number of times

 ## Example
 This example shows how to dynamically link `librpd_tracer.so` file to your application.

1) Make sure you run step 2 above to install rpd utilities.
2) Create empty rpd file with `python3 -m rocpd.schema --create ${OUTPUT_FILE}`. Here `${OUTPUT_FILE}` a rpd file.
3) If you are using CMake to build your application, add `target_link_libraries(/opt/rocm/lib/libroctracer64.so ${CMAKE_DL_LIBS})`. This will allow you to load libraries for `libroctracer` and `DL`.
4) In your app make the following changes:
    1) Declare pointer for `dlopen()` function. Somewhere at the beginning of your code:
        ```
        void* rocTracer_lib;
        ```
    2) Define function type of the functions you are interested in calling, and declare them:
        ```
        typedef void (*rt_func_t)();
        rt_func_t init_tracing;
        rt_func_t start_tracing;
        rt_func_t stop_tracing;
        ```
    3) Load dynamic library file:
        ```
        rocTracer_lib = dlopen("librpd_tracer.so", RTLD_LAZY); //defer resolution until the first reference via RTLD_LAZY
            std::cout << rocTracer_lib << std::endl; // points to some memory location, e.g. 0x89a...
            if (!rocTracer_lib) {
                fputs (dlerror(), stderr);
                exit(1);
            }
        ```
    4) Initializing/starting/stopping follow the same syntax, below shows example for `init_tracing()` function, place them appropriately in your application:
        ```
            dlerror(); //clear any previous errors.
            init_tracing = (rt_func_t)dlsym(rocTracer_lib, "_Z12init_tracingv"); 
            if(!init_tracing)
            {
                std::cout << init_tracing << std::endl;
                fputs(dlerror(), stderr);
                exit(1);
            }
        ```
    5) You can run the function you linked via `dlsym` as (following the example in above step):
        ```
        init_tracing();
        ```
<b>Note:</b> You can utilize `nm -gd <PATH_TO_librpd_tracer.so>` to find out symbol names in your library.
