
PREFIX = /usr/local

HIP_PATH ?= $(or $(wildcard /opt/rocm/),$(wildcard /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel))
CUDA_PATH ?= $(wildcard /usr/local/cuda/)
CPPTRACE_INCLUDE_PATH ?= $(wildcard ../cpptrace_install/include)
RLOG_PATH ?= $(wildcard ../rlog)
RLOG_LIB_PATH ?= /usr/local

ROCM_VERSION ?= $(shell cat $(HIP_PATH)/.info/version)

RPD_LIBS = -lsqlite3 -lfmt -ldl
RPD_INCLUDES =
RPD_SRCS = Schema.cpp Table.cpp BufferedTable.cpp OpTable.cpp KernelApiTable.cpp CopyApiTable.cpp ApiTable.cpp StringTable.cpp UStringTable.cpp MetadataTable.cpp MonitorTable.cpp StackFrameTable.cpp CounterTable.cpp ApiIdList.cpp DbResource.cpp Config.cpp Storage.cpp Logger.cpp Unwind.cpp RoctxDataSource.cpp NvtxDataSource.cpp RlogClient.cpp ChronoSyncDataSource.cpp Firefly.cpp Utility.cpp TcpConnection.cpp NetWriterBackend.cpp RemoteDataSource.cpp

ifneq (,$(HIP_PATH))
        $(info rocm_version is ${ROCM_VERSION})
        ifneq (,$(wildcard $(HIP_PATH)/include/rocprofiler-sdk))
        ifneq ($(shell echo ${ROCM_VERSION} | cut -d. -f1), 6)
                $(info Building with rocprofiler-sdk)
		RPD_LIBS += -lrocprofiler-sdk
		RPD_SRCS += RocprofDataSource.cpp
        endif
        endif

        ifneq (,$(wildcard $(HIP_PATH)/include/roctracer))
                $(info Building with roctracer)
		RPD_LIBS += -lroctracer64
		RPD_SRCS += RoctracerDataSource.cpp
        endif

        ifneq (,$(wildcard $(HIP_PATH)/include/hip/amd_detail/hip_profiler_ext.h))
                $(info Building with ClrDataSource)
                RPD_SRCS += ClrDataSource.cpp
        endif

        RPD_LIBS += -L$(HIP_PATH)/lib
        RPD_LIBS += -lamdhip64

        RPD_INCLUDES += -D__HIP_PLATFORM_AMD__
        RPD_INCLUDES += -I$(HIP_PATH)/include -I$(HIP_PATH)/include/hsa
        RPD_INCLUDES += -Ijson/include

        # rocm-trace-lite (RTL) data source — requires submodule init
        ifneq (,$(wildcard ../rocm-trace-lite/src/trace_db.h))
                $(info Building with rocm-trace-lite)
                RTL_SRC = ../rocm-trace-lite/src
                RPD_SRCS += RtlDataSource.cpp
                RPD_INCLUDES += -I$(RTL_SRC)
                RPD_LIBS += -lhsa-runtime64

                RTL_SUBMODULE_SRCS = $(RTL_SRC)/hsa_intercept.cpp $(RTL_SRC)/hip_api_intercept.cpp $(RTL_SRC)/trace_db.cpp
                RTL_SUBMODULE_OBJS = hsa_intercept.o hip_api_intercept.o trace_db_rtl.o
        else
                $(warning rocm-trace-lite submodule not initialized — skipping RtlDataSource (run: git submodule update --init))
        endif
endif

ifneq ($(CUDA_PATH),)
        $(info Building with cupti)
        RPD_LIBS += -L/usr/local/cuda/lib64 -lcupti
        RPD_INCLUDES += -I/usr/local/cuda/include
        RPD_SRCS += CuptiDataSource.cpp
endif

ifneq (,$(RLOG_PATH))
        $(info Building with rlog)
        RPD_LIBS += -L$(RLOG_LIB_PATH)/lib -lrlog
        RPD_INCLUDES += -I$(RLOG_PATH)/include
        RPD_SRCS += $(RLOG_PATH)/src/rlog.cpp RlogDataSource.cpp
endif

ifneq ($(CPPTRACE_INCLUDE_PATH),)
        $(info Building with cpptrace)
        RPD_INCLUDES += -DRPD_STACKFRAME_SUPPORT
        RPD_LIBS += -L$(CPPTRACE_INCLUDE_PATH)/../lib -lcpptrace -ldwarf -lz -lzstd
        RPD_INCLUDES += -I $(CPPTRACE_INCLUDE_PATH)
endif

RPD_OBJS = $(RPD_SRCS:.cpp=.o)
RPD_DEPS = $(RPD_OBJS:.o=.d)

RTL_SUBMODULE_OBJS ?=

RPD_TRACER = librpd_tracer.so
RPD_EMBEDDED = librpd_embedded.so
RPD_SCRIPT = runTracer.sh loadTracer.sh

PYTHON = python3
PIP = pip3


all: $(RPD_TRACER) $(RPD_EMBEDDED) rpdrun

.PHONY: all

SCHEMA_DIR = ../rocpd_python/rocpd/schema_data

tableSchema.h: $(SCHEMA_DIR)/tableSchema.cmd
	printf 'static const unsigned char tableSchema_cmd[] = {\n' > $@
	xxd -i < $< >> $@
	printf ', 0x00\n};\n' >> $@

utilitySchema.h: $(SCHEMA_DIR)/utilitySchema.cmd
	printf 'static const unsigned char utilitySchema_cmd[] = {\n' > $@
	xxd -i < $< >> $@
	printf ', 0x00\n};\n' >> $@

Schema.o: tableSchema.h utilitySchema.h


rpd_capi_tracer.o: rpd_capi.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -DRPD_TRACER_BUILD -std=c++17 -fPIC -g -O3 -MMD -MP

rpd_capi_embedded.o: rpd_capi.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3 -MMD -MP

$(RPD_TRACER): $(RPD_OBJS) rpd_capi_tracer.o $(RTL_SUBMODULE_OBJS)
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 $(RPD_LIBS) -Wl,--version-script=marker_shim.ver -g

$(RPD_EMBEDDED): $(RPD_OBJS) rpd_capi_embedded.o $(RTL_SUBMODULE_OBJS)
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 $(RPD_LIBS) -Wl,--version-script=marker_shim.ver -g

rpdrun: rpdrun.c
	$(CC) -o $@ $< -ldl

%.o: %.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3 -MMD -MP

# rocm-trace-lite submodule objects (source lives outside rpd_tracer/)
hsa_intercept.o: $(RTL_SRC)/hsa_intercept.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3

hip_api_intercept.o: $(RTL_SRC)/hip_api_intercept.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3

trace_db_rtl.o: $(RTL_SRC)/trace_db.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3

-include $(RPD_DEPS)
-include rpd_capi_tracer.d rpd_capi_embedded.d

.PHONY: install
install: all
	cp $(RPD_TRACER) $(RPD_EMBEDDED)  $(PREFIX)/lib/
	cp $(RPD_SCRIPT) rpdrun $(PREFIX)/bin/
	ldconfig
	$(PIP) install .

.PHONY: uninstall
uninstall:
	rm $(PREFIX)/lib/$(RPD_TRACER) $(PREFIX)/lib/$(RPD_EMBEDDED)
	rm $(PREFIX)/bin/$(RPD_SCRIPT)

.PHONY: clean
clean:
	rm -f *.o *.so *.d tableSchema.h utilitySchema.h rpdrun
