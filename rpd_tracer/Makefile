
PREFIX = /usr/local

HIP_PATH ?= $(wildcard /opt/rocm/)
CUDA_PATH ?= $(wildcard /usr/local/cuda/)
CPPTRACE_INCLUDE_PATH ?= $(wildcard ../cpptrace_install/include)

ROCM_VERSION ?= $(shell cat /opt/rocm/.info/version)

RPD_LIBS = -lsqlite3 -lfmt
RPD_INCLUDES =
RPD_SRCS = Schema.cpp Table.cpp BufferedTable.cpp OpTable.cpp KernelApiTable.cpp CopyApiTable.cpp ApiTable.cpp StringTable.cpp UStringTable.cpp MetadataTable.cpp MonitorTable.cpp StackFrameTable.cpp ApiIdList.cpp DbResource.cpp Logger.cpp Unwind.cpp RoctxDataSource.cpp NvtxDataSource.cpp

ifneq (,$(HIP_PATH))
        $(info rocm_version is ${ROCM_VERSION})
        #USE_ROCPROFILER_SDK = false
        ifneq (,$(wildcard /opt/rocm/include/rocprofiler-sdk))
        ifneq ($(shell echo ${ROCM_VERSION} | cut -d. -f1), 6)
            # Disable rocprofiler-sdk (for performance reasons)
            # USE_ROCPROFILER_SDK = true
        endif
        endif

        ifeq ($(USE_ROCPROFILER_SDK), true)
                $(info Building with rocprofiler-sdk)
		RPD_LIBS += -lrocprofiler-sdk
		RPD_SRCS += RocprofDataSource.cpp
        else
                $(info Building with roctracer)
		RPD_LIBS += -lroctracer64
		RPD_SRCS += RoctracerDataSource.cpp
		RPD_INCLUDES += -I/opt/rocm/include/roctracer
        endif

        RPD_LIBS += -L/opt/rocm/lib
        RPD_LIBS += -lamdhip64
        RPD_SRCS += AmdSmiDataSource.cpp

        RPD_INCLUDES += -D__HIP_PLATFORM_AMD__
        RPD_INCLUDES += -I/opt/rocm/include -I/opt/rocm/include/hsa
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

ifneq ($(CPPTRACE_INCLUDE_PATH),)
        $(info Building with cpptrace)
        RPD_INCLUDES += -DRPD_STACKFRAME_SUPPORT
        RPD_LIBS += -L$(CPPTRACE_INCLUDE_PATH)/../lib -lcpptrace -ldwarf -lz -lzstd -ldl
        RPD_INCLUDES += -I $(CPPTRACE_INCLUDE_PATH)
endif

RPD_OBJS = $(RPD_SRCS:.cpp=.o)
RTL_SUBMODULE_OBJS ?=

RPD_MAIN = librpd_tracer.so
RPD_SCRIPT = runTracer.sh loadTracer.sh

PYTHON = python3
PIP = pip3


all: | $(RPD_MAIN) 

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


$(RPD_MAIN): $(RPD_OBJS) $(RTL_SUBMODULE_OBJS)
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 $(RPD_LIBS) -Wl,--version-script=marker_shim.ver -g

.cpp.o:
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3

# rocm-trace-lite submodule objects (source lives outside rpd_tracer/)
hsa_intercept.o: $(RTL_SRC)/hsa_intercept.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3

hip_api_intercept.o: $(RTL_SRC)/hip_api_intercept.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3

trace_db_rtl.o: $(RTL_SRC)/trace_db.cpp
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3

#$(PREFIX)/lib/lib$(RPD_MAIN):
#	ln -s $(PREFIX)/lib/$(RPD_MAIN) $@

.PHONY: install
#install: | $(PREFIX)/lib/lib$(RPD_MAIN)
install: all
	cp $(RPD_MAIN)  $(PREFIX)/lib/
	cp $(RPD_SCRIPT) $(PREFIX)/bin/
	ldconfig
	$(PIP) install .

.PHONY: uninstall
uninstall:
	rm $(PREFIX)/lib/$(RPD_MAIN)
	rm $(PREFIX)/bin/$(RPD_SCRIPT)
.PHONY: clean
clean:
	rm -f *.o *.so *.d tableSchema.h utilitySchema.h
