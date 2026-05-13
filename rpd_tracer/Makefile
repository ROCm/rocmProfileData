
PREFIX = /usr/local

HIP_PATH ?= $(wildcard /opt/rocm/)
CUDA_PATH ?= $(wildcard /usr/local/cuda/)
CPPTRACE_INCLUDE_PATH ?= $(wildcard ../cpptrace_install/include)
RLOG_PATH ?= $(wildcard ../rlog)
RLOG_LIB_PATH ?= /usr/local

ROCM_VERSION ?= $(shell cat /opt/rocm/.info/version)

RPD_LIBS = -lsqlite3 -lfmt
RPD_INCLUDES =
RPD_SRCS = Schema.cpp Table.cpp BufferedTable.cpp OpTable.cpp KernelApiTable.cpp CopyApiTable.cpp ApiTable.cpp StringTable.cpp UStringTable.cpp MetadataTable.cpp MonitorTable.cpp StackFrameTable.cpp ApiIdList.cpp DbResource.cpp Logger.cpp Unwind.cpp RoctxDataSource.cpp NvtxDataSource.cpp RlogClient.cpp

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
        RPD_LIBS += -L$(CPPTRACE_INCLUDE_PATH)/../lib -lcpptrace -ldwarf -lz -lzstd -ldl
        RPD_INCLUDES += -I $(CPPTRACE_INCLUDE_PATH)
endif

RPD_OBJS = $(RPD_SRCS:.cpp=.o)


RPD_TRACER = librpd_tracer.so
RPD_EMBEDDED = librpd_embedded.so
RPD_SCRIPT = runTracer.sh loadTracer.sh

PYTHON = python3
PIP = pip3


all: $(RPD_TRACER) $(RPD_EMBEDDED)

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

$(RPD_TRACER): $(RPD_OBJS) rpd_capi_tracer.o
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 $(RPD_LIBS) -Wl,--version-script=marker_shim.ver -g

$(RPD_EMBEDDED): $(RPD_OBJS) rpd_capi_embedded.o
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 $(RPD_LIBS) -Wl,--version-script=marker_shim.ver -g

.cpp.o:
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++17 -fPIC -g -O3

#$(PREFIX)/lib/lib$(RPD_MAIN):
#	ln -s $(PREFIX)/lib/$(RPD_MAIN) $@

-include rpd_capi_tracer.d rpd_capi_embedded.d

.PHONY: install
#install: | $(PREFIX)/lib/lib$(RPD_MAIN)
install: all
	cp $(RPD_TRACER) $(RPD_EMBEDDED)  $(PREFIX)/lib/
	cp $(RPD_SCRIPT) $(PREFIX)/bin/
	ldconfig
	$(PIP) install .

.PHONY: uninstall
uninstall:
	rm $(PREFIX)/lib/$(RPD_TRACER) $(PREFIX)/lib/$(RPD_EMBEDDED)
	rm $(PREFIX)/bin/$(RPD_SCRIPT)
.PHONY: clean
clean:
	rm -f *.o *.so *.d tableSchema.h utilitySchema.h
