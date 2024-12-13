
PREFIX = /usr/local

HIP_PATH?= $(wildcard /opt/rocm/)
CUDA_PATH?= $(wildcard /usr/local/cuda/)

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

RPD_LIBS = -lsqlite3 -lfmt
RPD_INCLUDES =
RPD_SRCS = Table.cpp BufferedTable.cpp OpTable.cpp KernelApiTable.cpp CopyApiTable.cpp ApiTable.cpp StringTable.cpp MetadataTable.cpp MonitorTable.cpp ApiIdList.cpp DbResource.cpp Logger.cpp

ifneq (,$(HIP_PATH))
        $(info Building with roctracer)
        RPD_LIBS += -L/opt/rocm/lib -lroctracer64 -lroctx64 -lamdhip64 -lrocm_smi64
        RPD_INCLUDES += -I/opt/rocm/include -I/opt/rocm/include/roctracer -I/opt/rocm/include/hsa
        RPD_SRCS += RoctracerDataSource.cpp #RocmSmiDataSource.cpp
        RPD_INCLUDES += -D__HIP_PLATFORM_AMD__
endif

ifneq ($(CUDA_PATH),)
        $(info Building with cupti)

        # Find CUDA include dirs
        RPD_INCLUDES += $(patsubst %,-I%,$(shell find /usr/local/lib/python*/dist-packages/nvidia -type d -name include))
        RPD_INCLUDES += $(patsubst %,-I%,$(shell find /usr/local/lib/python*/dist-packages/triton/backends/nvidia -type d -name include))

        # Find any libcupti.so* file
        CUPTI_LIB_FILE := $(shell find /usr/local/lib/python*/dist-packages/nvidia/cuda_cupti/lib -name "libcupti.so*" 2>/dev/null | head -n1)
        RPD_LIBS += -L$(dir $(CUPTI_LIB_FILE))
        RPD_LIBS += -l:$(notdir $(CUPTI_LIB_FILE))
        RPD_LIBS += -Wl,-rpath,$(dir $(CUPTI_LIB_FILE))

        RPD_SRCS += CuptiDataSource.cpp
endif

RPD_OBJS = $(RPD_SRCS:.cpp=.o)


RPD_MAIN = librpd_tracer.so
RPD_SCRIPT = runTracer.sh loadTracer.sh

PYTHON = python3
PIP = pip3


all: | $(RPD_MAIN) 

.PHONY: all 


$(RPD_MAIN): $(RPD_OBJS)
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 $(RPD_LIBS) -g

.cpp.o:
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g -O3

#$(PREFIX)/lib/lib$(RPD_MAIN):
#	ln -s $(PREFIX)/lib/$(RPD_MAIN) $@

.PHONY: install
#install: | $(PREFIX)/lib/lib$(RPD_MAIN)
install: all
	cp $(RPD_MAIN)  $(PREFIX)/lib/
	cp $(RPD_SCRIPT) $(PREFIX)/bin/
	ldconfig
	$(PYTHON) setup.py install

.PHONY: uninstall
uninstall:
	rm $(PREFIX)/lib/$(RPD_MAIN)
	rm $(PREFIX)/bin/$(RPD_SCRIPT)
.PHONY: clean
clean:
	rm -f *.o *.so 
