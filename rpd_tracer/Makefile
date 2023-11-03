
PREFIX = /usr/local

HIP_PATH?= $(wildcard /opt/rocm/hip)
CUDA_PATH?= $(wildcard /usr/local/cuda/)
SMUDUMP?= $(wildcard ../smutrace/Makefile)
ROOT_DIR = $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

ifeq (,$(HIP_PATH))
HIP_PATH= $(wildcard /opt/rocm)
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

RPD_LIBS = -lsqlite3 -lfmt
RPD_INCLUDES =
RPD_SRCS = Table.cpp OpTable.cpp KernelApiTable.cpp CopyApiTable.cpp ApiTable.cpp StringTable.cpp MetadataTable.cpp MonitorTable.cpp ApiIdList.cpp DbResource.cpp Logger.cpp

ifneq (,$(HIP_PATH))
        $(info Building with roctracer)
        RPD_LIBS += -L/opt/rocm/lib -lroctracer64 -lroctx64 -lamdhip64 -lrocm_smi64
        RPD_INCLUDES += -I/opt/rocm/include -I/opt/rocm/include/roctracer -I/opt/rocm/include/hsa
        RPD_SRCS += RoctracerDataSource.cpp RocmSmiDataSource.cpp SmuDumpDataSource.cpp
        RPD_INCLUDES += -D__HIP_PLATFORM_AMD__
endif

ifneq ($(CUDA_PATH),)
        $(info Building with cupti)
        RPD_LIBS += -L/usr/local/cuda/lib64 -lcupti
        RPD_INCLUDES += -I/usr/local/cuda/include
        RPD_SRCS += CuptiDataSource.cpp
endif

RPD_OBJS = $(RPD_SRCS:.cpp=.o)


RPD_MAIN = librpd_tracer.so
RPD_SCRIPT = runTracer.sh

PYTHON = python3
PIP = pip3


all: | $(RPD_MAIN) 

.PHONY: all 


$(RPD_MAIN): $(RPD_OBJS)
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 $(RPD_LIBS) -g
ifneq (,$(SMUDUMP))
	sudo make -C $(ROOT_DIR)../smutrace
endif

.cpp.o:
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g

#$(PREFIX)/lib/lib$(RPD_MAIN):
#	ln -s $(PREFIX)/lib/$(RPD_MAIN) $@

.PHONY: install
#install: | $(PREFIX)/lib/lib$(RPD_MAIN)
install:
	cp $(RPD_MAIN)  $(PREFIX)/lib/
	cp $(RPD_SCRIPT) $(PREFIX)/bin/
	echo $(ROOT_DIR)../tools/rpd2tracing.py --format object ./trace.rpd ./trace.json >> $(PREFIX)/bin/$(RPD_SCRIPT)
	ldconfig
	$(PYTHON) setup.py install
ifneq (,$(SMUDUMP))
	sudo make install -C $(ROOT_DIR)../smutrace
endif

.PHONY: config
config:
ifneq (,$(SMUDUMP))
	sudo make config -C $(ROOT_DIR)../smutrace
endif

.PHONY: uninstall
uninstall:
	rm $(PREFIX)/lib/$(RPD_MAIN)
	rm $(PREFIX)/bin/$(RPD_SCRIPT)
.PHONY: clean
clean:
	rm -f *.o *.so 
ifneq (,$(SMUDUMP))
	sudo make clean -C $(ROOT_DIR)../smutrace
endif
