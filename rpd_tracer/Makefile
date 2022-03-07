HIP_PATH?= $(wildcard /opt/rocm/hip)
PREFIX = /opt/rocm
ifeq (,$(HIP_PATH))
    HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

ROC_LIBS = -lroctracer64 -lroctx64 -lsqlite3 -lfmt
ROC_INCLUDE = -I/opt/rocm/include -I/opt/rocm/roctracer/include/ -I/opt/rocm/include/hsa

CXX=$(HIPCC)

RPD_MAIN = rpd_tracer.so
RTG_MAIN = rtg_tracer.so
RPD_SCRIPT = runTracer.sh

PYTHON = python3
PIP = pip3


RTG_DIR = $(wildcard ../rtg_tracer/rtg_tracer)

all: | $(RPD_MAIN) $(RTG_DIR)

.PHONY: all $(RTG_DIR)

$(RTG_DIR):
	@echo "rtg_tracer exists.  Building"
	$(MAKE) $(RTG_MAIN)

RPD_SRCS = Table.cpp OpTable.cpp  KernelApiTable.cpp CopyApiTable.cpp ApiTable.cpp StringTable.cpp MetadataTable.cpp ApiIdList.cpp
RPD_OBJS = $(RPD_SRCS:.cpp=.o)


$(RPD_MAIN): rpd_tracer.o $(RPD_OBJS)
	$(CXX) -o $@ $< $(RPD_OBJS) -shared -rdynamic -std=c++11 -L/opt/rocm/lib $(ROC_LIBS) -g

.cpp.o:
	$(CXX) -o $@ -c $< $(ROC_INCLUDE) -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g -O3

RTG_SRCS =
RTG_SRCS += ../rtg_tracer/rtg_tracer/rtg_out_printf.cpp
RTG_SRCS += ../rtg_tracer/rtg_tracer/rtg_out_rpd.cpp
RTG_OBJS = $(RTG_SRCS:.cpp=.o)

../rtg_tracer/rtg_tracer/rtg_tracer.o: ../rtg_tracer/rtg_tracer/rtg_tracer.cpp
	$(CXX) -o $@ -c $< $(ROC_INCLUDE) -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g -O3 -DRPD_TRACER

$(RTG_MAIN): ../rtg_tracer/rtg_tracer/rtg_tracer.o $(RTG_OBJS) $(RPD_OBJS)
	$(CXX) -o $@ $< $(RPD_OBJS) $(RTG_OBJS) -shared -rdynamic -std=c++11 -L/opt/rocm/lib $(ROC_LIBS) -g

$(PREFIX)/lib/lib$(RPD_MAIN):
	ln -s $(PREFIX)/lib/$(RPD_MAIN) $@

.PHONY: install
install: | $(PREFIX)/lib/lib$(RPD_MAIN)
	cp $(RPD_MAIN)  $(PREFIX)/lib/
	cp $(RPD_SCRIPT) $(PREFIX)/bin/
        
	$(PYTHON) setup.py install

.PHONY: uninstall
uninstall:
	rm $(PREFIX)/lib/$(RPD_MAIN)
	rm $(PREFIX)/bin/$(RPD_SCRIPT)
.PHONY: clean
clean:
	rm -f *.o *.so ../rtg_tracer/rtg_tracer/*.o
