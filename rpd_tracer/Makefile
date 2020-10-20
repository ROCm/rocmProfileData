HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
    HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

ROC_LIBS = -lroctracer64 -lroctx64 -lsqlite3
ROC_INCLUDE = -I/opt/rocm/include -I/opt/rocm/roctracer/include/ -I/opt/rocm/include/hsa

CXX=$(HIPCC)

SRCS = rpd_tracer.cpp Table.cpp OpTable.cpp ApiTable.cpp StringTable.cpp
OBJS = $(SRCS:.cpp=.o)

MAIN = rpd_tracer.so

$(MAIN): $(OBJS)
	$(CXX) -o $@ $(OBJS) -shared -rdynamic -std=c++11 -L/opt/rocm/lib $(ROC_LIBS) -g

.cpp.o:
	$(CXX) -o $@ -c $< $(ROC_INCLUDE) -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g -O3

#rpd_tracer.o: rpd_tracer.cpp
#	$(CXX) -o $@ -c $< $(ROC_INCLUDE) -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g

clean:
	rm -f *.o rpd_tracer.so
