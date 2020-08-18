HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
    HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

ROC_LIBS = -lroctracer64 -lroctx64 -lsqlite3
ROC_INCLUDE = -I/opt/rocm/include -I/opt/rocm/roctracer/include/ -I/opt/rocm/include/hsa

CXX=$(HIPCC)

rpd_tracer.so: rpd_tracer.o
	$(CXX) -o $@ $< -shared -rdynamic -std=c++11 -L/opt/rocm/lib $(ROC_LIBS) -g

rpd_tracer.o: rpd_tracer.cpp
	$(CXX) -o $@ -c $< $(ROC_INCLUDE) -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g

clean:
	rm -f rpd_tracer.o rpd_tracer.so
