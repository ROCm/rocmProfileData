PYTHON ?= python3

.PHONY:
all: cpptrace rlog rpd rocpd remote

.PHONY: install
install: all
	$(MAKE) install -C rocpd_python
	$(MAKE) install -C rpd_tracer
	$(MAKE) install -C remote
	$(MAKE) install -C rpd_viewer

.PHONY: uninstall
uninstall:
	$(MAKE) uninstall -C rocpd_python
	$(MAKE) uninstall -C rpd_tracer
	$(MAKE) uninstall -C remote
	$(MAKE) uninstall -C rpd_viewer

.PHONY: clean
clean: cpptrace-clean rlog-clean
	$(MAKE) clean -C rocpd_python
	$(MAKE) clean -C rpd_tracer
	$(MAKE) clean -C remote
	$(MAKE) clean -C rpd_viewer

.PHONY: rpd
rpd:
	$(MAKE) -C rpd_tracer
.PHONY: rocpd
rocpd:
	$(MAKE) -C rocpd_python
.PHONY: remote
remote:
	$(MAKE) -C remote 
.PHONY: cpptrace

RLOG_CMAKE?= $(wildcard rlog/CMakeLists.txt)
ifneq ($(RLOG_CMAKE),)
.PHONY: rlog rlog-clean
rlog:
	cd rlog; cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local; cmake --build build; cmake --install build
rlog-clean:
	rm -rf rlog/build
else
rlog:
rlog-clean:
endif

CPPTRACE_MAKE?= $(wildcard cpptrace/Makefile)
ifneq ($(CPPTRACE_MAKE),)
cpptrace:
	cd cpptrace; cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../cpptrace_install; cmake --build build; cmake --install build; cd ../cpptrace_install; if [ ! -d ./lib ]; then ln -s lib64 lib; fi
cpptrace-clean:
	$(MAKE) clean -C cpptrace
	rm -r cpptrace_install
else
cpptrace:
cpptrace-clean:
endif
