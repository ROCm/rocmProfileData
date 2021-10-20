PYTHON ?= python3

.PHONY: install
install:
	$(MAKE) install -C rocpd_python
	$(MAKE) install -C rpd_tracer

.PHONY: uninstall
uninstall:
	$(MAKE) uninstall -C rocpd_python
	$(MAKE) uninstall -C rpd_tracer

rpd:
	$(MAKE) -C rpd_tracer

rocpd:
	$(MAKE) -c rocpd_python

all: rpd rocpd
