PYTHON ?= python3

.PHONY:
all: rpd rocpd remote

.PHONY: install
install: all
	$(MAKE) install -C rocpd_python
	$(MAKE) install -C rpd_tracer
	$(MAKE) install -C remote

.PHONY: uninstall
uninstall:
	$(MAKE) uninstall -C rocpd_python
	$(MAKE) uninstall -C rpd_tracer
	$(MAKE) uninstall -C remote

.PHONY: clean
clean:
	$(MAKE) clean -C rocpd_python
	$(MAKE) clean -C rpd_tracer
	$(MAKE) clean -C remote

.PHONY: rpd
rpd:
	$(MAKE) -C rpd_tracer
.PHONY: rocpd
rocpd:
	$(MAKE) -C rocpd_python
.PHONY: remote
remote:
	$(MAKE) -C remote 
