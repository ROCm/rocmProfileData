PYTHON = python3
PIP = pip3
PACKAGE = rocpd

install:
	@echo "Building rocpd_python"
	$(PIP) install --user .

develop:
	@echo "Building $(PACKAGE)"
	$(PYTHON) setup.py develop

uninstall:
	@echo "Uninstalling $(PACKAGE)"
	$(PIP) uninstall $(PACKAGE)
	
clean:
	@echo "Cleaning build directory"
	rm -rf build rocpd.egg-info dist
