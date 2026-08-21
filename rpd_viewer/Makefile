PYTHON = python3
PIP = pip3
PACKAGE = rpd-viewer

install:
	@echo "Building rpd_dash"
	pip install .

develop:
	@echo "Building $(PACKAGE)"
	$(PYTHON) setup.py develop

uninstall:
	@echo "Uninstalling $(PACKAGE)"
	$(PIP) uninstall $(PACKAGE)

clean:
	@echo "Cleaning build directory"
	rm -rf build rpd_viewer.egg-info dist
