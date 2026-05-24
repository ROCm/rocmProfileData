CXX = /opt/rocm/llvm/bin/clang++
CXXFLAGS = -std=c++11 -g -O2 -I.. -pthread

TESTS = test_seqlock test_measurement test_regression test_convergence

# Firefly.o and Utility.o built from parent directory
PARENT_OBJS = ../Firefly.o ../Utility.o

.PHONY: all test clean

all: $(TESTS)

test: $(TESTS)
	@failed=0; \
	for t in $(TESTS); do \
		echo "--- Running $$t ---"; \
		./$$t || failed=1; \
		echo; \
	done; \
	if [ $$failed -eq 0 ]; then echo "All tests passed."; else echo "SOME TESTS FAILED"; exit 1; fi

test_seqlock: test_seqlock.cpp $(PARENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_measurement: test_measurement.cpp $(PARENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_regression: test_regression.cpp $(PARENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

test_convergence: test_convergence.cpp $(PARENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(PARENT_OBJS):
	$(MAKE) -C .. $(notdir $@)

clean:
	rm -f $(TESTS)
