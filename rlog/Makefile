
PREFIX = /usr/local

RLOG_LIBS = 
RLOG_INCLUDES =
RLOG_SRCS = RLogger.cpp

RLOG_OBJS = $(RLOG_SRCS:.cpp=.o)


RLOG_MAIN = librlog.so
RLOG_HEADER = rlog.h

PYTHON = python3
PIP = pip3


all: | $(RLOG_MAIN) 

.PHONY: all 


$(RLOG_MAIN): $(RLOG_OBJS)
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 $(RLOG_LIBS) -g

.cpp.o:
	$(CXX) -o $@ -c $< $(RLOG_INCLUDES) -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g -O3


.PHONY: install
install: all
	cp $(RLOG_MAIN)  $(PREFIX)/lib/
	cp $(RLOG_HEADER)  $(PREFIX)/include/
	ldconfig

.PHONY: uninstall
uninstall:
	rm $(PREFIX)/lib/$(RLOG_MAIN)
.PHONY: clean
clean:
	rm -f *.o *.so 
