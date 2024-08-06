
PREFIX = /usr/local

RPD_INCLUDES =
RPDREMOTE_SRCS = Remote.cpp


RPDREMOTE_OBJS = $(RPDREMOTE_SRCS:.cpp=.o)


RPDREMOTE_MAIN = librpd_remote.so


all: | $(RPDREMOTE_MAIN) 

.PHONY: all 


$(RPDREMOTE_MAIN): $(RPDREMOTE_OBJS)
	$(CXX) -o $@ $^ -shared -rdynamic -std=c++11 -g

.cpp.o:
	$(CXX) -o $@ -c $< $(RPD_INCLUDES) -std=c++11 -fPIC -g -O3


.PHONY: install
#install: | $(PREFIX)/lib/lib$(RPDREMOTE_MAIN)
install:
	cp rpdRemote.sh $(PREFIX)/bin/rpdRemote
	cp $(RPDREMOTE_MAIN)  $(PREFIX)/lib/
	ldconfig

.PHONY: uninstall
uninstall:
	rm $(PREFIX)/lib/$(RPDREMOTE_MAIN)

.PHONY: clean
clean:
	rm -f *.o *.so 
