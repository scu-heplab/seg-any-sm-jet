CXX = g++
CXXFLAGS_BASE = -std=c++11 -O2 -Wall -fPIC

TARGET = sajm.out
SRCS = main.cpp extract.cpp
OBJS = $(SRCS:.cpp=.o)

DELPHES_DIR ?=
ifeq ($(DELPHES_DIR),)
    $(error DELPHES_DIR is not set. Please set it via environment variable or command line.)
endif
ifneq ("$(wildcard $(DELPHES_DIR)/libDelphes.so)","")
    # All good, Delphes found.
else
    $(error DELPHES_DIR is set to '$(DELPHES_DIR)', but libDelphes.so was not found there. Please check the path.)
endif

ROOT_CFLAGS = $(shell root-config --cflags)
ROOT_LIBS = $(shell root-config --libs)
DELPHES_CFLAGS = -I$(DELPHES_DIR) -I$(DELPHES_DIR)/external
DELPHES_LDFLAGS = -L$(DELPHES_DIR)
DELPHES_LIBS = -lDelphes

CXXFLAGS = $(CXXFLAGS_BASE) $(ROOT_CFLAGS) $(DELPHES_CFLAGS)
LDFLAGS = $(DELPHES_LDFLAGS)
LIBS = $(ROOT_LIBS) $(DELPHES_LIBS)


all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "==> Linking executable: $@"
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS) && \
	echo "==> Cleaning up intermediate object files..." && \
	rm -f $(OBJS)
	@echo "==> Build successful! Directory is clean. Executable: './$(TARGET)'"

%.o: %.cpp
	@echo "==> Compiling: $<"
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	@echo "==> Cleaning up build files..."
	rm -f $(OBJS) $(TARGET)
	@echo "==> Done."

.PHONY: all clean