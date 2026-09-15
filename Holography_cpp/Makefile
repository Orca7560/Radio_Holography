CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic
ifeq ($(OS),Windows_NT)
  EXEEXT := .exe
else
  EXEEXT :=
endif

PROGRAMS := Holography corr_fringe fringe_search group_up_txt scanning_effect
TARGETS := $(addsuffix $(EXEEXT),$(PROGRAMS))

all: $(TARGETS)

Holography$(EXEEXT): Holography.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
corr_fringe$(EXEEXT): corr_fringe.cpp common.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
fringe_search$(EXEEXT): fringe_search.cpp common.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
group_up_txt$(EXEEXT): group_up_txt.cpp common.hpp
	$(CXX) $(CXXFLAGS) $< -o $@
scanning_effect$(EXEEXT): scanning_effect.cpp common.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
