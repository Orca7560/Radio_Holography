CXX ?= g++
CXXFLAGS ?= -std=c++11 -O3 -Wall -Wextra -pedantic
BIN_DIR := bin
PREFIX ?= $(HOME)/.local
INSTALL_DIR ?= $(PREFIX)/bin
ifeq ($(OS),Windows_NT)
  EXEEXT := .exe
else
  EXEEXT :=
endif

# Holography itself remains the Python/Matplotlib implementation (Holography_v9.py).
PROGRAMS := corr_fringe fringe_search group_up_txt scanning_effect
TARGETS := $(addprefix $(BIN_DIR)/,$(addsuffix $(EXEEXT),$(PROGRAMS)))

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/corr_fringe$(EXEEXT): corr_fringe.cpp common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BIN_DIR)/fringe_search$(EXEEXT): fringe_search.cpp common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BIN_DIR)/group_up_txt$(EXEEXT): group_up_txt.cpp common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BIN_DIR)/scanning_effect$(EXEEXT): scanning_effect.cpp common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

install: all
	mkdir -p $(INSTALL_DIR)
	cp $(TARGETS) $(INSTALL_DIR)/
	@echo "Installed commands in $(INSTALL_DIR)"
	@echo "Ensure $(INSTALL_DIR) is included in PATH."

uninstall:
	rm -f $(addprefix $(INSTALL_DIR)/,$(addsuffix $(EXEEXT),$(PROGRAMS)))

clean:
	rm -f $(TARGETS)

.PHONY: all install uninstall clean
