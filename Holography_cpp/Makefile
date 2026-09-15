CXX ?= g++
CXXFLAGS ?= -std=c++11 -O3 -Wall -Wextra -pedantic

# Intermediate build artifacts stay in this project directory.
BUILD_DIR := build

# `sudo make install` installs commands for all users.
PREFIX ?= /usr/local
INSTALL_DIR ?= $(PREFIX)/bin

ifeq ($(OS),Windows_NT)
  EXEEXT := .exe
else
  EXEEXT :=
endif

# Holography itself remains the Python/Matplotlib implementation (Holography.py).
PROGRAMS := corr_fringe fringe_search group_up_txt scanning_effect
TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix $(EXEEXT),$(PROGRAMS)))

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/corr_fringe$(EXEEXT): corr_fringe.cpp common.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BUILD_DIR)/fringe_search$(EXEEXT): fringe_search.cpp common.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BUILD_DIR)/group_up_txt$(EXEEXT): group_up_txt.cpp common.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BUILD_DIR)/scanning_effect$(EXEEXT): scanning_effect.cpp common.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

install: all
	mkdir -p $(INSTALL_DIR)
	cp $(TARGETS) $(INSTALL_DIR)/
	@echo "Installed commands in $(INSTALL_DIR)"

uninstall:
	rm -f $(addprefix $(INSTALL_DIR)/,$(addsuffix $(EXEEXT),$(PROGRAMS)))

clean:
	rm -f $(TARGETS)

.PHONY: all install uninstall clean
