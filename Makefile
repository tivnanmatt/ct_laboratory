# Variables
PYTHON = python
BUILD_DIR = build
SRC_DIR = src

# Automatically detect Conda-installed GCC/G++
CUDA_HOME ?= $(shell dirname $(dirname $(which nvcc)))
CC ?= $(shell which gcc)
CXX ?= $(shell which g++)

# Add torch lib path to LD_LIBRARY_PATH
export LD_LIBRARY_PATH := /home/staticct/miniconda3/envs/ct_lab/lib/python3.12/site-packages/torch/lib:$(LD_LIBRARY_PATH)

all: build develop

build:
	mkdir -p $(BUILD_DIR)
	CUDA_HOME=$(CUDA_HOME) CC=$(CC) CXX=$(CXX) $(PYTHON) setup.py build

develop:
	CUDA_HOME=$(CUDA_HOME) CC=$(CC) CXX=$(CXX) $(PYTHON) setup.py develop --user

install:
	CUDA_HOME=$(CUDA_HOME) CC=$(CC) CXX=$(CXX) $(PYTHON) setup.py install --user

clean:
	rm -rf $(BUILD_DIR) build dist *.egg-info
	$(PYTHON) -m pip uninstall ct_laboratory -y || true
