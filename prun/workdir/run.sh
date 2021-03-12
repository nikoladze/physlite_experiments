#!/bin/bash

export PATH=$(pwd)/conda-env/bin:$PATH
unset PYTHONPATH
export PYTHONPATH=$(pwd)/xrootd-site-packages

type python

python -m physlite_experiments.scripts.run_analysis_example $1
