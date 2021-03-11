#!/bin/bash

tar xfz conda-env.tgz

export PATH=$(pwd)/conda-env/bin:$PATH
unset PYTHONPATH

type python

python -m physlite_experiments.scripts.run_analysis_example $1
