# Experiments with columnar data analysis of DAOD_PHYSLITE

This is a collection of experiments, prototypes and proof-of-concept code to do columnar data analysis with DAOD_PHYSLITE in python.

## Setup
Install via (e.g in a virtual environment)

```
git clone https://gitlab.cern.ch/nihartma/physlite-experiments.git
cd physlite-experiments
pip install -e ./
```

## Table of content

Some notable scripts, modules and notebooks:

* [physlite_experiments.ipynb](notebooks/physlite_experiments.ipynb): Summary of the current status of what can be done in terms reading data from ROOT files, converting to other formats and representing as awkward array
* [physlite_events.py](physlite_experiments/physlite_events.py): Attempt to mimic [coffea NanoEvents](https://github.com/CoffeaTeam/coffea/tree/master/coffea/nanoevents).
* [coffea_nanoevents.ipynb](notebooks/coffea_nanoevents.ipynb): Demonstration of the PHYSLITE schema implemented within coffea. This should make coffea NanoEvents work with PHYSLITE.
* [analysis_example.py](physlite_experiments/analysis_example.py): Example analysis, trying to reproduce object selections for Electrons, Muons and Jets to compare to a SUSYTools analysis.
* [columnar_vs_st.ipynb](notebooks/columnar_vs_st.ipynb): Notebook for validating that analysis and running a few timing studies.
* [to_parquet.py](physlite_experiments/scripts/to_parquet.py): Script to convert DAOD_PHYSLITE Aux branches to parquet
* [convert_to_basic_root.py](physlite_experiments/scripts/convert_to_basic_root.py): Scripts to convert DAOD_PHYSLITE Aux branches into more basic ROOT formats (e.g. without any custom classes or with only one level of jagged plain arrays)
* [proper_xrdfile.py](physlite_experiments/proper_xrdfile.py): Example for reading a parquet file via xrootd
* [prun](prun): Example script for submission to PanDA (using a conda environment tarball)
* [k8s](k8s): Setup for running the analysis example on a kubernetes-deployed dask cluster

## Submit to panda via container on cvmfs

Build and push new container image

```
docker build -t nikolaihartmann/physlite-experiments:latest .
docker image push nikolaihartmann/physlite-experiments:latest
```

Example for prun submission

```
prun --containerImage /cvmfs/unpacked.cern.ch/registry.hub.docker.com/nikolaihartmann/physlite-experiments:latest \
     --exec="python -m physlite_experiments.scripts.run_analysis_example %IN" \
     --inDS="<input-dataset-name>" \
     --outDS="<output-dataset-name>"
```


