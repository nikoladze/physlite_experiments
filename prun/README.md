# prun setup

## create conda environment tarball

```
conda env create -f physlite-experiments_environment.yml -p $(pwd)/conda-env
```

## update package in conda env before submitting

```
conda activate $(pwd)/conda-env
export PYTHONNOUSERSITE=1
python -m pip install ../
tar cfz conda-env.tgz conda-env
mv conda-env.tgz workdir
```

## submit

```
cd workdir
source submit.sh
```
