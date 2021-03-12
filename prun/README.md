# prun setup

## create conda environment tarball

```
conda env create -f physlite-experiments_environment.yml -p $(pwd)/workdir/conda-env
```

## update package in conda env before submitting

```
conda activate $(pwd)/workdir/conda-env
export PYTHONNOUSERSITE=1
python -m pip install ../
```

## create tarball

```
pushd workdir && tar cfz ../tarball.tgz * && popd
```

## submit

```
source submit.sh
```
