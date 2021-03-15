FROM continuumio/miniconda3
RUN conda install -c conda-forge xrootd
ADD . ./physlite-experiments
RUN pip install --no-cache ./physlite-experiments
