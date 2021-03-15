FROM continuumio/miniconda3
RUN conda install -c conda-forge xrootd
RUN git clone https://gitlab.cern.ch/nihartma/physlite-experiments.git
RUN pip install --no-cache ./physlite-experiments
