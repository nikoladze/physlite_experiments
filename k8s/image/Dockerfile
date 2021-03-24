FROM centos:7

RUN yum install -y epel-release.noarch && \
    yum clean all && \
    rm -rf /var/cache/yum
RUN yum upgrade -y && \
    yum clean all && \
    rm -rf /var/cache/yum

RUN yum -y install https://repo.ius.io/ius-release-el7.rpm && \
    yum install -y python36u-pip voms-clients-java gfal2-all gfal2-util gfal2-python3 xrootd-client\
                   nordugrid-arc-client nordugrid-arc-plugins-gfal \
                   nordugrid-arc-plugins-globus nordugrid-arc-plugins-s3 \
                   nordugrid-arc-plugins-xrootd && \
    yum clean all && \
    rm -rf /var/cache/yum

RUN yum install -y which openssh-clients xrootd xrootd-client cmake3 openssl python3-devel openssl-devel uuid-devel uuid libuuid-devel gcc-c++ && \
    yum clean all && \
    rm -rf /var/cache/yum

# Upgrade pip & setuptools and install Rucio
RUN python3.6 -m pip install --no-cache-dir --upgrade pip && \
    python3.6 -m pip install --no-cache-dir --upgrade setuptools && \
    python3.6 -m pip install --no-cache-dir --pre rucio-clients && \
    python3.6 -m pip install --no-cache-dir jinja2 j2cli pyyaml


RUN python3.6 -m pip install --no-cache-dir --upgrade wheel setuptools
RUN python3.6 -m pip install --no-cache-dir --upgrade xrootd uproot

RUN python3.6 -m pip install --no-cache-dir --upgrade dask distributed jupyterlab

COPY k8s/image/init.sh /init.sh

COPY k8s/image/grid-security /etc/grid-security
COPY k8s/image/vomses /etc/vomses
COPY k8s/image/rucio.cfg /opt/rucio/etc/rucio.cfg
COPY k8s/image/ca.crt /opt/rucio/etc/rucio_ca.crt


RUN chmod +x /init.sh
ENTRYPOINT ["/init.sh"]

RUN useradd jovyan
USER jovyan
WORKDIR /home/jovyan

ENV PATH /home/jovyan/.local:$PATH
ENV PYTHONPATH /home/jovyan/.local/lib/python3.6/site-packages:$PYTHONPATH

COPY --chown=jovyan . physlite-experiments
RUN cd physlite-experiments; pip3 install --user --editable .
COPY k8s/image/dask_test_k8s.ipynb dask_test_k8s.ipynb

CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
