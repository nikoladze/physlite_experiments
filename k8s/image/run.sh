docker run --rm -it \
-e RUCIO_ACCOUNT=lheinric \
-e X509_USER_CERT=$PWD/globus/usercert.pem \
-e X509_USER_KEY=$PWD/globus/userkey_decrypted.pem \
-e JUPYTER_TOKEN=dask \
-v $PWD/globus:$PWD/globus \
-p 10888:8888 \
-it lukasheinrich/atlas100daskworker \
#  bash
# jupyter lab --ip 0.0.0.0
# sh -c "voms-proxy-init --voms atlas; python3 -c 'import uproot; print(uproot.open(\"root://lcg-lrz-rootd.grid.lrz.de:1094/pnfs/lrz-muenchen.de/data/atlas/dq2/atlaslocalgroupdisk/rucio/data17_13TeV/26/59/DAOD_PHYSLITE.22958105._000001.pool.root.1\"))'"
 