#!/usr/bin/env python

from to_parquet import to_parquet

def convert_and_upload(xaod_path, prefix):

    filename = xaod_path.split("/")[-1].split(".pool")[0]
    output_filename = prefix+"_"+filename+".parquet"
    to_parquet(xaod_path, output_filename)

    from rucio.client import Client
    from rucio.client.uploadclient import UploadClient
    client = Client()
    upload_client = UploadClient(client)

    # force webdav
    from rucio.rse import rsemanager as rsemgr
    upload_client.rses = {"LRZ-LMU_LOCALGROUPDISK" : rsemgr.get_rse_info("LRZ-LMU_LOCALGROUPDISK")}
    upload_client.rses["LRZ-LMU_LOCALGROUPDISK"]["protocols"] = [
         p for p in upload_client.rses["LRZ-LMU_LOCALGROUPDISK"]["protocols"] if p["scheme"] == "davs"
    ]

    upload_client.upload([dict(
        path=output_filename,
        rse="LRZ-LMU_LOCALGROUPDISK",
        lifetime=3600*24*60,
        did_scope="user.nihartma",
        register_after_upload=True
    )])

    import os
    os.unlink(output_filename)

    return output_filename


if __name__ == "__main__":

    "execute this file in ipython to interactively debug jobs"

    import dask
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    cluster = SLURMCluster()
    client = Client(cluster)
    cluster.adapt(minimum=1, maximum=200)

    with open("replicas_physlite_data.txt") as f:
        paths = [l.strip() for l in f]

    futures = [client.submit(convert_and_upload, p, "testparquet1") for p in paths]
