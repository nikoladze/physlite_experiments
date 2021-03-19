#!/usr/bin/env python

import json
import uproot
import math
import awkward as ak

from physlite_experiments.physlite_events import (
    physlite_events, get_lazy_form, get_branch_forms, Factory, LazyGet, from_parquet
)
from physlite_experiments.analysis_example import get_obj_sel
from physlite_experiments.utils import subdivide


def run(filename, max_chunksize=10000):
    output = {
        collection: {
            flag : 0
            for flag in ["baseline", "passOR", "signal"]
        } for collection in ["Electrons", "Muons", "Jets"]
    }
    nevents = 0
    with uproot.open(
        f"{filename}:CollectionTree",
        # vector reads are currently memory leaking
        # see https://github.com/xrootd/xrootd/issues/1400
        # probably not much an issue for this script
        # but in case this becomes problematic fall back to MultithreadedXRootDSource for now
        #xrootd_handler=uproot.MultithreadedXRootDSource
        xrootd_handler=uproot.XRootDSource
    ) as tree:
        if max_chunksize is not None and tree.num_entries > max_chunksize:
            n_chunks = math.ceil(tree.num_entries / max_chunksize)
        else:
            n_chunks = 1
        form = json.dumps(get_lazy_form(get_branch_forms(tree)))
        entry_start = 0
        for num_entries in subdivide(tree.num_entries, n_chunks):
            print("Processing", num_entries, "entries")
            entry_stop = entry_start + num_entries
            cache = {}
            container = LazyGet(
                tree, entry_start=entry_start, entry_stop=entry_stop,
                cache=cache
            )
            factory = Factory(form, entry_stop - entry_start, container)
            events = factory.events
            events_decorated = get_obj_sel(events)
            entry_start = entry_stop
            for collection in output:
                for flag in output[collection]:
                    output[collection][flag] += ak.count_nonzero(
                        events_decorated[collection][flag]
                    )
            nevents += len(events)
    return output, nevents


def run_parquet(filename):
    import pyarrow.parquet as pq

    output = {
        collection: {
            flag : 0
            for flag in ["baseline", "passOR", "signal"]
        } for collection in ["Electrons", "Muons", "Jets"]
    }
    nevents = 0
    f = pq.ParquetFile(filename)
    for row_group in range(f.num_row_groups):
        print("Processing row group", row_group)
        events = from_parquet(filename, row_groups=row_group)
        events_decorated = get_obj_sel(events)
        for collection in output:
            for flag in output[collection]:
                output[collection][flag] += ak.count_nonzero(
                    events_decorated[collection][flag]
                )
        print("Contained", len(events_decorated), "events")
        nevents += len(events_decorated)

    return output, nevents



if __name__ == "__main__":

    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("input_files")
    parser.add_argument("--max-chunksize", help="only for root files", type=int)
    args = parser.parse_args()

    for filename in args.input_files.split(","):
        print("Processing", filename)
        start = time.time()
        if filename.endswith(".parquet"):
            print(run_parquet(filename))
        else:
            print(run(filename, max_chunksize=args.max_chunksize))
        print(f"Took {time.time() - start:.2f} seconds")
