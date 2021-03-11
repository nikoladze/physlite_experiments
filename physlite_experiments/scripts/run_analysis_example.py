#!/usr/bin/env python

import json
import uproot
import math
import awkward as ak

from physlite_experiments.physlite_events import (
    physlite_events, get_lazy_form, get_branch_forms, Factory, LazyGet
)
from physlite_experiments.analysis_example import get_obj_sel


def subdivide(l, n):
    """
    get the number of entries for subdividing l ntries into n approximately
    same sized chunks (like in numpy.array_split)
    """
    return [l // n + 1] * (l % n) + [l // n] * (n - l % n)


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
        array_cache=None,
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
            container = LazyGet(
                tree, entry_start=entry_start, entry_stop=entry_stop
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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_files")
    args = parser.parse_args()

    for filename in args.input_files.split(","):
        print(run(filename, max_chunksize=50000))
