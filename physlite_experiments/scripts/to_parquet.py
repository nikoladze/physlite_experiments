#!/usr/bin/env python

import math
import awkward as ak
import uproot
from physlite_experiments.utils import filter_name, zip_physlite, subdivide

# this hack won't be needed anymore when uproot uses awkward forth
# see https://github.com/scikit-hep/awkward-1.0/pull/661
from physlite_experiments.deserialization_hacks import tree_arrays


def to_ak(root_filename, zip=False, verbose=False):
    if verbose:
        new_filter = lambda name: filter_name(name, verbose=True)
    else:
        new_filter = filter_name
    with uproot.open(f"{root_filename}:CollectionTree") as tree:
        array = tree_arrays(tree, filter_name=new_filter)
        if zip:
            return zip_physlite(array)
        else:
            return ak.zip(array, depth_limit=1)


def to_parquet(
        input_daod,
        output_parquet,
        zip=False,
        verbose=False,
        max_partition_size=None,
        **kwargs
):
    entry_stop = kwargs.pop("entry_stop", None)
    ar = to_ak(input_daod, zip=zip, verbose=verbose)
    if entry_stop is not None:
        ar = ar[:entry_stop]
    if max_partition_size is not None and len(ar) > max_partition_size:
        npartitions = math.ceil(len(ar) / max_partition_size)
        ar = ak.repartition(ar, subdivide(len(ar), npartitions))
    ak.to_parquet(ar, output_parquet, **kwargs)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Convert \"almost all\" Aux branches in DAOD_PHYSLITE to parquet")
    parser.add_argument("input_daod", help="input daod path")
    parser.add_argument("output_parquet", help="output parquet filename/path")
    parser.add_argument("--zip", action="store_true", help="zip Collections (e.g. group Electrons, Muons, Jets)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print all skipped branches")
    parser.add_argument("--entry-stop", help="only convert up to this number of entries", type=int)
    parser.add_argument("--max-partition-size", help="subdivide into row groups of maximum this size", type=int)
    args = parser.parse_args()

    to_parquet(**vars(args))
