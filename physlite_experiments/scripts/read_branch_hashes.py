#!/usr/bin/env python3

"""
Read and printout hash to branchname mapping (used for ElementLinks) for the
EventFormat branch in the MetaData tree of xAOD files. This branch is stored
with memberwise splitting and can currently (Dec 2022) not yet be read with
uproot.

See https://github.com/scikit-hep/uproot5/issues/38
"""

import sys
import json
import uproot
import numpy as np

def read_strings(data, start=6):
    ns = np.frombuffer(data[start: start + 4].tobytes(), dtype=">i4")[0]
    pos = start + 4
    strings = []
    for i in range(ns):
        nc = data[pos]
        pos += 1
        strings.append(data[pos: pos + nc].tobytes())
        pos += nc
    return strings, pos


def hash_dict(data):
    branch_names, p = read_strings(data, start=6)
    container_names, p = read_strings(data, start=p + 6)
    _, p = read_strings(data, start=p + 6)
    hashes = np.frombuffer(data[p + 10:].tobytes(), dtype=">u4")
    return dict(zip(hashes, [i.decode() for i in branch_names]))


if __name__ == "__main__":
    rootfile = sys.argv[1]
    with uproot.open(rootfile) as f:
        metadata = f["MetaData"]
        branchname = [k for k in metadata.keys() if "EventFormat" in k][0]
        data = metadata[branchname].basket(0).data
    print(json.dumps({int(k): str(v) for k, v in hash_dict(data).items()}, indent=4))
