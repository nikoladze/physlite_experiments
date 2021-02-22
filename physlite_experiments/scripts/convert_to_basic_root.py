#!/usr/bin/env python

import ROOT
import uproot
import awkward as ak
import numpy as np
from array import array
from physlite_experiments.deserialization_hacks import tree_arrays
from tqdm import tqdm

typename_dict = {
    'bool' : 'char',
    'float32' : 'float',
    'float64' : 'double',
    'int16' : 'short',
    'int32' : 'int',
    'int8' : 'char',
    'string' : 'string',
    'uint16' : 'unsigned short',
    'uint32' : 'unsigned int',
    'uint64' : 'unsigned long',
    'uint8' : 'unsigned char',
}

letter_dict_root = {
    'bool' : 'B',
    'float32' : 'F',
    'float64' : 'D',
    'int16' : 'S',
    'int32' : 'I',
    'int8' : 'B',
    'uint16' : 's',
    'uint32' : 'i',
    'uint64' : 'l',
    'uint8' : 'b',
}

letter_dict_array = {
    'bool' : 'b',
    'float32' : 'f',
    'float64' : 'd',
    'int16' : 'h',
    'int32' : 'i',
    'int8' : 'B',
    'uint16' : 'H',
    'uint32' : 'I',
    'uint64' : 'L',
    'uint8' : 'b',
}


def filter_branch(branch):
    k = branch.name

    if not "Aux" in k:
        return False

    # the following don't contain data (in split files)
    if k.endswith("."):
        return
    if "SG::" in k:
        return
    if k.endswith("Base"):
        return

    # are often empty
    # see https://github.com/scikit-hep/uproot4/issues/126
    if "DescrTags" in k:
        return

    interpretation = str(branch.interpretation)

    # skip triple-jagged vectors and sets
    if (interpretation.count("AsVector") > 1) and ("AsString" in interpretation):
        return False
    if interpretation.count("AsVector") > 2:
        return False
    if "AsSet" in interpretation:
        # what are these anyways?
        return False

    return True


def read_physlite_flat(rootfile):
    f = uproot.open(rootfile)
    tree = f["CollectionTree"]
    array_dict = tree_arrays(tree, filter_branch=filter_branch)
    d_exploded = {}
    for key, ak_array in array_dict.items():
        keys = ak_array.fields
        if len(keys) == 0:
            d_exploded[key] = ak_array
        for subkey in keys:
            d_exploded[f"{key}.{subkey}"] = ak_array[subkey]
    return d_exploded


def write_branch_dict_root(branch_dict, rootfile, entry_stop=None):

    # allocate vectors and numbers
    branch_objects = {}
    for branch_name, branch_array in branch_dict.items():
        typestr = str(ak.type(branch_array))
        nptype = typestr.split("*")[-1].strip()
        if typestr.count("var") == 2:
            branch_objects[branch_name] = (
                ROOT.std.vector[f"{typename_dict[nptype]}"](),
                ROOT.std.vector[f"vector<{typename_dict[nptype]}>"]()
            )
        elif typestr.count("var") == 1:
            branch_objects[branch_name] = ROOT.std.vector[f"{typename_dict[nptype]}"]()
        else:
            branch_objects[branch_name] = array(f"{letter_dict_array[nptype]}", [0])

    typestrings = [str(ak.type(i)) for i in branch_dict.values()]

    f = ROOT.TFile.Open(rootfile, "RECREATE")
    tree = ROOT.TTree("tree", "tree")

    # create branches
    for branch_name, branch_array in branch_dict.items():
        typestr = str(ak.type(branch_array))
        nptype = typestr.split("*")[-1].strip()
        if "var" in typestr:
            o = branch_objects[branch_name]
            if isinstance(o, tuple):
                o = o[1]
            tree.Branch(branch_name, o)
        else:
            tree.Branch(
                branch_name,
                branch_objects[branch_name],
                f"{branch_name}/{letter_dict_root[nptype]}"
            )

    # loop over entries and fill tree
    nevents = len(next(iter(branch_dict.values())))
    for e in tqdm(range(nevents) if entry_stop is None else range(entry_stop)):
        for ts, [bname, bval] in zip(typestrings, branch_dict.items()):
            if ts.count("var") == 1:
                branch_objects[bname].clear()
                for v in bval[e]:
                    branch_objects[bname].push_back(v)
            elif ts.count("var") == 2:
                tmpvec, vec = branch_objects[bname]
                vec.clear()
                for i in bval[e]:
                    tmpvec.clear()
                    for j in i:
                        tmpvec.push_back(j)
                        vec.push_back(tmpvec)
            else:
                branch_objects[bname][0] = bval[e]
        tree.Fill()

    tree.Write()
    f.Close()

BUFSIZE = 10000

def write_branch_dict_root_flat(branch_dict, rootfile, entry_stop=None):

    branch_objects = {}
    for branch_name, branch_array in branch_dict.items():
        typestr = str(ak.type(branch_array))
        nptype = typestr.split("*")[-1].strip()
        branch_objects[branch_name] = {}
        if typestr.count("var") == 2:
            branch_objects[branch_name]["n"] = array(f"i", [0])
            branch_objects[branch_name]["n1"] = np.empty(BUFSIZE, dtype=np.int32)
            branch_objects[branch_name]["data"] = np.empty(BUFSIZE, dtype=getattr(np, nptype))
        elif typestr.count("var") == 1:
            branch_objects[branch_name]["n"] = array(f"i", [0])
            branch_objects[branch_name]["n1"] = np.empty(BUFSIZE, dtype=np.int32)
            branch_objects[branch_name]["data"] = np.empty(BUFSIZE, dtype=getattr(np, nptype))
        else:
            branch_objects[branch_name]["data"] = array(f"{letter_dict_array[nptype]}", [0])


    # TODO: continue here

    typestrings = [str(ak.type(i)) for i in branch_dict.values()]

    f = ROOT.TFile.Open(rootfile, "RECREATE")
    tree = ROOT.TTree("tree", "tree")

    # create branches
    for branch_name, branch_array in branch_dict.items():
        typestr = str(ak.type(branch_array))
        nptype = typestr.split("*")[-1].strip()
        if "var" in typestr:
            o = branch_objects[branch_name]
            if isinstance(o, tuple):
                o = o[1]
            tree.Branch(branch_name, o)
        else:
            tree.Branch(
                branch_name,
                branch_objects[branch_name],
                f"{branch_name}/{letter_dict_root[nptype]}"
            )

    # loop over entries and fill tree
    nevents = len(next(iter(branch_dict.values())))
    for e in tqdm(range(nevents) if entry_stop is None else range(entry_stop)):
        for ts, [bname, bval] in zip(typestrings, branch_dict.items()):
            if ts.count("var") == 1:
                branch_objects[bname].clear()
                for v in bval[e]:
                    branch_objects[bname].push_back(v)
            elif ts.count("var") == 2:
                tmpvec, vec = branch_objects[bname]
                vec.clear()
                for i in bval[e]:
                    tmpvec.clear()
                    for j in i:
                        tmpvec.push_back(j)
                        vec.push_back(tmpvec)
            else:
                branch_objects[bname][0] = bval[e]
        tree.Fill()

    tree.Write()
    f.Close()



if __name__ == "__main__":

    # branch_dict = read_physlite_flat("user.nihartma.22884623.EXT0._000001.DAOD_PHYSLITE.test.pool.root")
    # write_branch_dict_root(branch_dict, "physlite_flat.root")

    pass
