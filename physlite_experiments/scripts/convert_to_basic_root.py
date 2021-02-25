#!/usr/bin/env python

import ROOT
import uproot
import awkward as ak
import numpy as np
from array import array
from tqdm import tqdm
import warnings
from physlite_experiments.deserialization_hacks import tree_arrays
from physlite_experiments.utils import example_file

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

    if "-" in k:
        # we want to do MakeClass later, which can't deal with that
        return False

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


def write_branch_dict_root_flat(branch_dict, rootfile, entry_stop=None, lz4=False):

    if lz4:
        f = ROOT.TFile.Open(rootfile, "RECREATE", "", ROOT.CompressionSettings(ROOT.kLZ4, 1))
    else:
        f = ROOT.TFile.Open(rootfile, "RECREATE")
    tree = ROOT.TTree("tree", "tree")
    tree.SetAutoFlush(3 * 1024 ** 3)

    BUFSIZE = 10000

    # create objects and branches
    count_branches = {}
    branch_objects = {}
    for branch_name, branch_array in branch_dict.items():
        typestr = str(ak.type(branch_array))
        nptype = typestr.split("*")[-1].strip()
        if nptype == "string":
            warnings.warn(f"Skipping {branch_name} of type {nptype}")
            continue
        if branch_name in [
            "xTrigDecisionAux.lvl2PassedPhysics",
            "xTrigDecisionAux.efPassedPhysics",
            "xTrigDecisionAux.lvl2PassedRaw",
            "xTrigDecisionAux.efPassedRaw",
            "xTrigDecisionAux.lvl2PassedThrough",
            "xTrigDecisionAux.efPassedThrough",
            "xTrigDecisionAux.lvl2Prescaled",
            "xTrigDecisionAux.efPrescaled",
            "xTrigDecisionAux.lvl2Resurrected",
            "xTrigDecisionAux.efResurrected",
            "EventInfoAuxDyn.mcEventWeights",
        ]:
            warnings.warn(f"Skipping {branch_name} (doesn't fit naming scheme)")
            continue
        branch_objects[branch_name] = {}
        if typestr.count("var") == 2:
            branch_objects[branch_name]["nm"] = array(f"i", [0])
            branch_objects[branch_name]["n"] = array(f"i", [0])
            branch_objects[branch_name]["m"] = np.empty(BUFSIZE, dtype=np.int32)
            branch_objects[branch_name]["data"] = np.empty(BUFSIZE, dtype=getattr(np, nptype))
        elif typestr.count("var") == 1:
            branch_objects[branch_name]["n"] = array(f"i", [0])
            branch_objects[branch_name]["data"] = np.empty(BUFSIZE, dtype=getattr(np, nptype))
        else:
            branch_objects[branch_name]["data"] = array(f"{letter_dict_array[nptype]}", [0])

        if "var" in typestr:
            count_branch = branch_name.split(".")[0]
            if not count_branch in count_branches:
                # n-entries level 1
                tree.Branch(
                    f"n{count_branch}",
                    branch_objects[branch_name]["n"],
                    f"n{count_branch}/I",
                )
                count_branches[count_branch] = branch_objects[branch_name]["n"]
            if typestr.count("var") == 1:
                tree.Branch(
                    branch_name,
                    branch_objects[branch_name]["data"],
                    f"{branch_name}[n{count_branch}]/{letter_dict_root[nptype]}"
                )
            if typestr.count("var") == 2:
                # n-entries level 1 (flattened)
                tree.Branch(
                    f"n{branch_name}",
                    branch_objects[branch_name]["nm"],
                    f"n{branch_name}/I",
                )
                # data (also flattened)
                tree.Branch(
                    f"{branch_name}",
                    branch_objects[branch_name]["data"],
                    f"{branch_name}[n{branch_name}]/{letter_dict_root[nptype]}"
                )
                # n-entries level 2
                # (length of count branch)
                tree.Branch(
                    f"m{branch_name}",
                    branch_objects[branch_name]["m"],
                    f"m{branch_name}[n{count_branch}]/I",
                )
        else:
            tree.Branch(
                f"{branch_name}",
                branch_objects[branch_name]["data"],
                f"{branch_name}/{letter_dict_root[nptype]}"
            )

    tree.SetBasketSize("*", 1024 ** 3)

    # loop over entries and fill tree
    nevents = len(next(iter(branch_dict.values())))
    typestrings = [str(ak.type(i)) for i in branch_dict.values()]
    for e in tqdm(range(nevents) if entry_stop is None else range(entry_stop)):
        for ts, [bname, bval] in zip(typestrings, branch_dict.items()):
            if not bname in branch_objects:
                continue
            event = bval[e]
            if "var" in ts:
                count_branch = bname.split(".")[0]
                n = len(event)
                count_branches[count_branch][0] = n
            if ts.count("var") == 1:
                branch_objects[bname]["data"][:n] = ak.to_numpy(event)
            elif ts.count("var") == 2:
                flat = ak.to_numpy(ak.flatten(event))
                nm = len(flat)
                branch_objects[bname]["nm"][0] = nm
                branch_objects[bname]["data"][:nm] = flat
                branch_objects[bname]["m"][:n] = ak.to_numpy(ak.num(event))
            else:
                branch_objects[bname]["data"][0] = event
        tree.Fill()

    tree.Write()
    f.Close()

    return tree


def unflatten_double_jagged(branch_dict, key):
    m = branch_dict["m" + key]
    c = branch_dict[key]
    return ak.Array(
        ak.layout.ListOffsetArray64(
            m.layout.offsets,
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(
                    np.cumsum(np.append([0], m.layout.content))
                ),
                c.layout.content
            )
        )
    )


def unflatten(branch_dict):
    branch_dict = dict(branch_dict)
    for mkey in [k for k in branch_dict if k.startswith("m")]:
        akey = mkey[1:]
        branch_dict[akey] = unflatten_double_jagged(branch_dict, akey)
    return branch_dict


def test_flat_root(tmpdir):
    branch_dict = read_physlite_flat(example_file())
    root_path = str(tmpdir / "test.root")
    write_branch_dict_root_flat(branch_dict, root_path)
    with uproot.open(f"{root_path}:tree") as tree:
        new_branch_dict = {k : v.array() for k, v in tree.iteritems()}
    new_branch_dict = unflatten(new_branch_dict)
    for k in branch_dict:
        if not k in new_branch_dict:
            continue
        try:
            assert ak.all(branch_dict[k] == new_branch_dict[k])
        except:
            print(k)
            raise


if __name__ == "__main__":

    pass

    # branch_dict = read_physlite_flat("user.nihartma.22884623.EXT0._000001.DAOD_PHYSLITE.test.pool.root")
    # # for nentries in [1250, 2500, 5000, 10000]:
    # #     output_file = f"physlite_flat_lz4_{nentries}.root"
    # #     print(f"Writing {output_file}")
    # #     write_branch_dict_root_flat(branch_dict, output_file, entry_stop=nentries, lz4=True)
    # write_branch_dict_root_flat(branch_dict, "test.root", entry_stop=10)

    branch_dict = read_physlite_flat(example_file())
    write_branch_dict_root_flat(branch_dict, "test.root")
    tree = uproot.open("test.root:tree")
    arrays = {k: v.array() for k, v in tree.iteritems()}
    arrays_unflat = unflatten(arrays)
