import os
import awkward as ak
import uproot


def example_file(
    filename="DAOD_PHYSLITE.art_split99.pool.root",
    url="https://cernbox.cern.ch/index.php/s/xTuPIvSJsP3QmXa/download",
):
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}")
        import requests

        res = requests.get(url)
        with open(filename, "wb") as of:
            of.write(res.content)
    return filename


def filter_name(name, verbose=False):

    def filter(name):
        if not "Aux" in name:
            return False

        # the following don't contain data (in split files)
        if name.endswith("."):
            return False
        if "SG::" in name:
            return False
        if name.endswith("Base"):
            return False

        # are often empty
        # see https://github.com/scikit-hep/uproot4/issues/126
        # -> now fixed, but my custom deserialization does not work yet with them
        if "DescrTags" in name:
            return False

        return True

    keep = filter(name)
    if verbose and (not keep):
        print("Skipping", name)

    return keep


def zip_physlite(array_dict):
    regrouped = {}
    for k_top in set(k.split(".")[0] for k in array_dict):
        if k_top == "EventInfoAux":
            # skip that for now - let's use EventInfoAuxDyn
            continue
        if k_top == "EventInfoAuxDyn":
            k_top = "EventInfoAux"
        # zip will put together jagged arrays with common offsets
        def ak_zip(depth_limit=2):
            return ak.zip(
                {k.replace(k_top, "")[1:] : array_dict[k] for k in array_dict if k_top in k},
                depth_limit=depth_limit
            )
        # for some containers this will work 2 levels, for some only up to 1
        try:
            v = ak_zip(depth_limit=2)
        except ValueError:
            v = ak_zip(depth_limit=1)
        regrouped[k_top.replace("Analysis", "").replace("AuxDyn", "").replace("Aux", "")] = v
    # lets restructure such that we get TrigMatchedObjets.<trigger-name>
    # instead of AnalysisHLT_<trigger_name>.TrigMatchedObjects
    trig_matched_objects = ak.zip(
        {
            k.replace("HLT_", "") : regrouped[k].TrigMatchedObjects
            for k in regrouped if "HLT" in k
        },
        depth_limit=1
    )
    for k in list(regrouped.keys()):
        if "HLT" in k:
            regrouped.pop(k)
    regrouped["TrigMatchedObjects"] = trig_matched_objects
    return ak.zip(regrouped, depth_limit=1)


def subdivide(l, n):
    """
    get the number of entries for subdividing l ntries into n approximately
    same sized chunks (like in numpy.array_split)
    """
    return [l // n + 1] * (l % n) + [l // n] * (n - l % n)
