"""
This is an quick & dirty attempt to create a lazy-loadable awkward array for DAOD_PHYSLITE,
encapsulating the full event data model, similar to coffea NanoEvents
(see https://github.com/CoffeaTeam/coffea/tree/master/coffea/nanoevents)
"""

from copy import deepcopy
import json
import uproot
from uproot.interpretation.objects import CannotBeAwkward
import awkward as ak
import numpy as np
from deserialization_hacks import branch_to_array
from behavior import xAODParticle, xAODTrackParticle

behavior_dict = {
    "Electrons": "xAODElectron",
    "Muons": "xAODMuon",
    "Jets": "xAODParticle",
    "TauJets" : "xAODParticle",
    "CombinedMuonTrackParticles": "xAODTrackParticle",
    "ExtrapolatedMuonTrackParticles": "xAODTrackParticle",
    "GSFTrackParticles": "xAODTrackParticle",
    "InDetTrackParticles": "xAODTrackParticle",
    "MuonSpectrometerTrackParticle": "xAODTrackParticle",
    "TruthBoson" : "xAODParticle",
    "TruthBosonsWithDecayParticles" : "xAODParticle",
    "TruthBosonsWithDecayVertices" : "xAODParticle",
    "TruthBottom" : "xAODParticle",
    "TruthElectrons" : "xAODParticle",
    "TruthEvents" : "xAODParticle",
    "TruthMuons" : "xAODParticle",
    "TruthNeutrinos" : "xAODParticle",
    "TruthPhotons" : "xAODParticle",
    "TruthPrimaryVertices" : "xAODParticle",
    "TruthTop" : "xAODParticle",
    "TruthTaus" : "xAODParticle",
    "TruthForwardProtons" : "xAODParticle",
}


def get_branch_names():

    branch_names = {}

    # TODO: parse this from MetaData/EventFormat once member wise splitting can
    # be read from uproot
    with open("branch_names_hashes_log.txt") as f:
        for l in f:
            l = l.strip()
            if l.startswith("**"):
                continue
            if not l.startswith("*"):
                continue
            fields = l.split()
            if fields[1] == "Row":
                continue
            branch_name, branch_hash = fields[5], fields[7]
            branch_names[int(branch_hash)] = branch_name

    return branch_names


def get_branch_forms(uproot_tree):
    forms = {}

    def add(key, branch):
        try:
            ak_form = branch.interpretation.awkward_form(None)
            forms[key] = ak_form
        except CannotBeAwkward:
            print("Can't interpret", key)

    # TODO: include the non-"Dyn"-branches - like MET association
    for key, branch in uproot_tree.iteritems(filter_name="*AuxDyn.*"):
        if "/" in key:
            continue
        if len(branch.branches) == 0:
            add(key, branch)
        else:
            for sub_branch in branch.branches:
                sub_key = sub_branch.name
                add(sub_key, sub_branch)
    return forms


def get_lazy_form(branch_forms):
    def apply(parent_form, form, key, root_key):
        form = deepcopy(form)
        if form["class"] == "ListOffsetArray64":
            apply(form, form["content"], "content", f"{root_key}%content")
            form["form_key"] = f"{root_key}%content%offsets"
        elif form["class"] == "RecordArray":
            for field in form["contents"]:
                apply(
                    form["contents"],
                    form["contents"][field],
                    field,
                    f"{root_key}%{field}",
                )
        else:
            form["form_key"] = f"{root_key}%content"
        parent_form[key] = form

    form = {"class": "RecordArray", "contents": {}}

    for key, ak_form in branch_forms.items():
        key_fields = key.split(".")
        top_key = key_fields[0]
        sub_key = ".".join(key_fields[1:])
        ak_top_key = top_key.replace("Analysis", "").replace("AuxDyn", "")
        form_dict = json.loads(ak_form.tojson())

        # top level will be "zipped"
        if form_dict["class"] == "ListOffsetArray64":
            if not ak_top_key in form["contents"]:
                parameters = {}
                form["contents"][ak_top_key] = {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": {
                        "class": "RecordArray",
                        "contents": {},
                        "parameters": parameters,
                    },
                    "form_key": f"{key}%offsets",
                }
                if ak_top_key in behavior_dict:
                    form["contents"][ak_top_key]["parameters"] = {
                        "__record__": behavior_dict[ak_top_key],
                    }
                    parameters["__record__"] = behavior_dict[ak_top_key]
            apply(
                form["contents"][ak_top_key]["content"]["contents"],
                form_dict["content"],
                sub_key,
                key,
            )
        else:
            # TODO non-nested collections like EventInfo
            ...

    return form


class LazyGet:
    def __init__(self, tree, verbose=False):
        self.tree = tree
        self.verbose = verbose

    def __getitem__(self, key):
        if self.verbose:
            print("Loading", key)
        part, key, component = key.split("-")
        attrs = key.split("%")
        key = attrs[0]
        attrs = attrs[1:]
        ar = branch_to_array(self.tree[key]).layout
        for attr in attrs:
            if attr in ["content", "offsets"]:
                ar = getattr(ar, attr)
            else:
                ar = ar[attr]
        return np.array(ar)


def physlite_events(uproot_tree, json_form=None, verbose=False):
    if json_form is None:
        form = get_lazy_form(get_branch_forms(uproot_tree))
        json_form = json.dumps(form)
    ar_container = [0]
    ar = ak.from_buffers(
        json_form,
        uproot_tree.num_entries,
        LazyGet(uproot_tree, verbose=verbose),
        lazy=True,
        behavior={"__events__": ar_container},
    )
    ar.branch_names = get_branch_names()
    ar_container[0] = ar
    return ar


if __name__ == "__main__":

    from importlib import reload
    import behavior

    f = uproot.open("user.nihartma.22884623.EXT0._000001.DAOD_PHYSLITE.test.pool.root")
    tree = f["CollectionTree"]
    events = physlite_events(tree, verbose=True)

    # TODO: try out `to_buffers` to load everything and see what goes wrong
