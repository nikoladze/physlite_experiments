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
    "Electrons": "xAODParticle",
    "Muons": "xAODParticle",
    "Jets": "xAODParticle",
    "CombinedMuonTrackParticles": "xAODTrackParticle",
    "ExtrapolatedMuonTrackParticles": "xAODTrackParticle",
    "GSFTrackParticles": "xAODTrackParticle",
    "InDetTrackParticles": "xAODTrackParticle",
    "MuonSpectrometerTrackParticle": "xAODTrackParticle",
}


def get_branch_forms(uproot_tree):
    forms = {}
    # TODO: include the non-"Dyn"-branches - like MET association
    for key, branch in uproot_tree.iteritems(filter_name="*AuxDyn.*"):
        if "/" in key:
            continue
        try:
            ak_form = branch.interpretation.awkward_form(None)
            forms[key] = ak_form
        except CannotBeAwkward:
            print("Can't interpret", key)
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
        top_key, sub_key = key.split(".")
        ak_top_key = top_key.replace("Analysis", "").replace("AuxDyn", "")
        form_dict = json.loads(ak_form.tojson())

        # top level will be "zipped"
        if form_dict["class"] == "ListOffsetArray64":
            if not ak_top_key in form["contents"]:
                form["contents"][ak_top_key] = {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": {"class": "RecordArray", "contents": {}},
                    "form_key": f"{key}%offsets",
                }
                if ak_top_key in behavior_dict:
                    form["contents"][ak_top_key]["parameters"] = {
                        "__record__": behavior_dict[ak_top_key]
                    }
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
        ar = branch_to_array(tree[key]).layout
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
    return ak.from_buffers(
        json_form, uproot_tree.num_entries, LazyGet(f, verbose=verbose), lazy=True
    )


if __name__ == "__main__":

    f = uproot.open("user.nihartma.22884623.EXT0._000001.DAOD_PHYSLITE.test.pool.root")
    tree = f["CollectionTree"]
    events = physlite_events(tree, verbose=True)

    # TODO: try out `to_buffers` to load everything and see what goes wrong
