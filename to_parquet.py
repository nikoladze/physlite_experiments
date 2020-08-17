#!/usr/bin/env python

import awkward1 as ak

def to_ak_uproot4(root_filename, zip=False):
    "Convert easily readable AuxDyn Branches"

    import uproot4

    with uproot4.open(root_filename) as f:
        read_branches = []
        t = f["CollectionTree"]
        for k in t.keys(filter_name="*AuxDyn*"):
            # skip <vector<vector<...>> of custom types
            if "Links" in k:
                continue
            if "TrigMatchedObjects" in k:
                continue
            if "GhostTrack" in k:
                continue
            # only read m_persIndex, m_persKey for Link
            if "Link" in k and not "m_" in k:
                continue
            if "/" in k:
                k = k.split("/")[1]
            # these vector<vector<...>> are readable but slow ...
            if k in [
                'AnalysisJetsAuxDyn.NumTrkPt500',
                'AnalysisJetsAuxDyn.SumPtTrkPt500',
                'AnalysisJetsAuxDyn.NumTrkPt1000',
                'AnalysisJetsAuxDyn.TrackWidthPt1000',
                'CaloCalTopoClustersAuxDyn.e_sampl',
                'EventInfoAuxDyn.streamTagRobs',
                'EventInfoAuxDyn.streamTagDets',
            ]:
                continue
            read_branches.append(k)
        if zip:
            ar = t.arrays(read_branches, how="zip")
            # AnalysisJets will be zipped with BTagging info
            jet_keys = [k.replace("AnalysisJetsAuxDyn.", "") for k in ak.keys(ar["jagged0"])]
            new_dict = {}
            for k in ak.keys(ar):
                if k == "jagged0":
                    new_dict["AnalysisJetsAuxDyn"] = ak.zip(
                        {jet_key : jet_ar for jet_key, jet_ar in zip(jet_keys, ak.unzip(ar[k]))}
                    )
                else:
                    new_dict[k] = ar[k]
            ar = ak.zip(new_dict, depth_limit=1)
            assert all([not "jagged" in k for k in ak.keys(ar)])
            return ar
        else:
            return t.arrays(read_branches)


def to_ak(root_filename):

    import uproot

    with uproot.open(root_filename) as f:
        read_branches = []
        t = f["CollectionTree"]
        for k in t.keys():
            k = k.decode()
            if not "AuxDyn" in k:
                continue
            # skip <vector<vector<...>> of custom types
            if "Links" in k:
                continue
            if "TrigMatchedObjects" in k:
                continue
            if "GhostTrack" in k:
                continue
            # only read m_persIndex, m_persKey for Link
            if "Link" in k and not "m_" in k:
                continue
            if "/" in k:
                k = k.split("/")[1]
            # these vector<vector<...>> are readable but slow ...
            if k in [
                'AnalysisJetsAuxDyn.NumTrkPt500',
                'AnalysisJetsAuxDyn.SumPtTrkPt500',
                'AnalysisJetsAuxDyn.NumTrkPt1000',
                'AnalysisJetsAuxDyn.TrackWidthPt1000',
                'CaloCalTopoClustersAuxDyn.e_sampl',
                'EventInfoAuxDyn.streamTagRobs',
                'EventInfoAuxDyn.streamTagDets',
            ]:
                continue
            # will be 'O' with uproot3
            if any([s in k for s in ["definingParametersCovMatrix", "MET_Core", "e_sampl", "eta_sampl"]]):
                continue
            read_branches.append(k)
        ar_dict = t.arrays(read_branches)
        ar = ak.zip({k : ak.from_awkward0(v) for k, v in ar_dict.items()}, depth_limit=1)
    return ar


def to_parquet(root_filename, output_parquet):
    ar = to_ak(root_filename)
    ak.to_parquet(ar, output_parquet)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Convert \"simple\" branches in DAOD_PHYSLITE to parquet")
    parser.add_argument("input_daod", help="input daod path")
    parser.add_argument("output_parquet", help="output parquet filename/path")
    args = parser.parse_args()

    to_parquet(args.input_daod, args.output_parquet)
