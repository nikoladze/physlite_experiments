#!/usr/bin/env python

import pyarrow.parquet as pq
import awkward1 as ak

electron_vars = [
    "pt",
    "eta",
    "phi",
    "DFCommonElectronsLHLooseBL",
    "DFCommonElectronsLHTight",
    "topoetcone20",
    "ptvarcone20_TightTTVA_pt1000",
]

muon_vars = [
    "pt",
    "eta",
    "phi",
    "DFCommonGoodMuon",
    "topoetcone20",
    "ptvarcone30",
]

jet_vars = [
    "pt",
    "eta",
    "phi",
    "Jvt",
    # "NumTrkPt500" # ... currently vector<vector ...
]

columns = sum(
    [
        [f"{cont}.{v}" for v in vars]
        for cont, vars in [
                ("AnalysisElectronsAuxDyn", electron_vars),
                ("AnalysisJetsAuxDyn", jet_vars),
                ("AnalysisMuonsAuxDyn", muon_vars)
        ]
    ],
    []
)

#ar = ak.from_parquet("test2.parquet", columns=columns)

if __name__ == "__main__":

    import pyarrow.parquet as pq
    from proper_xrdfile import XRDFile
    f = XRDFile()
    f.open("root://lcg-lrz-rootd.grid.lrz.de:1094/pnfs/lrz-muenchen.de/data/atlas/dq2/atlaslocalgroupdisk/rucio/user/nihartma/c4/40/testdata_physlite.parquet")

    import awkward1 as ak
    ar = ak.from_parquet(f, columns=columns)
