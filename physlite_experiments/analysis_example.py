import awkward as ak
import numpy as np

def has_overlap(obj1, obj2, filter_dr):
    """
    Return mask array where obj1 has overlap with obj2 based on a filter
    function on deltaR (and pt of the first one)
    """
    obj1x, obj2x = ak.unzip(ak.cartesian([obj1, obj2], nested=True))
    dr = obj1x.delta_r(obj2x)
    return ak.any(filter_dr(dr, obj1.pt), axis=-1)


def match_dr(dr, pt, cone_size=0.2):
    return dr < cone_size


def match_boosted_dr(dr, pt, max_cone_size=0.4):
    return dr < np.minimum(*ak.broadcast_arrays(10000.0 / pt + 0.04, max_cone_size))


def muon_track_links(muons):
    return muons["inDetTrackParticleLink.m_persIndex"].mask[
        muons["inDetTrackParticleLink.m_persKey"] != 0
    ]


def jet_ghost_links(jets):
    return jets.GhostTrack.m_persIndex.mask[jets.GhostTrack.m_persKey != 0]


def is_associated(idx1, idx2):
    """
    Used for muon-jet ghost association
    """
    # TODO: check if keys also match
    xx1, xx2 = ak.unzip(ak.cartesian([idx1, idx2], nested=True))
    return ak.fill_none(ak.any(ak.any(xx1 == xx2, axis=-1), axis=-1), False)


def has_overlap_mujet(obj1, obj2, filter_dr):
    """
    Check if either ghost associated or delta_r match
    """
    if "GhostTrack" in obj1.fields:
        ghost_match = is_associated(jet_ghost_links(obj1), muon_track_links(obj2))
    else:
        ghost_match = is_associated(muon_track_links(obj1), jet_ghost_links(obj2))
    return has_overlap(obj1, obj2, filter_dr) | ghost_match


def mu_pflow_or_requirement(jets, muons):
    """
    Taken from this beautiful piece:
    https://gitlab.cern.ch/atlas/athena/-/blob/c8de5319c743b68e3e4935a5e34c7ccb8da778bf/PhysicsAnalysis/AnalysisCommon/AssociationUtils/Root/MuPFJetOverlapTool.cxx#L167
    I have no idea what's going on here ...
    """

    GeV = 1000.0
    lowNtrk_x1 = 0.7
    lowNtrk_x2 = 0.85
    lowNtrk_y0 = 15.0 * GeV
    lowNtrk_y1 = 15.0 * GeV
    lowNtrk_y2 = 30.0 * GeV
    highNtrk_x1 = 0.6
    highNtrk_x2 = 0.9
    highNtrk_y0 = 5.0 * GeV
    highNtrk_y1 = 5.0 * GeV
    highNtrk_y2 = 30.0 * GeV
    numJetTrk = 4

    mu_id_pt = muons.id_pt
    mu_topoetcone40 = muons.topoetcone40

    nTrk = jets.NumTrkPt500_0
    sumTrkPt = jets.SumPtTrkPt500_0

    def requirements(x1, x2, y0, y1, y2):
        return (
            (mu_topoetcone40 < y0)
            | (mu_topoetcone40 < y0 + (y2 - y1) / (x2 - x1) * (mu_id_pt / sumTrkPt - x1))
            | (mu_id_pt / sumTrkPt > x2)
        )

    return ak.where(
        nTrk < numJetTrk,
        requirements(lowNtrk_x1, lowNtrk_x2, lowNtrk_y0, lowNtrk_y1, lowNtrk_y2),
        requirements(highNtrk_x1, highNtrk_x2, highNtrk_y0, highNtrk_y1, highNtrk_y2),
    )


def has_mu_pflow_overlap(jets, muons, cone_size=0.4):
    muons["id_pt"] = muons.trackParticle.pt
    jets["NumTrkPt500_0"] = ak.firsts(jets.NumTrkPt500, axis=-1)
    jets["SumPtTrkPt500_0"] = ak.firsts(jets.SumPtTrkPt500, axis=-1)
    jets_x, muons_x = ak.unzip(ak.cartesian([jets, muons], nested=True))
    return ak.any(
        mu_pflow_or_requirement(jets_x, muons_x) & (jets_x.delta_r(muons_x) < cone_size),
        axis=-1
    )


def get_obj_sel(evt):

    ## Object selection
    # ---------------------------------------------------------
    # TODO: sort jets

    # mask events without primary vertex (important for data)
    evt = evt.mask[ak.num(evt.PrimaryVertices.x) > 0]

    # lepton selection
    evt["Electrons", "baseline"] = (
        (evt.Electrons.DFCommonElectronsLHLooseBL == 1)
        & (evt.Electrons.pt > 7000)
        & (np.abs(evt.Electrons.eta) < 2.47)
    )
    evt["Electrons", "signal"] = (
        evt.Electrons.baseline
        & (evt.Electrons.DFCommonElectronsLHTight == 1)
        & (evt.Electrons.topoetcone20 / evt.Electrons.pt < 0.2)
        & (evt.Electrons.ptvarcone20_TightTTVA_pt1000 / evt.Electrons.pt < 0.15)
    )
    evt["Muons", "baseline"] = (
        (~ak.is_none(evt.Muons.trackParticle, axis=1)) # require ID track
        & (evt.Muons.DFCommonGoodMuon == 1)
        & (evt.Muons.pt > 6000)
        & (np.abs(evt.Muons.eta) < 2.7)
    )
    evt["Muons", "signal"] = (
        evt.Muons.baseline
        & (np.abs(evt.Muons.eta) < 2.5)
        & (evt.Muons.topoetcone20 / evt.Muons.pt < 0.3)
        & (evt.Muons.ptvarcone30 / evt.Muons.pt < 0.15)
    )

    # jet selection
    evt["Jets", "baseline"] = evt.Jets.pt >= 20000
    jvt_pt_min = 20000
    jvt_pt_max = 60000
    jvt_eta_max = 2.4
    evt["Jets", "passJvt"] = (
        ((evt.Jets.pt >= jvt_pt_min) & (evt.Jets.pt < jvt_pt_max) & (evt.Jets.Jvt > 0.2) & (np.abs(evt.Jets.eta) < jvt_eta_max))
        | ((evt.Jets.pt >= jvt_pt_max) | (evt.Jets.pt < jvt_pt_min) | (np.abs(evt.Jets.eta) >= jvt_eta_max))
    )
    evt["Jets", "signal"] = evt.Jets.baseline & evt.Jets.passJvt & (np.abs(evt.Jets.eta) < 4.5)

    ## Overlap removal
    # --------------------------------------------------------------
    # run the funny MuPFlowJetOR procedure ...
    evt["Jets", "passOR"] = (
        evt.Jets.baseline
        & ~has_mu_pflow_overlap(evt.Jets, evt.Muons[evt.Muons.baseline])
    )
    # remove Jets overlapping with Electrons
    evt["Jets", "passOR"] = (
        evt.Jets.passOR
        & (
            ~has_overlap(
                evt.Jets,
                evt.Electrons[evt.Electrons.baseline],
                match_dr
            )
        )
    )
    # remove Electrons overlapping (boosted cone) with remaining Jets (if they pass jvt)
    evt["Electrons", "passOR"] = (
        evt.Electrons.baseline
        & (
            ~has_overlap(
                evt.Electrons,
                evt.Jets[evt.Jets.passOR & evt.Jets.passJvt],
                match_boosted_dr
            )
        )
    )
    # remove Jets overlapping with Muons
    evt["Jets", "passOR"] = (
        evt.Jets.passOR
        & ~(
            has_overlap_mujet(evt.Jets, evt.Muons[evt.Muons.baseline], match_dr)
            & (evt.Jets.NumTrkPt500[:, :, 0] < 3)
        )
    )
    # remove Muons overlapping (boosted cone) with remaining Jets (if they pass jvt)
    evt["Muons", "passOR"] = (
        evt.Muons.baseline
        & (
            ~has_overlap_mujet(
                evt.Muons,
                evt.Jets[evt.Jets.baseline & evt.Jets.passOR & evt.Jets.passJvt],
                match_boosted_dr
            )
        )
    )
    return evt
