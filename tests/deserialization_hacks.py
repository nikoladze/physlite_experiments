import pytest
import uproot
import awkward as ak
import numpy as np
from physlite_experiments.deserialization_hacks import branch_to_array
from physlite_experiments.utils import example_file


@pytest.mark.parametrize("use_forth", [True, False])
def test_vector_vector_int(use_forth):
    with uproot.open(example_file()) as f:
        branch = f["CollectionTree"]["AnalysisJetsAuxDyn.NumTrkPt500"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True, use_forth=use_forth))


@pytest.mark.parametrize("use_forth", [True, False])
def test_vector_vector_double(use_forth):
    with uproot.open(example_file()) as f:
        branch = f["CollectionTree"]["METAssoc_AnalysisMETAux.trkpx"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True, use_forth=use_forth))


@pytest.mark.parametrize("use_forth", [True, False])
def test_vector_vector_float(use_forth):
    with uproot.open(example_file()) as f:
        branch = f["CollectionTree"]["AnalysisJetsAuxDyn.TrackWidthPt1000"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True, use_forth=use_forth))


@pytest.mark.parametrize("use_forth", [True, False])
def test_vector_vector_elementlink(use_forth):
    with uproot.open(example_file()) as f:
        branch = f["CollectionTree"]["AnalysisElectronsAuxDyn.trackParticleLinks"]
        array1 = branch.array()
        array2 = branch_to_array(branch, force_custom=True, use_forth=use_forth)
        assert set(array1.fields) == set(array2.fields)
        for field in array1.fields:
            assert ak.all(array1[field] == array2[field])


@pytest.mark.parametrize("use_forth", [True, False])
def test_vector_string(use_forth):
    with uproot.open(example_file()) as f:
        branch = f["CollectionTree"]["EventInfoAux.streamTagNames"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True, use_forth=use_forth))


def test_vector_vector_vector():
    with uproot.open(example_file()) as f:
        branch = f["CollectionTree"]["METAssoc_AnalysisMETAux.overlapIndices"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True))


def test_vector_vector_vector2():
    with uproot.open(example_file()) as f:
        branch = f["CollectionTree"]["METAssoc_AnalysisMETAux.overlapTypes"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True))


@pytest.mark.parametrize("use_forth", [True, False])
def test_start_stop(use_forth):
    with uproot.open(example_file()) as f:
        branch = f["CollectionTree"]["AnalysisElectronsAuxDyn.trackParticleLinks"]
        rnd = (
            [0]
            + sorted(np.random.randint(branch.num_entries, size=5))
            + [branch.num_entries]
        )
        for start, stop in zip(rnd[:-1], rnd[1:]):
            array1 = branch.array(entry_start=start, entry_stop=stop)
            array2 = branch_to_array(
                branch,
                force_custom=True,
                entry_start=start,
                entry_stop=stop,
                use_forth=use_forth
            )
            for field in array1.fields:
                assert ak.all(array1[field] == array2[field])
