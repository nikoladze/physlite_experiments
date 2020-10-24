import uproot4
from uproot4 import AsObjects, AsVector, AsString
import os
import numba
import numpy as np
import awkward1 as ak

__all__ = ["example_file", "interpretation_is_vector_vector", "branch_to_array", "tree_arrays"]

def example_file(
    filename="DAOD_PHYSLITE.art_split99.pool.root",
    url="https://cernbox.cern.ch/index.php/s/xTuPIvSJsP3QmXa/download"
):
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}")
        import requests
        res = requests.get(url)
        with open(filename, "wb") as of:
            of.write(res.content)
    return filename


@numba.njit(cache=True)
def _read_big_endian_int(data):
    "unfortunately np.frombuffer doesn't work in numba with big endian"
    res = 0
    factor = 1
    for i in range(len(data) - 1, -1, -1):
        res += factor * data[i]
        factor *= 256
    return res


@numba.njit(cache=True)
def parse_vector_header(d, pos):
    num_entries = _read_big_endian_int(d[pos + 6: pos + 10])
    return pos + 10, num_entries


@numba.njit(cache=True)
def _read_vector_vector(basket_data, num_entries, data_size=4, data_header_size=0, num_entries_size=4):
    """
    Deserialize raw data bytes that represent a list of
    vector<vector<some_data_type>>. Only the outermost vector is assumed to
    have a header.

    Parameters:
    -----------
    basket_data: array of bytes (eg. basket.raw_data)
    border: last index containing any data (e.g. from basket.border, the rest will typically be the event byte offsets)
    num_entries: number of events in this basket (e.g. from basket.num_entries)
    data_size: number of bytes for each element
    data_header_size: number of header bytes to skip over (e.g 20 for ElementLink)
    num_entries_size: number of bytes that encode the number of entries for the inner vectors.

    """
    d = basket_data
    # estimate - might need to grow the inner offsets (can have many empty entries)
    buf_size = len(basket_data) // data_size
    offsets_lvl1 = np.empty(num_entries + 1, dtype=np.int64)
    offsets_lvl2 = np.empty(buf_size, dtype=np.int64)
    offsets_lvl1[0] = 0
    offsets_lvl2[0] = 0
    actual_data = np.empty(len(basket_data), dtype=np.uint8)

    pos = 0
    i_offset_lvl1 = 1
    i_offset_lvl2 = 1
    i_data = 0
    for i_entry in range(num_entries):
        pos, num_entries_lvl1 = parse_vector_header(d, pos)
        offsets_lvl1[i_offset_lvl1] = offsets_lvl1[i_offset_lvl1 - 1] + num_entries_lvl1
        i_offset_lvl1 += 1
        for i_entry_lvl1 in range(num_entries_lvl1):
            num_entries_lvl2 = _read_big_endian_int(d[pos: pos + num_entries_size])
            if i_offset_lvl2 >= len(offsets_lvl2):
                # grow if nescessary
                offsets_lvl2 = np.concatenate(
                    (offsets_lvl2, np.empty(buf_size, dtype=np.int64))
                )
            offsets_lvl2[i_offset_lvl2] = offsets_lvl2[i_offset_lvl2 - 1] + num_entries_lvl2
            i_offset_lvl2 += 1
            pos += num_entries_size
            for i_entry_lvl2 in range(num_entries_lvl2):
                pos += data_header_size
                for _ in range(data_size):
                    actual_data[i_data] = d[pos]
                    i_data += 1
                    pos += 1

    return offsets_lvl1, offsets_lvl2[:i_offset_lvl2], actual_data[:i_data]


def _branch_to_array_vector_vector(branch, dtype=np.dtype(">i4"), data_size=4, data_header_size=0, num_entries_size=4):
    oo, oi, ad = [], [], []
    for i in range(branch.num_baskets):
        basket = branch.basket(i)
        oo_i, oi_i, ad_i = _read_vector_vector(
            basket.data.tobytes(),
            basket.num_entries,
            data_size=data_size,
            data_header_size=data_header_size,
            num_entries_size=num_entries_size
        )
        ad.append(ad_i)
        if len(oo) == 0:
            oo.append(oo_i)
            oi.append(oi_i)
        else:
            # add last offset from previous basket
            if len(oo_i) > 1:
                oo.append(oo_i[1:] + oo[-1][-1])
            if len(oi_i) > 1:
                oi.append(oi_i[1:] + oi[-1][-1])
    oo, oi, ad = [np.concatenate(i) for i in [oo, oi, ad]]
    ad = np.frombuffer(ad.tobytes(), dtype=dtype)
    # storing in parquet needs contiguous arrays
    if ad.dtype.fields is None:
        ad = ak.Array(ad.newbyteorder().byteswap()).layout
    else:
        ad = ak.zip(
            {
                k : np.ascontiguousarray(ad[k]).newbyteorder().byteswap()
                for k in ad.dtype.fields
            }
        ).layout
    return ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(oo),
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(oi),
                ad
            )
        )
    )


def _branch_to_array_vector_vector_elementlink(branch):
    return _branch_to_array_vector_vector(
        branch, dtype=np.dtype([("m_persKey", ">i4"), ("m_persIndex", ">i4")]), data_size=8, data_header_size=20
    )


def _branch_to_array_vector_string(branch):
    array = _branch_to_array_vector_vector(branch, dtype=np.uint8, data_size=1, num_entries_size=1)
    array.layout.content.setparameter("__array__", "string")
    array.layout.content.content.setparameter("__array__", "char")
    return array


def interpretation_is_vector_vector(interpretation):
    "... there is probably a better way"
    if not isinstance(interpretation, AsObjects):
        return False
    if not hasattr(interpretation, "_model"):
        return False
    if not isinstance(interpretation._model, AsVector):
        return False
    if not interpretation._model.header:
        return False
    if not isinstance(interpretation._model.values, AsVector):
        return False
    if interpretation._model.values.header:
        return False
    return True


def branch_to_array(branch, force_custom=False):
    "Try to deserialize with the custom functions and fall back to uproot"
    if branch.interpretation == AsObjects(AsVector(True, AsString(False))):
        return _branch_to_array_vector_string(branch)
    if interpretation_is_vector_vector(branch.interpretation):
        values = branch.interpretation._model.values.values
        if isinstance(values, np.dtype):
            return _branch_to_array_vector_vector(
                branch, dtype=values, data_size=values.itemsize, data_header_size=0
            )
        else:
            try:
                if "ElementLink_3c_DataVector" in values.__name__:
                    return _branch_to_array_vector_vector_elementlink(branch)
            except:
                pass
    if force_custom:
        raise TypeError(f"No custom deserialization for interpretation {branch.interpretation}")
    return branch.array()


def tree_arrays(tree, filter_branch=None):
    """
    Read all branches from a tree into arrays (using custom deserialization if
    possible). Optionally takes a filter function that takes a branch and returns
    True or False.

    Returns a dictionary of (awkward) arrays.
    """

    array_dict = {}

    def fill_dict(branch):
        for sub in branch.branches:
            fill_dict(sub)
        if len(branch.branches) > 0:
            return
        if filter_branch is not None and not filter_branch(branch):
            return
        array_dict[branch.name] = branch_to_array(branch)

    fill_dict(tree)

    return array_dict



def test_vector_vector_int():
    with uproot4.open(example_file()) as f:
        branch = f["CollectionTree"]["AnalysisJetsAuxDyn.NumTrkPt500"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True))


def test_vector_vector_double():
    with uproot4.open(example_file()) as f:
        branch = f["CollectionTree"]["METAssoc_AnalysisMETAux.trkpx"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True))


def test_vector_vector_float():
    with uproot4.open(example_file()) as f:
        branch = f["CollectionTree"]["AnalysisJetsAuxDyn.TrackWidthPt1000"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True))


def test_vector_vector_elementlink():
    with uproot4.open(example_file()) as f:
        branch = f["CollectionTree"]["AnalysisElectronsAuxDyn.trackParticleLinks"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True))


def test_vector_string():
    with uproot4.open(example_file()) as f:
        branch = f["CollectionTree"]["EventInfoAux.streamTagNames"]
        assert ak.all(branch.array() == branch_to_array(branch, force_custom=True))
