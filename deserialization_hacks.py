import uproot4
from uproot4 import AsObjects, AsVector, AsString
import os
import numba
import numpy as np
import awkward1 as ak


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


@numba.njit
def _read_big_endian_int(data):
    "unfortunately np.frombuffer doesn't work in numba with big endian"
    res = 0
    factor = 1
    for i in range(len(data) - 1, -1, -1):
        res += factor * data[i]
        factor *= 256
    return res


@numba.njit
def _read_vector_vector(basket_data, border, num_entries, data_size=4, data_header_size=0, offset_size=4):
    """
    Deserialize raw data bytes that represent a list of vector<vector<some_data_type>>

    Parameters:
    -----------
    basket_data: array of bytes (eg. basket.raw_data)
    border: last index containing any data (e.g. from basket.border, the rest will typically be the event byte offsets)
    num_entries: number of events in this basket (e.g. from basket.num_entries)
    data_size: number of bytes for each element
    data_header_size: number of header bytes to skip over (e.g 20 for ElementLink)
    offset_size: number of bytes that encode the offsets
    """
    d = basket_data
    # estimate - might need to grow the inner offsets (can have many empty entries)
    buf_size = len(basket_data) // data_size
    offsets_outer = np.empty(num_entries + 1, dtype=np.int64)
    offsets_inner = np.empty(buf_size, dtype=np.int64)
    offsets_outer[0] = 0
    offsets_inner[0] = 0
    actual_data = np.empty(len(basket_data), dtype=np.uint8)
    start = 0
    i_outer = 1
    total_entries = 1
    total_bytes = 0
    while start < border:
        nbytes = _read_big_endian_int(d[start + 2: start + 4])
        stop = start + 4 + nbytes
        inner_start = start + 10
        n_outer = _read_big_endian_int(d[start + 6: start + 10])
        offsets_outer[i_outer] = offsets_outer[i_outer - 1] + n_outer
        while inner_start < stop:
            n = _read_big_endian_int(d[inner_start: inner_start + offset_size])
            inner_stop = inner_start + n * (data_size + data_header_size) + offset_size
            if total_entries >= len(offsets_inner):
                # increase the size if not sufficient
                offsets_inner = np.concatenate(
                    (offsets_inner, np.empty(buf_size, dtype=np.int64))
                )
            offsets_inner[total_entries] = offsets_inner[total_entries - 1] + n
            i = inner_start + offset_size
            while i < inner_stop:
                i += data_header_size
                for ii in range(data_size):
                    actual_data[total_bytes] = d[i + ii]
                    total_bytes += 1
                i += data_size
            total_entries += 1
            inner_start = inner_stop
        i_outer += 1
        start = inner_start
    return offsets_outer, offsets_inner[:total_entries], actual_data[:total_bytes]


def _branch_to_array_vector_vector(branch, dtype=np.dtype(">i4"), data_size=4, data_header_size=0, offset_size=4):
    oo, oi, ad = [], [], []
    for i in range(branch.num_baskets):
        basket = branch.basket(i)
        oo_i, oi_i, ad_i = _read_vector_vector(
            basket.raw_data.tobytes(),
            basket.border,
            basket.num_entries,
            data_size=data_size,
            data_header_size=data_header_size,
            offset_size=offset_size
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
    array = _branch_to_array_vector_vector(branch, dtype=np.uint8, data_size=1, offset_size=1)
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
