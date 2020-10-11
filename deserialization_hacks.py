import uproot4
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
def _read_vector_vector_int(basket_data, border, num_entries):
    d = basket_data
    buf_size = len(basket_data) // 4
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
            n = _read_big_endian_int(d[inner_start: inner_start + 4])
            inner_stop = inner_start + n * 4 + 4
            offsets_inner[total_entries] = offsets_inner[total_entries - 1] + n
            for i in range(inner_start + 4, inner_stop):
                actual_data[total_bytes] = d[i]
                total_bytes += 1
            total_entries += 1
            inner_start = inner_stop
        i_outer += 1
        start = inner_start
    return offsets_outer, offsets_inner[:total_entries], actual_data[:total_bytes]


def branch_to_array_vector_vector_int(branch):
    oo, oi, ad = [], [], []
    for i in range(branch.num_baskets):
        basket = branch.basket(i)
        oo_i, oi_i, ad_i = _read_vector_vector_int(
            basket.raw_data.tobytes(), basket.border, basket.num_entries
        )
        ad.append(ad_i)
        if len(oo) == 0:
            oo.append(oo_i)
            oi.append(oi_i)
        else:
            # add last offset from previous basket
            oo.append(oo_i[1:] + oo[-1][-1])
            oi.append(oi_i[1:] + oi[-1][-1])
    oo, oi, ad = [np.concatenate(i) for i in [oo, oi, ad]]
    ad = np.frombuffer(ad.tobytes(), dtype=np.dtype(">i4"))
    ad = np.array(ad, dtype="<i4")
    return ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(oo),
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(oi),
                ak.layout.NumpyArray(ad)
            )
        )
    )


def test_vector_vector_int():
    with uproot4.open(example_file()) as f:
        branch = f["CollectionTree"]["AnalysisJetsAuxDyn.NumTrkPt500"]
        assert ak.all(branch.array() == branch_to_array_vector_vector_int(branch))
