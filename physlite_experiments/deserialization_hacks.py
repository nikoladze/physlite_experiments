import uproot
from uproot import AsObjects, AsVector, AsString
import os
import numba
import numpy as np
import awkward as ak
import awkward.forth
import queue

__all__ = [
    "interpretation_is_vector_vector",
    "branch_to_array",
    "tree_arrays",
    "patch_nanoevents",
]


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
    num_entries = _read_big_endian_int(d[pos + 6 : pos + 10])
    return pos + 10, num_entries


def _read_vector_vector(basket_data, num_entries, use_forth=False, **kwargs):
    if not use_forth:
        kwargs.pop("byte_offsets", None)
        return _read_vector_vector_numba(basket_data.tobytes(), num_entries, **kwargs)
    else:
        return _read_nested_vector_forth(np.array(basket_data), num_entries, ndim=2, **kwargs)


@numba.njit(cache=True)
def _read_vector_vector_numba(
    basket_data, num_entries, data_size=4, data_header_size=0, num_entries_size=4
):
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
            num_entries_lvl2 = _read_big_endian_int(d[pos : pos + num_entries_size])
            if i_offset_lvl2 >= len(offsets_lvl2):
                # grow if nescessary
                offsets_lvl2 = np.concatenate(
                    (offsets_lvl2, np.empty(buf_size, dtype=np.int64))
                )
            offsets_lvl2[i_offset_lvl2] = (
                offsets_lvl2[i_offset_lvl2 - 1] + num_entries_lvl2
            )
            i_offset_lvl2 += 1
            pos += num_entries_size
            for i_entry_lvl2 in range(num_entries_lvl2):
                pos += data_header_size
                for _ in range(data_size):
                    actual_data[i_data] = d[pos]
                    i_data += 1
                    pos += 1

    return offsets_lvl1, offsets_lvl2[:i_offset_lvl2], actual_data[:i_data]


def _read_nested_vector_forth(
    basket_data,
    num_entries,
    byte_offsets,
    data_size=4,
    data_header_size=0,
    num_entries_size=4,
    ndim=2,
):
    forth = [
        "input data",
        "input byte_offsets",
    ]
    for i in range(ndim):
        forth.append(f"output offsets{i} int64")
    forth.append("output content int8")
    for i in range(ndim):
        forth.append(f"0 offsets{i} <- stack")
    forth += [
        "begin",
        "  byte_offsets i-> stack",
        "  6 + data seek",
        "  data !i-> stack"
        "  dup offsets0 +<- stack",
    ]
    for i in range(1, ndim):
        forth.append("0 do")
        if num_entries_size == 4:
            forth.append("data !i-> stack")
        elif num_entries_size == 1:
            forth.append("data !b-> stack")
        else:
            raise NotImplementedError(
                f"No implementation for `num_entries_size` == {num_entries_size}"
            )
        forth.append(f"dup offsets{i} +<- stack")
    if data_header_size == 0:
        forth += [
            f"{data_size} *",
            "data #!b-> content"
        ]
    else:
        forth += [
            " 0 do",
            f"  {data_header_size} data skip"
            f"  {data_size} data #!b-> content",
            " loop",
        ]
    for i in range(ndim - 1):
        forth.append("loop")
    forth.append("again")
    machine = awkward.forth.ForthMachine32("\n".join(forth))
    machine.run(
        {"data": basket_data, "byte_offsets": byte_offsets},
        raise_read_beyond=False,
        raise_seek_beyond=False,
    )
    return [
        np.asarray(i) for i in [
            machine.output_Index64(f"offsets{j}")
            for j in range(ndim)
        ] + [
            machine.output_NumpyArray("content"),
        ]
    ]


def _read_vector_vector_vector(basket_data, num_entries, use_forth=False, **kwargs):
    if not use_forth:
        kwargs.pop("byte_offsets", None)
        return _read_vector_vector_vector_numba(basket_data.tobytes(), num_entries, **kwargs)
    else:
        return _read_nested_vector_forth(np.array(basket_data), num_entries, ndim=3, **kwargs)
    pass


@numba.njit(cache=True)
def _read_vector_vector_vector_numba(
    basket_data, num_entries, data_size=4, data_header_size=0, num_entries_size=4
):
    """
    Deserialize raw data bytes that represent a list of
    vector<vector<vector<some_data_type>>>. Only the outermost vector is assumed to
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
    offsets_lvl3 = np.empty(buf_size, dtype=np.int64)
    offsets_lvl1[0] = 0
    offsets_lvl2[0] = 0
    offsets_lvl3[0] = 0
    actual_data = np.empty(len(basket_data), dtype=np.uint8)

    pos = 0
    i_offset_lvl1 = 1
    i_offset_lvl2 = 1
    i_offset_lvl3 = 1
    i_data = 0
    for i_entry in range(num_entries):
        pos, num_entries_lvl1 = parse_vector_header(d, pos)
        offsets_lvl1[i_offset_lvl1] = offsets_lvl1[i_offset_lvl1 - 1] + num_entries_lvl1
        i_offset_lvl1 += 1
        for i_entry_lvl1 in range(num_entries_lvl1):
            num_entries_lvl2 = _read_big_endian_int(d[pos : pos + num_entries_size])
            if i_offset_lvl2 >= len(offsets_lvl2):
                # grow if nescessary
                offsets_lvl2 = np.concatenate(
                    (offsets_lvl2, np.empty(buf_size, dtype=np.int64))
                )
            offsets_lvl2[i_offset_lvl2] = (
                offsets_lvl2[i_offset_lvl2 - 1] + num_entries_lvl2
            )
            i_offset_lvl2 += 1
            pos += num_entries_size
            for i_entry_lvl2 in range(num_entries_lvl2):
                num_entries_lvl3 = _read_big_endian_int(d[pos : pos + num_entries_size])
                if i_offset_lvl3 >= len(offsets_lvl3):
                    # grow if nescessary
                    offsets_lvl3 = np.concatenate(
                        (offsets_lvl3, np.empty(buf_size, dtype=np.int64))
                    )
                offsets_lvl3[i_offset_lvl3] = (
                    offsets_lvl3[i_offset_lvl3 - 1] + num_entries_lvl3
                )
                i_offset_lvl3 += 1
                pos += num_entries_size
                for i_entry_lvl3 in range(num_entries_lvl3):
                    pos += data_header_size
                    for _ in range(data_size):
                        actual_data[i_data] = d[pos]
                        i_data += 1
                        pos += 1

    return (
        offsets_lvl1,
        offsets_lvl2[:i_offset_lvl2],
        offsets_lvl3[:i_offset_lvl3],
        actual_data[:i_data],
    )


def _get_baskets(branch, entry_start=None, entry_stop=None):
    notifications = queue.Queue()
    source = branch._file._source

    basket_chunks = []
    basket_ids = {}
    entry_starts, entry_stops = (
        branch.member("fBasketEntry")[:-1],
        branch.member("fBasketEntry")[1:],
    )
    basket_entries = branch.member("fBasketEntry")
    for i in range(branch.num_baskets):

        if entry_start is not None and entry_stops[i] <= entry_start:
            continue
        if entry_stop is not None and entry_starts[i] >= entry_stop:
            break

        start = branch.member("fBasketSeek")[i]
        stop = start + branch.basket_compressed_bytes(i)
        basket_chunks.append((int(start), int(stop)))
        basket_ids[start, stop] = i

    def chunk_to_basket(chunk, basket_num):
        cursor = uproot.source.cursor.Cursor(chunk.start)
        return uproot.models.TBasket.Model_TBasket.read(
            chunk,
            cursor,
            {"basket_num": basket_num},
            branch._file,
            branch._file,
            branch,
        )

    source.chunks(basket_chunks, notifications)
    result_baskets = {}
    for i in range(len(basket_chunks)):
        chunk = notifications.get(timeout=10)
        basket_num = basket_ids[chunk.start, chunk.stop]
        result_baskets[basket_num] = chunk_to_basket(chunk, basket_num)

    return result_baskets


def _get_start_stop(first_basket_start, num_entries, entry_start, entry_stop):
    stop = entry_stop or num_entries
    start = entry_start or 0
    num_entries = stop - start
    this_entry_start = start - first_basket_start
    this_entry_stop = this_entry_start + num_entries
    return this_entry_start, this_entry_stop


def _branch_to_array_vector_vector(
    branch,
    dtype=np.dtype(">i4"),
    data_size=4,
    data_header_size=0,
    num_entries_size=4,
    entry_start=None,
    entry_stop=None,
    use_forth=False,
):
    offsets_lvl1, offsets_lvl2, data = [], [], []
    baskets = _get_baskets(branch, entry_start=entry_start, entry_stop=entry_stop)
    for i in sorted(baskets):
        basket = baskets[i]
        offsets_lvl1_i, offsets_lvl2_i, data_i = _read_vector_vector(
            basket.data,
            basket.num_entries,
            byte_offsets=basket.byte_offsets,
            data_size=data_size,
            data_header_size=data_header_size,
            num_entries_size=num_entries_size,
            use_forth=use_forth
        )
        data.append(data_i)
        if len(offsets_lvl1) == 0:
            offsets_lvl1.append(offsets_lvl1_i)
            offsets_lvl2.append(offsets_lvl2_i)
        else:
            # add last offset from previous basket
            if len(offsets_lvl1_i) > 1:
                offsets_lvl1.append(offsets_lvl1_i[1:] + offsets_lvl1[-1][-1])
            if len(offsets_lvl2_i) > 1:
                offsets_lvl2.append(offsets_lvl2_i[1:] + offsets_lvl2[-1][-1])
    offsets_lvl1, offsets_lvl2, data = [
        np.concatenate(i) for i in [offsets_lvl1, offsets_lvl2, data]
    ]
    data = np.frombuffer(data.tobytes(), dtype=dtype)
    # storing in parquet needs contiguous arrays
    if data.dtype.fields is None:
        data = ak.Array(data.newbyteorder().byteswap()).layout
    else:
        data = ak.zip(
            {
                k: np.ascontiguousarray(data[k]).newbyteorder().byteswap()
                for k in data.dtype.fields
            }
        ).layout
    if entry_start is not None or entry_stop is not None:
        start, stop = _get_start_stop(
            baskets[min(baskets)].entry_start_stop[0],
            branch.num_entries,
            entry_start,
            entry_stop,
        )
        offsets_lvl1 = offsets_lvl1[start: stop + 1]
        offsets_lvl2 = offsets_lvl2[offsets_lvl1[0]: offsets_lvl1[-1] + 1]
        data = data[offsets_lvl2[0]: offsets_lvl2[-1]]
        offsets_lvl1 -= offsets_lvl1[0]
        offsets_lvl2 -= offsets_lvl2[0]
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(offsets_lvl1),
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(offsets_lvl2),
                data,
            ),
        )
    )
    return array


def _branch_to_array_vector_vector_vector(
    branch, dtype=np.dtype(">i4"), data_size=4, data_header_size=0, num_entries_size=4, use_forth=False
):
    offsets_lvl1, offsets_lvl2, offsets_lvl3, data = [], [], [], []
    baskets = _get_baskets(branch)
    for i in range(branch.num_baskets):
        basket = baskets[i]
        (
            offsets_lvl1_i,
            offsets_lvl2_i,
            offsets_lvl3_i,
            data_i,
        ) = _read_vector_vector_vector(
            basket.data,
            basket.num_entries,
            data_size=data_size,
            data_header_size=data_header_size,
            num_entries_size=num_entries_size,
            byte_offsets=basket.byte_offsets,
            use_forth=use_forth,
        )
        data.append(data_i)
        if len(offsets_lvl1) == 0:
            offsets_lvl1.append(offsets_lvl1_i)
            offsets_lvl2.append(offsets_lvl2_i)
            offsets_lvl3.append(offsets_lvl3_i)
        else:
            # add last offset from previous basket
            if len(offsets_lvl1_i) > 1:
                offsets_lvl1.append(offsets_lvl1_i[1:] + offsets_lvl1[-1][-1])
            if len(offsets_lvl2_i) > 1:
                offsets_lvl2.append(offsets_lvl2_i[1:] + offsets_lvl2[-1][-1])
            if len(offsets_lvl3_i) > 1:
                offsets_lvl3.append(offsets_lvl3_i[1:] + offsets_lvl3[-1][-1])
    offsets_lvl1, offsets_lvl2, offsets_lvl3, data = [
        np.concatenate(i) for i in [offsets_lvl1, offsets_lvl2, offsets_lvl3, data]
    ]
    data = np.frombuffer(data.tobytes(), dtype=dtype)
    # storing in parquet needs contiguous arrays
    if data.dtype.fields is None:
        data = ak.Array(data.newbyteorder().byteswap()).layout
    else:
        data = ak.zip(
            {
                k: np.ascontiguousarray(data[k]).newbyteorder().byteswap()
                for k in data.dtype.fields
            }
        ).layout
    return ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(offsets_lvl1),
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(offsets_lvl2),
                ak.layout.ListOffsetArray64(
                    ak.layout.Index64(offsets_lvl3),
                    data,
                ),
            ),
        ),
    )


def _branch_to_array_vector_vector_elementlink(branch, **kwargs):
    return _branch_to_array_vector_vector(
        branch,
        dtype=np.dtype([("m_persKey", ">i4"), ("m_persIndex", ">i4")]),
        data_size=8,
        data_header_size=20,
        **kwargs,
    )


def _branch_to_array_vector_string(branch, **kwargs):
    array = _branch_to_array_vector_vector(
        branch, dtype=np.uint8, data_size=1, num_entries_size=1, **kwargs
    )
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
    if isinstance(interpretation._model.values.values, AsVector):
        # vector<vector<vector
        return False
    return True


_other_custom = {
    "AsObjects(AsVector(True, AsVector(False, AsVector(False, dtype('>u8')))))": (
        lambda branch, **kwargs: _branch_to_array_vector_vector_vector(
            branch, dtype=np.dtype(">u8"), data_size=8, **kwargs
        )
    ),
    "AsObjects(AsVector(True, AsVector(False, AsVector(False, dtype('uint8')))))": (
        lambda branch, **kwargs: _branch_to_array_vector_vector_vector(
            branch, dtype=np.dtype(">i1"), data_size=1, **kwargs
        )
    ),
    "AsObjects(AsVector(True, AsSet(False, dtype('>u4'))))": (
        lambda branch, **kwargs: _branch_to_array_vector_vector(
            branch, dtype=np.dtype(">u4"), data_size=4, **kwargs
        )
    ),
}


def branch_to_array(branch, force_custom=False, **kwargs):
    "Try to deserialize with the custom functions and fall back to uproot"
    if branch.interpretation == AsObjects(AsVector(True, AsString(False))):
        return _branch_to_array_vector_string(branch, **kwargs)
    elif interpretation_is_vector_vector(branch.interpretation):
        values = branch.interpretation._model.values.values
        if isinstance(values, np.dtype):
            return _branch_to_array_vector_vector(
                branch,
                dtype=values,
                data_size=values.itemsize,
                data_header_size=0,
                **kwargs,
            )
        else:
            try:
                if "ElementLink_3c_DataVector" in values.__name__:
                    return _branch_to_array_vector_vector_elementlink(branch, **kwargs)
            except:
                pass
    elif str(branch.interpretation) in _other_custom:
        return _other_custom[str(branch.interpretation)](branch, **kwargs)
    if force_custom:
        raise TypeError(
            f"No custom deserialization for interpretation {branch.interpretation}"
        )
    kwargs.pop("use_forth", None)
    return branch.array(**kwargs)


def tree_arrays(tree, filter_name=None, filter_branch=None, use_forth=False):
    """
    Read all branches from a tree into arrays (using custom deserialization if
    possible). Optionally takes a filter function that takes a branch name and returns
    True or False.

    Returns a dictionary of (awkward) arrays.
    """

    array_dict = {}

    def fill_dict(branch):
        for sub in branch.branches:
            fill_dict(sub)
        if len(branch.branches) > 0:
            return
        if filter_name is not None and not filter_name(branch.name):
            return
        if filter_branch is not None and not filter_branch(branch):
            return
        array_dict[branch.name] = branch_to_array(branch, use_forth=use_forth)

    fill_dict(tree)

    return array_dict


def patch_nanoevents(verbose=False):
    """
    Patch the `extract_column` method of `UprootSourceMapping` in
    `coffea.nanoevents` to make use of the deserialization hacks
    """
    from coffea.nanoevents.mapping import UprootSourceMapping
    from coffea.nanoevents.schemas import PHYSLITESchema

    def extract_column(self, columnhandle, start, stop):
        if verbose:
            print("extracting", columnhandle)
        return branch_to_array(columnhandle, entry_start=start, entry_stop=stop)

    UprootSourceMapping.extract_column = extract_column
    PHYSLITESchema._hack_for_elementlink_int64 = False
