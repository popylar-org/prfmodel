"""Read MATLAB ``table`` objects out of MAT-file v5 archives."""

from __future__ import annotations
import logging
import struct
import zlib
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.io import loadmat

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# A MAT-file opens with a 128 byte header that ends with the offset of the object system, the version of
# the format, and an endianness marker.
_HEADER_BYTES = 128
_SUBSYSTEM_OFFSET = slice(116, 124)
_VERSION_OFFSET = slice(124, 126)
_ENDIAN_OFFSET = slice(126, 128)
_VERSION_5 = 0x0100
_LITTLE_ENDIAN = b"IM"

# MAT-file v5 data types and array classes, from the MAT-File Format reference.
_TYPE_MATRIX = 14
_TYPE_COMPRESSED = 15
_CLASS_OPAQUE = 17

# The class metadata block opens with a version, a string count and six region offsets, after which the
# names of the classes and properties follow as null terminated strings.
_METADATA_HEADER_WORDS = 10
_METADATA_STRING_START = 4 * _METADATA_HEADER_WORDS
_METADATA_MIN_VERSION = 2
_METADATA_MAX_VERSION = 4

# The object region holds one reserved entry followed by six words per object, of which the last two are the
# index of the property block and the identifier of the object itself.
_OBJECT_WORDS = 6
_OBJECT_PROPERTY_BLOCK = 4
_OBJECT_ID = 5

# A property is a name index, a kind, and a value whose meaning depends on the kind.
_PROPERTY_WORDS = 3
_PROPERTY_KIND_NAME = 0
_PROPERTY_KIND_CELL = 1
_PROPERTY_KIND_LITERAL = 2

# Values of kind ``_PROPERTY_KIND_CELL`` index the file wrapper cells from the third cell onwards: the first
# holds the metadata block and the second is an unused placeholder.
_FIRST_VALUE_CELL = 2

# The subsystem is a headerless MAT-file stream, so it needs a plausible 128 byte header in front of it
# before scipy will read it, and its single variable needs a name before scipy will return it.
_SUBSYSTEM_DESCRIPTION = b"MATLAB 5.0 MAT-file, subsystem".ljust(124, b" ")
_SUBSYSTEM_PREFIX_BYTES = 8
_SUBSYSTEM_NAME = "mcos"
_SUBSYSTEM_NAME_OFFSET = 40
_TYPE_INT8 = 1

_TABLE_CLASS = b"table"
_REQUIRED_PROPERTIES = ("data", "varnames", "nrows", "nvars")


def _read_element(buffer: bytes, position: int) -> tuple[int, bytes, int]:
    """Read one data element, returning its type, its payload, and the position of the next element."""
    header = struct.unpack("<I", buffer[position : position + 4])[0]
    if header >> 16:
        # Small data element format: the byte count and the type share the first four bytes.
        n_bytes, data_type = header >> 16, header & 0xFFFF
        return data_type, buffer[position + 4 : position + 4 + n_bytes], position + 8
    n_bytes = struct.unpack("<I", buffer[position + 4 : position + 8])[0]
    end = position + 8 + n_bytes
    return header, buffer[position + 8 : end], end + (-n_bytes % 8)


def _read_array_payload(matrix: bytes) -> bytes:
    """Skip the flags, dimensions and name of an array element and return its raw data."""
    position = 0
    for _ in range(3):
        _, _, position = _read_element(matrix, position)
    _, payload, _ = _read_element(matrix, position)
    return payload


def _iter_variables(raw: bytes, end: int) -> Iterator[bytes]:
    """Yield the body of every variable in a MAT-file, decompressing the ones that are deflated."""
    position = _HEADER_BYTES
    while position < end - 8:
        data_type, n_bytes = struct.unpack("<II", raw[position : position + 8])
        body = raw[position + 8 : position + 8 + n_bytes]
        position += 8 + n_bytes
        if data_type == _TYPE_COMPRESSED:
            _, body, _ = _read_element(zlib.decompress(body), 0)
        elif data_type != _TYPE_MATRIX:
            continue
        yield body


def _read_table_object_ids(raw: bytes, end: int) -> dict[str, int]:
    """Map the name of every ``table`` variable in a MAT-file to the identifier of its object."""
    object_ids = {}

    for body in _iter_variables(raw, end):
        _, flags, position = _read_element(body, 0)
        if not flags or flags[0] != _CLASS_OPAQUE:
            continue

        _, name, position = _read_element(body, position)
        _, _system, position = _read_element(body, position)
        _, class_name, position = _read_element(body, position)
        if class_name != _TABLE_CLASS:
            continue

        _, reference, _ = _read_element(body, position)
        words = np.frombuffer(_read_array_payload(reference), dtype="<u4")

        # The reference holds a marker, the shape of the object array, one identifier per object, and the
        # class. A table variable is always a single object.
        n_dimensions = int(words[1])
        identifiers = words[2 + n_dimensions : -1]
        if identifiers.size != 1:
            msg = f"Expected variable '{name.decode()}' to hold a single table, got {identifiers.size} objects."
            raise ValueError(msg)

        object_ids[name.decode()] = int(identifiers[0])

    return object_ids


def _read_subsystem_cells(raw: bytes, offset: int) -> np.ndarray:
    """Read the object system appended to a MAT-file and return its file wrapper cells."""
    data_type, body, _ = _read_element(raw, offset)
    if data_type == _TYPE_COMPRESSED:
        _, body, _ = _read_element(zlib.decompress(body), 0)
    workspace = _read_array_payload(body)

    # The workspace is a MAT-file stream whose only variable is unnamed. Its first four bytes carry the
    # version and endianness that a header would otherwise hold.
    stream = bytearray(workspace[_SUBSYSTEM_PREFIX_BYTES:])
    name = struct.pack("<I", (len(_SUBSYSTEM_NAME) << 16) | _TYPE_INT8) + _SUBSYSTEM_NAME.encode()
    stream[_SUBSYSTEM_NAME_OFFSET : _SUBSYSTEM_NAME_OFFSET + len(name)] = name

    header = _SUBSYSTEM_DESCRIPTION + workspace[:4]
    subsystem = loadmat(BytesIO(header + bytes(stream)), struct_as_record=False, squeeze_me=True)
    return np.asarray(subsystem[_SUBSYSTEM_NAME].MCOS["arr"][0]).ravel()


def _read_property_names(metadata: bytes, end: int, n_names: int) -> list[str]:
    names = metadata[_METADATA_STRING_START:end].split(b"\x00")[:n_names]
    return [name.decode() for name in names]


def _read_property_blocks(words: np.ndarray, names: list[str]) -> list[dict[str, tuple[int, int]]]:
    """Split the property region into one block of name, kind and value triples per object."""
    blocks = []
    position = 0

    while position < words.size:
        n_properties = int(words[position])
        position += 1

        block = {}
        for _ in range(n_properties):
            name_index, kind, value = (int(word) for word in words[position : position + _PROPERTY_WORDS])
            block[names[name_index - 1]] = (kind, value)
            position += _PROPERTY_WORDS

        blocks.append(block)
        # Every block is padded to a multiple of eight bytes, which is an even number of words.
        position += position % 2

    return blocks


def _read_object_properties(cells: np.ndarray) -> dict[int, dict[str, tuple[int, int]]]:
    """Map the identifier of every object in the object system to its properties."""
    metadata = np.asarray(cells[0]).ravel().tobytes()
    header = np.frombuffer(metadata[:_METADATA_STRING_START], dtype="<u4")

    version = int(header[0])
    if not _METADATA_MIN_VERSION <= version <= _METADATA_MAX_VERSION:
        msg = (
            f"Unsupported MATLAB object system version {version}, expected "
            f"{_METADATA_MIN_VERSION} to {_METADATA_MAX_VERSION}."
        )
        raise ValueError(msg)

    n_names = int(header[1])
    regions = [int(offset) for offset in header[2:8]]
    names = _read_property_names(metadata, regions[0], n_names)

    objects = np.frombuffer(metadata[regions[2] : regions[3]], dtype="<u4").reshape(-1, _OBJECT_WORDS)
    blocks = _read_property_blocks(np.frombuffer(metadata[regions[3] : regions[4]], dtype="<u4"), names)

    # The first entry of the object region is reserved.
    return {int(row[_OBJECT_ID]): blocks[int(row[_OBJECT_PROPERTY_BLOCK])] for row in objects[1:]}


def _resolve(value: tuple[int, int], cells: np.ndarray, names: list[str]) -> object:
    kind, index = value
    if kind == _PROPERTY_KIND_CELL:
        return cells[_FIRST_VALUE_CELL + index]
    if kind == _PROPERTY_KIND_NAME:
        return names[index - 1]
    if kind == _PROPERTY_KIND_LITERAL:
        return index
    msg = f"Unsupported MATLAB property kind {kind}."
    raise ValueError(msg)


def _to_scalar(value: object) -> object:
    """Unwrap the one by one arrays that MATLAB uses for scalars and strings."""
    array = np.asarray(value)
    if array.size == 0:
        return None
    if array.size == 1:
        return array.item()
    return array


def _to_column(values: object) -> object:
    """Turn one column of a MATLAB table into something a DataFrame accepts."""
    array = np.asarray(values)
    if array.dtype == object:
        return [_to_scalar(value) for value in array.ravel()]
    if array.ndim > 1 and array.shape[-1] > 1:
        # A column of a table may itself be a matrix, which only fits in a DataFrame row by row.
        return list(array.reshape(len(array), -1))
    return array.ravel()


def _to_frame(properties: dict[str, tuple[int, int]], cells: np.ndarray, names: list[str]) -> pd.DataFrame:
    missing = [name for name in _REQUIRED_PROPERTIES if name not in properties]
    if missing:
        msg = f"The MATLAB table is missing the {', '.join(missing)} propert{'y' if len(missing) == 1 else 'ies'}."
        raise ValueError(msg)

    columns = np.asarray(_resolve(properties["varnames"], cells, names)).ravel()
    data = np.asarray(_resolve(properties["data"], cells, names)).ravel()
    n_rows = int(np.asarray(_resolve(properties["nrows"], cells, names)).item())

    if columns.size != data.size:
        msg = f"The MATLAB table has {columns.size} column names but {data.size} columns."
        raise ValueError(msg)

    pairs = zip(columns, data, strict=True)
    frame = pd.DataFrame({str(_to_scalar(name)): _to_column(column) for name, column in pairs})
    if len(frame) != n_rows:
        msg = f"The MATLAB table reports {n_rows} rows but its columns hold {len(frame)}."
        raise ValueError(msg)

    return frame


def _validate_header(raw: bytes, path: str | os.PathLike) -> None:
    if len(raw) < _HEADER_BYTES or raw[_ENDIAN_OFFSET] != _LITTLE_ENDIAN:
        msg = f"{path} is not a little endian version 5 MAT-file."
        raise ValueError(msg)

    version = struct.unpack("<H", raw[_VERSION_OFFSET])[0]
    if version != _VERSION_5:
        msg = f"{path} uses MAT-file version {version:#06x}, but only version 5 ({_VERSION_5:#06x}) is supported."
        raise ValueError(msg)


def read_matlab_tables(path: str | os.PathLike) -> dict[str, pd.DataFrame]:
    """
    Read every MATLAB ``table`` variable in a MAT-file into a DataFrame.

    MATLAB stores a ``table`` as an object rather than as an array, which puts it out of reach of
    :func:`scipy.io.loadmat`: the tables of a file all collapse into a single ``None`` key that holds an
    opaque placeholder, and their contents end up in an undocumented ``__function_workspace__`` entry. This
    function reads that object system and rebuilds the tables, so a file can be opened with
    :func:`scipy.io.loadmat` for its arrays and with this function for its tables.

    Only version 5 MAT-files are supported, which is what MATLAB writes unless ``-v7.3`` is requested.

    Parameters
    ----------
    path : str or os.PathLike
        Path of the MAT-file to read.

    Returns
    -------
    dict of str to pandas.DataFrame
        The tables in the file, keyed by the name of the variable that holds them. Empty if the file holds
        no tables. Columns of text become columns of :class:`str`, and columns that are matrices in MATLAB
        become columns of :class:`numpy.ndarray`.

    Raises
    ------
    ValueError
        If the file is not a little endian version 5 MAT-file, if its object system uses an unsupported
        version, or if a table in it is not laid out as expected.

    Examples
    --------
    >>> from prfmodel.examples import read_matlab_tables
    >>> tables = read_matlab_tables("p01_freq_spectra.mat")  # doctest: +SKIP
    >>> sorted(tables)  # doctest: +SKIP
    ['channels', 'events']
    >>> tables["channels"].shape  # doctest: +SKIP
    (11, 48)
    >>> tables["channels"]["name"].tolist()[:4]  # doctest: +SKIP
    ['OT05', 'OT06', 'OT07', 'OT08']

    """
    raw = Path(path).read_bytes()
    _validate_header(raw, path)

    offset = struct.unpack("<Q", raw[_SUBSYSTEM_OFFSET])[0]
    if not offset:
        logger.info("%s holds no object system, so it holds no MATLAB tables.", path)
        return {}

    object_ids = _read_table_object_ids(raw, offset)
    if not object_ids:
        logger.info("Found no MATLAB tables in %s.", path)
        return {}

    cells = _read_subsystem_cells(raw, offset)
    properties = _read_object_properties(cells)
    metadata = np.asarray(cells[0]).ravel().tobytes()
    header = np.frombuffer(metadata[:_METADATA_STRING_START], dtype="<u4")
    names = _read_property_names(metadata, int(header[2]), int(header[1]))

    return {name: _to_frame(properties[object_id], cells, names) for name, object_id in object_ids.items()}
