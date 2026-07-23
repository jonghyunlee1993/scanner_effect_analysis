"""Version 3 RGB-only per-slide HDF5 store.

Layout (row ``i`` identifies the same registered coordinate for every scanner):

``coords``       ``(T, 2)`` int64
``tile_id``      ``(T,)`` int64
``present``      ``(T, S)`` uint8
``geom_ok``      ``(T, S)`` uint8
``q_reg``        ``(T, S)`` float32
``<scanner>/rgb`` ``(T,)`` PNG-encoded variable-length bytes

Derived model inputs are intentionally absent. They are built differentiably from
RGB during training.
"""
import cv2
import h5py
import numpy as np

SCHEMA_VERSION = 3
CHANNELS = ("rgb",)
_VLEN = h5py.vlen_dtype(np.uint8)


def encode(arr):
    """Encode a uint8 image as a one-dimensional PNG byte array."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.reshape(-1)


def decode(buf):
    """Decode a PNG byte array into a uint8 image."""
    raw = np.frombuffer(np.asarray(buf, np.uint8), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)


def write_slide(path, slide_id, scanners, records, attrs):
    """Write RGB tuples and v3 supervision fields to one slide store."""
    t = len(records)
    with h5py.File(path, "w") as handle:
        handle.attrs["slide_id"] = slide_id
        handle.attrs["scanners"] = list(scanners)
        for key, value in attrs.items():
            handle.attrs[key] = value
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.create_dataset("coords", data=np.asarray([r["coords"] for r in records], np.int64))
        handle.create_dataset("tile_id", data=np.asarray([r["tile_id"] for r in records], np.int64))
        handle.create_dataset("present", data=np.asarray([r["present"] for r in records], np.uint8))
        handle.create_dataset("geom_ok", data=np.asarray([r["geom_ok"] for r in records], np.uint8))
        handle.create_dataset("q_reg", data=np.asarray([r["q_reg"] for r in records], np.float32))
        for scanner in scanners:
            dataset = handle.create_group(scanner).create_dataset("rgb", (t,), dtype=_VLEN)
            for i, record in enumerate(records):
                dataset[i] = encode(record[scanner]["rgb"])


def read_tuple(handle, scanners, i):
    """Decode the RGB image for tuple ``i`` from every scanner."""
    return {scanner: {"rgb": decode(handle[scanner]["rgb"][i])} for scanner in scanners}
