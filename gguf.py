r"""
Zoomies - reading a .gguf header.

Everything here comes from the header alone: the metadata keys and the table
of tensors. No weights are read, so it is fast and never touches a GPU.

The header is cached (by path, size and modified time) in
%LOCALAPPDATA%\Zoomies\cache\gguf.json, trimmed to what Zoomies uses: the
scalar metadata, short arrays such as per-layer head counts, the length of
long ones such as the vocabulary, the chat template, and tensor bytes summed
per layer. Finding the chat template means stepping over a few hundred
thousand vocabulary strings, which is worth doing once, not per click.
"""

import os
import re
import struct
import threading

import state

CACHE_PATH = os.path.join(state.CACHE_DIR, "gguf.json")
CACHE_VERSION = 1

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
SHORT_ARRAY = 1024              # longer arrays keep only their length

_LAYER = re.compile(r"^blk\.(\d+)\.")

_lock = threading.Lock()
_cache = None


class Header:
    def __init__(self, data):
        self.kv = data["kv"]                   # key -> value (or array length)
        self.lengths = data["lengths"]         # long array key -> length
        self.layer_bytes = data["layer_bytes"]  # {"0": bytes, ...}
        self.other_bytes = data["other_bytes"]  # tensor name -> bytes
        self.total_bytes = data["total_bytes"]

    @property
    def arch(self):
        return str(self.kv.get("general.architecture") or "")

    def get(self, key, default=None):
        """An architecture key without its prefix: get("block_count")."""
        return self.kv.get("%s.%s" % (self.arch, key), default)

    @property
    def chat_template(self):
        return self.kv.get("tokenizer.chat_template") or ""


def read(path):
    """The Header for a .gguf file. Raises OSError / ValueError."""
    global _cache
    st = os.stat(path)
    key = "%s|%d|%d" % (os.path.normcase(path), st.st_size, int(st.st_mtime))
    with _lock:
        if _cache is None:
            saved = state.read_json(CACHE_PATH, {}) or {}
            _cache = (saved.get("files") or {}) \
                if saved.get("version") == CACHE_VERSION else {}
        if key in _cache:
            return Header(_cache[key])
    data = _parse(path, st.st_size)
    with _lock:
        _cache[key] = data
        state.write_json(CACHE_PATH, {"version": CACHE_VERSION, "files": _cache})
    return Header(data)


def _parse(path, file_size):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError("GGUF version %d is too old" % version)
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        r = _Reader(f)
        kv, lengths = {}, {}
        for _ in range(n_kv):
            key = r.string().decode("utf-8", "replace")
            kind = r.u32()
            value, length = r.value(kind)
            if length is not None:
                lengths[key] = length
            else:
                kv[key] = value
        tensors = []
        for _ in range(n_tensors):
            name = r.string().decode("utf-8", "replace")
            n_dims = r.u32()
            r.take(8 * n_dims)
            r.u32()                                    # ggml type
            tensors.append((name, r.u64()))
        align = int(kv.get("general.alignment") or 32)
        data_start = r.offset()
        data_start += (-data_start) % align
    # Sizes from the offsets: each tensor runs to the next one, which is
    # exact for every quant type without a table of block sizes.
    tensors.sort(key=lambda t: t[1])
    layer_bytes, other_bytes = {}, {}
    for i, (name, offset) in enumerate(tensors):
        end = tensors[i + 1][1] if i + 1 < len(tensors) else file_size - data_start
        size = end - offset
        m = _LAYER.match(name)
        if m:
            layer_bytes[m.group(1)] = layer_bytes.get(m.group(1), 0) + size
        else:
            other_bytes[name] = size
    return {"kv": kv, "lengths": lengths, "layer_bytes": layer_bytes,
            "other_bytes": other_bytes,
            "total_bytes": sum(layer_bytes.values()) + sum(other_bytes.values())}


class _Reader:
    """Reads through a large buffer - stepping over a vocabulary one f.read()
    per token is what makes the naive version slow."""

    CHUNK = 1 << 22

    def __init__(self, f):
        self.f, self.buf, self.pos, self.base = f, b"", 0, f.tell()

    def offset(self):
        return self.base + self.pos

    def take(self, n):
        if self.pos + n > len(self.buf):
            self.base += self.pos
            self.buf = self.buf[self.pos:] + self.f.read(max(n, self.CHUNK))
            self.pos = 0
            if n > len(self.buf):
                raise ValueError("GGUF header ends early")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64())

    def value(self, kind):
        """(value, None), or (None, length) for an array too long to keep."""
        if kind == _STRING:
            return self.string().decode("utf-8", "replace"), None
        if kind in _SCALAR:
            fmt = _SCALAR[kind]
            return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0], None
        if kind != _ARRAY:
            raise ValueError("unsupported GGUF value type %d" % kind)
        item, count = self.u32(), self.u64()
        if count > SHORT_ARRAY:
            if item == _STRING:
                for _ in range(count):
                    self.take(self.u64())
            elif item in _SCALAR:
                self.take(struct.calcsize(_SCALAR[item]) * count)
            else:
                raise ValueError("unsupported GGUF array type %d" % item)
            return None, count
        return [self.value(item)[0] for _ in range(count)], None
