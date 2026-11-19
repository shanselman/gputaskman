import os
import struct
import tempfile
import unittest

from gputaskman.gguf import _read_cached

STRING = 8


def gguf_bytes(kvs: list[tuple[bytes, bytes]]) -> bytes:
    """A v3 GGUF header with string key/values and no tensors."""
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for key, value in kvs:
        out += struct.pack("<Q", len(key)) + key + struct.pack("<I", STRING) + struct.pack("<Q", len(value)) + value
    return out


class GgufHeaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = os.path.join(directory.name, "model.gguf")

    def read(self, data: bytes, size: int | None = None):
        with open(self.path, "wb") as f:
            f.write(data)
            if size:
                f.truncate(size)
        st = os.stat(self.path)
        return _read_cached(self.path, st.st_size, st.st_mtime)

    def test_reads_string_metadata(self):
        info = self.read(gguf_bytes([(b"general.architecture", b"llama"), (b"general.name", b"Tiny")]))
        self.assertIsNone(info.error)
        self.assertEqual((info.arch, info.name), ("llama", "Tiny"))

    def test_corrupt_key_length_is_an_error_not_a_huge_read(self):
        # A key claiming 8 MiB inside a 9 MiB file: no real key is that long, so stop instead of reading it.
        header = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 8 << 20)
        info = self.read(header, size=9 << 20)
        self.assertIsNotNone(info.error)
        self.assertEqual(info.metadata, {})

    def test_huge_string_value_is_skipped_and_parsing_continues(self):
        big = b"x" * (2 << 20)  # e.g. an embedded tokenizer.json
        info = self.read(gguf_bytes([(b"tokenizer.huggingface.json", big), (b"general.name", b"Tiny")]))
        self.assertIsNone(info.error)
        self.assertEqual(info.name, "Tiny")
        self.assertEqual(info.metadata["tokenizer.huggingface.json"], f"<string[{len(big)} bytes]>")

    def test_string_cut_short_is_truncation(self):
        header = gguf_bytes([(b"general.name", b"Tiny")])
        info = self.read(header[:-2])
        self.assertIn("EOFError", info.error)


if __name__ == "__main__":
    unittest.main()
