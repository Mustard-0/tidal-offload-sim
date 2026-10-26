import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from cryptography.exceptions import InvalidTag
from .transport import seal, unseal, pack, unpack


class TransportTests(unittest.TestCase):
    def test_authenticated_encryption_roundtrip_and_origin(self):
        key = bytes(range(32))
        packet = seal(b'private synthetic fixture',key,dict(cell='TMAPO_LR1',segment=1))
        raw, _ = unseal(packet,key,dict(cell='TMAPO_LR1',segment=1))
        self.assertEqual(raw,b'private synthetic fixture')
        self.assertNotIn(raw,packet)
        with self.assertRaises(ValueError):
            unseal(packet,key,dict(segment=2))
        with self.assertRaises(InvalidTag):
            unseal(packet[:-1]+bytes([packet[-1]^1]),key,{})

    def test_archive_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'source'
            source.mkdir()
            (source/'a.txt').write_text('fixture')
            raw = pack(source)
            unpack(raw,root/'target')
            self.assertEqual((root/'target/a.txt').read_text(),'fixture')
            with self.assertRaises(FileExistsError):
                unpack(raw,root/'target')

    def test_zip_traversal_rejected_before_extraction(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:
            z.writestr('../escape.txt','bad')
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'target'
            with self.assertRaises(ValueError):
                unpack(buffer.getvalue(),target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
