"""Authenticated encrypted archives for public-code runners; no plaintext results."""
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import zipfile
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b'LRSCREEN1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def seal(data, key, header):
    meta = json.dumps(header, sort_keys=True, separators=(',',':')).encode()
    nonce = os.urandom(12)
    return MAGIC+struct.pack('>I',len(meta))+meta+nonce+AESGCM(key).encrypt(nonce,data,meta)


def unseal(packet, key, expected):
    if not packet.startswith(MAGIC):
        raise ValueError('Unexpected encrypted archive format')
    n = struct.unpack('>I', packet[len(MAGIC):len(MAGIC)+4])[0]
    if n > 16384:
        raise ValueError('Oversized envelope metadata')
    start = len(MAGIC)+4
    meta = packet[start:start+n]
    header = json.loads(meta)
    if any(header.get(k) != v for k,v in expected.items()):
        raise ValueError('Archive origin differs from request')
    nonce = packet[start+n:start+n+12]
    return AESGCM(key).decrypt(nonce,packet[start+n+12:],meta), header


def pack(folder):
    out = io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(Path(folder).rglob('*')):
            if (p.is_file() and '__pycache__' not in p.parts and p.suffix != '.tmp'
                    and p.name not in ('STOP','STOP_AFTER_UPDATE','.worker.lock','owner.lock')):
                z.write(p,p.relative_to(folder).as_posix())
    return out.getvalue()


def unpack(data, folder):
    root = Path(folder).resolve()
    if root.exists():
        raise FileExistsError('Never overwrite a received segment')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set()
        for item in z.infolist():
            dest = (root/item.filename).resolve()
            if root not in dest.parents or item.filename in names or (item.external_attr >> 16)&0o170000 == 0o120000:
                raise ValueError('Unsafe archive entry')
            names.add(item.filename)
        if z.testzip() is not None:
            raise ValueError('Archive CRC failure')
        root.mkdir(parents=True)
        z.extractall(root)
