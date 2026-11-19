"""Typed, deterministic audit fingerprints including unordered environment sets."""
from collections import deque
import hashlib
import sys


def fingerprint(value):
    np = sys.modules.get('numpy')
    torch = sys.modules.get('torch')
    result = hashlib.sha256()

    def feed(tag, data=b''):
        tag = tag.encode('utf-8')
        result.update(str(len(tag)).encode()+b':'+tag)
        result.update(str(len(data)).encode()+b':'+data)

    def visit(v):
        if torch is not None and isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            feed('tensor',str((a.dtype,a.shape)).encode())
            feed('data',a.tobytes())
        elif np is not None and isinstance(v,np.ndarray):
            feed('ndarray',str((v.dtype,v.shape)).encode())
            feed('data',v.tobytes())
        elif np is not None and isinstance(v,np.generic):
            feed('numpy-scalar',str(v.dtype).encode())
            visit(v.item())
        elif isinstance(v,dict):
            feed('dict',str(len(v)).encode())
            for key in sorted(v,key=repr):
                visit(key);visit(v[key])
        elif isinstance(v,(set,frozenset)):
            feed(type(v).__name__,str(len(v)).encode())
            for item in sorted(v,key=repr):
                visit(item)
        elif isinstance(v,(list,tuple,deque)):
            feed(type(v).__name__,str(len(v)).encode())
            for item in v:
                visit(item)
        elif isinstance(v,(str,int,float,bool,bytes,type(None))):
            feed(type(v).__name__,repr(v).encode())
        elif hasattr(v,'__dict__') and not isinstance(v,type):
            feed('object',type(v).__qualname__.encode())
            visit(vars(v))
        else:
            raise TypeError('Unsupported checkpoint audit type '+str(type(v)))
    visit(value)
    return result.hexdigest()
