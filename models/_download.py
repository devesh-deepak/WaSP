import hashlib
import os
import urllib.request


def download(url, path, sha256_prefix=None):
    """Download url to path (atomically) and optionally check the SHA-256 prefix."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'  Downloading {url} -> {path}', flush=True)
    tmp = path + '.part'
    urllib.request.urlretrieve(url, tmp)
    if sha256_prefix is not None:
        check_sha256(tmp, sha256_prefix)
    os.replace(tmp, path)
    return path


def check_sha256(path, prefix):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    if not h.hexdigest().startswith(prefix):
        raise RuntimeError(f'{path}: SHA-256 {h.hexdigest()[:16]}... does not start with {prefix}')
