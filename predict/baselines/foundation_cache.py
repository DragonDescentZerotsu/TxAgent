"""Optional, checksum-verified local copies of immutable foundation inputs."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from predict.utils.json import sha256_file, write_json_atomic


def cached_input(source, binding):
    """Keep canonical paths in run identities; cache only bytes used for reading.

    A file binding is its expected SHA256. Directory bindings are pinned model
    releases with a primary weight path and SHA256. One host-local lock protects
    first publication; subsequent reads validate local file sizes and mtimes.
    """
    source = Path(source)
    cache = os.environ.get("TXAGENT_FOUNDATION_CACHE")
    if not cache:
        return source
    key = hashlib.sha256(json.dumps([str(source.resolve()), binding], sort_keys=True).encode()).hexdigest()
    root = Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = Path(tempfile.mkdtemp(prefix=key + ".", dir=root))
            try:
                directory = isinstance(binding, dict)
                files = sorted(p for p in source.rglob("*") if p.is_file() and
                               not {"__pycache__", ".cache", ".git"}.intersection(p.relative_to(source).parts)) if directory else [source]
                receipt = dict(binding=binding, source=str(source), files={})
                for file in files:
                    relative = file.relative_to(source) if directory else Path("payload")
                    destination = temporary / "data" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = sha256_file(file) if directory else binding
                    shutil.copyfile(file, destination)
                    if sha256_file(destination) != digest:
                        raise ValueError(f"Input changed during local caching: {file}")
                    destination.chmod(0o444)
                    stat = destination.stat()
                    receipt["files"][str(relative)] = dict(sha256=digest, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                if directory and receipt["files"][binding["weight"]]["sha256"] != binding["sha256"]:
                    raise ValueError(f"Pretrained checkpoint differs from pinned release: {source}")
                write_json_atomic(temporary / "receipt.json", receipt)
                temporary.rename(target)
            except BaseException:
                shutil.rmtree(temporary)
                raise
        receipt = json.loads((target / "receipt.json").read_text())
        if receipt["binding"] != binding or receipt["source"] != str(source):
            raise ValueError("Local input cache binding changed")
        for relative, record in receipt["files"].items():
            stat = (target / "data" / relative).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
                raise ValueError(f"Local cached input changed: {target / relative}")
    return target / "data" if isinstance(binding, dict) else target / "data/payload"
