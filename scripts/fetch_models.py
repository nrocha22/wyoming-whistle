"""Fetch Whistle weights and the native engine at pinned revisions, verify sha256.

Runs at image build time. The running server never touches the network.

    python scripts/fetch_models.py --arch amd64 --out /models --checksums models/CHECKSUMS.sha256
    python scripts/fetch_models.py --arch amd64 --out /tmp/m --write-checksums models/CHECKSUMS.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile

from huggingface_hub import hf_hub_download

WEIGHTS_REPO = "Cactus-Compute/whistle"
WEIGHTS_REVISION = "ca5287601bef25af26dcf2e1b2bdc0843a7c19e5"
WEIGHTS_FILE = "whistle.cact"

ENGINE_REPO = "Cactus-Compute/needle3"
ENGINE_REVISION = "93e88ede867e4311c39151285e74df513e4b835b"
# The engine version cactus-needle 3.2.0 pins (needle/agent/fetch.py, ENGINE_VERSIONS[3]).
ENGINE_VERSION = "3.3.0"
ENGINE_MEMBER = "needle/libneedle3.so"
ENGINE_LIB = "libneedle3.so"

ARCH_TAGS = {"amd64": "manylinux2014_x86_64", "arm64": "manylinux2014_aarch64"}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(arch: str, out: str) -> dict:
    os.makedirs(out, exist_ok=True)

    weights = hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE, revision=WEIGHTS_REVISION)
    shutil.copyfile(weights, os.path.join(out, WEIGHTS_FILE))

    wheel_name = f"python/cactus_needle-{ENGINE_VERSION}-py3-none-{ARCH_TAGS[arch]}.whl"
    wheel = hf_hub_download(ENGINE_REPO, wheel_name, revision=ENGINE_REVISION)
    with zipfile.ZipFile(wheel) as archive, open(os.path.join(out, ENGINE_LIB), "wb") as lib:
        lib.write(archive.read(ENGINE_MEMBER))

    return {
        WEIGHTS_FILE: sha256(os.path.join(out, WEIGHTS_FILE)),
        f"{ENGINE_LIB}:{arch}": sha256(os.path.join(out, ENGINE_LIB)),
    }


def read_checksums(path: str) -> dict:
    sums = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                digest, name = line.split()
                sums[name] = digest
    return sums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=sorted(ARCH_TAGS))
    parser.add_argument("--out", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checksums", help="verify against this file")
    group.add_argument("--write-checksums", help="add this arch's sums to this file")
    args = parser.parse_args()

    got = fetch(args.arch, args.out)

    if args.write_checksums:
        sums = read_checksums(args.write_checksums) if os.path.exists(args.write_checksums) else {}
        sums.update(got)
        with open(args.write_checksums, "w", encoding="utf-8") as handle:
            handle.write(f"# weights {WEIGHTS_REPO}@{WEIGHTS_REVISION}\n")
            handle.write(f"# engine {ENGINE_REPO}@{ENGINE_REVISION} v{ENGINE_VERSION}\n")
            for name in sorted(sums):
                handle.write(f"{sums[name]}  {name}\n")
        print(f"wrote {args.write_checksums}")
        return

    expected = read_checksums(args.checksums)
    bad = [name for name, digest in got.items() if expected.get(name) != digest]
    if bad:
        for name in bad:
            print(f"checksum mismatch for {name}: got {got[name]}, expected {expected.get(name)}", file=sys.stderr)
        sys.exit(1)
    print("checksums ok:", ", ".join(sorted(got)))


if __name__ == "__main__":
    main()
