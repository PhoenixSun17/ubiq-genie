"""Install-time model downloader; never imports ASR/VAD/CUDA modules.

The repository is the large-v3-turbo alias in Faster-Whisper v1.2.1. Its current
canonical Hugging Face owner is recorded separately because the alias redirects.
All downloads use an immutable revision, resume a local .partial file, and are
verified against published LFS SHA-256 or Git blob SHA-1 metadata before use.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote


MODEL_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
MODEL_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.json")


@contextmanager
def download_lock(model_dir):
    """Let Windows release the lock even if an installer is interrupted."""
    import msvcrt

    with (model_dir / ".download.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another ASR model installer is already downloading into this directory") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def curl(arguments, *, capture=False):
    result = subprocess.run(
        ["curl.exe", "-L", "--fail", "--show-error", "--silent", *arguments],
        stdout=subprocess.PIPE if capture else None,
        stderr=None,
        check=True,
    )
    return result.stdout


def metadata():
    url = f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true"
    payload = json.loads(curl(["--max-time", "90", "--retry", "3", url], capture=True))
    if payload.get("sha") != MODEL_REVISION or payload.get("private") or payload.get("gated"):
        raise RuntimeError("Expected the pinned public, ungated Faster-Whisper model revision")
    canonical = payload.get("id")
    if not isinstance(canonical, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", canonical):
        raise RuntimeError("Model metadata has an invalid canonical repository ID")
    siblings = {entry["rfilename"]: entry for entry in payload.get("siblings", [])}
    missing = [name for name in REQUIRED_FILES if name not in siblings]
    if missing:
        raise RuntimeError("Pinned model is missing required files: " + ", ".join(missing))
    return canonical, siblings


def verified_hashes(path, entry):
    size = path.stat().st_size
    expected_size = entry.get("size")
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"Model file size mismatch for {path.name}: {size} != {expected_size}")
    sha256 = hashlib.sha256()
    blob_sha1 = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            sha256.update(chunk)
            blob_sha1.update(chunk)
    digest = sha256.hexdigest()
    lfs = entry.get("lfs") or {}
    expected_sha256 = lfs.get("sha256") or lfs.get("oid")
    if expected_sha256:
        expected_sha256 = expected_sha256.removeprefix("sha256:")
        if digest != expected_sha256:
            raise RuntimeError(f"Model file SHA-256 mismatch: {path.name}")
    elif entry.get("blobId"):
        if blob_sha1.hexdigest() != entry["blobId"]:
            raise RuntimeError(f"Model file Git SHA-1 mismatch: {path.name}")
    else:
        raise RuntimeError(f"No upstream content hash available for model file: {path.name}")
    return {"sizeBytes": size, "sha256": digest}


def download_file(canonical, name, entry, model_dir):
    destination = model_dir / name
    if destination.is_file():
        print(f"Verifying existing ASR asset: {name}", flush=True)
        return verified_hashes(destination, entry)
    partial = destination.with_name(destination.name + ".partial")
    if partial.is_file() and partial.stat().st_size >= entry.get("size", float("inf")):
        try:
            hashes = verified_hashes(partial, entry)
        except RuntimeError:
            print(f"Restarting invalid temporary ASR asset: {partial.name}", flush=True)
            partial.unlink()
        else:
            partial.replace(destination)
            return hashes
    url = f"https://huggingface.co/{canonical}/resolve/{MODEL_REVISION}/{quote(name)}?download=true"
    print(f"Downloading ASR asset: {name} ({entry.get('size', 'unknown')} bytes)", flush=True)
    curl([
        "--connect-timeout", "30", "--retry", "4", "--retry-delay", "3", "--retry-all-errors",
        "--speed-time", "120", "--speed-limit", "1024", "--continue-at", "-", "--output", str(partial), url,
    ])
    hashes = verified_hashes(partial, entry)
    partial.replace(destination)
    print(f"Verified ASR asset: {name}", flush=True)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--requirements", required=True)
    args = parser.parse_args()
    runtime_root = Path(args.runtime_root).resolve()
    model_dir = runtime_root / "models" / "faster-whisper-large-v3-turbo"
    model_dir.mkdir(parents=True, exist_ok=True)
    with download_lock(model_dir):
        install_model(runtime_root, model_dir, Path(args.requirements).resolve())


def install_model(runtime_root, model_dir, requirements):
    canonical, siblings = metadata()
    artifacts = {}
    for name in REQUIRED_FILES:
        hashes = download_file(canonical, name, siblings[name], model_dir)
        artifacts[name] = {"path": str(model_dir / name), **hashes}
    manifest = {
        "schemaVersion": 1,
        "installedAtUtc": datetime.now(timezone.utc).isoformat(),
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version,
        "requestedRepoId": MODEL_REPO,
        "resolvedRepoId": canonical,
        "revision": MODEL_REVISION,
        "modelPath": str(model_dir),
        "requirementsPath": str(requirements),
        "requirementsSha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "installedPackages": dict(sorted((distribution.metadata["Name"], distribution.version)
                                         for distribution in importlib.metadata.distributions())),
        "artifacts": artifacts,
        "runtimeValidation": "not_run",
        "modelInference": "not_run",
    }
    manifest_path = runtime_root / "asr-manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Saved ASR installation manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
