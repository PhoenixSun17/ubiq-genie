"""Download pinned TTS assets only. Never import or run application/model code."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

MODEL_REVISION = "8f8857563c040d542de108a2c18b0425abd6c9c5"
EXTRA_REVISION = "0c47645e02a7bc3688d7b263b0042c81e3cd82cd"
MODEL_FILES = [
    "chinese-hubert-base/config.json", "chinese-hubert-base/preprocessor_config.json",
    "chinese-hubert-base/pytorch_model.bin", "chinese-roberta-wwm-ext-large/config.json",
    "chinese-roberta-wwm-ext-large/pytorch_model.bin", "chinese-roberta-wwm-ext-large/tokenizer.json",
    "s1v3.ckpt", "v2Pro/s2Gv2Pro.pth", "sv/pretrained_eres2netv2w24s4ep4.ckpt",
]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(entry):
    url, target, logs, metadata = entry
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_name(target.name + ".partial")
        log = logs / (hashlib.sha256(url.encode()).hexdigest()[:12] + ".download.log")
        with log.open("ab") as output:
            subprocess.run(["curl.exe", "-L", "--fail", "--show-error", "--retry", "4",
                            "--retry-delay", "3", "-C", "-", "-o", str(partial), url],
                           stdout=output, stderr=output, check=True)
        partial.replace(target)
    file_hash = digest(target)
    if target.stat().st_size != metadata["size"]:
        raise ValueError(f"Size mismatch for {target}; refusing an incomplete asset")
    if metadata.get("lfs"):
        if file_hash != metadata["lfs"]["sha256"]:
            raise ValueError(f"Upstream SHA256 mismatch for {target}")
    else:
        git_hash = hashlib.sha1(f"blob {target.stat().st_size}\0".encode() + target.read_bytes()).hexdigest()
        if git_hash != metadata["blobId"]:
            raise ValueError(f"Upstream Git blob hash mismatch for {target}")
    print(f"Verified {target.name}: {target.stat().st_size} bytes", flush=True)
    return {"url": url, "path": str(target), "bytes": target.stat().st_size, "sha256": file_hash, "upstreamHashVerified": True}


def repo_metadata(repo, revision):
    response = subprocess.run(["curl.exe", "-L", "--fail", "--silent", "--show-error",
                               f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"],
                              check=True, capture_output=True, text=True, encoding="utf-8")
    data = json.loads(response.stdout)
    if data["sha"] != revision:
        raise ValueError("Upstream model revision does not match the pinned revision")
    return {entry["rfilename"]: entry for entry in data["siblings"]}


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            (root / member.filename).resolve().relative_to(root)
        package.extractall(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    args = parser.parse_args()
    root = Path(args.runtime_root).resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    models = root / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"
    main_metadata = repo_metadata("lj1995/GPT-SoVITS", MODEL_REVISION)
    extra_metadata = repo_metadata("XXXXRT/GPT-SoVITS-Pretrained", EXTRA_REVISION)
    entries = [(f"https://huggingface.co/lj1995/GPT-SoVITS/resolve/{MODEL_REVISION}/{name}", models / name, logs, main_metadata[name])
               for name in MODEL_FILES]
    for name in ["G2PWModel.zip", "nltk_data.zip"]:
        entries.append((f"https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/{EXTRA_REVISION}/{name}", root / "downloads" / name, logs, extra_metadata[name]))
    for name in ["lid.176.bin", "lid.176.ftz"]:
        entries.append((f"https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/{EXTRA_REVISION}/pretrained_models/fast_langdetect/{name}", models / "fast_langdetect" / name, logs, extra_metadata[f"pretrained_models/fast_langdetect/{name}"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, entries))
    text = root / "GPT-SoVITS" / "GPT_SoVITS" / "text"
    if not (text / "G2PWModel" / "g2pW.onnx").exists():
        extract(root / "downloads" / "G2PWModel.zip", text)
        if (text / "G2PWModel_1.1").exists() and not (text / "G2PWModel").exists():
            (text / "G2PWModel_1.1").rename(text / "G2PWModel")
    if not (root / "tools" / "nltk_data").exists():
        extract(root / "downloads" / "nltk_data.zip", root / "tools")
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    (receipts / "tts-models.json").write_text(json.dumps({
        "modelRepository": "lj1995/GPT-SoVITS", "modelRevision": MODEL_REVISION,
        "extraRepository": "XXXXRT/GPT-SoVITS-Pretrained", "extraRevision": EXTRA_REVISION,
        "files": files, "inferenceValidated": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
