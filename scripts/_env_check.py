"""Verify an extraction environment by doing the work, not by importing names.

Two earlier passes at this environment looked healthy and were not: pip resolved
against ~/.local and left holes that only appear under PYTHONNOUSERSITE=1, and the
dependency list was written from memory rather than from the script.  So this
check parses the real imports, then exercises the two things that actually fail in
production -- opening a whole-slide image through libvips, and building the
encoder from its local checkpoint.
"""
import ast, importlib, pathlib, sys

assert not any("/.local/" in p for p in sys.path), f"user site leaked in: {sys.path}"

SCRIPT = pathlib.Path("/mnt/isilon/oldridge_lab/leej/prenorm/src/extract_plism_core_pfm.py")
STD = {"argparse", "json", "sys", "pathlib", "os", "__future__"}
wanted = set()
for node in ast.walk(ast.parse(SCRIPT.read_text())):
    if isinstance(node, ast.Import):
        wanted.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        wanted.add(node.module.split(".")[0])
wanted -= STD | {"trident"}

missing = []
for name in sorted(wanted):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__} {exc}")
if missing:
    print("  MISSING: " + "; ".join(missing))
    sys.exit(1)

import torch, timm, numpy as np
print(f"  torch {torch.__version__}  timm {timm.__version__}  imports {sorted(wanted)}")

# a real slide, a real crop, the real Lanczos3 reduction the extraction uses
import pyvips
slide = "/mnt/isilon/oldridge_lab/leej/prenorm/data/PLISM_dataset/original_wsi/MY_AT2.svs"
image = pyvips.Image.new_from_file(slide, access="random")
# AT2's SVS carries an alpha channel; the extraction flattens it and so must this
# check, or it tests a different image than production reads.
if image.hasalpha():
    image = image.flatten(background=255)
mpp = float(image.get("openslide.mpp-x"))
side = int(round(512 * 0.5052 / mpp))
tile = image.crop(50000, 40000, side, side).reduce(0.5052 / mpp, 0.5052 / mpp, kernel="lanczos3")
array = tile.numpy()[:512, :512]
assert array.shape == (512, 512, 3), array.shape
print(f"  pyvips {pyvips.__version__}: read {side}px at {mpp:.5f} um/px -> {array.shape} ok")

# the encoder itself, from the frozen checkpoint, on CPU
if len(sys.argv) > 1:
    import json
    encoder_id = sys.argv[1]
    contract = json.loads(pathlib.Path(
        "/mnt/isilon/oldridge_lab/leej/prenorm/outputs/e9_pfm_contract/checkpoint_manifest.json"
    ).read_text())
    model = {m["encoder_id"]: m for m in contract["models"]}[encoder_id]
    sys.path.insert(0, contract["trident_root"])
    from trident.patch_encoder_models.load import encoder_factory
    from PIL import Image
    encoder = encoder_factory(encoder_id, weights_path=model["checkpoint_path"]).eval()
    fov = model["native_fov_px"]
    crop = array[(512 - fov) // 2:(512 + fov) // 2, (512 - fov) // 2:(512 + fov) // 2]
    batch = torch.stack([encoder.eval_transforms(Image.fromarray(crop, mode="RGB"))])
    with torch.inference_mode():
        out = encoder(batch.float())
    assert out.shape[-1] == model["feature_dim"], (out.shape, model["feature_dim"])
    print(f"  {encoder_id}: crop {fov}px -> input {tuple(batch.shape[-2:])} -> "
          f"feature {tuple(out.shape)} matches contract dim {model['feature_dim']} ok")
print("  ALL CHECKS PASS")
