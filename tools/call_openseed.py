# MSc-thesis/tools/call_openseed.py
#!/usr/bin/env python
"""
Run OpenSeeD from VLMaps.

• Always uses the openseed6 env (change the path if you rename it)
• Lets you override --cfg / --weight if you want another checkpoint
• Works both as an importable helper and as a CLI tool
"""

import subprocess, tempfile, numpy as np, os
from pathlib import Path
import argparse

# ----------------------------------------------------------------------
OPENSEED_ENV = "/home/jovyan/teaching_material/msc/envs/openseed6"
OPENSEED_PY  = "/home/jovyan/teaching_material/msc/OpenSeeD/tools/openseed_dump_pixels.py"
DEFAULT_CFG  = "/home/jovyan/teaching_material/msc/OpenSeeD/configs/openseed/openseed_swinl_lang_decouple.yaml"
DEFAULT_WGT  = "/home/jovyan/teaching_material/msc/OpenSeeD/weights/openseed_swinl_pano_sota.pt"
# ----------------------------------------------------------------------

def _run_openseed(img:str, cfg:str, wgt:str) -> Path:
    """spawn the extractor, return path to the tmp .npy file"""
    img  = os.path.abspath(img)
    cfg  = cfg or DEFAULT_CFG
    wgt  = wgt or DEFAULT_WGT
    dump = Path(tempfile.mktemp(suffix=".npy"))

    cmd = [
        "conda", "run", "-p", OPENSEED_ENV,
        "python", OPENSEED_PY,
        "--cfg",   cfg,
        "--weight",wgt,
        "--image", img,
        "--out",   str(dump)
    ]
    subprocess.run(cmd, check=True)
    return dump


def openseed_pixels(image_path:str, cfg:str=None, weight:str=None) -> np.ndarray:
    """import-friendly helper that just returns the array"""
    npy = _run_openseed(image_path, cfg, weight)
    return np.load(npy)


# --------------------------- CLI section ---------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Call OpenSeeD from VLMaps")
    ap.add_argument("--image", required=True)
    ap.add_argument("--cfg",   default=None,
                    help="YAML; overrides the default Swin-L config")
    ap.add_argument("--weight",default=None,
                    help="Checkpoint; overrides the default Swin-L weight")
    ap.add_argument("--out",   default=None,
                    help="Optional .npy path to save")
    ap.add_argument("--summary", action="store_true", help="print shape/min/max")

    args = ap.parse_args()
    feats = openseed_pixels(args.image, args.cfg, args.weight)

    if args.summary or args.out is None:
        print("shape:", feats.shape,
              "| dtype:", feats.dtype,
              "| range:", (feats.min(), feats.max()))
    if args.out:
        np.save(args.out, feats)
        print("saved →", os.path.abspath(args.out))