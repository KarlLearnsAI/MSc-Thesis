# MSc-thesis/tools/call_openseed_text.py
#!/usr/bin/env python
"""
Return 512-D ℓ2-normalised text embeddings from OpenSeeD
by spawning the openseed6 conda env.

usage from VLMaps code:
    from tools.call_openseed_text import openseed_text
    z = openseed_text(["door","chair","ground"])   # (K,256)
"""
import subprocess, tempfile, numpy as np, os, sys
from pathlib import Path

OPENSEED_ENV = "/home/jovyan/teaching_material/msc/envs/openseed6"
SCRIPT       = "/home/jovyan/teaching_material/msc/OpenSeeD/tools/openseed_dump_text.py" # dump_text.py
CFG          = "/home/jovyan/teaching_material/msc/OpenSeeD/configs/openseed/openseed_swinl_lang_decouple.yaml"
WGT          = "/home/jovyan/teaching_material/msc/OpenSeeD/weights/openseed_swinl_pano_sota.pt"

def _run(labels, cfg, wgt):
    out = Path(tempfile.mktemp(suffix=".npy"))
    cmd = [
        "conda","run","-p",OPENSEED_ENV,"python",SCRIPT,
        "--cfg",cfg,"--weight",wgt,"--labels",",".join(labels),
        "--out",str(out)
    ]
    subprocess.run(cmd, check=True)
    return np.load(out)

def openseed_text(labels, cfg: str = CFG, weight: str = WGT):
    labels = [s.strip() for s in labels]
    return _run(labels, cfg, weight) # (K,512)