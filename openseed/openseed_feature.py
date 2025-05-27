# used for ONE COMBINED CONDA ENV
# -------------------------------------------------------------
# openseed_feature.py  (put this anywhere on the VLMaps PYTHONPATH)
# -------------------------------------------------------------
import torch, numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from openseed import build_model            # your OpenSeeD repo
from openseed.BaseModel import BaseModel
import yaml, os

# ---------- build & cache the model once ----------------------
_OPENSEED_MODEL = None                       # global singleton

def _load_openseed(cfg_file, ckpt, device):
    global _OPENSEED_MODEL
    if _OPENSEED_MODEL is not None:
        return _OPENSEED_MODEL               # reuse
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    cfg["WEIGHT"] = ckpt
    cfg.setdefault("MODEL", {})["DEVICE"] = device
    _OPENSEED_MODEL = (
        BaseModel(cfg, build_model(cfg))
        .from_pretrained(ckpt)
        .eval()
        .to(device)
    )
    # --- tiny patches exactly as in openseed_dump_pixels.py ----
    from openseed_dump_pixels import (
        _ensure_dummy_text_embeddings,
        _patch_openseed_encoder,
    )
    _ensure_dummy_text_embeddings(_OPENSEED_MODEL.model, device)
    _patch_openseed_encoder(_OPENSEED_MODEL.model)
    return _OPENSEED_MODEL

# ---------- the public API: *same* as get_lseg_feat -----------    
@torch.no_grad()
def get_openseed_feat(
        rgb_np,                              # <-- H×W×3 uint8, BGR/RGB both OK
        model_cfg   = "/…/openseed_swinl_lang_decouple.yaml",
        model_ckpt  = "/…/openseed_swinl_pano_sota.pt",
        device      = "cuda" if torch.cuda.is_available() else "cpu",
        **_ignored,                          # swallow LSeg-specific kwargs
):
    """
    Drop-in replacement for get_lseg_feat().
    Returns (C, H, W) float32 ℓ2-normalised OpenSeeD pixel embeddings.
    """
    # ---- load model (singleton) ------------------------------------------
    model = _load_openseed(model_cfg, model_ckpt, device)

    # ---- convert the input exactly as VLMaps gives it --------------------
    if isinstance(rgb_np, Image.Image):
        pil = rgb_np.convert("RGB")
    else:
        pil = Image.fromarray(rgb_np[..., ::-1] if rgb_np.shape[2] == 3 else rgb_np)

    H, W = pil.height, pil.width
    x = ToTensor()(pil) * 255  # to 0-255 like OpenSeeD expects
    feats = model.model.backbone(x.to(device).unsqueeze(0))
    out   = model.model.sem_seg_head.pixel_decoder.forward_features(feats, masks=None)
    mask_feats = out[0] if isinstance(out, (tuple, list)) else out       # (1,C,h,w)

    # interpolate back to the **original** size (LSeg behaviour)
    mask_feats = torch.nn.functional.interpolate(
        mask_feats, size=(H, W), mode="bilinear", align_corners=False
    )

    # per-pixel ℓ2 normalise, cast to float32, drop batch dim
    mask_feats = torch.nn.functional.normalize(mask_feats, dim=1)[0]     # (C,H,W)
    return mask_feats.cpu().float().numpy()
