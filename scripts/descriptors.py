"""
Descriptor text prototypes — the zero-shot fuel.

Phase-0 showed descriptor QUALITY is the decisive lever (rich +8..+15pp over bare; matches
SAGE's +14-16pp), even on a frozen 11M model. Strategies, weakest -> strongest:
    bare    = "{disease} on {crop} leaf"                  (class name only)
    crude   = bare + one generic keyword sentence
    rich    = bare + a detailed per-disease symptom description (source-grounded STYLE)
    grounded= load descriptors/<crop>.json if present (Phase A2 auditable source-grounded text),
              else fall back to rich.

`build_prototypes(model, tokenizer, classes, strategy, device)` returns an [N_classes, dim]
tensor of L2-normalized text-prototype embeddings.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# generic keyword stubs (the weak "crude" baseline)
CRUDE = {
    "rust": "orange to brown powdery pustules on the underside of the leaf",
    "blight": "rapidly spreading brown necrotic lesions and dead leaf tissue",
    "spot": "small dark circular spots with concentric rings on the leaf",
    "mildew": "a white or grey powdery fungal coating on the leaf surface",
    "canker": "sunken corky lesions with yellow halos on the leaf",
    "greening": "blotchy asymmetric yellow mottling of the leaf",
    "huanglongbing": "blotchy asymmetric yellow mottling of the leaf",
    "curl": "puckered, thickened, distorted and reddened curled leaves",
    "brown rot": "brown spreading rot with tan fungal spore masses",
    "scab": "olive-green to black velvety scabby lesions on the leaf",
    "mosaic": "a mottled light-and-dark green mosaic pattern on the leaf",
    "cercospora": "brown spots with grey centers and yellow halos on the leaf",
    "deficiency": "interveinal yellowing of the leaf from nutrient deficiency",
    "healthy": "a healthy green leaf with no disease symptoms",
}

# rich per-disease symptom descriptions (source-grounded STYLE; priority-ordered, specific first)
RICH = [
    ("huanglongbing", "blotchy asymmetric yellow mottling that does not mirror across the midrib, with "
                      "yellowed veins and a thickened leathery leaf; a hallmark of citrus greening"),
    ("greening", "blotchy asymmetric yellow mottling across the leaf, not matching on either side of the "
                 "midrib, with green islands and yellow veins"),
    ("citrus canker", "raised tan-to-brown corky lesions ringed by a yellow halo and a water-soaked margin"),
    ("leaf curl", "severely thickened, puckered and curled leaves, reddish to purple, later with a whitish bloom"),
    ("brown rot", "rapidly spreading firm brown rot bearing tufts of tan-grey powdery spores"),
    ("black spot", "small dark sunken circular spots with pale grey centres and a brittle cracked surface"),
    ("brown eye", "circular tan-to-brown spots with pale grey or white centres surrounded by a yellow halo"),
    ("cercospora", "circular brown spots with grey centres ringed by a bright yellow halo on the leaf"),
    ("leaf miner", "winding translucent serpentine mines and silvery tunnels within the leaf tissue"),
    ("red spider", "fine pale stippling and dull bronzing of the leaf with faint webbing"),
    ("spider mite", "fine pale stippling and bronzing of the leaf surface with delicate webbing"),
    ("shot hole", "small reddish-purple spots whose centres drop out to leave clean round shot holes"),
    ("bacterial spot", "small angular dark purple-to-brown spots confined by veins, often dropping to shot-holes"),
    ("powdery mildew", "white powdery fungal patches dusting the leaf surface, distorting young leaves"),
    ("downy mildew", "pale yellow angular blotches on the upper leaf with grey-purple downy mould beneath"),
    ("greasy spot", "yellow blistered mottling above with brown greasy translucent blisters underneath"),
    ("melanose", "numerous tiny raised dark-brown sandpaper-textured specks, sometimes in tear-streaks"),
    ("anthracnose", "sunken dark lesions with concentric rings and a tan papery centre"),
    ("phoma", "dark brown to black necrotic blotches at leaf margins and tips, with concentric zoning"),
    ("canker", "raised corky brown lesions with a yellow halo on the leaf and stem"),
    ("scab", "raised wart-like corky scabby pustules with cracked, distorted, wrinkled tissue"),
    ("rust", "yellow-orange powdery pustules under the leaf with matching pale chlorotic blotches above"),
    ("curl", "puckered, thickened and distorted curled leaves, often reddened"),
    ("mildew", "a white-to-grey powdery fungal coating spreading over the leaf surface"),
    ("mosaic", "a mottled light-and-dark green mosaic with mild puckering"),
    ("blight", "rapidly spreading brown necrotic lesions killing large areas of leaf tissue"),
    ("deficiency", "interveinal yellowing with the veins staying green, from nutrient deficiency"),
    ("nutrient", "interveinal yellowing while veins remain green, indicating nutrient deficiency"),
    ("mite", "fine pale stippling and bronzing of the leaf with faint webbing"),
    ("spot", "scattered dark circular leaf spots with concentric rings and yellow margins"),
    ("rot", "spreading soft brown rot of the tissue with fungal growth"),
    ("healthy", "a uniformly green, glossy, healthy leaf with no spots, mottling, lesions or distortion"),
]

_grounded_cache: dict = {}


def _grounded(crop, disease):
    """Source-grounded descriptor text from descriptors/<crop>.json (Phase A2), if available."""
    if crop not in _grounded_cache:
        p = C.DESCRIPTORS_DIR / f"{crop.lower()}.json"
        try:
            _grounded_cache[crop] = json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            _grounded_cache[crop] = {}
    entry = _grounded_cache[crop].get(disease) if isinstance(_grounded_cache[crop], dict) else None
    if isinstance(entry, dict):
        return entry.get("symptom_text") or entry.get("value")
    return None


def text_for(label: str, strategy: str = "rich", coverage: dict | None = None) -> str:
    crop, dis = label.split("|", 1)
    base = f"{dis} on {crop} leaf".replace("_", " ")
    k = dis.lower()
    if strategy == "bare":
        return base
    if strategy == "crude":
        hint = next((v for kw, v in CRUDE.items() if kw in k), "")
        return f"{base}: {hint}" if hint else base
    if strategy == "grounded":
        g = _grounded(crop, dis)
        if g:
            return f"{base}. {g}"
        # fall through to rich
    for kw, desc in RICH:           # rich (or grounded fallback)
        if kw in k:
            if coverage is not None:
                coverage[label] = kw
            return f"{base}. {desc}"
    if coverage is not None:
        coverage[label] = "(NO MATCH)"
    return base


def build_prototypes(model, tokenizer, classes, strategy="rich", device="cpu", coverage=None):
    import torch
    import torch.nn.functional as F
    protos = []
    with torch.no_grad():
        for c in classes:
            toks = tokenizer([t.format(text_for(c, strategy, coverage)) for t in C.PROMPT_TEMPLATES]).to(device)
            emb = F.normalize(model.encode_text(toks), dim=-1).mean(0)
            protos.append(F.normalize(emb, dim=-1))
    return torch.stack(protos).to(device)
