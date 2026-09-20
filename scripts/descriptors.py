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
import os
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


def _is_placeholder(text) -> bool:
    """True if this symptom_text is an unfilled stub rather than a real description.

    status=='filled' alone is NOT sufficient. Four records in the shipped registry
    (Coffee: Berry_Blotch, Cerscospora, Miner, Phoma) carry status='filled' while their
    symptom_text is still the literal generator stub "TODO: source-grounded symptom
    description for ...". Trusting the flag put those placeholder strings into CLIP as
    prototypes for 4 of the 51 scale-C held-out classes, which is exactly what the Methods
    section promises cannot happen. Coffee was consequently the grounded arm's worst crop
    (10.3% vs 17.5% ungrounded); excluding it, the arms converge to within 0.35pp.
    The text is authoritative, so gate on the text.
    """
    t = (text or "").strip()
    return (not t) or t.upper().startswith("TODO")


def _grounded_rec(crop, disease):
    """The filled descriptor record for (crop, disease) from descriptors/<crop>.json, or None.

    build_descriptors.py writes a LIST of records per crop; we index by disease and only use records
    with status=='filled' (stubs carry a 'TODO' placeholder that must NOT become a prototype — those
    fall back to rich). Filename match is case-insensitive (Coffee.json vs coffee.json)."""
    if crop not in _grounded_cache:
        idx = {}
        try:
            fname = f"{C.safe_name(crop)}.json"
            p = C.DESCRIPTORS_DIR / fname
            if not p.exists():  # case-insensitive fallback for case-sensitive filesystems
                p = next((q for q in C.DESCRIPTORS_DIR.glob("*.json")
                          if q.name.lower() == fname.lower()), p)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for rec in (data if isinstance(data, list) else data.values()):
                    if isinstance(rec, dict) and rec.get("status") == "filled"                             and not _is_placeholder(rec.get("symptom_text")):
                        idx[rec.get("disease")] = rec
        except Exception:
            idx = {}
        _grounded_cache[crop] = idx
    return _grounded_cache[crop].get(disease)


def _grounded(crop, disease):
    """Full source-grounded symptom paragraph (the 'grounded' strategy)."""
    rec = _grounded_rec(crop, disease)
    if isinstance(rec, dict):
        t = (rec.get("symptom_text") or "").strip()
        return t or None
    return None


def _grounded_visual(crop, disease):
    """ONLY the visual_symptoms field value (terser, no taxonomy/pathogen prose that CLIP ignores).
    The 'grounded_visual' strategy — tests whether stripping non-visual text helps prototype matching."""
    rec = _grounded_rec(crop, disease)
    if isinstance(rec, dict):
        v = (rec.get("fields", {}).get("visual_symptoms", {}).get("value") or "").strip()
        return v or None
    return None


# --- ungrounded LLM descriptors (the control for the grounding claim) --------------------------
# Same generator and same schema as the grounded registry, but with the "cite a retrievable source"
# constraint removed. Generated at several seeds because a single sample of LLM text cannot
# distinguish "grounding helps" from "this particular generation was lucky".
_arm_cache: dict = {}
def _seed() -> int:
    """The active descriptor seed, read AT CALL TIME.

    This was a module-level constant, which silently broke the whole seed study: evaluate.py
    imports zeroshot (and so descriptors) at line 27 but does not set PDE_UNGROUNDED_SEED until
    line 63, so the constant was already frozen at 0 before --ungrounded-seed was applied. All
    three "seeds" therefore evaluated seed 0's text and produced byte-identical accuracies, which
    reads as a spread of exactly zero rather than as a bug. Reading the environment per call costs
    nothing next to a forward pass and cannot go stale."""
    return int(os.environ.get("PDE_UNGROUNDED_SEED", "0"))

# Directory per seeded arm. `grounded_matched` exists because the shipped grounded registry records
# no generating model, so comparing it against a freshly generated ungrounded set would confound
# grounding with model version -- the same class of confound that invalidated the original claim.
# grounded_matched is generated by the same model, in the same run, as the ungrounded seeds.
ARM_DIRS = {
    "ungrounded": "descriptors_ungrounded",
    "grounded_matched": "descriptors_grounded_matched",
    # Truncation-safe variants. Every generated prototype exceeded CLIP's 77-token text
    # window (51/51 ungrounded, 35/40 matched), so the arms above are compared on their
    # leading sentences only. These arms hold the same text compressed to fit, which lets
    # the descriptor question be asked without truncation as a confound.
    "ungrounded_short": "descriptors_ungrounded_short",
    "grounded_matched_short": "descriptors_grounded_matched_short",
}


def _seeded_arm(arm, crop, disease):
    """symptom_text from <ARM_DIRS[arm]>/<seed>/<crop>.json, or None."""
    seed = _seed()
    key = (arm, crop, seed)
    if key not in _arm_cache:
        idx = {}
        try:
            p = C.REPO_ROOT / ARM_DIRS[arm] / str(seed) / f"{C.safe_name(crop)}.json"
            if p.exists():
                for rec in json.loads(p.read_text(encoding="utf-8")):
                    if isinstance(rec, dict) and rec.get("status") == "filled":
                        idx[rec.get("disease")] = rec
        except Exception:
            idx = {}
        _arm_cache[key] = idx
    rec = _arm_cache[key].get(disease)
    if isinstance(rec, dict):
        return (rec.get("symptom_text") or "").strip() or None
    return None


def text_for(label: str, strategy: str = "rich", coverage: dict | None = None) -> str:
    crop, dis = label.split("|", 1)
    base = f"{dis} on {crop} leaf".replace("_", " ")
    # Normalise underscores BEFORE matching. Labels are Crop|Disease_Name, and 13 of the 32 RICH
    # keys are multi-word ("powdery mildew", "citrus canker", "leaf curl", ...). Matching against
    # the raw "powdery_mildew" made every one of those keys unreachable for every label in the
    # dataset, so classes with a correct distinct entry in the bank silently fell through to a
    # coarser key ("mildew") or to no match at all. `base` already normalised; `k` did not.
    k = dis.lower().replace("_", " ")
    if strategy == "bare":
        return base
    if strategy == "crude":
        hint = next((v for kw, v in CRUDE.items() if kw in k), "")
        return f"{base}: {hint}" if hint else base
    if strategy in ARM_DIRS:
        u = _seeded_arm(strategy, crop, dis)
        if u:
            return f"{base}. {u}"
        # fall through to rich, exactly as `grounded` does, so coverage gaps are handled
        # identically in every arm and the comparison stays fair
    if strategy == "grounded":
        g = _grounded(crop, dis)
        if g:
            return f"{base}. {g}"
        # fall through to rich
    if strategy == "grounded_visual":
        gv = _grounded_visual(crop, dis)
        if gv:
            return f"{base}. {gv}"
        # fall through to rich
    for kw, desc in RICH:           # rich (or grounded fallback)
        if kw in k:
            if coverage is not None:
                coverage[label] = kw
            return f"{base}. {desc}"
    if coverage is not None:
        coverage[label] = "(NO MATCH)"
    return base


# --- established descriptor-generation baselines (the comparison Section 6 named as missing) ---
#
# Three additions, none of which touches the five strategies above. Section 6 concedes that the
# paper compares only strategies of its own making: `bare` uses a 3-template ensemble rather than
# CLIP's standard 80, and the per-class comparators (DCLIP, CuPL) were never implemented. These
# close both gaps. All three are TEXT-ONLY -- no new images, no training -- so they can be scored
# against cached image embeddings.
#
#   bare80  the `bare` class name, ensembled over the 80 OpenAI ImageNet templates instead of
#           C.PROMPT_TEMPLATES. This is the baseline the original CLIP zero-shot recipe uses, and
#           it is the strongest available class-name-only reference.
#   dclip   Menon and Vondrick (ICLR 2023). An LLM writes short visual descriptors per class; each
#           is embedded as "<class>, which has <descriptor>" and the class score is the MEAN of the
#           per-descriptor similarities.
#   cupl    Pratt et al. (ICCV 2023). An LLM writes full sentences per class from several question
#           templates; sentence embeddings are averaged into one prototype and no hand-written
#           template is applied.
#
# FIDELITY NOTE, and the reason `dclip` needs its own code path. DCLIP scores a class as
# mean_i cos(img, d_i). For L2-normalised img and d_i that equals <img, mean_i(d_i)>, i.e. the
# UNNORMALISED mean of the descriptor embeddings. Re-normalising the mean -- which is what every
# other strategy here does, and what CuPL does -- rescales each class prototype by a different
# factor and so changes the ranking between classes. Re-normalising would therefore be a different
# method wearing DCLIP's name, which is why `dclip` is exempted below.

# The 80 OpenAI ImageNet prompt templates, inlined rather than imported. open_clip has moved this
# list between modules across versions (`open_clip.zero_shot_metadata`, `open_clip.constants`,
# top-level), so importing it makes a published number depend on the installed version.
IMAGENET_80_TEMPLATES = [
    "a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.",
    "a photo of the hard to see {}.", "a low resolution photo of the {}.", "a rendering of a {}.",
    "graffiti of a {}.", "a bad photo of the {}.", "a cropped photo of the {}.",
    "a tattoo of a {}.", "the embroidered {}.", "a photo of a hard to see {}.",
    "a bright photo of a {}.", "a photo of a clean {}.", "a photo of a dirty {}.",
    "a dark photo of the {}.", "a drawing of a {}.", "a photo of my {}.",
    "the plastic {}.", "a photo of the cool {}.", "a close-up photo of a {}.",
    "a black and white photo of the {}.", "a painting of the {}.", "a painting of a {}.",
    "a pixelated photo of the {}.", "a sculpture of the {}.", "a bright photo of the {}.",
    "a cropped photo of a {}.", "a plastic {}.", "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.", "a blurry photo of the {}.", "a photo of the {}.",
    "a good photo of the {}.", "a rendering of the {}.", "a {} in a video game.",
    "a photo of one {}.", "a doodle of a {}.", "a close-up photo of the {}.",
    "a photo of a {}.", "the origami {}.", "the {} in a video game.", "a sketch of a {}.",
    "a doodle of the {}.", "a origami {}.", "a low resolution photo of a {}.",
    "the toy {}.", "a rendition of the {}.", "a photo of the clean {}.",
    "a photo of a large {}.", "a rendition of a {}.", "a photo of a nice {}.",
    "a photo of a weird {}.", "a blurry photo of a {}.", "a cartoon {}.", "art of a {}.",
    "a sketch of the {}.", "a embroidered {}.", "a pixelated photo of a {}.",
    "itap of the {}.", "a jpeg corrupted photo of the {}.", "a good photo of a {}.",
    "a plushie {}.", "a photo of the nice {}.", "a photo of the small {}.",
    "a photo of the weird {}.", "the cartoon {}.", "art of the {}.", "a drawing of the {}.",
    "a photo of the large {}.", "a black and white photo of a {}.", "the plushie {}.",
    "a dark photo of a {}.", "itap of a {}.", "graffiti of the {}.", "a toy {}.",
    "itap of my {}.", "a photo of a cool {}.", "a photo of a small {}.", "a tattoo of the {}.",
]

# Generated registries, written by scripts/build_descriptor_baselines.py. Same on-disk shape as
# ARM_DIRS (one JSON per crop, per seed) so the integrity gate and the seed plumbing are reused.
BASELINE_DIRS = {
    "dclip": "descriptors_dclip",
    "cupl": "descriptors_cupl",
}

# Prompt ensemble per strategy. Anything absent uses C.PROMPT_TEMPLATES, so the five original
# strategies are untouched. "{}" is the identity template: the generated text is embedded as
# written, which is what CuPL and DCLIP both do.
STRATEGY_TEMPLATES = {
    "bare80": IMAGENET_80_TEMPLATES,
    "dclip": ["{}"],
    "cupl": ["{}"],
}

# Strategies whose class prototype must NOT be re-normalised after averaging. See the fidelity
# note above: for DCLIP this is the difference between the published method and a variant of it.
NO_RENORM_STRATEGIES = {"dclip"}


def _baseline_variants(strategy, crop, disease):
    """The list of generated texts for one class under `dclip` or `cupl`, or None if absent.

    Returns None (not []) when the class has no record, so callers can fall through to `rich`
    exactly as the grounded and ungrounded arms do. Falling through keeps coverage handling
    identical across every arm, which is what makes the comparison fair.
    """
    seed = _seed()
    key = (strategy, crop, seed)
    if key not in _arm_cache:
        idx = {}
        try:
            p = C.REPO_ROOT / BASELINE_DIRS[strategy] / str(seed) / f"{C.safe_name(crop)}.json"
            if p.exists():
                for rec in json.loads(p.read_text(encoding="utf-8")):
                    if isinstance(rec, dict) and rec.get("status") == "filled":
                        items = rec.get("descriptors") or rec.get("sentences") or []
                        items = [str(x).strip() for x in items if str(x).strip()]
                        if items:
                            idx[rec.get("disease")] = items
        except Exception:
            idx = {}
        _arm_cache[key] = idx
    return _arm_cache[key].get(disease)


def texts_for(label: str, strategy: str = "rich", coverage: dict | None = None) -> list[str]:
    """Every text variant for one class. One element for all strategies except dclip/cupl.

    `text_for` stays the single-text entry point and is unchanged, so existing callers
    (evaluate.py, descriptor_coverage.py) behave exactly as before.
    """
    if strategy in BASELINE_DIRS:
        crop, dis = label.split("|", 1)
        base = f"{dis} on {crop} leaf".replace("_", " ")
        items = _baseline_variants(strategy, crop, dis)
        if items:
            if coverage is not None:
                coverage[label] = strategy
            if strategy == "dclip":
                # Menon and Vondrick's scoring template.
                return [f"{base}, which has {d.rstrip('.')}" for d in items]
            return items          # CuPL sentences are embedded as generated
        # no record -> fall through to rich, as every other generated arm does
        return [text_for(label, "rich", coverage)]
    if strategy == "bare80":
        return [text_for(label, "bare", coverage)]
    return [text_for(label, strategy, coverage)]


def build_prototypes(model, tokenizer, classes, strategy="rich", device="cpu", coverage=None):
    import torch
    import torch.nn.functional as F
    templates = STRATEGY_TEMPLATES.get(strategy, C.PROMPT_TEMPLATES)
    renorm = strategy not in NO_RENORM_STRATEGIES
    protos = []
    with torch.no_grad():
        for c in classes:
            # For every pre-existing strategy texts_for yields exactly one string and templates is
            # C.PROMPT_TEMPLATES, so this reduces to the original single-comprehension form and
            # reproduces the published prototypes bit for bit.
            prompts = [t.format(v) for v in texts_for(c, strategy, coverage) for t in templates]
            toks = tokenizer(prompts).to(device)
            emb = F.normalize(model.encode_text(toks), dim=-1).mean(0)
            protos.append(F.normalize(emb, dim=-1) if renorm else emb)
    return torch.stack(protos).to(device)
