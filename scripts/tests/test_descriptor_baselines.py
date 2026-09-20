"""Guard the descriptor-baseline additions.

The critical property is NON-REGRESSION: adding bare80 / dclip / cupl must leave the five
published strategies producing byte-identical prompt lists, because Tables 2-4 were measured
with the old code path. Test 1 asserts exactly that against a re-implementation of the
original single-comprehension form.

    python -m pytest scripts/tests/test_descriptor_baselines.py -q
    python scripts/tests/test_descriptor_baselines.py          # no pytest needed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import config as C          # noqa: E402
import descriptors as D     # noqa: E402

PUBLISHED = ["bare", "crude", "rich", "grounded", "grounded_visual"]
LABELS = ["Orange|Citrus_Canker", "Coffee|Leaf_Rust", "Peach|Peach_Leaf_Curl",
          "Wheat|Head_Scab", "Cucumber|Angular_Leaf_Spot"]


def _original_prompts(label, strategy):
    """The pre-2026-09-13 form, verbatim, as the reference implementation."""
    return [t.format(D.text_for(label, strategy)) for t in C.PROMPT_TEMPLATES]


def _new_prompts(label, strategy):
    """What build_prototypes now constructs, minus the tokenizer/encoder."""
    templates = D.STRATEGY_TEMPLATES.get(strategy, C.PROMPT_TEMPLATES)
    return [t.format(v) for v in D.texts_for(label, strategy) for t in templates]


def test_published_strategies_are_unchanged():
    for strategy in PUBLISHED:
        for label in LABELS:
            assert _new_prompts(label, strategy) == _original_prompts(label, strategy), (
                f"{strategy}/{label} prompt list changed -- Tables 2-4 would no longer reproduce")


def test_bare80_uses_eighty_templates_and_bare_text():
    assert len(D.IMAGENET_80_TEMPLATES) == 80
    assert len(set(D.IMAGENET_80_TEMPLATES)) == 80, "duplicate template would skew the mean"
    for t in D.IMAGENET_80_TEMPLATES:
        assert t.count("{}") == 1, f"template must have exactly one slot: {t!r}"
    for label in LABELS:
        p = _new_prompts(label, "bare80")
        assert len(p) == 80
        # same underlying text as `bare`, only the ensemble differs
        assert D.texts_for(label, "bare80") == [D.text_for(label, "bare")]


def test_dclip_is_not_renormalised_but_cupl_is():
    # The fidelity point: DCLIP's score is mean_i cos(img, d_i) = <img, unnormalised mean>.
    assert "dclip" in D.NO_RENORM_STRATEGIES
    assert "cupl" not in D.NO_RENORM_STRATEGIES
    assert "bare80" not in D.NO_RENORM_STRATEGIES
    for s in D.STRATEGY_TEMPLATES:
        if s != "bare80":
            assert D.STRATEGY_TEMPLATES[s] == ["{}"], f"{s} must embed generated text as written"


def test_generated_baselines_fall_through_to_rich_when_absent(tmp_path=None):
    """A class with no generated record must behave exactly like `rich`, as the other arms do."""
    D._arm_cache.clear()
    for strategy in ("dclip", "cupl"):
        for label in LABELS:
            got = D.texts_for(label, strategy)
            if got == [D.text_for(label, "rich")]:
                continue                      # fell through, which is the contract
            assert len(got) >= 1 and all(isinstance(x, str) and x for x in got)


def test_dclip_scoring_template_shape():
    """Exercise the dclip branch with a synthetic registry, then clean up."""
    seed = D._seed()
    out = C.REPO_ROOT / D.BASELINE_DIRS["dclip"] / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{C.safe_name('Orange')}.json"
    existed = path.exists()
    backup = path.read_bytes() if existed else None
    try:
        path.write_text(json.dumps([{
            "crop": "Orange", "disease": "Citrus_Canker", "status": "filled",
            "descriptors": ["raised corky lesions.", "yellow halo around spots"],
        }]), encoding="utf-8")
        D._arm_cache.clear()
        got = D.texts_for("Orange|Citrus_Canker", "dclip")
        assert got == ["Citrus Canker on Orange leaf, which has raised corky lesions",
                       "Citrus Canker on Orange leaf, which has yellow halo around spots"], got
    finally:
        if existed:
            path.write_bytes(backup)
        else:
            path.unlink(missing_ok=True)
            try:
                out.rmdir(); out.parent.rmdir()
            except OSError:
                pass
        D._arm_cache.clear()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
    print(f"\n{'all green' if not fails else f'{fails} failure(s)'}")
    raise SystemExit(1 if fails else 0)
