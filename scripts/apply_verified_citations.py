"""
Phase A2 (Option A) — apply HUMAN/WEB-VERIFIED citations over the LLM-recalled ones.

The Lava fill produced good symptom_text but model-recalled source_url/verbatim_quote (some URLs were
dead, some quotes were paraphrases not verbatim). For the headline held-out diseases we fetched the
actual source pages and copied the EXACT sentences. This script overwrites those records' citations
with the verified ones and stamps `"verified": true` (grounded/usable logic still keys on status
"filled", so nothing else changes). Re-runnable; leaves non-verified records untouched.

Sources chosen to be authoritative AND retrievable (APS blocks bots, EDIS URLs were dead):
UC IPM (ipm.ucanr.edu) and Wikipedia disease pages. Quotes copied verbatim on 2026-07-02.

    python scripts/apply_verified_citations.py
"""
from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# (crop, disease) -> {source, pathogen:(value,quote), affected_organs:(value,quote), visual_symptoms:(value,quote)}
V = {
    ("Coffee", "Rust"): {
        "source": "https://en.wikipedia.org/wiki/Coffee_leaf_rust",
        "pathogen": ("Hemileia vastatrix",
            "Hemileia vastatrix is a multicellular basidiomycete fungus of the order Pucciniales "
            "(previously also known as Uredinales) that causes coffee leaf rust (CLR)."),
        "affected_organs": ("Leaves (rarely young stems and fruit)",
            "It mainly attacks the leaves and is only rarely found on young stems and fruit."),
        "visual_symptoms": ("Yellow-orange powdery uredinia on the underside of leaves; chlorotic pale-yellow spots",
            "The mycelium with uredinia looks yellow-orange and powdery, and appears on the underside "
            "of leaves as points ~0.1 mm in diameter."),
    },
    # SAGE label "Cerscospora" is a misspelling of Cercospora coffeicola (brown eye spot).
    # Verified 2026-07-27 from the Wikipedia species page (reachable; APS/CABI block bots).
    ("Coffee", "Cerscospora"): {
        "source": "https://en.wikipedia.org/wiki/Cercospora_coffeicola",
        "pathogen": ("Cercospora coffeicola (teleomorph Mycosphaerella coffeicola)",
            "Mycosphaerella coffeicola is a sexually reproducing fungal plant pathogen."),
        "affected_organs": ("Leaves and berries",
            "On leaves, lesions begin as chlorotic (yellow) spots that expand to become deep brown "
            "and necrotic on the upper leaf surface."),
        "visual_symptoms": ("Brown necrotic leaf spots with a pale/light sporulating centre and a "
                            "yellow halo; lesions coalesce into large irregular necrotic areas",
            "These spots often have a discolored, light center where sporulation can occur, and many "
            "have a yellow 'halo' around the margins."),
    },
    # Berry blotch is the fruit phase of the same pathogen -> same verified page, berry-specific quotes.
    ("Coffee", "Berry_Blotch"): {
        "source": "https://en.wikipedia.org/wiki/Cercospora_coffeicola",
        "pathogen": ("Cercospora coffeicola (teleomorph Mycosphaerella coffeicola)",
            "Mycosphaerella coffeicola is a sexually reproducing fungal plant pathogen."),
        "affected_organs": ("Berries (green cherries and ripe red cherries)",
            "On green berries, this includes irregularly shaped brown, sunken lesions that are "
            "surrounded by a purple halo."),
        "visual_symptoms": ("Tan-to-brown sunken berry lesions with a purple halo, maturing to a "
                            "deeply depressed ashy centre that can reach the bean",
            "As the lesion matures, it becomes deeply depressed with an ashy center and may penetrate "
            "down to the coffee bean."),
    },
    # SAGE labels this a "disease" but it is an INSECT PEST (Leucoptera coffeella). The image-visible
    # signal is the larval MINE damage, so the descriptor describes the damage, not a pathogen.
    # Source: open-access peer-reviewed review (Insects 2021), verified 2026-08-01.
    ("Coffee", "Miner"): {
        "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8707027/",
        "pathogen": ("Leucoptera coffeella (coffee leaf miner) — an insect pest, not a pathogen",
            "CLM is a monophagous pest, coffee exclusive and the larvae are the causal agent of the "
            "crop damage"),
        "affected_organs": ("Leaves (mesophyll / palisade parenchyma); severe attack causes defoliation",
            "When feeding on the mesophyll of the coffee tree leaves, the insect creates mines that "
            "justify the common name"),
        "visual_symptoms": ("Irregular pale-to-brown mines on the upper leaf surface that enlarge from "
                            "millimetres to centimetres into necrotic patches, ending in leaf fall",
            "the injuries area evolves from some millimeters to several centimeters...ending up to the "
            "falling of the leaves"),
    },
    # Phoma/Ascochyta spot of coffee. NOTE: trade-technical source (reachable); CABI/APS block bots.
    ("Coffee", "Phoma"): {
        "source": "https://revistacultivar.com/articles/phoma-spot-or-ascochyta-spot-of-coffee",
        "pathogen": ("Phoma tarda and Phoma costarricensis",
            "Phoma tarda (RB Stewart) H. Verm. and Phoma costarricensis Echandi, 1957"),
        "affected_organs": ("Leaves, branches and floral rosettes (indirectly flowers and fruit)",
            "Phoma spot can also affect floral rosettes, indirectly causing necrosis of flowers and fruits"),
        "visual_symptoms": ("Dark circular leaf spots up to ~2 cm; leaf edges curl and crack; dark "
                            "sunken lesions girdling branches (dry shoots)",
            "circular spots of dark color and varying sizes, which can reach 2 cm in diameter"),
    },
    ("Peach", "Leaf_Curl"): {
        "source": "https://ipm.ucanr.edu/PMG/PESTNOTES/pn7426.html",
        "pathogen": ("Taphrina deformans",
            "Peach leaf curl, also known as leaf curl, is a disease caused by the fungus Taphrina deformans."),
        "affected_organs": ("Blossoms, fruit, leaves, and shoots",
            "Peach leaf curl affects the blossoms, fruit, leaves, and shoots of peaches, ornamental "
            "flowering peaches, and nectarines."),
        "visual_symptoms": ("Leaves thickened, puckered, curled and distorted; turn yellowish then grayish-white with velvety spores",
            "These areas become thickened and puckered, causing leaves to curl and severely distort. "
            "The thickened areas turn yellowish and then grayish white, as velvety spores are produced "
            "on the surface by the leaf curl fungus."),
    },
    ("Peach", "Brown_Rot"): {
        "source": "https://en.wikipedia.org/wiki/Monilinia_fructicola",
        "pathogen": ("Monilinia fructicola",
            "Monilinia fructicola is a species of fungus in the order Helotiales. A plant pathogen, "
            "it is the causal agent of brown rot of stone fruits."),
        "affected_organs": ("Blossoms, twigs, and fruit",
            "Brown rot causes blossom blight, twig blight; twig canker and fruit rot."),
        "visual_symptoms": ("Small circular brown fruit spots enlarging to rot the whole fruit; greyish spore tufts; mummified fruit",
            "Fruit rot appears as small, circular brown spots that increase rapidly in size causing "
            "the entire fruit to rot. Greyish spores appear in tufts on rotted areas."),
    },
    ("Orange", "Citrus_Canker"): {
        "source": "https://en.wikipedia.org/wiki/Citrus_canker",
        "pathogen": ("Xanthomonas citri",
            "Citrus canker is a disease affecting Citrus species caused by the bacterium Xanthomonas citri."),
        "affected_organs": ("Leaves, stems, and fruit",
            "Infection causes lesions on the leaves, stems, and fruit of citrus trees, including lime, "
            "oranges, and grapefruit."),
        "visual_symptoms": ("Raised brown water-soaked lesions with a yellow halo; older lesions corky",
            "Plants infected with citrus canker have characteristic lesions on leaves, stems, and fruit "
            "with raised, brown, water-soaked margins, usually with a yellow halo or ring effect around "
            "the lesion. Older lesions have a corky appearance, still in many cases retaining the halo effect."),
    },
    ("Orange", "Huanglongbing"): {
        "source": "https://en.wikipedia.org/wiki/Citrus_greening_disease",
        "pathogen": ("Candidatus Liberibacter spp.",
            "is a disease of citrus trees caused by bacteria of the genus Liberibacter."),
        "affected_organs": ("Leaves, twigs, fruit, and roots",
            "followed by splotchy mottling of the entire leaf, premature defoliation, dieback of twigs, "
            "decay of feeder rootlets and lateral roots"),
        "visual_symptoms": ("Asymmetrical (blotchy mottle) leaf yellowing; small lopsided fruit green at the bottom",
            "Nutrient deficiencies tend to be symmetrical along the leaf vein margin, while HLB has an "
            "asymmetrical yellowing around the vein."),
    },
}
# duplicate-label diseases in SAGE share a verified source
V[("Orange", "Canker")] = V[("Orange", "Citrus_Canker")]
V[("Orange", "Greening_Disease")] = V[("Orange", "Huanglongbing")]
V[("Peach", "Peach_Leaf_Curl")] = V[("Peach", "Leaf_Curl")]

# --- new held crops (Exp B/C): web-verified headline disease per crop (2026-07) ---
# (Bean rust + Cotton bacterial blight pages 404'd on fetch -> left honestly LLM-authored.)
V[("Banana", "Panama_Disease")] = {
    "source": "https://en.wikipedia.org/wiki/Panama_disease",
    "pathogen": ("Fusarium oxysporum f. sp. cubense", ""),
    "affected_organs": ("Feeder roots, rhizome, pseudostem, leaves",
        "The infection begins at the tips of the feeder roots and then moves on to the rhizome."),
    "visual_symptoms": ("Oldest leaves yellow, wilt and buckle; outer leaf sheaths split; xylem reddish-brown",
        "Externally, the oldest leaves start turning yellow and there is often a longitudinal splitting of "
        "the lower part of the outer leaf sheaths on the pseudostem. The leaves begin to wilt and may buckle "
        "at the base of the petiole."),
}
V[("Banana", "Yellow_And_Black_Sigatoka")] = {
    "source": "https://en.wikipedia.org/wiki/Mycosphaerella_fijiensis",
    "pathogen": ("Mycosphaerella fijiensis",
        "Black sigatoka is a leaf-spot disease of banana plants caused by the ascomycete fungus "
        "Mycosphaerella fijiensis"),
    "affected_organs": ("Banana leaves", ""),
    "visual_symptoms": ("Streaks parallel to secondary veins; rusty-brown paint-like specks darkening to sunken depressions",
        "In the early stages of the infection of the plant, the lesions have a rusty brown appearance and "
        "appear to be faint, paint-like specks on the leaves."),
}
V[("Cucumber", "Downy_Mildew")] = {
    "source": "https://en.wikipedia.org/wiki/Pseudoperonospora_cubensis",
    "pathogen": ("Pseudoperonospora cubensis",
        "Pseudoperonospora cubensis is a species of water mould known for causing downy mildew"),
    "affected_organs": ("Leaves only (not fruit, flowers, stems or roots)",
        "Regardless of which cucurbit is involved, only the leaves are infected, not fruit, flowers, "
        "stems or roots."),
    "visual_symptoms": ("Angular chlorotic lesions bound by leaf veins; gray-brown to purplish-black growth on the underside",
        "The pathogen causes angular chlorotic lesions on the foliage. These lesions appear angular because "
        "they are bound by leaf veins. During humid conditions, inspection of the underside of the leaf "
        "reveals gray-brown to purplish-black fungal growth."),
}
V[("Wheat", "Head_Scab")] = {
    "source": "https://en.wikipedia.org/wiki/Fusarium_ear_blight",
    "pathogen": ("Fusarium graminearum",
        "In wheat, Fusarium infects the head (hence the name 'Fusarium head blight')"),
    "affected_organs": ("Wheat head/spike and kernels", "kernels to shrivel up and become chalky white"),
    "visual_symptoms": ("Bright pink/orange sporulation on the spike; bleached, shrivelled chalky-white kernels",
        "Macroconidia are produced in sporodochia, which gives the spike a bright pink or orange color."),
}

FIELDS = ("pathogen", "affected_organs", "visual_symptoms")


def main():
    stamp = date.today().isoformat()
    n = 0
    for crop in sorted({c for c, _ in V}):
        p = C.DESCRIPTORS_DIR / f"{C.safe_name(crop)}.json"
        if not p.exists():
            print(f"  [skip] {p.name} missing"); continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        for rec in recs:
            key = (rec.get("crop"), rec.get("disease"))
            if key not in V:
                continue
            v = V[key]
            for f in FIELDS:
                val, quote = v[f]
                rec["fields"][f] = {"value": val, "source_url": v["source"], "verbatim_quote": quote}
            rec["verified"] = True
            rec["verified_source"] = v["source"]
            rec["verified_date"] = stamp
            rec["status"] = "filled"
            n += 1
            print(f"  verified {crop}/{rec['disease']}  <- {v['source']}")
        p.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[verified] stamped {n} records with page-matched verbatim citations.")


if __name__ == "__main__":
    main()
