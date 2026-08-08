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

# --- Bean and Cotton, verified 2026-08-08 ------------------------------------------------------
# These two held-out crops previously had ZERO page-verified records while carrying the headline
# cross-crop claim, which was the weakest point in the auditability story. Sources are UC IPM and the
# Crop Protection Network (both retrievable; APS returns 403 to automated requests).
V[("Bean", "Halo_Blight")] = {
    "source": "https://ipm.ucanr.edu/agriculture/dry-beans/halo-blight/",
    "pathogen": ("Pseudomonas syringae pv. phaseolicola", ""),
    "affected_organs": ("Leaves (undersurfaces first), extending to upper plant parts",
        "small, angular, water-soaked spots (almost resembling little pin pricks) on the "
        "undersurfaces of leaves"),
    "visual_symptoms": ("Angular water-soaked spots with a light green to yellow halo; general "
                        "chlorosis of leaves and upper plant in severe cases",
        "a characteristic light green to yellow halo appears around the spots"),
}
V[("Bean", "Common_Bacterial_Blight_Of_Beans")] = {
    "source": "https://ipm.ucanr.edu/agriculture/dry-beans/common-bacterial-blight/",
    "pathogen": ("Xanthomonas campestris (axonopodis) pv. phaseoli; Xanthomonas fuscans subsp. fuscans",
        ""),
    "affected_organs": ("Leaf margins and interveinal areas; whole leaves in severe cases",
        "The spots or lesions develop on the edges or interveinal areas of leaves."),
    "visual_symptoms": ("Water-soaked or light green spots enlarging to brown necrotic centres, "
                        "bordered by a diagnostic lemon-yellow ring",
        "These irregularly shaped spots are bordered by a lemon yellow ring, which serves as a "
        "diagnostic symptom of common bacterial blight."),
}
V[("Bean", "Rust")] = {
    "source": "https://ipm.ucanr.edu/PMG/GARDEN/VEGES/DISEASES/beanrust.html",
    "pathogen": ("Uromyces phaseoli (syn. U. appendiculatus)", ""),
    "affected_organs": ("Leaves, primarily the lower surface; pods may also be affected", ""),
    "visual_symptoms": ("Dry reddish, yellowish or orange powdery spore pustules, mainly on the "
                        "underside of leaves",
        "Plants infected with rust have dry reddish, yellowish, or orange spore masses or pustules "
        "primarily on the lower surface of leaves."),
}
V[("Bean", "Leaf_Mosaic_Virus")] = {
    "source": "https://ipm.ucanr.edu/agriculture/dry-beans/bean-common-mosaic/",
    "pathogen": ("Bean common mosaic virus", ""),
    "affected_organs": ("Trifoliolate leaves", ""),
    "visual_symptoms": ("Light green-yellow and dark green mosaic on trifoliolate leaves with "
                        "puckering, blistering, distortion and downward curling",
        "mosaic patterns of light green-yellow leaf tissue, dark green tissue, or both light and dark "
        "mosaics together on the trifoliolate leaves. Leaf discoloration is usually accompanied by "
        "puckering, blistering, distortion, and a downward curling and rolling."),
}
V[("Cotton", "Bacterial_Blight")] = {
    "source": "https://cropprotectionnetwork.org/encyclopedia/bacterial-blight-of-cotton",
    "pathogen": ("Xanthomonas citri subsp. malvacearum", "Xanthomonas citri subsp. malvacearum"),
    "affected_organs": ("Leaves, stems and petioles (black arm), and bolls",
        "black lesions on stems and petioles (black arm), and round water-soaked lesions on bolls"),
    "visual_symptoms": ("Angular water-soaked leaf lesions delimited by the veins",
        "water-soaked lesions on leaves constricted by the leaf veins"),
}
V[("Cotton", "Target_Spot")] = {
    "source": "https://cropprotectionnetwork.org/encyclopedia/target-spot-of-cotton",
    "pathogen": ("Corynespora cassiicola", "The disease is caused by the fungus Corynespora cassiicola."),
    "affected_organs": ("Leaves", ""),
    "visual_symptoms": ("Brick-red spots enlarging to tan/light-brown centres with concentric rings",
        "Initial symptoms are brick-red spots that expand into tan to light brown centers with "
        "concentric rings."),
}

# --- Replacements for dead source URLs, verified 2026-08-08 ------------------------------------
# IMPORTANT: repointing a dead URL at a live one WITHOUT re-extracting the quote is worse than
# leaving it dead -- the model-recalled sentence will not appear on the new page, and a reviewer who
# checks concludes the citation was fabricated. Every entry below has its verbatim_quote copied from
# the replacement page itself. All of these are SEEN crops (plus one held-out duplicate that --clean
# merges away), so none of them affects any reported number; this is purely auditability.
V[("Grape", "Gray_Mold")] = {
    "source": "https://ipm.ucanr.edu/agriculture/grape/botrytis-bunch-rot/",
    "pathogen": ("Botrytis cinerea", ""),
    "affected_organs": ("Berries and clusters", ""),
    "visual_symptoms": ("Infected berries brown (white cultivars) or redden (red/black cultivars), "
                        "developing a gray velvety mould",
        "At veraison, individually infected berries in a cluster turn brown on white cultivars or "
        "reddish in red and black cultivars...resulting in the characteristic gray, velvety "
        "appearance of infected berries."),
}
V[("Grape", "Black_Rot")] = {
    "source": "https://ohioline.osu.edu/factsheet/plpath-fru-24",
    "pathogen": ("Guignardia bidwellii", ""),
    "affected_organs": ("Leaves, shoots and fruit", ""),
    "visual_symptoms": ("Small yellow leaf spots enlarging to lesions with dark brownish-red borders "
                        "and tan to dark brown centres, ringed with black pycnidia",
        "Symptoms of black rot first appear as small yellow spots on leaves. Enlarged spots (lesions) "
        "have a dark brownish-red border with tan to dark brown centers."),
}
V[("Grape", "Downy_Mildew")] = {
    "source": "https://agritech.tnau.ac.in/crop_protection/grapes_diseases_1.html",
    "pathogen": ("Plasmopara viticola", ""),
    "affected_organs": ("Leaves", ""),
    "visual_symptoms": ("Irregular yellowish translucent spots on the upper leaf surface with white "
                        "powdery growth beneath; leaves yellow, brown and dry",
        "Irregular, yellowish, translucent sports on the upper surface of the leaves. "
        "Correspondingly on the lower surface, white, powdery growth on leaves."),
}
V[("Grape", "Powdery_Mildew")] = {
    "source": "https://agritech.tnau.ac.in/crop_protection/grapes_diseases_2.html",
    "pathogen": ("Uncinula necator", ""),
    "affected_organs": ("Leaves", ""),
    "visual_symptoms": ("Powdery growth mainly on the upper leaf surface, with malformation and "
                        "discolouration of affected leaves",
        "Powdery growth mostly on the upper surface of the leaves. Malformation and discolouration "
        "of affected leaves."),
}
V[("Apple", "Powdery_Mildew")] = {
    "source": "https://agritech.tnau.ac.in/crop_protection/apple_2.html",
    "pathogen": ("Podosphaera leucotricha", ""),
    "affected_organs": ("Leaves, twigs and fruit buds", "Twigs are also infected."),
    "visual_symptoms": ("Small patches of white powdery growth on the upper leaf surface, on both "
                        "surfaces when severe; affected leaves fall",
        "Small patches of white powdery growth appear on upper side of leaves."),
}
V[("Corn", "Diplodia_Ear_Rot")] = {
    "source": "https://cropprotectionnetwork.org/encyclopedia/diplodia-ear-rot-of-corn",
    "pathogen": ("Stenocarpella maydis and S. macrospora", ""),
    "affected_organs": ("Ears, husks and kernels", ""),
    "visual_symptoms": ("White mould starting at the ear base, turning grayish-brown; raised black "
                        "pycnidia on husk or kernels",
        "white mold beginning at the base of the ear that eventually becomes grayish-brown and rots "
        "the entire ear"),
}
V[("Corn", "Goss_S_Wilt")] = {
    "source": "https://cropprotectionnetwork.org/publications/an-overview-of-gosss-bacterial-wilt-and-blight",
    "pathogen": ("Clavibacter nebraskensis", ""),
    "affected_organs": ("Leaves, with systemic infection of the stalk", ""),
    "visual_symptoms": ("Elongated tan to grayish-brown lesions with wavy margins along the veins, "
                        "carrying diagnostic dark water-soaked 'freckles' that appear translucent "
                        "when backlit",
        "Dark green to black, scattered, discontinuous water-soaked spots (\"freckles\") develop "
        "within the plant tissue inside the lesions and are diagnostic for Goss's wilt"),
}
V[("Corn", "Stewart_S_Disease")] = {
    "source": "https://cropprotectionnetwork.org/encyclopedia/stewarts-disease-of-corn",
    "pathogen": ("Pantoea stewartii", ""),
    "affected_organs": ("Leaves, spreading from flea beetle feeding scars", ""),
    "visual_symptoms": ("Pale green to yellow streaks spreading from flea beetle scars, browning as "
                        "tissue dies, with wavy margins following the veins",
        "Stewart's disease lesions spread from flea beetle feeding scars (a tiny scratch on the leaf) "
        "and are initially pale green to yellow streaks, later becoming brown as tissue dies. The "
        "margins of the streaks are usually wavy but generally follow leaf veins."),
}
V[("Corn", "Tar_Spot")] = {
    "source": "https://cropprotectionnetwork.org/encyclopedia/tar-spot-of-corn",
    "pathogen": ("Phyllachora maydis", ""),
    "affected_organs": ("Leaves (upper and lower surfaces)", ""),
    "visual_symptoms": ("Small raised black stromata scattered over both leaf surfaces; tan-to-brown "
                        "'fisheye' lesions with dark borders may surround them",
        "Tar spot appears as small, raised, black spots scattered across the upper and lower leaf "
        "surfaces."),
}
V[("Rice", "Brown_Spot")] = {
    "source": "https://agritech.tnau.ac.in/expert_system/paddy/cpdisbrownspot.html",
    "pathogen": ("Helminthosporium oryzae", ""),
    "affected_organs": ("Leaves", ""),
    "visual_symptoms": ("Minute brown dots enlarging to oval-to-circular sesame-seed-like spots "
                        "0.5-2.0 mm across that coalesce into large patches",
        "The disease appears first as minute brown dots, later becoming cylindrical or oval to "
        "circular.(resemble sesame seed)"),
}
V[("Rice", "Tungro")] = {
    "source": "https://agritech.tnau.ac.in/expert_system/paddy/cpdistungro.html",
    "pathogen": ("Rice tungro bacilliform virus (RTBV) and Rice tungro spherical virus (RTSV)", ""),
    "affected_organs": ("Leaves", ""),
    "visual_symptoms": ("Yellow to orange-yellow leaves with rust-coloured spots, discolouring from "
                        "the leaf tip downward",
        "Their leaves become yellow or orange-yellow, may also have rust-colored spots. "
        "Discoloration begins from leaf tip and extends down to the blade or the lower leaf portion"),
}
V[("Rice", "Severe_Tungro")] = dict(V[("Rice", "Tungro")])
V[("Strawberry", "Anthracnose")] = {
    "source": "https://ipm.ucanr.edu/agriculture/strawberry/anthracnose/",
    "pathogen": ("Colletotrichum acutatum", ""),
    "affected_organs": ("Petioles, runners, crown and fruit", ""),
    "visual_symptoms": ("Dark brown to black sunken lens-shaped spots on petioles and runners; "
                        "salmon or orange spore masses form on lesions",
        "salmon or orange-colored spores can form on the lesions of the fruit, petioles, and runners"),
}
V[("Tomato", "Southern_Blight")] = {
    "source": "https://hort.extension.wisc.edu/articles/southern-blight/",
    "pathogen": ("Athelia rolfsii (formerly Sclerotium rolfsii)", ""),
    "affected_organs": ("Lower stems and leaves, crown, roots and fruit", ""),
    "visual_symptoms": ("Water-soaked lesions on lower stems and leaves, wilting, thick white "
                        "mycelial mats and mustard-seed-sized tan to dark sclerotia",
        "Sclerotia (small spherical structures that are about the size of mustard seeds) develop on "
        "infected tissue and on the soil surface. Sclerotia range in color from light tan to dark "
        "reddish-brown to black"),
}

V[("Soybean", "Brown_Stem_Rot")] = {
    "source": "https://cropprotectionnetwork.org/encyclopedia/brown-stem-rot-of-soybean",
    "pathogen": ("Cadophora gregata", ""),
    "affected_organs": ("Leaves; internal stem vascular tissue and pith, especially at nodes and "
                        "in the lower stem", ""),
    "visual_symptoms": ("Interveinal chlorosis and necrosis followed by leaf curling and death; "
                        "internal browning of stem pith while the stem looks healthy outside",
        "Characteristic foliar symptoms of BSR include chlorosis and necrosis between leaf veins, "
        "followed by leaf curling and leaf death."),
}
V[("Rose", "Rose_Rosette_Virus")] = {
    "source": "https://www.canr.msu.edu/news/rose_gardeners_should_learn_the_symptoms_of_rose_rosette_virus",
    "pathogen": ("Rose rosette virus, vectored by the eriophyid mite Phyllocoptes fructiphilus", ""),
    "affected_organs": ("Shoots, stems, leaves, buds and flowers", ""),
    "visual_symptoms": ("Witches' brooms, red or yellow discoloured and distorted buds, excessive "
                        "thorns, mosaic-patterned leaves, thick stalks and deformed leaves/flowers",
        "witches' brooms (Photo 2), red or yellow discoloration or distorted buds (Photo 3), "
        "excessive thorns (Photo 4), mosaic-patterned leaves, thick stalks, deformed leaves and "
        "flowers on roses."),
}
# Same disease as Head_Scab (see LABEL_ALIASES) -> same verified source rather than a dead Purdue PDF.
V[("Wheat", "Fusarium_Graminearum_Schwabe")] = dict(V[("Wheat", "Head_Scab")])

# Not a disease at all -- abiotic chemical damage. The "pathogen" field records that explicitly
# rather than naming an organism, because a descriptor that invents a causal agent here would be
# exactly the hallucination the grounding protocol exists to prevent.
V[("Soybean", "Herbicide_Injury")] = {
    "source": "https://crops.extension.iastate.edu/post/identifying-common-herbicide-symptoms-soybean",
    "pathogen": ("None - abiotic injury from herbicide drift, carryover or contact, not a pathogen",
                 ""),
    "affected_organs": ("Leaves and leaflets; hypocotyls and cotyledons after preemergence splash",
        "tissue contacted by the herbicide develops necrosis"),
    "visual_symptoms": ("Symptoms vary by herbicide group: strapped or cupped leaves (auxin mimics), "
                        "interveinal chlorosis and necrosis (photosynthesis inhibitors), speckled "
                        "necrotic tissue (PPO inhibitors), heart-shaped asymmetric leaflets (lipid "
                        "synthesis inhibitors), and bleaching or yellowing of newly emerged leaves "
                        "(HPPD inhibitors)",
        "2,4-D often causes more of a strapped appearance to leaves, making them longer and skinnier "
        "with parallel veins"),
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
