"""
Supplementary File S1: ipe_analysis.py
Information Prediction Error (IPE) Pipeline
EUvsDisinfo Corpus 2015-2022

Reproduces all results reported in Chapter 2 and Chapter 3.
Requirements: pandas, numpy, scipy, scikit-learn (standard Python scientific stack)
No internet access, API keys, or pre-trained model downloads required.

Usage:
    python ipe_analysis.py --data euvsdisinfo_all_texts_rename.csv
"""

import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.preprocessing import LabelBinarizer
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1: PREPROCESSING
# =============================================================================

_STRIP = re.compile(r"[^a-z\s]")

def preprocess(text: str) -> str:
    """Lowercase, strip non-alphabetic characters, truncate to 512 tokens."""
    t = str(text).lower()
    t = _STRIP.sub(" ", t)
    return " ".join(t.split()[:512])


# =============================================================================
# SECTION 2: MORAL-EMOTIONAL INTENSITY (MEI) LEXICON v2
# =============================================================================
# Lexicon constructed from:
#   - Extended Moral Foundations Dictionary (eMFD): Hopp et al. (2021)
#     https://doi.org/10.3758/s13428-020-01433-0
#   - NRC Valence-Arousal-Dominance Lexicon: Mohammad (2018)
#     https://doi.org/10.18653/v1/P18-1017
# New terms (v2) derived from eMFD/NRC-VAD literature where available;
# author judgment calibrated to existing scale where coverage absent.

MORAL_LEXICON = {
    # Care / Harm
    "kill": 0.98, "murder": 0.99, "massacre": 0.99, "slaughter": 0.99,
    "torture": 0.98, "atrocity": 0.97, "genocide": 0.99, "exterminate": 0.99,
    "annihilate": 0.97, "destroy": 0.82, "harm": 0.85, "hurt": 0.78,
    "wound": 0.75, "injure": 0.75, "victim": 0.80, "innocent": 0.72,
    "child": 0.65, "civilian": 0.70, "refugee": 0.62,
    "abuse": 0.90, "rape": 0.99, "assault": 0.88, "attack": 0.78,
    "bomb": 0.80, "shell": 0.72, "execute": 0.92,
    "death": 0.85, "dead": 0.80, "die": 0.72,
    # Sexual violence / child harm (v2)
    "pedophilia": 0.99, "pedophile": 0.99, "pedophilic": 0.99,
    "groom": 0.92, "grooming": 0.97, "molest": 0.98, "molestation": 0.98,
    "predator": 0.90, "underage": 0.88,
    "sex offender": 0.96, "child abuse": 0.99,
    "sexual assault": 0.97, "sexual violence": 0.97,
    # Fairness / Cheating
    "corrupt": 0.88, "corruption": 0.90, "fraud": 0.87, "lie": 0.82,
    "deceive": 0.85, "cheat": 0.83, "manipulate": 0.84, "exploit": 0.86,
    "steal": 0.85, "loot": 0.84, "plunder": 0.88, "sanction": 0.60,
    "injustice": 0.88, "unfair": 0.72, "illegal": 0.75, "criminal": 0.82,
    "puppet": 0.80, "stooge": 0.78, "bribe": 0.87, "blackmail": 0.90,
    # Loyalty / Betrayal
    "traitor": 0.96, "betrayal": 0.94, "betray": 0.93, "treason": 0.95,
    "collaborator": 0.82, "spy": 0.78, "enemy": 0.80,
    "invader": 0.85, "occupier": 0.84, "aggressor": 0.87, "patriot": 0.72,
    "sovereignty": 0.68, "independence": 0.62, "homeland": 0.65,
    "identity": 0.52,
    # Demographic / civilizational betrayal (v2)
    "replacement": 0.88, "islamization": 0.90, "islamisation": 0.90,
    "overrun": 0.82, "swamp": 0.75, "ethnocide": 0.95,
    "russophobia": 0.85, "russophobic": 0.85,
    "denazification": 0.72, "desovietization": 0.70,
    "great replacement": 0.96, "white genocide": 0.99,
    "demographic replacement": 0.95, "population replacement": 0.94,
    # Authority / Subversion
    "coup": 0.90, "junta": 0.88, "regime": 0.82, "dictator": 0.90,
    "tyranny": 0.88, "oppression": 0.86, "totalitarian": 0.87,
    "illegitimate": 0.85, "overthrow": 0.83, "usurp": 0.82,
    "occupation": 0.78, "colonize": 0.75, "subvert": 0.80,
    "undermine": 0.72, "destabilize": 0.75,
    # Conspiracy / elite subversion (v2)
    "globalist": 0.85, "globalism": 0.80, "zionist": 0.85, "zionism": 0.82,
    "cabal": 0.88, "indoctrinate": 0.88, "indoctrination": 0.90,
    "brainwash": 0.88, "censorship": 0.80, "suppress": 0.78,
    "deep state": 0.88, "new world order": 0.88, "great reset": 0.85,
    "cover up": 0.82,
    # Purity / Degradation
    "nazi": 0.99, "fascist": 0.97, "fascism": 0.97, "nazism": 0.99,
    "degenerate": 0.90, "contaminate": 0.85, "pollute": 0.80,
    "poison": 0.87, "filth": 0.88, "pervert": 0.86, "deviant": 0.84,
    "immoral": 0.82, "evil": 0.90, "satanic": 0.92, "demonic": 0.90,
    "abomination": 0.93,
    # Moral / cultural degradation (v2)
    "degeneracy": 0.90, "depravity": 0.88, "perversion": 0.87, "perverse": 0.86,
    "feminize": 0.75, "feminise": 0.75, "emasculate": 0.80,
    "woke": 0.70, "wokeism": 0.75, "sharia": 0.85, "caliphate": 0.88,
    "jihad": 0.90, "infidel": 0.82, "blasphemy": 0.82,
    "civilization": 0.60, "civilizational": 0.68,
    "gender ideology": 0.82, "cultural marxism": 0.85,
    "existential threat": 0.85,
    # Geopolitical high-charge
    "terrorist": 0.97, "terrorism": 0.97, "extremist": 0.90,
    "radical": 0.75, "jihadist": 0.88, "militant": 0.78,
    "aggression": 0.82, "annexation": 0.75, "weapon": 0.68, "nuclear": 0.78,
    "war crime": 0.97, "chemical weapon": 0.96, "biological weapon": 0.95,
}

AROUSAL_LEXICON = {
    "massacre": 0.97, "genocide": 0.97, "torture": 0.95, "murder": 0.95,
    "rape": 0.96, "terror": 0.95, "terrorist": 0.93, "panic": 0.92,
    "atrocity": 0.93, "horror": 0.92, "outrage": 0.91, "shock": 0.88,
    "explosion": 0.90, "bomb": 0.88, "execute": 0.91, "slaughter": 0.95,
    "exterminate": 0.94, "annihilate": 0.92, "brutal": 0.88, "savage": 0.87,
    "barbaric": 0.88, "monstrous": 0.89, "evil": 0.87, "devil": 0.85,
    "nazi": 0.95, "fascist": 0.94, "coup": 0.85,
    "catastrophe": 0.88, "disaster": 0.86, "collapse": 0.80,
    "war": 0.82, "invasion": 0.83, "aggression": 0.82, "assault": 0.86,
    "attack": 0.80, "kill": 0.92, "death": 0.85, "dead": 0.82,
    "victim": 0.78, "refugee": 0.72, "destroy": 0.78, "devastate": 0.82,
    "oppression": 0.78, "tyranny": 0.80,
    "betrayal": 0.85, "traitor": 0.90, "treason": 0.88,
    "lie": 0.70, "fraud": 0.72, "corrupt": 0.78, "blackmail": 0.85,
    "hate": 0.82, "hatred": 0.83, "violence": 0.85, "abuse": 0.85,
    "criminal": 0.72, "imprison": 0.70,
    "nuclear": 0.80, "pandemic": 0.75, "plague": 0.80,
    "poison": 0.82, "contaminate": 0.78, "toxic": 0.78,
    # Identity arousal (v2)
    "pedophilia": 0.99, "pedophile": 0.98, "groom": 0.92, "grooming": 0.95,
    "molest": 0.97, "predator": 0.90, "underage": 0.85,
    "sex offender": 0.92, "child abuse": 0.98, "sexual assault": 0.95,
    "replacement": 0.82, "great replacement": 0.95, "white genocide": 0.98,
    "islamization": 0.88, "islamisation": 0.88, "overrun": 0.80,
    "degeneracy": 0.88, "depravity": 0.87, "perversion": 0.85,
    "indoctrination": 0.87, "brainwash": 0.85,
    "sharia": 0.85, "caliphate": 0.88, "jihad": 0.90, "infidel": 0.80,
    "globalist": 0.80, "cabal": 0.85, "zionist": 0.82,
    "russophobia": 0.82, "russophobic": 0.80,
    "censorship": 0.78, "suppress": 0.75,
    "woke": 0.68, "degenerate": 0.88, "abomination": 0.92, "filth": 0.86,
    "existential": 0.80, "extinction": 0.88, "ethnocide": 0.90,
    "feminize": 0.70, "feminise": 0.70, "emasculate": 0.78,
    "demographic replacement": 0.93, "gender ideology": 0.78,
    "cultural marxism": 0.82, "deep state": 0.82,
    "new world order": 0.85, "great reset": 0.82,
}


def lemma_candidates(word: str) -> set:
    """Generate approximate root forms via rule-based suffix stripping."""
    forms = {word}
    for suffix, repl in [
        ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"),
        ("ly", ""), ("ness", ""), ("tion", ""), ("ies", "y"),
        ("ers", "er"), ("ists", "ist"), ("ized", "ize"),
        ("ising", "ise"), ("izing", "ize"), ("ised", "ise"),
        ("ise", "ize"), ("ization", "ize"), ("s", ""),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            forms.add(word[: -len(suffix)] + repl)
    return forms


def compute_mei(text: str) -> float:
    """
    Compute Moral-Emotional Intensity for a single text.

    Algorithm:
    1. Preprocess text.
    2. Scan for 2-word phrases (matched first, positions flagged).
    3. Score remaining single tokens via lemmatization.
    4. Contribution = moral_score x arousal_score (both lexicons)
                    = score x 0.5 (one lexicon only)
                    = 0 (neither)
    5. Raw MEI = mean contribution across ALL tokens (not just matched).
    """
    text_clean = preprocess(text)
    tokens = text_clean.split()
    if not tokens:
        return 0.0
    total = 0.0
    matched = set()

    # Pass 1: 2-word phrases
    for i in range(len(tokens) - 1):
        phrase = tokens[i] + " " + tokens[i + 1]
        m = MORAL_LEXICON.get(phrase, 0.0)
        a = AROUSAL_LEXICON.get(phrase, 0.0)
        if m > 0 or a > 0:
            if m > 0 and a > 0:
                c = m * a
            elif m > 0:
                c = m * 0.5
            else:
                c = a * 0.5
            total += c
            matched.add(i)
            matched.add(i + 1)

    # Pass 2: single tokens
    for i, token in enumerate(tokens):
        if i in matched:
            continue
        cands = lemma_candidates(token)
        m = max((MORAL_LEXICON.get(c, 0.0) for c in cands), default=0.0)
        a = max((AROUSAL_LEXICON.get(c, 0.0) for c in cands), default=0.0)
        if m > 0 and a > 0:
            c = m * a
        elif m > 0:
            c = m * 0.5
        elif a > 0:
            c = a * 0.5
        else:
            c = 0.0
        total += c

    return total / len(tokens)


# =============================================================================
# SECTION 3: SEMANTIC DISTANCE
# =============================================================================

def build_tfidf_and_compute_sd(disinfo_texts, info_texts):
    """
    Fit TF-IDF on joint corpus, compute per-pair cosine distance.

    Parameters
    ----------
    disinfo_texts : list[str]   preprocessed disinformation claims
    info_texts    : list[str]   preprocessed matched fact-checks

    Returns
    -------
    sd_raw : np.ndarray   raw cosine distances (not yet normalized)
    tfidf  : fitted TfidfVectorizer (preserved for later transforms)
    """
    all_texts = disinfo_texts + info_texts
    tfidf = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf.fit(all_texts)

    n = len(disinfo_texts)
    sd_raw = np.zeros(n)
    dv = tfidf.transform(disinfo_texts)
    iv = tfidf.transform(info_texts)

    # Batch cosine similarity (avoids materializing full n×n matrix)
    for i in range(0, n, 512):
        sims = cosine_similarity(dv[i : i + 512], iv[i : i + 512])
        sd_raw[i : i + 512] = np.diag(sims)

    sd_raw = 1.0 - sd_raw  # convert similarity to distance
    return sd_raw, tfidf


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# =============================================================================
# SECTION 4: US CORPUS FILTER
# =============================================================================

def apply_us_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only claims where the US is genuinely the primary subject.
    Applied after the EUvsDisinfo country tag pre-filter (tag contains 'us').

    Five criteria (OR logic — any match retains the claim):
    1. Domestic US political vocabulary
    2. US as primary accused state actor
    3. American political figures as central subject
    4. Identity-destabilizing content targeting US audiences
    5. US as explicit agent/driver of described action
    """
    text = df["Disinformation"].fillna("").str.lower()
    country = df["Country"].fillna("").str.lower()

    us_tag = country.str.contains(r"\bus\b", regex=True)

    def c(p):
        return text.str.contains(p, regex=True, na=False)

    domestic = (
        c(r"\belection\b") | c(r"\bdemocrat\b") | c(r"\brepublican\b")
        | c(r"\bcongress\b") | c(r"\bfbi\b") | c(r"\bcia\b")
        | c(r"\bdeep state\b") | c(r"\bracial\b") | c(r"\bblack lives\b")
        | c(r"\bimpeach\b") | c(r"\bvoting\b") | c(r"\bballot\b")
        | c(r"\belection fraud\b") | c(r"\bfake news\b")
        | c(r"\bwhite house\b") | c(r"\bstate department\b")
        | c(r"\bpentagon\b")
    )
    accused = (
        c(r"\bbiolab\b") | c(r"\bbiological weapon\b") | c(r"\busaid\b")
        | c(r"\bnational endowment\b") | c(r"\bnumber.?one.*terror\b")
        | c(r"\bregime change\b") | c(r"\bcolor revolution\b")
        | c(r"\bcolour revolution\b") | c(r"\belection.*interfere\b")
        | c(r"\bamerican.*empire\b")
    )
    figures = (
        c(r"\btrump\b") | c(r"\bbiden\b") | c(r"\bobama\b")
        | c(r"\bclinton\b") | c(r"\bnuland\b") | c(r"\bblinken\b")
        | c(r"\bpompeo\b") | c(r"\bsoros\b") | c(r"\bsecretary of state\b")
    )
    identity_us = (
        c(r"\bwoke\b") | c(r"\bantifa\b") | c(r"\bblm\b")
        | c(r"\bpatriots\b") | c(r"\bgreat reset\b")
        | c(r"\bamerica.*declin\b") | c(r"\bamerica.*collaps\b")
    )
    us_agent = (
        c(r"\bamerican.*policy\b") | c(r"\bamerican.*military\b")
        | c(r"\bamerican.*funded\b") | c(r"\bamerican.*backed\b")
        | c(r"\bus policy\b") | c(r"\bus military\b")
        | c(r"\bus government\b") | c(r"\bwashington.*want\b")
        | c(r"\bwashington.*fund\b")
    )

    mask = us_tag & (domestic | accused | figures | identity_us | us_agent)
    return df[mask].copy()


# =============================================================================
# SECTION 5: HAND-CRAFTED RHETORICAL PATTERN FEATURES
# =============================================================================

def make_handcrafted_features(text_series: pd.Series) -> pd.DataFrame:
    """
    12 binary features encoding rhetorical structural patterns
    identified through systematic analysis of 241 annotator notes.

    Each feature fires (=1) when a specific rhetorical pattern is
    detectable in the claim text via keyword/phrase matching.
    These patterns cannot be reliably captured by TF-IDF alone because
    they describe structural relationships between word clusters rather
    than individual vocabulary items.
    """
    t = text_series.str.lower().fillna("")
    def c(p):
        return t.str.contains(p, regex=True, na=False).astype(float)

    return pd.DataFrame({
        # F1: Migrant/refugee combined with violence vocabulary
        "f_migrant_crime": (
            c(r"migrant|refugee|immigrant")
            * c(r"rape|attack|assault|crime|sexual|murder|kill")
        ).values,
        # F2: Nazi label on society/culture (not just state)
        "f_nazi_civilizational": c(
            r"nazi.*identit|nazi.*civiliz|nazi.*cultur|bandera|"
            r"nazi.*sympathi|west.*nazi|nazi.*west|fascist.*cultur|"
            r"fascism.*society|nazi.*europe"
        ).values,
        # F3: WWII/Cold War history invoked as civilizational vessel
        "f_history_vessel": c(
            r"world war|wwii|third reich|hitler|molotov|ribbentrop|"
            r"nazi germany|cold war|second world"
        ).values,
        # F4: Western civilizational decline / gender degeneration
        "f_west_decline": c(
            r"feminiz|emascul|western.*declin|civilizat.*declin|"
            r"western.*weak|western.*degener|moral.*declin|masculin|"
            r"traditional.*values|christian.*civiliz|western.*collaps|"
            r"cultural.*decline"
        ).values,
        # F5: Demographic replacement / great replacement vocabulary
        "f_demographic_threat": c(
            r"replacement|great replacement|islamiz|"
            r"demographic.*threat|native.*population|"
            r"overrun|ethnic.*replac|white.*genoc|population.*replac"
        ).values,
        # F6: Elite conspiracy against the people
        "f_puppet_conspiracy": c(
            r"puppet.*state|puppet.*govern|outside.*power|"
            r"soros|deep state|globalist|zionist.*conspir|"
            r"cia.*coup|designed.*destroy"
        ).values,
        # F7: Us-versus-them populist polarization
        "f_us_them_divide": c(
            r"us.*them|real.*american|real.*european|"
            r"our.*people|true.*patriot|elite.*people|"
            r"establishment.*people|real.*russian"
        ).values,
        # F8: Geopolitical wrapper / identity payload (both present)
        "f_geo_identity_mix": (
            c(r"nato|sanction|military|diplomat|alliance")
            * c(r"cultur|identit|tradition|value|christian|"
                r"islam|migrant|religion|civiliz")
        ).values,
        # F9: Russian identity / Russophobia attack
        "f_russophobia": c(
            r"russophob|anti.russian|russian.*speaker|"
            r"russian.*identit|desoviet|denazif|"
            r"russian.*language.*ban"
        ).values,
        # F10: LGBTQ / gender ideology as values attack
        "f_gender_values": c(
            r"lgbt|gender.*ideology|woke|gay.*propaganda|"
            r"homosexual.*agenda|transgender|gender.*indoctrin|"
            r"gay.*flag|pedophil|grooming"
        ).values,
        # F11: Media censorship as civilizational attack on values
        "f_media_censorship": c(
            r"rt.*ban|ban.*rt|youtube.*remov|censor.*free|"
            r"information.*aggress|free speech.*destroy|"
            r"fake news|mainstream.*media|media.*war|media.*censor"
        ).values,
        # F12: Nazi label applied specifically to Ukrainian STATE
        "f_nazi_ukraine_state": c(
            r"nazi.*ukraine|ukraine.*nazi|nazi.*kyiv|"
            r"nazi.*junta|nazi.*zelensky|fascist.*ukraine|"
            r"ukraine.*fascist|nazi.*regime.*ukraine|"
            r"nazi.*ukrainian.*authorit"
        ).values,
    }, index=text_series.index)


# =============================================================================
# SECTION 6: FULL PIPELINE
# =============================================================================

def run_pipeline(csv_path: str) -> pd.DataFrame:
    """
    Full IPE pipeline from raw EUvsDisinfo CSV to scored output.

    Produces ipe_results_v3.csv with columns:
        SD, MEI_disinfo, MEI_info, IPE,
        GEO_predicted, ID_predicted, high_ID,
        plus all original metadata columns.
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Year"] = df["Date_parsed"].dt.year
    df = df.dropna(subset=["Year", "Disinformation", "Information"])
    df["Year"] = df["Year"].astype(int)
    print(f"  Corpus: {len(df)} claims, {df['Year'].min()}–{df['Year'].max()}")

    # Preprocess
    print("Preprocessing...")
    df["disinfo_clean"] = df["Disinformation"].apply(preprocess)
    df["info_clean"] = df["Information"].apply(preprocess)

    # Semantic Distance
    print("Computing Semantic Distance...")
    sd_raw, tfidf = build_tfidf_and_compute_sd(
        df["disinfo_clean"].tolist(), df["info_clean"].tolist()
    )
    df["sd_raw"] = sd_raw
    df["SD"] = minmax_normalize(sd_raw)

    # MEI
    print("Computing MEI (v2 lexicon, phrase-aware)...")
    df["mei_disinfo_raw"] = df["disinfo_clean"].apply(compute_mei)
    df["mei_info_raw"] = df["info_clean"].apply(compute_mei)
    df["MEI_disinfo"] = minmax_normalize(df["mei_disinfo_raw"].values)
    df["MEI_info"] = minmax_normalize(df["mei_info_raw"].values)

    # IPE
    df["IPE"] = df["SD"] * df["MEI_disinfo"]
    print(f"  SD mean={df['SD'].mean():.3f}  MEI mean={df['MEI_disinfo'].mean():.4f}  "
          f"IPE mean={df['IPE'].mean():.4f}")

    # Country subsets for classifier training (loaded separately if annotation
    # file is available; otherwise GEO/ID columns are omitted)
    try:
        labeled = pd.read_csv("labeled_data.csv")
        _train_and_predict(df, labeled, tfidf)
    except FileNotFoundError:
        print("  labeled_data.csv not found — skipping GEO/ID prediction.")
        df["GEO_predicted"] = np.nan
        df["ID_predicted"] = np.nan
        df["high_ID"] = np.nan

    # Output
    out_cols = ["Date", "Year", "Country", "Title", "Disinformation", "Information",
                "SD", "MEI_disinfo", "MEI_info", "IPE",
                "sd_raw", "mei_disinfo_raw",
                "GEO_predicted", "ID_predicted", "high_ID"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv("ipe_results_v3.csv", index=False)
    print(f"Output saved: ipe_results_v3.csv ({len(df)} rows)")
    return df


def _train_and_predict(df, labeled, tfidf):
    """Train GEO and ID regressors; add predictions to df in-place."""
    from scipy.sparse import hstack, csr_matrix
    import joblib

    labeled["text_clean"] = labeled["text"].str.lower().str.replace(
        r"[^a-z\s]", " ", regex=True
    )

    # Feature matrix for labeled set
    X_tfidf_labeled = tfidf.transform(labeled["text_clean"])
    hc_labeled = make_handcrafted_features(
        pd.Series(labeled["text_clean"].values, index=labeled.index)
    )
    num_labeled = _make_numeric(labeled, tfidf_fitted=True)

    X_labeled = hstack([
        X_tfidf_labeled,
        csr_matrix(hc_labeled.values),
        csr_matrix(num_labeled.values),
    ])

    y_geo = labeled["GEO_y"].values
    y_id  = labeled["ID_y"].values
    y_id_bin = (y_id >= 2).astype(int)

    X_tr, X_val, yg_tr, yg_val, yi_tr, yi_val = train_test_split(
        X_labeled, y_geo, y_id,
        test_size=0.20, random_state=42, stratify=y_id_bin
    )

    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, min_samples_leaf=5, random_state=42)
    geo_model = GradientBoostingRegressor(**params).fit(X_tr, yg_tr)
    id_model  = GradientBoostingRegressor(**params).fit(X_tr, yi_tr)

    # Validation report
    geo_pred = np.clip(geo_model.predict(X_val), 0, 3)
    id_pred  = np.clip(id_model.predict(X_val),  0, 3)
    geo_mae  = mean_absolute_error(yg_val, geo_pred)
    id_mae   = mean_absolute_error(yi_val, id_pred)
    print(f"  GEO: MAE={geo_mae:.3f}  within-1="
          f"{np.mean(np.abs(yg_val-geo_pred)<=1)*100:.1f}%")
    print(f"  ID:  MAE={id_mae:.3f}  within-1="
          f"{np.mean(np.abs(yi_val-id_pred)<=1)*100:.1f}%")

    # Full corpus prediction
    full_text = df["disinfo_clean"]
    X_tfidf_full = tfidf.transform(full_text)
    hc_full  = make_handcrafted_features(full_text)
    num_full = _make_numeric(df, tfidf_fitted=True)
    X_full   = hstack([
        X_tfidf_full,
        csr_matrix(hc_full.values),
        csr_matrix(num_full.values),
    ])

    df["GEO_predicted"] = np.clip(geo_model.predict(X_full), 0, 3).round(3)
    df["ID_predicted"]  = np.clip(id_model.predict(X_full),  0, 3).round(3)
    df["high_ID"] = (df["ID_predicted"] >= 1.5).astype(int)

    joblib.dump(geo_model, "geo_model.pkl")
    joblib.dump(id_model,  "id_model.pkl")
    print("  Models saved: geo_model.pkl, id_model.pkl")


def _make_numeric(df, tfidf_fitted=False):
    """Build 8 numeric metadata features."""
    text_col = "disinfo_clean" if "disinfo_clean" in df.columns else "text_clean"
    out = pd.DataFrame(index=df.index)
    out["ipe"]  = df.get("IPE",  df.get("ipe",  0)).fillna(0).astype(float)
    out["sd"]   = df.get("SD",   df.get("sd",   0)).fillna(0).astype(float)
    out["mei"]  = df.get("MEI_disinfo", df.get("MEI", df.get("mei", 0))).fillna(0).astype(float)
    out["year_norm"] = (df.get("Year", df.get("year", 2019)).fillna(2019) - 2015) / 7
    out["claim_len"] = df[text_col].str.split().str.len().fillna(0) / 100
    country = df.get("Country", df.get("country", "")).fillna("").str.lower()
    out["country_germany"] = country.str.contains("germany").astype(float)
    out["country_us"]      = country.str.contains(r"\bus\b", regex=True).astype(float)
    out["country_ukraine"] = country.str.contains("ukraine").astype(float)
    return out


# =============================================================================
# SECTION 7: DESCRIPTIVE STATISTICS AND HYPOTHESIS TESTS
# =============================================================================

def run_analysis(df: pd.DataFrame):
    """Reproduce main analysis tables from Chapters 2–3."""
    mei = "MEI_disinfo" if "MEI_disinfo" in df.columns else "MEI"

    print("\n=== TABLE 3.1: Corpus-wide descriptive statistics ===")
    for col, label in [("SD", "Semantic Distance"), (mei, "MEI"), ("IPE", "IPE")]:
        s = df[col]
        print(f"  {label}: mean={s.mean():.3f}  median={s.median():.3f}  "
              f"std={s.std():.3f}  zero%={(s==0).mean()*100:.1f}%")

    if "MEI_info" in df.columns:
        t, p = ttest_ind(df["MEI_info"].dropna(), df[mei].dropna())
        print(f"\n  Fact-check vs disinformation MEI: t={t:.2f}, p={p:.4f}")

    if "GEO_predicted" in df.columns and df["GEO_predicted"].notna().any():
        print("\n=== TABLE 3.3: Cross-national comparison ===")
        def has(col, kw):
            return col.fillna("").str.lower().str.contains(kw, regex=False)

        de = df[has(df["Country"], "germany")]
        ua = df[has(df["Country"], "ukraine")]
        # US refined
        text = df["Disinformation"].fillna("").str.lower()
        ctry = df["Country"].fillna("").str.lower()
        us_mask = ctry.str.contains(r"\bus\b", regex=True) & (
            text.str.contains(r"\btrump\b|\bbiden\b|\bobama\b|\bclinton\b|\belection\b|\bdeep state\b", regex=True, na=False)
        )
        us = df[us_mask]

        for label, sub in [("Germany", de), ("US (refined)", us), ("Ukraine", ua), ("Full", df)]:
            gm = sub["GEO_predicted"].mean()
            im = sub["ID_predicted"].mean()
            hid = sub["high_ID"].mean() * 100
            ipe = sub["IPE"].mean()
            print(f"  {label:15s}: N={len(sub):6d}  GEO={gm:.3f}  ID={im:.3f}  "
                  f"High-ID={hid:.1f}%  IPE={ipe:.4f}")

        f, p = f_oneway(de["ID_predicted"], us["ID_predicted"], ua["ID_predicted"])
        print(f"\n  ANOVA ID across countries: F={f:.2f}, p={p:.4f}")

        print("\n=== TABLE 3.4: Temporal trends ===")
        print(f"  {'Year':6s} {'N':6s} {'GEO':7s} {'ID':7s} {'High-ID%':9s} {'IPE':8s}")
        for year, grp in df.groupby("Year"):
            if pd.notna(year):
                print(f"  {int(year):6d} {len(grp):6d} {grp['GEO_predicted'].mean():7.3f} "
                      f"{grp['ID_predicted'].mean():7.3f} "
                      f"{grp['high_ID'].mean()*100:8.1f}% "
                      f"{grp['IPE'].mean():8.4f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IPE Analysis Pipeline")
    parser.add_argument("--data", default="euvsdisinfo_all_texts_rename.csv",
                        help="Path to EUvsDisinfo CSV file")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip pipeline; load ipe_results_v3.csv and run analysis")
    args = parser.parse_args()

    if args.analyze_only:
        df = pd.read_csv("ipe_results_v3.csv")
        df["Year"] = df["Year"].fillna(2019).astype(int)
        run_analysis(df)
    else:
        df = run_pipeline(args.data)
        run_analysis(df)
