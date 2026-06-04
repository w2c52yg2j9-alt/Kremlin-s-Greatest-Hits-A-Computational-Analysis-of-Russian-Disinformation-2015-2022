# Kremlin's Greatest Hits: A Computational Analysis of Russian Disinformation, 2015–2022

**Master's thesis — Harvard University, Department of Quantitative Social Sciences**  
*Awarded best thesis in the department, April 2026*

---

## What This Project Does

This repository contains the full computational pipeline behind my master's thesis, which asks: *why does pro-Kremlin disinformation stick?*

The answer I propose is neurological. The human brain fires a strong salience signal — called a **Reward Prediction Error (RPE)** — when it encounters something simultaneously *surprising* and *emotionally charged*. I argue that effective disinformation exploits exactly this mechanism, whether intentionally or through iterative feedback from audience engagement.

To test this, I built the **Information Prediction Error (IPE) framework**: a computational model that measures the cognitive potency of disinformation claims along two dimensions, then applies it to **14,497 verified pro-Kremlin claims** from the EUvsDisinfo archive (2015–2022).

---

## The Framework

```
IPE = Semantic Distance × Moral-Emotional Intensity
```

**Semantic Distance (SD)** — How far does the false claim depart from verified reality?  
Measured as TF-IDF cosine distance between each disinformation claim and its matched fact-check. A claim describing the same event in entirely different terms scores high; a subtle distortion scores low.

**Moral-Emotional Intensity (MEI)** — How much affective charge does the claim carry?  
Measured against two published lexicons — the Extended Moral Foundations Dictionary (eMFD) and the NRC Valence-Arousal-Dominance lexicon — using a custom phrase-aware scoring algorithm. I expanded the lexicon by 60 terms across six semantic clusters (demographic threat, elite conspiracy, gender ideology, sexual violence, religious identity, Russian identity attack) after auditing systematic gaps in the initial version.

**IPE** is their product. A claim that is surprising but emotionally inert scores near zero. A claim that is emotionally charged but says what the audience already expected also scores near zero. Only claims combining both properties score high — which is precisely the AND-gate the RPE mechanism requires.

---

## Research Questions

1. **Temporal:** Has the cognitive architecture of pro-Kremlin disinformation changed over time in the direction RPE habituation predicts? (Escalating identity content during crisis periods; declining as audiences adapt)

2. **Cross-national:** Does the composition of disinformation vary across Germany, the United States, and Ukraine in ways reflecting strategic calibration to each audience's specific vulnerabilities?

---

## Technical Stack

| Component | Implementation |
|---|---|
| Text preprocessing | Custom tokenizer, rule-based lemmatizer (no external NLP dependencies) |
| Semantic Distance | `TfidfVectorizer` (15k features, bigrams, sublinear TF) + cosine distance |
| MEI scoring | Custom dual-lexicon phrase-aware algorithm (296 terms) |
| Narrative classification | `GradientBoostingRegressor` (n=300, depth=4) trained on 700 annotated claims |
| Feature engineering | TF-IDF + 12 hand-crafted rhetorical pattern features + 8 numeric metadata features |
| Statistical analysis | OLS regression, one-way ANOVA, pairwise t-tests, chi-square tests |
| Libraries | `pandas`, `numpy`, `scipy`, `scikit-learn` |

No internet access, API keys, or pre-trained model downloads required at any stage.

---

## Repository Structure

```
├── ipe_analysis.py        # Full pipeline: preprocessing → scoring → classification → analysis
├── requirements.txt       # Python dependencies
├── data/
│   └── README.md          # Data sourcing instructions (EUvsDisinfo is publicly available)
└── README.md
```

### Key sections in `ipe_analysis.py`

| Section | What it does |
|---|---|
| `MORAL_LEXICON` / `AROUSAL_LEXICON` | 296-term dual lexicon with per-term scores |
| `compute_mei()` | Phrase-aware MEI scoring algorithm |
| `build_tfidf_and_compute_sd()` | TF-IDF fitting + batched cosine distance |
| `make_handcrafted_features()` | 12 binary rhetorical pattern features derived from 241 annotation notes |
| `run_pipeline()` | End-to-end: load CSV → preprocess → score → classify → output |
| `run_analysis()` | Reproduces all tables from Chapters 2–3: descriptive stats, cross-national comparison, temporal trends |

---

## How to Run

```bash
pip install -r requirements.txt

# Full pipeline (requires euvsdisinfo CSV)
python ipe_analysis.py --data euvsdisinfo_all_texts_rename.csv

# Analysis only (if you already have scored output)
python ipe_analysis.py --analyze-only
```

**Data:** The EUvsDisinfo archive is publicly available at [euvsdisinfo.eu](https://euvsdisinfo.eu). Download and export the full database as CSV. The pipeline expects columns: `Date`, `Disinformation`, `Information`, `Country`.

To download: visit the link above → use the export function to download as CSV
→ rename the file to `euvsdisinfo_all_texts_rename.csv`
→ place it in this `data/` folder


### labeled_data.csv
700 disinformation claims manually annotated by the author with GEO and ID scores
(0–3 scale). Includes 241 annotation notes documenting rhetorical patterns.
Used to train the Gradient Boosting classifier.



Then run:
    python ipe_analysis.py --data data/euvsdisinfo_all_texts_rename.csv
---

## Selected Findings

- Pro-Kremlin disinformation overwhelmingly constructs **alternative realities** (high SD) rather than subtle distortions — the corpus mean SD is 0.74
- Temporal evolution follows an **episodic rather than linear** pattern: identity-destabilizing content spikes during the 2015–16 refugee crisis and the 2022 invasion, then declines as audiences habituate
- Clear evidence of **strategic audience calibration**: Germany receives crisis-driven identity activation; the US receives structural polarisation content; Ukraine receives sustained existential delegitimisation
- Cross-national ANOVA on ID scores is statistically significant (p < 0.001)

---

## Methodological Notes

The MEI lexicon was expanded from 14.5% corpus coverage (v1) to 60.1% (v2) after systematic auditing of claims in the identity-destabilization register, where atrocity-focused vocabulary alone systematically underscored affective charge.

The narrative classifier (GEO vs. ID) was trained on a 700-claim stratified sample with oversampling of the top IPE quartile. A codebook of 60 calibration examples guided annotation; 241 notes documenting override decisions were subsequently analysed to derive the 12 hand-crafted rhetorical features. LDA baseline achieved Cohen's κ = 0.17; the trained classifier substantially outperforms this.

---

## Context

This pipeline was built as a **supplementary file** to a 70-page thesis. It is designed to be fully reproducible: all results reported in Chapters 2 and 3 can be regenerated from the raw EUvsDisinfo CSV using this script alone.

The study is **supply-side**: it measures the structural properties of disinformation content, not its reception or behavioural impact. High IPE indicates cognitive potential, not confirmed persuasion.

---

*Harvard University · M.A. Regional Studies–Russia, Eastern Europe, and Central Asia with Quantitative Concentration · 2026*
