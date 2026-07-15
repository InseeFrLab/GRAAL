"""
Audit interactif des labels du jeu d'entraînement, de bout en bout.

On ne fait pas confiance aux labels du jeu d'entraînement (libelle -> nace2025).
Ce script enchaîne, sur un même échantillon, les briques déjà utilisées
séparément ailleurs dans le repo :

  1. MatchVerifier juge si le label existant lui semble correct
     (src/agents/closers/match_verifier.py)
  2. Les trois méthodes de classification (Navigator, Agentic-RAG, modèle
     supervisé de production) proposent chacune un code (src/main.py)
  3. CodeChooser arbitre entre le label d'origine et les 3 prédictions
     (src/agents/closers/code_chooser.py)
  4. MatchVerifier rejuge le choix final de CodeChooser

C'est la démarche d'"évaluation manuelle" décrite dans
presentation/presentation.qmd (ground truth + prédiction + jugement des
closers, mis bout à bout), appliquée ici au jeu d'entraînement plutôt qu'à un
jeu de test annoté.

Pensé pour être exécuté cellule par cellule (VS Code Jupyter interactive)
plutôt qu'en batch : les appels `await` nécessitent un kernel Jupyter
(top-level await), comme déjà fait dans explorations.py pour classify_navigator.

Pour un rendu HTML via Quarto, il faut d'abord convertir ce script en notebook
avec un kernel pointant sur le venv du projet (Quarto ne détecte pas tout
seul un .py au format "percent", et le kernel `python3` par défaut n'a pas les
dépendances du projet) :

    uv run python -m ipykernel install --user --name graalbis \
        --display-name "Python (GRAALbis .venv)"          # une seule fois
    uv run jupytext --to notebook --set-kernel graalbis \
        evaluate_train_labels.py -o evaluate_train_labels.ipynb
    quarto render evaluate_train_labels.ipynb --to html --execute
"""

# %% Imports
import os

import pandas as pd

from src.agents.closers.code_chooser import CodeChoice, CodeChooser
from src.agents.closers.match_verifier import MatchVerificationInput, MatchVerifier
from src.config import neo4j_config
from src.evaluation.build_eval_set import load_dataframe
from src.evaluation.metrics import normalize_code
from src.evaluation.verify_train_labels import TRAIN_SET_PATH, verify_rows
from src.main import classify_agentic_rag, classify_navigator, classify_supervised
from src.neo4j_graph.graph import Graph

# %% Config
# Cost per row: 1 verify + 3 reclassify + (at most) 1 arbitrate + 1 re-verify calls,
# and Navigator/Agentic-RAG are themselves multi-turn agents, so this adds up fast.
# Start small, bump this up once a run looks right.
N_SAMPLES = 10
SEED = 42
TEXT_COLUMN = "libelle"
CODE_COLUMN = "nace2025"
OUTPUT_PATH = "data/eval/train_verification/train_label_reclassification.parquet"

# %% Sample the train set
df = load_dataframe(TRAIN_SET_PATH)
sample = df.sample(n=N_SAMPLES, seed=SEED)
labels = sample[TEXT_COLUMN].to_list()
train_codes = sample[CODE_COLUMN].to_list()
print(f"Sampled {len(labels)} train rows")

graph = Graph(neo4j_config)
verifier = MatchVerifier(graph)

# %% Step 1 - verify the original train labels
original_verifications = await verify_rows(verifier, sample.to_dicts(), TEXT_COLUMN, CODE_COLUMN)

# %% Step 2 - reclassify with the other methods
navigator_preds = await classify_navigator(labels)
agentic_rag_preds = await classify_agentic_rag(labels)
supervised_preds = await classify_supervised(labels)


# %% Step 3 - arbitrate with CodeChooser among the 4 candidates
async def choose_best(activity: str, candidates: list[str | None]) -> CodeChoice | None:
    """Dedupe candidate codes by normalized value, let CodeChooser pick the best.

    Returns None if the candidates all collapse to a single unique code: there is
    nothing to arbitrate.
    """
    unique = {}
    for code in candidates:
        if not code:
            continue
        norm = normalize_code(code)
        if norm and norm not in unique:
            unique[norm] = code
    codes = list(unique.values())
    if len(codes) < 2:
        return None
    chooser = CodeChooser(graph, num_choices=len(codes))
    return await chooser(activity=activity, codes=codes)


chosen = []
for i, activity in enumerate(labels):
    candidates = [
        train_codes[i],
        getattr(navigator_preds[i], "code", None),
        getattr(agentic_rag_preds[i], "code", None),
        getattr(supervised_preds[i], "code", None),
    ]
    chosen.append(await choose_best(activity, candidates))

# %% Step 4 - close the loop: re-verify CodeChooser's final pick
final_verifications = []
for i, choice in enumerate(chosen):
    if choice is None:
        # Only one unique candidate: nothing was arbitrated, reuse the original verdict.
        final_verifications.append(original_verifications[i])
        continue
    verification = await verifier(
        MatchVerificationInput(
            activity=labels[i],
            code=choice.chosen_code,
            proposed_explanation=choice.explanation,
            proposed_confidence=choice.confidence,
        )
    )
    final_verifications.append(
        {
            "llm_is_match": verification.is_match,
            "llm_confidence": verification.confidence,
            "llm_explanation": verification.explanation,
        }
    )


# %% Assemble the comparison table
def agrees(code: str | None, ref_norm: str | None) -> bool:
    return bool(code) and normalize_code(code) == ref_norm


rows = []
for i, activity in enumerate(labels):
    ref_norm = normalize_code(train_codes[i])
    nav_code = getattr(navigator_preds[i], "code", None)
    rag_code = getattr(agentic_rag_preds[i], "code", None)
    sup_code = getattr(supervised_preds[i], "code", None)
    choice = chosen[i]
    chooser_code = choice.chosen_code if choice else train_codes[i]

    rows.append(
        {
            "libelle": activity,
            "nace2025": train_codes[i],
            "verifier_is_match": original_verifications[i]["llm_is_match"],
            "verifier_confidence": original_verifications[i]["llm_confidence"],
            "verifier_explanation": original_verifications[i]["llm_explanation"],
            "navigator_code": nav_code,
            "navigator_agrees": agrees(nav_code, ref_norm),
            "agentic_rag_code": rag_code,
            "agentic_rag_agrees": agrees(rag_code, ref_norm),
            "supervised_code": sup_code,
            "supervised_agrees": agrees(sup_code, ref_norm),
            "chooser_code": chooser_code,
            "chooser_confidence": choice.confidence if choice else None,
            "chooser_explanation": (
                choice.explanation if choice else "Candidats identiques, pas d'arbitrage nécessaire."
            ),
            "chooser_agrees_with_label": agrees(chooser_code, ref_norm),
            "final_verifier_is_match": final_verifications[i]["llm_is_match"],
            "final_verifier_confidence": final_verifications[i]["llm_confidence"],
            "final_verifier_explanation": final_verifications[i]["llm_explanation"],
        }
    )

comparison = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
comparison.to_parquet(OUTPUT_PATH)
print(f"Comparison table written to {OUTPUT_PATH}")


# %% Display - highlighted comparison table
def highlight_disagreements(row: pd.Series) -> list[str]:
    styles = [""] * len(row)
    agreement_columns = {
        "navigator_code": "navigator_agrees",
        "agentic_rag_code": "agentic_rag_agrees",
        "supervised_code": "supervised_agrees",
        "chooser_code": "chooser_agrees_with_label",
    }
    for code_col, agree_col in agreement_columns.items():
        if not row[agree_col]:
            styles[row.index.get_loc(code_col)] = "background-color: #ffcccc"
    if not row["verifier_is_match"]:
        styles[row.index.get_loc("nace2025")] = "background-color: #ffe0b3"
    if row["final_verifier_is_match"] and not row["verifier_is_match"]:
        styles[row.index.get_loc("chooser_code")] = "background-color: #ccffcc"
    return styles


comparison.style.apply(highlight_disagreements, axis=1)

# %% Display - summary numbers
n = len(comparison)
summary = {
    "n": n,
    "pct_labels_flagged_by_verifier": float(1 - comparison["verifier_is_match"].mean()),
    "navigator_agreement_rate": float(comparison["navigator_agrees"].mean()),
    "agentic_rag_agreement_rate": float(comparison["agentic_rag_agrees"].mean()),
    "supervised_agreement_rate": float(comparison["supervised_agrees"].mean()),
    "pct_rescued_by_chooser": (
        float(
            (
                ~comparison["verifier_is_match"]
                & ~comparison["chooser_agrees_with_label"]
                & comparison["final_verifier_is_match"]
            ).sum()
            / n
        )
        if n
        else None
    ),
}
print(summary)

# %% Display - standout relabeling candidates
relabeling_candidates = comparison[
    ~comparison["verifier_is_match"]
    & ~comparison["chooser_agrees_with_label"]
    & comparison["final_verifier_is_match"]
]
print(f"\n{len(relabeling_candidates)} candidate(s) for relabeling:")
for _, r in relabeling_candidates.iterrows():
    print(f"  {r['libelle']!r}: {r['nace2025']} -> {r['chooser_code']} ({r['chooser_explanation']})")

# %%
