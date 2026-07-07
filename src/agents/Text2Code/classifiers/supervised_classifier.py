"""Wrapper autour du modèle supervisé de production (torchTextClassifiers, MLflow).

Ce classifieur n'est PAS un agent LLM : il charge le modèle de *deep learning*
actuellement en production (cf. cadrage §1.1, entraîné avec le package
`torchTextClassifiers`, distribué via MLflow) et l'expose avec le même contrat
de sortie (`MatchVerificationInput`) que `NavigatorAgenticClassifier` et
`AgenticRAGClassifier`, pour qu'il puisse être comparé aux méthodes agentiques
dans `src/evaluation/run_eval.py` (cf. cadrage §3.3-B, note de conception).

Le modèle est chargé une seule fois par processus (cache module-level) via
l'API MLflow pyfunc, indépendante du framework d'entraînement sous-jacent.

**Non validé à ce stade** : le format exact renvoyé par `.predict()` pour ce
modèle précis (nom des colonnes de sortie, présence d'un score de confiance)
n'a pas pu être vérifié dans cet environnement (pas d'accès au tracking
MLflow). `_parse_prediction` gère plusieurs formats plausibles et journalise
un avertissement s'il doit se rabattre sur une confiance par défaut — à
corriger après un premier essai réel sur le Datalab.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_model = None  # cache module-level : un seul chargement par processus


def _load_model():
    global _model
    if _model is None:
        import mlflow

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        model_uri = os.environ["MLFLOW_MODEL_URI"]  # ex. "models:/apet_classifier/Production"

        if model_uri.startswith(("http://", "https://")):
            raise ValueError(
                f"MLFLOW_MODEL_URI={model_uri!r} looks like a browser link to the MLflow UI, "
                "not a model URI. mlflow.pyfunc.load_model() expects the 'models:' scheme, "
                "e.g. 'models:/<model_name>/<version_or_stage>' (for '.../#/models/"
                "FastText-pytorch/versions/9', use 'models:/FastText-pytorch/9'). Also check "
                "that MLFLOW_TRACKING_URI points to the MLflow server where that model is "
                "actually registered, not a different (e.g. personal) instance."
            )

        logger.info(f"Loading supervised model '{model_uri}' from MLflow ({tracking_uri})")

        mlflow.set_tracking_uri(tracking_uri)
        _model = mlflow.pyfunc.load_model(model_uri)
    return _model


def _parse_prediction(raw_prediction, text: str) -> tuple[str, float]:
    """Extrait (code, confiance) d'une sortie `.predict()` MLflow pyfunc.

    Gère les formats plausibles pour un modèle de classification logué en
    pyfunc : DataFrame avec colonnes code/confiance explicites, DataFrame à
    une seule colonne (pas de confiance dans la sortie), ou liste/array brute.
    """
    try:
        import pandas as pd

        if isinstance(raw_prediction, pd.DataFrame):
            row = raw_prediction.iloc[0]
            code_col = next(
                (c for c in ("code", "prediction", "label", "apet2025") if c in row.index), None
            )
            confidence_col = next(
                (c for c in ("confidence", "score", "probability") if c in row.index), None
            )
            if code_col is not None:
                code = str(row[code_col])
                confidence = float(row[confidence_col]) if confidence_col is not None else None
                if confidence is None:
                    logger.warning(
                        f"No confidence column in model output for '{text}'; defaulting to 1.0"
                    )
                return code, confidence if confidence is not None else 1.0
    except ImportError:
        pass

    # Repli : liste/array de labels, éventuellement de tuples (label, score)
    first = raw_prediction[0] if hasattr(raw_prediction, "__getitem__") else raw_prediction
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return str(first[0]), float(first[1])

    logger.warning(f"Unrecognized model output shape for '{text}'; defaulting confidence to 1.0")
    return str(first), 1.0


class SupervisedClassifier:
    """Classifieur de référence : modèle supervisé de production via MLflow."""

    def __init__(self):
        self.model = _load_model()

    async def __call__(self, activity: str):
        from src.agents.closers.match_verifier import MatchVerificationInput

        code, confidence = await asyncio.to_thread(self._predict, activity)

        return MatchVerificationInput(
            activity=activity,
            code=code,
            proposed_explanation="Prédiction du modèle supervisé de production (torchTextClassifiers)",
            proposed_confidence=confidence,
        )

    def _predict(self, activity: str) -> tuple[str, float]:
        import pandas as pd

        raw_prediction = self.model.predict(pd.DataFrame({"text": [activity]}))
        return _parse_prediction(raw_prediction, activity)
