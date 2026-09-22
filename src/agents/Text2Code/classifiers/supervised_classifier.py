"""Wrapper autour du modèle supervisé de production, exposé via l'API déployée
codif-ape-API (https://github.com/InseeFrLab/codif-ape-API).

Sert de baseline de référence, comparée aux méthodes agentiques dans
`src/evaluation/run_eval.py` (cf. cadrage §3.3-B, note de conception), avec le
même contrat de sortie (`MatchVerificationInput`) que `NavigatorAgenticClassifier`
et `AgenticRAGClassifier`.

Nécessite les variables d'environnement CODIF_APE_API_USERNAME et
CODIF_APE_API_PASSWORD (authentification HTTP Basic de l'API). CODIF_APE_API_URL
est optionnelle (défaut : l'instance de production sur le SSP Cloud).
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://codification-ape2025-pytorch.lab.sspcloud.fr"


class SupervisedClassifier:
    """Classifieur de référence : modèle supervisé de production via l'API codif-ape-API."""

    def __init__(self):
        base_url = os.environ.get("CODIF_APE_API_URL", DEFAULT_API_URL)
        auth = (
            os.environ["CODIF_APE_API_USERNAME"],
            os.environ["CODIF_APE_API_PASSWORD"],
        )
        self.client = httpx.AsyncClient(base_url=base_url, auth=auth, timeout=60.0)

    async def __call__(self, activity: str):
        from src.agents.closers.match_verifier import MatchVerificationInput

        code, confidence, libelle = await self._predict(activity)

        return MatchVerificationInput(
            activity=activity,
            code=code,
            proposed_explanation=(
                f"Prédiction du modèle supervisé de production (codif-ape-API) : {libelle}"
            ),
            proposed_confidence=confidence,
        )

    async def _call(self, activity: str, nb_echos_max: int) -> dict:
        """Réponse brute de l'API pour une activité : `{"1": {...}, "2": {...}, "IC": ...}`.

        Les prédictions sont numérotées de 1 à N par probabilité décroissante, N pouvant
        être *inférieur* à `nb_echos_max` — l'API s'arrête d'elle-même quand la queue de
        distribution devient négligeable (un libellé sans ambiguïté ne renvoie qu'un seul
        écho, même si on en demande cinq).
        """
        payload = {"forms": [{"description_activity": activity}]}
        # nb_echos_max=1 makes the API crash with a 500 (server-side bug); ask for 2 at
        # the very least and trim on our side.
        response = await self.client.post(
            "/predict/", params={"nb_echos_max": max(nb_echos_max, 2)}, json=payload
        )
        response.raise_for_status()
        return response.json()[0]

    async def _predict(self, activity: str) -> tuple[str, float, str]:
        result = await self._call(activity, 1)
        top_prediction = result["1"]
        return top_prediction["code"], result["IC"], top_prediction["libelle"]

    async def top_k(self, activity: str, k: int = 5) -> list[dict]:
        """Les `k` codes les plus probables, du plus probable au moins probable.

        Le classement du modèle de production, là où `__call__` n'en garde que la tête :
        c'est ce qui en fait une source de candidats pour l'annotateur (cf.
        src.evaluation.match_verifier_suggestions), et la seule des trois dont le score
        soit une probabilité calibrée plutôt qu'un rang.

        Renvoie au plus `k` entrées `{code, libelle, proba}`, potentiellement moins (cf.
        `_call`).
        """
        result = await self._call(activity, k)
        ranked = [
            {
                "code": result[key]["code"],
                "libelle": result[key]["libelle"],
                "proba": result[key]["probabilite"],
            }
            # Les clés de rang sont des chaînes ("1", "2", ...) mêlées aux métadonnées
            # ("IC", "MLversion") : on ne garde que les premières, triées numériquement
            # — "10" se glisserait avant "2" dans un tri lexicographique.
            for key in sorted((key for key in result if key.isdigit()), key=int)
        ]
        return ranked[:k]
