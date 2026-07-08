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

    async def _predict(self, activity: str) -> tuple[str, float, str]:
        payload = {"forms": [{"description_activity": activity}]}
        # nb_echos_max=1 makes the API crash with a 500 (server-side bug); ask
        # for 2 and keep only the top prediction.
        response = await self.client.post("/predict/", params={"nb_echos_max": 2}, json=payload)
        response.raise_for_status()
        result = response.json()[0]

        top_prediction = result["1"]
        return top_prediction["code"], result["IC"], top_prediction["libelle"]
