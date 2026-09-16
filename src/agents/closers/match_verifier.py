"""Juge si un code peut être celui d'un libellé d'activité.

L'agent tourne en un seul appel, sans outil (cf. `get_tools`) : tout ce dont il a
besoin — la notice officielle du code — est déjà dans le prompt, que celui-ci
compose depuis Neo4j.

Deux choix guident tout le reste de ce module :

- Le prompt énonce une règle de décision, parce que « est-ce que ce code
  correspond ? » n'en est pas une. Les libellés viennent de déclarations
  d'entreprises : tronqués, abrégés, souvent plus vagues que la nomenclature. Sans
  critère explicite le modèle s'en invente un, et celui qu'il s'invente rejette
  l'imprécision autant que l'erreur — un libellé sous-déterminé n'est pourtant pas
  un mauvais label. D'où : on rejette une incompatibilité, pas une imprécision.
- Le verdict est binaire, mais son incertitude ne l'est pas, et `confidence`
  (auto-déclarée par le modèle) mesure mal cette incertitude. `p_match` la lit là
  où elle est réellement : la distribution des logprobs sur l'unique token
  `true`/`false` généré (cf. `p_match_from_logprobs`). Gratuite — aucun token
  supplémentaire — et calibrable a posteriori : un seuil sur `p_match` transforme
  un point de fonctionnement unique en une courbe ROC.
"""

import logging
import math
import re

from pydantic import BaseModel, Field

from agents.model_settings import ModelSettings
from src.agents.base_agent import BaseAgent
from src.neo4j_graph.graph import Graph

logger = logging.getLogger(__name__)

# Le modèle paie chaque mot d'explication en latence (c'est l'essentiel de ce qu'il
# décode) et l'annotateur le paie en lecture. Une justification qui cite un critère
# de la notice tient largement dans cette limite.
MAX_EXPLANATION_WORDS = 40

# Le token booléen à lire dans les logprobs est celui qui suit immédiatement la clé
# `"is_match"` du JSON final. Cherché sur le texte reconstitué token à token, pas sur
# la chaîne finale : c'est la position dans la séquence de tokens qui nous intéresse.
_IS_MATCH_KEY = re.compile(r'"is_match"\s*:\s*$')


class MatchVerification(BaseModel):
    """Exactement ce que le modèle génère — rien d'autre n'a sa place ici.

    Un champ présent dans ce schéma est un champ que le modèle doit remplir : lui
    demander de ne pas remplir un champ de comptabilité interne (« do not fill this
    in ») ne marche pas, il le remplit quand même, et le decoding guidé lui fait
    parfois déborder du bruit dans les champs voisins. Ce qui se calcule en Python
    après l'appel vit donc dans MatchVerificationResult, que le modèle ne voit jamais.
    """

    # L'ordre des champs est l'ordre de génération (cf. base_agent.py) : l'explication
    # est demandée avant le verdict qu'elle justifie, pour que le raisonnement informe
    # la décision au lieu de rationaliser une décision déjà prise quelques tokens plus
    # tôt. C'est aussi ce qui rend le bloc <think> du modèle redondant, et donc
    # désactivable (cf. MatchVerifier.get_model_settings).
    explanation: str = Field(
        description=f"Justification en {MAX_EXPLANATION_WORDS} mots maximum, écrite avant "
        "le verdict ci-dessous, appuyée sur un élément concret de la notice (critère qui "
        "correspond, exclusion qui contredit) — jamais l'annonce d'une vérification à venir."
    )
    is_match: bool = Field(
        description="true si le code peut être celui de cette activité, false seulement "
        "si la notice le rend incompatible avec elle"
    )
    confidence: float = Field(
        description="À quel point tu es sûr de ton propre verdict, entre 0 et 1 (et non "
        "la probabilité que le code soit le bon)",
        ge=0,
        le=1,
    )


class MatchVerificationResult(MatchVerification):
    """Le verdict du modèle, plus ce que Python sait de l'appel qui l'a produit.

    Type de retour de MatchVerifier ; hérite des champs générés et y ajoute ce qui est
    attaché après coup — jamais demandé au modèle (cf. MatchVerification).
    """

    p_match: float | None = Field(
        default=None,
        description="P(is_match = true) lue dans les logprobs du token booléen généré "
        "(cf. p_match_from_logprobs). None si l'endpoint n'a pas renvoyé de logprobs.",
    )
    tool_call_count: int | None = None
    attempt_count: int | None = None

    def __str__(self):
        return self.model_dump_json()


class MatchVerificationInput(BaseModel):
    activity: str = Field(description="The textual label of the activity to verify")
    # Field order matches generation order for structured output (cf. base_agent.py):
    # the explanation is asked for before the code it justifies, so the model's
    # reasoning can actually inform the choice instead of just rationalizing one it
    # already committed to a few tokens earlier.
    proposed_explanation: str | None = Field(
        default=None,
        description="The explanation provided for the proposed code, if any (absent for a "
        "raw ground-truth label with no accompanying model rationale)",
    )
    code: str = Field(description="The code that has been associated with the activity")
    proposed_confidence: float | None = Field(
        default=None,
        description="The confidence level of the proposed match, between 0 and 1, if any",
        ge=0,
        le=1,
    )
    tool_call_count: int | None = Field(
        default=None,
        description="Do not fill this in — populated automatically after the call completes.",
    )
    attempt_count: int | None = Field(
        default=None,
        description="Do not fill this in — populated automatically after the call completes.",
    )


def _output_logprobs(result):
    """Logprobs du dernier message généré d'un `Runner.run`, ou None."""
    for response in reversed(result.raw_responses):
        for item in reversed(response.output):
            for content in getattr(item, "content", None) or []:
                logprobs = getattr(content, "logprobs", None)
                if logprobs:
                    return logprobs
    return None


def p_match_from_logprobs(result) -> float | None:
    """P(is_match = true) lue sur l'unique token booléen de la réponse.

    Le modèle décode exactement un token pour `is_match`, et l'endpoint renvoie la
    distribution des alternatives à cette position : la probabilité du verdict s'y lit
    directement, sans rien demander au modèle ni décoder un token de plus — là où
    `confidence`, qu'on lui demande, mesure surtout sa propension à écrire 0.9.

    Renormalisée sur {true, false} : les autres candidats à cette position sont des
    artefacts du decoding guidé (espaces, tabulations, amorces de clé), pas des
    verdicts. Renvoie None si aucun token booléen n'est identifiable (pas de logprobs
    renvoyés par l'endpoint), et 0.0 ou 1.0 si l'alternative est tombée hors des
    `top_logprobs` demandés — un plancher dû à la troncature, pas une certitude.

    Le token retenu est le dernier qui suit la clé `"is_match"` : quand le modèle
    raisonne avant de répondre (cf. get_model_settings), les logprobs couvrent aussi
    son bloc <think>, où un booléen peut très bien apparaître au milieu d'un brouillon
    de JSON.
    """
    logprobs = _output_logprobs(result)
    if not logprobs:
        return None

    text = ""
    boolean_token = None
    for token_logprob in logprobs:
        if _IS_MATCH_KEY.search(text):
            boolean_token = token_logprob
        text += token_logprob.token
    if boolean_token is None:
        return None

    p_true = p_false = 0.0
    # `top_logprobs` inclut le token effectivement émis ; le fallback couvre un
    # endpoint qui renverrait les logprobs sans les alternatives.
    for alternative in boolean_token.top_logprobs or [boolean_token]:
        if alternative.token.strip().lower() == "true":
            p_true += math.exp(alternative.logprob)
        elif alternative.token.strip().lower() == "false":
            p_false += math.exp(alternative.logprob)
    if p_true + p_false == 0:
        logger.warning(f"No boolean candidate in logprobs for token {boolean_token.token!r}")
        return None
    return p_true / (p_true + p_false)


class MatchVerifier(BaseAgent):
    def __init__(self, graph: Graph, enable_thinking: bool = False):
        # Lu par get_model_settings, que super().__init__ appelle.
        self.enable_thinking = enable_thinking
        super().__init__(graph)

    def get_agent_name(self) -> str:
        return "MatchVerifier Agent"

    def get_tools(self):
        """Aucun outil : juger une paire (libellé, code) est un jugement en un coup.

        L'agent n'en appelait aucun de toute façon — le prompt contient déjà la notice
        du code, seule information dont il a besoin — mais les schémas des cinq outils
        du graphe lui coûtaient ~500 tokens de prompt par appel, et laissaient ouverte
        la possibilité d'un détour multi-tours sur une question qui n'en demande pas.
        """
        return []

    def get_max_turns(self) -> int:
        """Un seul tour : sans outil, il n'y a rien à faire d'un second."""
        return 1

    def get_model_settings(self) -> ModelSettings:
        """Logprobs demandés, raisonnement désactivé par défaut.

        `top_logprobs` : la distribution à la position du token `true`/`false`, d'où
        `p_match` est tirée (cf. p_match_from_logprobs). Gratuit : les logprobs portent
        sur des tokens de toute façon décodés.

        `enable_thinking=False` : Qwen3 sous vLLM raisonne par défaut, ce qui triplait
        le nombre de tokens décodés (≈1200 contre ≈240 mesurés sur un appel type) pour
        une réponse identique — le schéma de sortie demande déjà l'explication avant le
        verdict, donc le raisonnement-avant-conclusion est payé deux fois, et la
        deuxième fois il est jeté au parsing. Le paramètre reste ouvert pour pouvoir
        rejouer la comparaison sur un autre modèle ou un autre prompt.
        """
        settings = {"temperature": 0, "top_logprobs": 5}
        if not self.enable_thinking:
            settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return ModelSettings(**settings)

    def get_instructions(self) -> str:
        return """
                Tu es un agent spécialisé dans la vérification de la validité d'une
                correspondance entre un libellé textuel et le code qui lui a été associé.
            """

    def get_output_type(self):
        return MatchVerification

    def wrap_output(self, model_output, result) -> MatchVerificationResult:
        return MatchVerificationResult(
            **model_output.model_dump(), p_match=p_match_from_logprobs(result)
        )

    def build_prompt(self, match_verification_input: MatchVerificationInput) -> str:
        """Construire le prompt pour l'agent de vérification de correspondance.

        Ne dit jamais d'où vient le code. Le même prompt sert à juger un label de
        référence et la proposition d'un classifieur, et l'origine n'est pas un
        argument : annoncer une vérité terrain invite à la valider, annoncer une
        proposition de modèle invite à la corriger. Seule l'explication qui accompagne
        éventuellement le code est montrée — c'est un raisonnement à évaluer, pas une
        provenance (sa confidence, elle, n'est délibérément pas affichée : c'est une
        ancre, pas un argument).
        """
        # La notice entière, exclusions comprises : c'est exactement là que se tranchent
        # les cas limites (« vente à domicile » n'est pas 47.91Z parce que la notice de
        # 47.91Z le dit), et le modèle n'a aucun autre moyen de les connaître.
        notice = self.graph.get_notice(match_verification_input.code)
        if notice:
            code_definition_line = f"Notice officielle du code :\n{notice}"
        else:
            code_definition_line = (
                "Aucune notice trouvée dans la base pour ce code : appuie-toi sur "
                "ta connaissance de la nomenclature."
            )

        explanation_line = ""
        if match_verification_input.proposed_explanation:
            explanation_line = (
                "\nExplication avancée pour ce code : "
                f"{match_verification_input.proposed_explanation}\n"
            )

        prompt = f"""
        Vérifie si le code suivant peut être celui de l'activité décrite.

        Activité : {match_verification_input.activity}

        Code : {match_verification_input.code}
        {code_definition_line}
        {explanation_line}
        Règle de décision — les libellés sont saisis par les déclarants : tronqués,
        abrégés, souvent moins précis que la nomenclature.
        - Rejette une incompatibilité : la notice, en particulier ses exclusions,
          rattache explicitement l'activité à un autre code, ou le coeur de l'activité
          décrite est absent du champ du code.
        - Ne rejette pas une imprécision : un libellé qui décrit une partie plausible du
          champ du code est une correspondance valide, même s'il n'en couvre pas tout,
          même s'il pourrait aussi relever d'un autre code, et même s'il est trop vague
          pour que tu puisses trancher. C'est l'incompatibilité qui doit être démontrée.
        - Un libellé qui mentionne plusieurs activités correspond dès lors que le code
          couvre l'une d'elles de façon plausible.

        Réponds en fournissant, dans cet ordre :
        1. Une explication de {MAX_EXPLANATION_WORDS} mots maximum qui justifie ta
           décision par un élément concret de la notice — raisonne avant de conclure,
           n'annonce jamais une vérification à venir ("je vais vérifier..."), la
           vérification est déjà faite au moment où tu réponds.
        2. Un booléen : true si la correspondance est valide, false si elle est
           incompatible.
        3. Ta confiance dans ce verdict, entre 0 et 1.
        """
        return prompt
