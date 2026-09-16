"""Juge si un code peut être celui d'un libellé d'activité, et cherche mieux.

L'agent ne répond plus seulement « oui / non » sur la paire qu'on lui tend : il note
la correspondance, va chercher le meilleur code concurrent, note celui-là aussi, et
ne tranche qu'ensuite. Le verdict binaire reste la sortie qu'on consomme, mais il
devient la conclusion d'une comparaison au lieu d'être une impression sur un code
isolé — c'est la seule façon d'avoir une réponse à « aucun autre code ne ferait-il
mieux ? », question à laquelle un agent qui ne voit qu'un code ne peut pas répondre.

Trois choix guident tout le reste de ce module :

- **Il faut une alternative à chaque appel, même quand le code convient.** Demander
  une alternative seulement en cas de rejet en fait une justification après coup :
  le modèle décide d'abord, puis se cherche un coupable. Toujours l'exiger inverse
  la charge — il doit avoir regardé ailleurs *avant* de valider, et `alternative_score`
  dit noir sur blanc ce que cet ailleurs valait. Le coût est quelques tokens ; le
  bénéfice est qu'un `is_match=true` signifie « j'ai comparé » et non « je n'ai rien
  vu ».
- **Trois entrées, trois rôles.** Le résumé de la nomenclature (dans les
  instructions, donc préfixe stable et mis en cache par vLLM d'un appel à l'autre)
  dit quels codes existent ; la notice du code jugé, mise dans le prompt, dit ce que
  *celui-là* couvre ; l'outil `get_notices` (cf. Graph.get_notice_tools) sert à lire
  la notice des candidats que le résumé a fait remonter. C'est le minimum qui permet
  de proposer une alternative défendable, et rien de plus : un seul outil, en batch,
  pour que l'exploration tienne en un tour de dialogue (cf. get_max_turns).
- **Le verdict est binaire, mais son incertitude ne l'est pas**, et `is_match_score`
  (auto-déclaré par le modèle) mesure mal cette incertitude. `p_match` la lit là où
  elle est réellement : la distribution des logprobs sur l'unique token `true`/`false`
  généré (cf. `p_match_from_logprobs`). Gratuite — aucun token supplémentaire — et
  calibrable a posteriori : un seuil sur `p_match` transforme un point de
  fonctionnement unique en une courbe ROC.

Le raisonnement du modèle est activé (cf. get_model_settings) : chercher un meilleur
code dans 1059 positions n'est pas la même tâche que valider une paire, et elle ne
tient pas dans l'explication de 50 mots qu'on affiche à l'annotateur. Ce qui coûte du
temps est donc borné explicitement — longueur du raisonnement demandée dans les
instructions, `max_tokens` en filet — parce que cet agent tourne sur des centaines de
milliers de lignes de production et qu'une seconde de plus par ligne s'y compte en
heures.
"""

import logging
import math
import re

from pydantic import BaseModel, Field

from agents.model_settings import ModelSettings
from src.agents.base_agent import BaseAgent
from src.neo4j_graph.build_nace_summary import build_summary_text
from src.neo4j_graph.graph import Graph

logger = logging.getLogger(__name__)

# Le modèle paie chaque mot d'explication en latence (c'est l'essentiel de ce qu'il
# décode une fois son raisonnement terminé) et l'annotateur le paie en lecture. Une
# justification qui oppose le code jugé et l'alternative en citant un critère de
# notice tient dans cette limite.
MAX_EXPLANATION_WORDS = 50

# Longueur visée du bloc <think>, en mots — une consigne, et rien de plus. Mesuré :
# ce modèle raisonne 1 500 à 2 000 mots sur cette tâche quoi qu'on lui demande. Les
# consignes plus dures font pire, pas mieux (« 3 phrases maximum » double le
# raisonnement : le modèle se met à compter ses phrases), et l'endpoint ignore les
# réglages serveur qui fixeraient un budget (`chat_template_kwargs.thinking_budget`,
# `reasoning_effort` : réponses identiques au token près). La consigne reste parce
# qu'elle ne coûte rien, mais le seul levier réel sur le temps d'inférence de cet
# agent est `enable_thinking` (cf. get_model_settings), qui est tout ou rien.
MAX_REASONING_WORDS = 120

# Plafond dur de tokens décodés (raisonnement + JSON). Il ne raccourcit pas le cas
# moyen — rien ne le raccourcit, cf. MAX_REASONING_WORDS — mais il borne la queue :
# sans lui, un libellé qui fait boucler le raisonnement coûte le temps d'une dizaine
# de lignes normales. Réglé au-dessus de la plus longue réponse observée (4 506 tokens
# sur un échantillon de mise au point, médiane ≈ 2 300) parce qu'une troncature n'est
# pas une réponse courte : le bloc <think> précède le JSON, donc un décodage coupé en
# plein raisonnement ne rend aucun JSON du tout — la ligne est rejouée, puis
# abandonnée, et on a payé le plafond deux fois pour rien.
MAX_OUTPUT_TOKENS = 6000

# Niveau le plus fin du résumé de nomenclature injecté dans les instructions : la
# nomenclature entière, codes terminaux compris. Un résumé tronqué aux niveaux hauts
# coûterait moins de tokens mais ne permettrait pas de nommer une alternative
# *terminale*, qui est précisément ce qu'on demande.
SUMMARY_MAX_LEVEL = 5

# Écart de score à partir duquel l'alternative l'emporte, et plancher en-dessous
# duquel le code jugé est déclaré incompatible. Les deux sont écrits dans le prompt
# plutôt que appliqués en Python après coup : c'est la cohérence entre les scores et
# le booléen qu'on veut obtenir du modèle, et un `is_match` recalculé en Python
# masquerait justement les cas où il ne l'obtient pas.
ALTERNATIVE_WINS_MARGIN = 20
INCOMPATIBLE_BELOW = 40

# Le token booléen à lire dans les logprobs est celui qui suit immédiatement la clé
# `"is_match"` du JSON final. Cherché sur le texte reconstitué token à token, pas sur
# la chaîne finale : c'est la position dans la séquence de tokens qui nous intéresse.
# La clé voisine `"is_match_score"` ne peut pas être confondue avec elle — le guillemet
# fermant fait partie du motif.
_IS_MATCH_KEY = re.compile(r'"is_match"\s*:\s*$')


class MatchAssessment(BaseModel):
    """Exactement ce que le modèle génère — rien d'autre n'a sa place ici.

    Un champ présent dans ce schéma est un champ que le modèle doit remplir : lui
    demander de ne pas remplir un champ de comptabilité interne (« do not fill this
    in ») ne marche pas, il le remplit quand même, et le decoding guidé lui fait
    parfois déborder du bruit dans les champs voisins. Ce qui se calcule en Python
    après l'appel vit donc dans MatchVerificationResult, que le modèle ne voit jamais.

    L'ordre des champs — explication, scores, verdict — est celui qu'on demande, et la
    raison en est classique : un modèle qui écrit `is_match` en premier passe le reste
    de la réponse à le justifier. Mais c'est un voeu, pas une garantie. Le décodage
    guidé de l'endpoint (vLLM) ne respecte pas l'ordre des propriétés du schéma : sur
    un même prompt on voit sortir `explanation` en tête, ou `alternative_score`, ou
    `match_score`. C'est l'une des raisons d'activer le raisonnement (cf.
    MatchVerifier.get_model_settings) : le bloc <think>, lui, précède toujours le JSON,
    et fait le travail que l'ordre des champs ne fait plus.

    Les scores sont des entiers en pourcentage plutôt que des flottants sur [0, 1] :
    moins de tokens décodés, et une échelle sur laquelle le modèle est nettement moins
    grégaire (la moitié des flottants auto-déclarés valent 0.9).
    """

    explanation: str = Field(
        description=f"Justification globale en {MAX_EXPLANATION_WORDS} mots maximum. "
        "Elle oppose le code jugé et "
        "l'alternative en s'appuyant sur un élément concret de leurs notices (critère qui "
        "correspond, exclusion qui contredit) — jamais l'annonce d'une vérification à "
        "venir."
    )
    match_score: int = Field(
        description="Correspondance entre le libellé et le code jugé, en pourcentage. 100 "
        "= l'activité décrite est en plein dans le champ du code ; 50 = le libellé est "
        "compatible mais trop vague ou partiellement couvert ; 0 = la notice l'exclut "
        "explicitement ou le coeur de l'activité est hors du champ du code.",
        ge=0,
        le=100,
    )
    alternative_code: str = Field(
        description="Le code terminal de la nomenclature, différent du code jugé, qui "
        "correspondrait le mieux à cette activité. Toujours rempli, y compris quand le "
        "code jugé convient : c'est le meilleur concurrent, pas un remplaçant. Format "
        "exact du résumé de la nomenclature (ex: 47.91A)."
    )
    alternative_score: int = Field(
        description="Correspondance entre le libellé et alternative_code, en pourcentage, "
        "sur la même échelle que match_score.",
        ge=0,
        le=100,
    )
    is_match: bool = Field(
        description="true si le code jugé reste un codage acceptable de cette activité, "
        f"false s'il est incompatible (match_score < {INCOMPATIBLE_BELOW}) ou si "
        f"l'alternative fait nettement mieux (alternative_score dépasse match_score d'au "
        f"moins {ALTERNATIVE_WINS_MARGIN} points)."
    )
    is_match_score: int = Field(
        description="Netteté de ce verdict, en pourcentage : 100 quand l'un des deux codes "
        "domine largement l'autre, 50 quand les deux scores sont à égalité et que le "
        "verdict aurait pu basculer. C'est l'écart entre match_score et alternative_score "
        "qui le gouverne, pas ton assurance générale.",
        ge=0,
        le=100,
    )


class MatchVerificationResult(MatchAssessment):
    """Le verdict du modèle, plus ce que Python sait de l'appel qui l'a produit.

    Type de retour de MatchVerifier ; hérite des champs générés et y ajoute ce qui est
    attaché après coup — jamais demandé au modèle (cf. MatchAssessment).
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
    `is_match_score`, qu'on lui demande, mesure surtout sa propension à écrire 90.

    Renormalisée sur {true, false} : les autres candidats à cette position sont des
    artefacts du decoding guidé (espaces, tabulations, amorces de clé), pas des
    verdicts. Renvoie None si aucun token booléen n'est identifiable (pas de logprobs
    renvoyés par l'endpoint), et 0.0 ou 1.0 si l'alternative est tombée hors des
    `top_logprobs` demandés — un plancher dû à la troncature, pas une certitude.

    Le token retenu est le dernier qui suit la clé `"is_match"` : le modèle raisonne
    avant de répondre (cf. get_model_settings), donc les logprobs couvrent aussi son
    bloc <think>, où un booléen peut très bien apparaître au milieu d'un brouillon de
    JSON.
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
    def __init__(
        self, graph: Graph, enable_thinking: bool = True, summary_max_level: int | None = None
    ):
        # Lus par get_model_settings/get_instructions, que super().__init__ appelle.
        self.enable_thinking = enable_thinking
        # Le résumé est lu dans Neo4j à la construction, pas dans le fichier texte de
        # build_nace_summary.py comme SummaryAgenticClassifier : le vérificateur a de
        # toute façon besoin de la base (c'est elle qui lui donne les notices), et un
        # fichier généré absent — il est gitignoré — ferait échouer un agent qui, lui,
        # tourne en production.
        level = SUMMARY_MAX_LEVEL if summary_max_level is None else summary_max_level
        self.summary = build_summary_text(graph.get_summary_tree(level), level)
        super().__init__(graph)

    def get_agent_name(self) -> str:
        return "MatchVerifier Agent"

    def get_tools(self):
        """Le seul outil utile ici : lire la notice de codes candidats, en batch.

        Les cinq outils de navigation du graphe sont ceux d'un agent qui découvre la
        hiérarchie ; celui-ci l'a entière dans ses instructions. Ce qui lui manque
        quand un code du résumé lui fait de l'oeil, c'est la notice de ce code — ses
        inclusions, et surtout ses exclusions, qui sont l'endroit où se tranchent les
        cas limites. `get_notices` la lui donne pour plusieurs codes d'un coup, ce qui
        rend l'exploration possible sans la rendre lente : avec le raisonnement activé,
        un tour de dialogue de plus, c'est un bloc <think> de plus, donc le temps de la
        ligne doublé (cf. Graph.get_notice_tools et get_max_turns).

        À noter, et à surveiller : sur 34 lignes de mise au point, le modèle ne l'a
        appelé **aucune fois**, y compris sur des cas serrés où les instructions le lui
        demandent explicitement. Il se juge suffisamment informé par le résumé et par la
        notice du code jugé. L'outil reste en place — il coûte un schéma, pas un appel,
        et c'est le seul recours du modèle quand une exclusion déciderait — mais tant
        que `tool_call_count` reste à zéro dans les runs, c'est le résumé et la notice
        du prompt qui font tout le travail, et il ne faut pas créditer l'outil de la
        qualité des alternatives.
        """
        return self.graph.get_notice_tools()

    def get_max_turns(self) -> int:
        """Deux tours : une ronde d'outils, puis la réponse.

        Un tour (l'ancien réglage, sans outil) rendrait l'outil inutilisable. Mais un
        budget large — le `MAX_TURNS` global de 15 — serait ici bien pire qu'une simple
        latitude inutile : le raisonnement étant activé, *chaque* tour décode son propre
        bloc <think> de quelque 2 500 tokens, soit une quinzaine de secondes. Un tour de
        plus ne coûte pas un aller-retour réseau, il double le temps de la ligne, et
        trois tours dépasseraient le timeout client de 60 s (cf. base_agent.py).

        Deux, donc, et c'est pour cela que `get_notices` prend une *liste* de codes :
        le modèle n'a droit qu'à une seule ronde, mais il peut y demander toutes les
        notices qu'il veut. Un modèle qui tenterait quand même un second appel d'outil
        fait échouer la ligne (MaxTurnsExceeded) plutôt que de la faire traîner — un
        échec bruyant et rejoué une fois par l'éval, ce qui est le bon compromis tant
        que le cas reste rare.
        """
        return 2

    def get_model_settings(self) -> ModelSettings:
        """Logprobs demandés, raisonnement activé, longueur plafonnée.

        `top_logprobs` : la distribution à la position du token `true`/`false`, d'où
        `p_match` est tirée (cf. p_match_from_logprobs). Gratuit : les logprobs portent
        sur des tokens de toute façon décodés.

        `enable_thinking=True` : c'est un renversement du réglage précédent, et ce qui
        l'a renversé est le changement de tâche. Tant que l'agent ne faisait que valider
        une paire, le bloc <think> refaisait à perte le travail que le champ
        `explanation`, généré avant le verdict, faisait déjà — d'où sa désactivation.
        Chercher le meilleur concurrent parmi un millier de positions, décider s'il faut
        en lire les notices, puis comparer deux codes chiffres à l'appui, ne tient pas
        dans une explication de quelques dizaines de mots destinée à un humain : ce
        travail-là a besoin d'un brouillon, et le brouillon a besoin d'être jeté.

        `max_tokens` : le filet qui borne la queue de distribution (cf.
        MAX_OUTPUT_TOKENS) — raisonnement et JSON confondus, puisque c'est le total
        décodé qui se paie en secondes.
        """
        settings = {
            "temperature": 0,
            "top_logprobs": 5,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if not self.enable_thinking:
            settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return ModelSettings(**settings)

    def get_instructions(self) -> str:
        """Le rôle, la nomenclature entière, et les règles de lecture d'un libellé.

        Tout ce qui ne dépend pas de la ligne jugée vit ici plutôt que dans le prompt :
        les instructions sont identiques d'un appel à l'autre, donc vLLM les garde en
        cache de préfixe et les ~21 000 tokens du résumé ne se paient qu'une fois pour
        tout un run, au lieu d'une fois par ligne.
        """
        return f"""
        Tu es un expert de la nomenclature NACE/NAF. Ton travail est d'auditer des
        couples (libellé d'activité, code) : dire si le code tient, et surtout dire
        quel autre code aurait pu faire mieux.

        Voici la nomenclature complète, code et nom de chaque position :

        {self.summary}

        Ce résumé ne donne que le code et le nom : il ne contient pas les notices
        officielles (inclusions, exclusions, règle d'affectation). L'outil
        `get_notices` te donne la notice de plusieurs codes d'un seul appel. Tu n'y as
        droit qu'une fois, alors demande tous tes candidats dans le même appel. Sers-t'en
        quand, et seulement quand, la notice déciderait : si ton alternative te paraît
        aussi bonne que le code jugé — moins de 20 points d'écart — lis sa notice
        avant de trancher, car c'est dans les exclusions que se règlent ces cas-là. Si
        l'écart est franc, conclus directement. La notice du code jugé, elle, t'est déjà
        donnée dans le message : ne la redemande jamais.

        Comment lire un libellé — ils sont saisis par les déclarants : tronqués,
        abrégés, en majuscules, souvent moins précis que la nomenclature, parfois
        accompagnés d'un code saisi à la main qui n'engage personne.
        - Plusieurs activités **conflictuelles** (qui relèvent de codes différents et
          s'excluent) : c'est **la première citée** qui décide du code. L'ordre du
          libellé est l'ordre de l'importance, toujours.
        - Plusieurs activités qui ne se contredisent pas (une énumération, un métier
          décrit par ses tâches) : ne code pas un élément de la liste, code
          l'impression d'ensemble — ce qu'est l'activité prise comme un tout.
        - Une imprécision n'est pas une erreur : un libellé trop vague pour trancher
          finement reste correctement codé par un code dont il décrit une partie
          plausible du champ.

        Comment trancher :
        - `match_score` note le code jugé, `alternative_score` note le meilleur
          concurrent que tu aies trouvé. Cherche vraiment ce concurrent : propose-le à
          chaque fois, même quand le code jugé est manifestement bon, pour être sûr
          qu'aucun autre code ne correspond mieux.
        - Le code jugé garde le bénéfice du doute : tu ne le rejettes que si sa notice
          le rend incompatible avec l'activité (match_score < {INCOMPATIBLE_BELOW}) ou
          si l'alternative le dépasse d'au moins {ALTERNATIVE_WINS_MARGIN} points. À
          scores voisins, `is_match` reste true.

        Raisonne avant de répondre, mais brièvement : {MAX_REASONING_WORDS} mots de
        réflexion au maximum, le temps de repérer les candidats, d'en lire les notices
        s'il le faut, et de comparer. Tu es appelé sur des centaines de milliers de
        libellés.
        """

    def get_output_type(self):
        return MatchAssessment

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

        Ne contient que ce qui change d'une ligne à l'autre : les règles, la
        nomenclature et le format de réponse sont dans les instructions, où ils
        restent en cache (cf. get_instructions).
        """
        # La notice entière, exclusions comprises : c'est exactement là que se tranchent
        # les cas limites (« vente à domicile » n'est pas 47.91Z parce que la notice de
        # 47.91Z le dit), et la mettre ici évite au modèle de dépenser un tour d'outil
        # pour l'information dont il a besoin à tous les coups.
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
        Audite ce couple (activité, code).

        Activité : {match_verification_input.activity}

        Code jugé : {match_verification_input.code}
        {code_definition_line}
        {explanation_line}
        Réponds dans cet ordre :
        1. `explanation` : {MAX_EXPLANATION_WORDS} mots maximum, opposant le code jugé
           et ton alternative sur un élément concret de notice.
        2. `match_score` : correspondance libellé / code jugé, en pourcentage.
        3. `alternative_code` : le code terminal, différent de {match_verification_input.code},
           qui correspondrait le mieux à cette activité — toujours, même si le code jugé
           te paraît bon.
        4. `alternative_score` : correspondance libellé / alternative, en pourcentage.
        5. `is_match` : le code jugé tient-il ?
        6. `is_match_score` : netteté de ce verdict, en pourcentage.
        """
        return prompt
