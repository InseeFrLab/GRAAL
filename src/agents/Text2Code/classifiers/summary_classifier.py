"""Classifieur donnant à l'agent la structure globale de la nomenclature d'emblée,
plutôt qu'une navigation guidée pas à pas (`BaseClassifier`/`NavigatorAgenticClassifier`)
ou un point de départ par similarité (`AgenticRAGClassifier`).

Contrairement à `BaseClassifier`, qui pilote pas à pas la navigation en Python,
`SummaryAgenticClassifier` laisse le modèle libre pendant l'exploration : un
`Runner.run` en `tool_choice` non forcé, où le modèle décide lui-même s'il appelle un
outil, lequel, avec quel code, et quand s'arrêter. C'est un choix de conception assumé
pour ce classifieur (donner le plus de liberté possible à l'agent) : il n'y a donc pas
de garde-fou Python empêchant la remontée d'un code non terminal pendant l'exploration
— seulement le repli déterministe de `__call__` si le code final n'est pas terminal.
Seule contrainte : le premier tour doit appeler un outil (`tool_choice="required"`).

L'exploration se fait *sans* `output_type`, et la réponse structurée est demandée dans
un second `Runner.run` (`tool_choice="none"`), comme `BaseClassifier` sépare
exploration et finalisation. Un `output_type` envoie un `response_format` JSON que le
backend (vLLM) impose au décodage : le modèle ne peut alors plus émettre d'appel
d'outil, et concluait systématiquement depuis le seul résumé, sans jamais consulter
une notice (0 appel d'outil sur tout un échantillon du golden set NAF 2025).

Utilise les outils *stateless* de `Graph` (`get_code_information`, `get_children`,
`get_descendants`, `get_siblings`, `get_parent`), qui prennent chacun un `code` en
argument, plutôt que les outils du `Navigator` relatifs à une position courante : le
modèle choisit directement quel code interroger, sans notion de position à faire
évoluer pas à pas.

Le résumé est chargé depuis un fichier texte pré-généré par `build_nace_summary.py`
(pas recalculé à chaque appel). Le nom de la nomenclature et de l'objet classé viennent
du profil `NOMENCLATURE` (cf. src.nomenclature) ; le profil `naf2025` reproduit les
formulations d'origine.
"""

import logging

from agents import ItemHelpers, MaxTurnsExceeded, Runner
from agents.model_settings import ModelSettings
from src.agents.base_agent import BaseAgent, count_tool_calls
from src.agents.closers.match_verifier import MatchVerificationInput
from src.neo4j_graph.graph import Graph
from src.nomenclature import NOMENCLATURE

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_PATH = "data/nace_summary.txt"


class SummaryAgenticClassifier(BaseAgent):
    def __init__(self, graph: Graph, summary_path: str = DEFAULT_SUMMARY_PATH):
        with open(summary_path, "r", encoding="utf-8") as f:
            self.summary = f.read()
        super().__init__(graph)
        # Exploration : sans output_type (cf. docstring du module : un response_format
        # interdirait tout appel d'outil côté backend). Au moins un appel d'outil est
        # imposé (tool_choice="required") : livré à lui-même, le modèle concluait depuis
        # le seul résumé sur 57 % du golden set (55 % de bonnes réponses, contre 69 %
        # quand il consultait une notice). Le SDK repasse en "auto" dès le premier
        # appel (reset_tool_choice, par défaut), la suite de l'exploration reste libre.
        self.exploration_agent = self.agent.clone(
            output_type=None,
            model_settings=self.agent.model_settings.resolve(ModelSettings(tool_choice="required")),
            reset_tool_choice=True,
        )
        # Finalisation : mêmes instructions (préfixe de prompt identique), outils
        # déclarés mais interdits — retirer `tools` alors que l'historique contient des
        # tool_calls bloque le rendu du chat template côté backend (cf. BaseClassifier).
        # Sans thinking : le raisonnement a déjà eu lieu pendant l'exploration, ce tour
        # ne fait que le mettre au format ; le thinking y coûtait ~900 tokens (~6 s).
        self.finalize_agent = self.agent.clone(
            model_settings=ModelSettings(
                temperature=0,
                tool_choice="none",
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
        )

    def get_agent_name(self) -> str:
        return "Summary Agentic Classifier"

    def get_output_type(self):
        return MatchVerificationInput

    def get_instructions(self) -> str:
        return f"""
        Vous êtes un expert en classification {NOMENCLATURE.summary_name}.

        Voici un résumé de la nomenclature pour vous orienter :

        {self.summary}

        Ce résumé ne donne que le code et le nom de chaque position : il ne contient pas
        la notice officielle complète (description, inclusions, exclusions, règle
        d'affectation). À vous de décider, avec les outils à votre disposition
        (get_code_information, get_children, get_descendants, get_siblings, get_parent),
        quels codes approfondir pour confirmer ou affiner la position la plus spécifique
        possible. Vous êtes libre d'explorer comme vous le souhaitez, avec une seule
        exigence : avant de conclure, consultez la notice (get_code_information) du code
        terminal que vous comptez retenir, et vérifiez que l'activité n'en est pas exclue
        — un nom de poste proche dans le résumé ne suffit pas à trancher.

{NOMENCLATURE.reading_rules}

        Renvoyez votre réponse finale dans cet ordre : justifiez votre choix (raisonnez
        avant de conclure), indiquez le code terminal retenu (is_final = 1
        obligatoirement), puis votre niveau de confiance entre 0 et 1.
        """

    def build_prompt(self, activity: str) -> str:
        return f"{NOMENCLATURE.item_label} à classifier : {activity}"

    async def _explore(self, prompt: str) -> tuple[list, int]:
        """Exploration libre ; renvoie (historique de conversation, nb d'appels d'outils).

        Si le budget de tours est épuisé, l'historique accumulé jusque-là est conservé
        et la finalisation conclut avec ce que le modèle a déjà appris.
        """
        try:
            result = await Runner.run(
                self.exploration_agent, prompt, max_turns=self.get_max_turns()
            )
            return result.to_input_list(), count_tool_calls(result)
        except MaxTurnsExceeded as e:
            logger.warning("SummaryAgenticClassifier: exploration hit max_turns, finalizing")
            data = e.run_data
            if data is None:
                return ItemHelpers.input_to_new_input_list(prompt), 0
            conversation = ItemHelpers.input_to_new_input_list(data.input) + [
                item.to_input_item() for item in data.new_items
            ]
            n_calls = sum(1 for item in data.new_items if item.type == "tool_call_item")
            return conversation, n_calls

    async def __call__(self, activity: str) -> MatchVerificationInput:
        conversation, tool_calls = await self._explore(self.build_prompt(activity))
        conversation = conversation + [
            {
                "role": "user",
                "content": (
                    "Formalisez maintenant votre réponse finale à partir de ce qui précède, "
                    "sans appeler d'outil : justification, code terminal (is_final = 1), "
                    "puis confiance entre 0 et 1."
                ),
            }
        ]
        final = await Runner.run(self.finalize_agent, conversation, max_turns=2)
        result = final.final_output
        result.tool_call_count = tool_calls + count_tool_calls(final)
        logger.info(f"Result of SummaryAgenticClassifier: \n {result}")
        # Same rationale as BaseClassifier._run_navigator_loop: the model is not a
        # reliable source for this field, so it's always overridden with the caller's
        # own known value.
        result.activity = activity
        # Le code tel qu'il existe dans le graphe : forme pointée, ou classe à sous-classe
        # unique résolue vers celle-ci (« 70.20 » -> « 70.20Y », cf.
        # Graph.resolve_code) — sinon une bonne réponse écrite sous sa forme NACE
        # européenne passerait pour un code non terminal.
        info = self.graph.get_code_information(result.code)
        result.code = info.get("code") or result.code
        if info.get("is_final"):
            return result

        # Code non terminal (« 68.32 ») ou inexistant (« 47.91H », « 70.22Z » de la NAF
        # 2008) : plutôt que de descendre arbitrairement vers la première feuille, on
        # redemande au modèle de choisir parmi les codes terminaux sous le plus proche
        # ancêtre existant, avec tout le contexte de son exploration.
        leaves = self.graph.get_leaves(result.code)
        if len(leaves) == 1:
            # Une seule feuille possible (« 47.91H » -> « 47.91Y ») : pas de choix à
            # demander, le raisonnement du modèle reste valable.
            result.proposed_explanation = (
                f"[leaf choice from {result.code}] {result.proposed_explanation}"
            )
            result.code = leaves[0]["code"]
            return result
        if leaves:
            retry = await self._choose_leaf(final.to_input_list(), result.code, leaves)
            if retry is not None:
                retry.activity = activity
                retry.tool_call_count = result.tool_call_count
                return retry

        # Dernier recours (aucune feuille, ou choix encore hors liste) : descente
        # déterministe ; même convention « confiance 0.0 = repli forcé » que
        # BaseClassifier._fallback_output.
        logger.warning(
            f"SummaryAgenticClassifier returned non-final code {result.code!r}, "
            "falling back to nearest leaf"
        )
        return MatchVerificationInput(
            activity=activity,
            code=self.graph.first_leaf_from(
                self.graph.nearest_existing_code(result.code) or result.code
            ),
            proposed_explanation=(
                f"[non-final fallback] Code non terminal retourné par le modèle : {result.code}."
            ),
            proposed_confidence=0.0,
            tool_call_count=result.tool_call_count,
        )

    async def _choose_leaf(
        self, conversation: list, code: str, leaves: list[dict]
    ) -> MatchVerificationInput | None:
        """Seconde finalisation : choisir parmi `leaves`. None si le choix n'y figure pas."""
        listing = "\n".join(f"- {leaf['code']} : {leaf['name']}" for leaf in leaves)
        conversation = conversation + [
            {
                "role": "user",
                "content": (
                    f"Le code « {code} » n'est pas un code terminal existant de la "
                    "nomenclature. Choisissez le plus approprié parmi les codes "
                    f"terminaux suivants :\n{listing}\n\n"
                    "Répondez sans appeler d'outil : justification, code terminal choisi "
                    "(is_final = 1), puis confiance entre 0 et 1."
                ),
            }
        ]
        final = await Runner.run(self.finalize_agent, conversation, max_turns=2)
        result = final.final_output
        chosen = self.graph.get_code_information(result.code).get("code")
        if chosen not in {leaf["code"] for leaf in leaves}:
            logger.warning(f"SummaryAgenticClassifier: leaf choice {result.code!r} not in list")
            return None
        result.code = chosen
        result.proposed_explanation = f"[leaf choice from {code}] {result.proposed_explanation}"
        return result
