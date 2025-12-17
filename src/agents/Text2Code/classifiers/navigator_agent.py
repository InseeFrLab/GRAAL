from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier
from src.agents.base_agent import BaseAgent
from src.agents.closers.code_chooser import CodeChooser
from src.agents.closers.match_verifier import MatchVerificationInput

from src.navigator.Navigator import Navigator

class NavigatorAgenticClassifier(BaseClassifier):
    def __init__(self, navigator: Navigator):
        super().__init__(navigator)

    def get_agent_name(self) -> str:
        return "Navigator Agentic Classifier"
    
    def build_prompt(self) -> str:
        return """
            Vous êtes un expert en classification. Votre mission est de trouver le code de classification le plus spécifique et approprié pour une activité donnée.

            Classez les propositions de la plus probable à la moins probable. Attribuez un score à chaque code.
            Vous avez accès à unensemble d'outils pour naviguer dans l'arborescence des codes NACE/NAF:

            1. COMPRENDRE LA POSITION ACTUELLE
            - Commencez TOUJOURS par get_context_summary() pour comprendre où vous êtes
            - Utilisez get_current_node() pour voir les détails du nœud actuel

            2. EXPLORER LES OPTIONS
            - Utilisez get_children() pour voir les options au niveau suivant
            - Lisez attentivement les descriptions, ce qui est INCLUS et EXCLU
            - Utilisez get_siblings() si aucun enfant ne correspond bien

            3. NAVIGUER INTELLIGEMMENT
            - Descendez avec go_down(code) vers le nœud le plus pertinent
            - Si vous vous trompez de branche, utilisez go_up() puis explorez les siblings
            - Continuez jusqu'à atteindre un code FINAL (is_final: true)

            4. JUSTIFIER VOS CHOIX
            - Expliquez pourquoi vous choisissez chaque nœud
            - Citez les éléments des descriptions qui correspondent à l'activité
            - Mentionnez les alternatives considérées et pourquoi vous les avez écartées

            5. RÉSULTAT FINAL
            Retournez un JSON au format :
            {{
                "best_code": "XX.XX.X",
                "path": ["root", "A", "10", "10.71", "10.71C"],
                "confidence": "high|medium|low",
                "reasoning": "Explication détaillée du choix",
                "alternatives": [
                    {{"code": "XX.XX.Y", "reason": "Pourquoi ce code a été considéré mais écarté"}}
                ]
            }}

            RÈGLES IMPORTANTES :
            - Ne descendez QUE vers des codes qui sont des enfants directs
            - Un code FINAL (is_final: true) est le résultat recherché
            - Les descriptions "Includes" et "Excludes" sont cruciales pour le choix
            - Si vous hésitez entre plusieurs options, explorez-les toutes avant de décider
            """
