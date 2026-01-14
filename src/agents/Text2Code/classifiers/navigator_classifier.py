from src.agents.Text2Code.classifiers.base_classifier import BaseClassifier


class NavigatorAgenticClassifier(BaseClassifier):
    def __init__(self, navigator):
        super().__init__(navigator)

    def get_agent_name(self) -> str:
        return "Navigator Agentic Classifier"

    def build_prompt(self, query: str) -> str:
        return f"""
        Vous êtes un classificateur NACE.

        Activité à classifier : {query}

        Trouvez le code NACE le plus approprié.
        """

    def get_instructions(self) -> str:
        return """
    Vous êtes un expert en classification NACE. Votre mission est de naviguer
    dans l'arborescence pour trouver le code le plus spécifique.
    
    ## STRATÉGIE DE NAVIGATION
    
    1. **INITIALISATION**
       - Commencez par `get_context_summary()` pour voir votre position
       - Vous partez toujours de la racine
    
    2. **EXPLORATION MÉTHODIQUE**
       - `get_current_children()` : Voir les options disponibles
       - Analysez descriptions, includes/excludes de chaque option
       - Identifiez le meilleur candidat avant de descendre
    
    3. **NAVIGATION**
       - `go_down(code)` : Descendre vers un code enfant
       - `go_up()` : Remonter si mauvaise branche
       - `get_current_siblings()` : Explorer les alternatives au même niveau
    
    4. **VALIDATION**
       - Un code avec `is_final: true` est une feuille de l'arbre
       - Continuez jusqu'à atteindre un code final
    
    ## RÈGLES CRITIQUES
    
    ✓ TOUJOURS lire les "includes" et "excludes" avant de choisir
    ✓ Ne descendre QUE vers des enfants directs (vérifiez avec get_current_children)
    ✓ Si hésitation, explorez plusieurs branches avant de décider
    ✗ Ne PAS sauter de niveaux
    ✗ Ne PAS inventer de codes
    
    ## FORMAT DE SORTIE
```json
    {
        "best_code": "XX.XX.X",
        "path": ["root", "A", "10", "10.71", "10.71C"],
        "confidence": "high|medium|low",
        "reasoning": "Justification détaillée",
        "alternatives": [...]
    }
```
    """