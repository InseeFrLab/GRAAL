from pydantic import BaseModel, Field

from src.agents.base_agent import BaseAgent
from src.neo4j_graph.graph import Graph


class MatchVerificationResult(BaseModel):
    # Field order matches generation order for structured output (cf. base_agent.py):
    # the explanation is asked for before the verdict/confidence it justifies, so the
    # model's reasoning can actually inform the decision instead of just rationalizing
    # one it already committed to a few tokens earlier.
    explanation: str = Field(
        description="Concise explanation, written before your verdict below, that reasons "
        "about a concrete element (e.g. a matching or conflicting element of the code's "
        "official definition) — never a restatement of what you are about to check."
    )
    is_match: bool = Field(description="Indicates whether the match is valid or not")
    confidence: float = Field(
        description="Confidence level of the verification, between 0 and 1", ge=0, le=1
    )
    tool_call_count: int | None = Field(
        default=None,
        description="Do not fill this in — populated automatically after the call completes.",
    )
    attempt_count: int | None = Field(
        default=None,
        description="Do not fill this in — populated automatically after the call completes.",
    )

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


class MatchVerifier(BaseAgent):
    def __init__(self, graph: Graph):
        super().__init__(graph)

    def get_agent_name(self) -> str:
        return "MatchVerifier Agent"

    def get_instructions(self) -> str:
        return """
                Tu es un agent spécialisé dans la vérification de la validité d'une
                correspondance entre un libellé textuel et le code qui lui a été associé.
            """

    def get_output_type(self):
        return MatchVerificationResult

    def build_prompt(self, match_verification_input: MatchVerificationInput) -> str:
        """
        Construire le prompt pour l'agent de vérification de correspondance.
        """
        if match_verification_input.proposed_explanation:
            explanation_line = (
                f"Explication proposée : {match_verification_input.proposed_explanation}"
            )
        else:
            explanation_line = (
                "Aucune explication n'est fournie : il s'agit probablement d'un code de "
                "référence (vérité terrain), pas de la proposition d'un modèle. Juge la "
                "correspondance activité/code sur le fond, sans présumer qu'elle est correcte."
            )

        code_info = self.graph.get_code_information(match_verification_input.code)
        if code_info.get("description"):
            code_definition_line = (
                f"Définition officielle du code ({code_info.get('name')}) : "
                f"{code_info['description']}"
            )
        else:
            code_definition_line = (
                "Aucune définition trouvée dans la base pour ce code : appuie-toi sur "
                "ta connaissance de la nomenclature."
            )

        prompt = f"""
        Vérifie si le code suivant correspond bien à l'activité décrite.

        Activité : {match_verification_input.activity}

        Code proposé : {match_verification_input.code}
        {code_definition_line}
        {explanation_line}

        Réponds en fournissant, dans cet ordre :
        1. Une explication concise qui justifie ta décision avec un élément concret (ex. un
           critère de la définition officielle qui correspond ou qui contredit l'activité) —
           raisonne avant de conclure, n'annonce jamais une vérification à venir ("je vais
           vérifier..."), la vérification est déjà faite au moment où tu réponds.
        2. Un booléen indiquant si la correspondance est valide.
        3. Un niveau de confiance entre 0 et 1.
        """
        return prompt
