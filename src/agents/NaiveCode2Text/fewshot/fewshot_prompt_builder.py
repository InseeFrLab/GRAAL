import logging

logger = logging.getLogger(__name__)


def build_fewshot_system_prompt(
        prompt_path: str,
        language: str = "English",
        nb_labels: int = 10,
        ) -> str:
    """
    Import system prompt for fewshots with the correct language and specify the number of labels.

    Args:
        prompt_path (str): The path for importation.
        language (str): English or French.
        nb_labels (int): The number of labels to generate.

    Returns:
        str: The system prompt.
    """
    if language == "French":
        suffix = "_fr"
    elif language == "English":
        suffix = "_en"
    else:
        logger.warn("Supported languages are English and French. Switching to English...")
        suffix = "_en"

    # Importing prompt
    file_path = prompt_path + "system_prompt_fewshot" + suffix + ".txt"
    with open(file=file_path, mode='r') as f:
        system_prompt = f.read()

    # Indicating the correct number of labels
    system_prompt = system_prompt.replace("{nb_labels}", str(nb_labels))
    return system_prompt


def add_fewshot_user_prompt(
        fewshot: list,
        language: str = "English"
        ) -> str:
    """
    Add few-shot to user prompt with the correct language.

    Args:
        fewshot (list): The list of exampls to give for few-shot.
        language (str): English or French.

    Returns:
        str: The text to add to user prompt.
    """

    if len(fewshot) == 0:
        return ""

    if language == "English":
        prompt = "\nHere are examples of labels for the code:"
    elif language == "French":
        prompt = "\nVoici des exemples de libellés pour le code:"
    else:
        logger.warn("Supported languages are English and French. Switching to English...")
        prompt = "\nHere are examples of labels for the code:"

    for example in fewshot:
        prompt += "\n- " + example

    return prompt
