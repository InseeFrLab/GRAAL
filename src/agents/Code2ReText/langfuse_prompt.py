import logging
from langfuse import Langfuse

logger = logging.getLogger(__name__)


def build_system_prompt(
        prompt_path: str,
        language: str = "English",
        nb_labels: int = 10,
        use_fewshot: bool = False,
        supervisor: bool = False
        ) -> str:
    """
    Import system prompt with the correct language and specify the number of labels.

    Args:
        prompt_path (str): The path for importation.
        language (str): English or French.
        nb_labels (int): The number of labels to generate.

    Returns:
        str: The system prompt.
    """
    langfuse_client = Langfuse()

    # Selecting style
    if use_fewshot:
        style = "fewshot"
    else:
        style = "zeroshot"

    # Selecting language
    if language == "French":
        suffix = "-fr"
    elif language == "English":
        suffix = "-en"
    else:
        logger.warn("Supported languages are English and French. Switching to English...")
        suffix = "-en"

    if supervisor:
        style = "supervisor"
        suffix = ""

    # Importing prompt
    prompt_path = prompt_path + style + suffix
    system_prompt = langfuse_client.get_prompt(prompt_path, label="production").get_langchain_prompt()
    system_prompt = system_prompt.replace("{nb_labels}", str(nb_labels))

    return system_prompt
