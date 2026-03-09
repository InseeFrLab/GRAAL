import logging

import numpy.random as npr

logger = logging.getLogger(__name__)


def select_random_items(
        all_items: list,
        min_items: int = 1,
        max_items: int = None,
        geom_prob: float = 0.7
        ) -> list:
    """
    Select random items from a list (no replacement).
    The number of elements selected is drawn according to a geometric law.

    Args:
        all_items (list): The list of the items to sample from.
        min_items (int): The minimum number of items to pick
                        (offset for the geometric law)
        max_items (int): The maximum number of items to pick
                        (if None, equal to the number of items)
        geom_prob (float): parameter to give for the geometric law

    Returns:
        list: Random items sampled from the original list
    """
    # Set max_items to appropriate value
    if max_items is None:
        max_items = len(all_items)-1
    max_items = min(max_items, len(all_items)-1)
    min_items = max(min(min_items, len(all_items)-1), 0)

    # Sample
    random_items = npr.choice(
            all_items,
            min(npr.geometric(geom_prob)-1+min_items, max_items),
            replace=False
        )
    return random_items


def split_spec_and_select(
        all_spec: list,
        examples_divider: str,
        spec_geom_prob: float = 0.7,
        spec_min: int = 1,
        spec_max: int = None,
        examples_geom_prob: float = 0.5,
        examples_min: int = 1,
        examples_max: int = None,
        ) -> list:
    """
    Select random specifications from a list (no replacement).
    Handle the case where specifications contain examples.

    Args:
        all_spec (list): The list of the specifications to sample from.
        examples_divider (str): Divider for examples inside each specification.
        spec_geom_prob (float): parameter to give for the geometric law for spec.
        spec_min (int): The minimum number of specifications to pick
                        (offset for the geometric law)
        spec_max (int): The maximum number of specifications to pick
                        (if None, equal to the number of items)
        examples_geom_prob (float): parameter to give for the geometric law for examples.
        examples_min (int): The minimum number of examples to pick
                        (offset for the geometric law)
        examples_max (int): The maximum number of examples to pick
                        (if None, equal to the number of items)

    Returns:
        list: Random spec with examples sampled from the original list
    """
    # First check to unintended selection
    if len(all_spec) == 0:
        return all_spec

    final_spec = []

    # First, select spec to keep
    if len(all_spec) >= 2:
        random_spec = select_random_items(
            all_items=all_spec,
            min_items=spec_min,
            max_items=spec_max,
            geom_prob=spec_geom_prob
        )
    else:
        random_spec = all_spec

    # Then, select examples to keep within each spec that remains
    for spec in random_spec:
        spec = spec.split(examples_divider)

        # If the spec contains examples
        if len(spec) >= 2:
            sub_spec = select_random_items(
                all_items=spec[1:],
                min_items=examples_min,
                max_items=examples_max,
                geom_prob=examples_geom_prob
            )

            # Add the spec description
            spec = spec[0]

            # Add examples
            for s in sub_spec:
                spec += s

            final_spec.append(spec)
        else:
            final_spec.append(spec[0])

    return final_spec


def build_system_prompt(
        prompt_path: str,
        language: str = "English",
        nb_labels: int = 10,
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

    # Selecting language
    if language == "French":
        suffix = "_fr"
    elif language == "English":
        suffix = "_en"
    else:
        logger.warn("Supported languages are English and French. Switching to English...")
        suffix = "_en"

    # Importing prompt
    file_path = prompt_path + "system_prompt" + suffix + ".txt"
    with open(file=file_path, mode='r') as f:
        system_prompt = f.read()

    # Indicating the correct number of labels
    system_prompt = system_prompt.replace("{nb_labels}", str(nb_labels))
    return system_prompt


def build_user_prompt(
        code_details: dict,
        language: str = "English",
        nb_labels: int = 10,
        includes_divider: str = "\n-",
        examples_divider: str = "\n",
        excludes_divider: str = "\n",
        random_spec_sampling: bool = False,
        random_includes_geom_prob: float = 0.7,
        random_includes_min: int = 1,
        random_includes_max: int = None,
        random_examples_geom_prob: float = 0.5,
        random_examples_min: int = 1,
        random_examples_max: int = None,
        ) -> str:
    """
    Build user prompt.

    Args:
        code_details (dict): A dictionnary that includes details about the code.
            Necessary keys:
            - code: the code itself
            - name: the title of the code
            - includes: what the code includes
            - includes_also: what the code includes also
            - excludes: what the code excludes
            These keys should be the same as in the Neo4j database.
        language (str): English or French
        nb_labels (int): The number of labels to generate.
        includes_divider (str): Divider for includes inside includes and includes_also.
        examples_divider (str): Divider for examples inside each include.
        random_spec_sampling (bool): Select a random sample from all includes and examples.
        random_includes_geom_prob (float): parameter to give for the geometric law for includes.
        random_includes_min (int): The minimum number of includes to pick
                        (offset for the geometric law)
        random_includes_max (int): The maximum number of includes to pick
                        (if None, equal to the number of items)
        random_examples_geom_prob (float): parameter to give for the geometric law for examples.
        random_examples_min (int): The minimum number of examples to pick
                        (offset for the geometric law)
        random_examples_max (int): The maximum number of examples to pick
                        (if None, equal to the number of items)

    Returns:
        list: Random spec with examples sampled from the original list
    """
    # Extracting useful information
    if code_details["includes"]:
        all_includes = code_details["includes"].split(includes_divider)

        if len(all_includes) >= 2:              # Introduction sentence
            all_includes = all_includes[1:]

        # Case with includes_also: extend the Includes
        if code_details["includes_also"]:
            all_includes_also = code_details["includes_also"].split(includes_divider)

            if len(all_includes_also) >= 2:     # Introduction sentence
                all_includes_also = all_includes_also[1:]

            all_includes += all_includes_also

        if random_spec_sampling and len(all_includes) >= 2:
            # Select includes randomly
            random_includes = split_spec_and_select(
                all_spec=all_includes,
                examples_divider=examples_divider,
                spec_geom_prob=random_includes_geom_prob,
                spec_min=random_includes_min,
                spec_max=random_includes_max,
                examples_geom_prob=random_examples_geom_prob,
                examples_min=random_examples_min,
                examples_max=random_examples_max
            )
        else:
            # Select all includes
            random_includes = all_includes
    else:
        random_includes = []

    # Add all excludes
    if code_details["excludes"]:
        all_excludes = code_details["excludes"].split(excludes_divider)
        if len(all_excludes) >= 2:
            all_excludes = all_excludes[1:]
    else:
        all_excludes = []

    # For English
    if language == "English":
        user_prompt = "I would like to generate labels corresponding to the following code:"

        user_prompt += f"\n\nCode : {code_details["code"]}"

        if code_details["name"]:
            user_prompt += f"\n\nTitle : {code_details["name"]}"

        if len(random_includes) >= 1:
            user_prompt += "\n\nIncluded domains:"
            for include in random_includes:
                user_prompt += include

        if len(all_excludes) >= 1:
            user_prompt += "\n\nExcluded domains:"
            for exclude in all_excludes:
                user_prompt += "\n" + exclude

        user_prompt += f"\n\nInstruction:\nGenerate {nb_labels} different, diverse, and realistic" \
            + "labels that strictly correspond to this code, fully complying with the official" \
            + " description."

    # For French
    elif language == "French":
        user_prompt = "Je souhaite générer des libellés correspondant au code suivant :"

        user_prompt += f"\n\nCode : {code_details["code"]}"

        if code_details["name"]:
            user_prompt += f"\n\nTitre : {code_details["name"]}"

        if len(random_includes) >= 1:
            user_prompt += "\n\n Domaines inclus :"
            for include in random_includes:
                user_prompt += include

        if len(all_excludes) >= 1:
            user_prompt += "\n\n Domaines exclus :"
            for exclude in all_excludes:
                user_prompt += "\n" + exclude

        user_prompt += f"\n\n Consigne :\nGénère {nb_labels} libellés différents, variés et" \
            + " réalistes correspondant strictement à ce code, en respectant la notice."

    return user_prompt
