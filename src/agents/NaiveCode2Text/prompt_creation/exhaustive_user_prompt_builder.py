"""
Use to retrieve all specifications from code information.
"""

import logging

logger = logging.getLogger(__name__)


def split_spec_and_group(
        all_spec: list,
        examples_divider: str,
        n_spec: int
        ) -> list:
    """
    Select all specifications from a list and group them by chunks.
    Handle the case where specifications contain examples.
    One example counts as one spec.
    If there are more examples than n_spec, split the examples and repeat the spec
    in the next prompt.
    If n_spec is equal to 1, examples are not split.

    Args:
        all_spec (list): The list of the specifications to sample from.
        examples_divider (str): Divider for examples inside each specification.
        n_spec (int): The max number of specifications per group

    Returns:
        list of list: Every group of spec with examples taken from the original list
    """
    # First check to unintended selection
    if len(all_spec) == 0:
        return all_spec

    assert n_spec >= 1, f"n_spec ({n_spec}) should be greater or equal than 1"
    if n_spec == 1:
        return all_spec

    final_spec = []
    current_spec = []

    # Then, select examples to keep within each spec that remains
    for spec in all_spec:
        spec = spec.split(examples_divider)
        main_spec = spec[0]
        current_spec.append(main_spec)

        # If the spec contains examples
        if len(spec) >= 2:

            # Spec should not be without its examples
            if len(current_spec) == n_spec:
                final_spec.append(current_spec[-1:])
                current_spec = [main_spec]

            examples = spec[1:]

            # Add examples until n_spec is reached
            for example in examples:
                current_spec.append(example)
                if len(current_spec) == n_spec:
                    final_spec.append(current_spec)
                    current_spec = [main_spec]

        if len(current_spec) == n_spec:
            final_spec.append(current_spec)
            current_spec = []

    if len(current_spec) >= 1:
        final_spec.append(current_spec)

    return final_spec


def build_single_user_prompt(
        includes: list,
        excludes: list,
        code_details: dict,
        language: str = "English",
        nb_labels: int = 10
        ) -> str:
    """
    Build one user prompt from given code details and specifications.

    Args:
        includes (list): list of includes to specify
        excludes (list): list of excludes to specify
        code_details (dict): A dictionnary that includes details about the code.
            Necessary keys:
            - code: the code itself
            - name: the title of the code
            These keys should be the same as in the Neo4j database.
        language (str): English or French
        nb_labels (int): The number of labels to generate.

    Returns:
        str: User prompt for the code
    """
    # For English
    if language == "English":
        user_prompt = "I would like to generate labels corresponding to the following code:"

        user_prompt += f"\n\nCode : {code_details["code"]}"

        if code_details["name"]:
            user_prompt += f"\n\nTitle : {code_details["name"]}"

        if len(includes) >= 1:
            user_prompt += "\n\nIncluded domains:"
            for include in includes:
                user_prompt += include

        if len(excludes) >= 1:
            user_prompt += "\n\nExcluded domains:"
            for exclude in excludes:
                user_prompt += "\n" + exclude

        user_prompt += f"\n\nInstruction:\nGenerate {nb_labels} different, diverse, and realistic" \
            + "labels that strictly correspond to this code, fully complying with the official" \
            + " description."

    # For French
    elif language == "French":
        user_prompt = "Génère des libellés correspondant au code suivant :"

        user_prompt += f"\n\nCode : {code_details["code"]}"

        if code_details["name"]:
            user_prompt += f"\n\nTitre : {code_details["name"]}"

        if len(includes) >= 1:
            user_prompt += "\n\n Domaines inclus :"
            for include in includes:
                user_prompt += include

        if len(excludes) >= 1:
            user_prompt += "\n\n Domaines exclus :"
            for exclude in excludes:
                user_prompt += "\n" + exclude

        user_prompt += f"\n\n Consigne :\nGénère {nb_labels} libellés différents, variés et" \
            + " réalistes correspondant strictement à ce code, en respectant la notice."

    return user_prompt


def build_user_prompts(
        code_details: dict,
        language: str = "English",
        nb_labels: int = 10,
        includes_divider: str = "\n-",
        examples_divider: str = "\n",
        excludes_divider: str = "\n",
        n_spec: int = 5
        ) -> list:
    """
    Build all user prompts for the code, each with mach n_spec specifications.

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
        n_spec (int): Number of specifications desired per prompt.

    Returns:
        list[str]: User prompts for the same code
    """
    includes_list = []

    # Extracting useful information
    if code_details["includes"]:
        all_includes = code_details["includes"].split(includes_divider)

        if len(all_includes) >= 2:              # Introduction sentence
            all_includes = all_includes[1:]

        includes_list += all_includes

    # Case with includes_also: extend the Includes
    if code_details["includes_also"]:
        all_includes_also = code_details["includes_also"].split(includes_divider)

        if len(all_includes_also) >= 2:     # Introduction sentence
            all_includes_also = all_includes_also[1:]

        includes_list += all_includes_also

    # Group
    if len(includes_list) >= 1:
        group_includes = split_spec_and_group(
            all_spec=includes_list,
            examples_divider=examples_divider,
            n_spec=n_spec
        )
    else:
        group_includes = []

    # Add all excludes
    if code_details["excludes"]:
        all_excludes = code_details["excludes"].split(excludes_divider)
        if len(all_excludes) >= 2:
            all_excludes = all_excludes[1:]
    else:
        all_excludes = []

    user_prompts = []

    # If no include, still creates a prompt
    if len(group_includes) == 0:
        user_prompt = build_single_user_prompt(
            includes=[],
            excludes=all_excludes,
            code_details=code_details,
            language=language,
            nb_labels=nb_labels
        )

        user_prompts.append(user_prompt)

    else:
        for includes in group_includes:
            user_prompt = build_single_user_prompt(
                includes=includes,
                excludes=all_excludes,
                code_details=code_details,
                language=language,
                nb_labels=nb_labels
            )

            user_prompts.append(user_prompt)

    return user_prompts
