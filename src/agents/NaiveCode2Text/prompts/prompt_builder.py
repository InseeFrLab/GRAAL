import numpy.random as npr


def select_random_spec(all_spec, min_spec, max_spec, geom_prob=0.7):
    """
    Select random specifications from a list.
    The number of elements selected is drawn according to a geometric law.
    """
    random_spec = npr.choice(
            all_spec,
            min(npr.geometric(geom_prob)-1+min_spec, max_spec),
            replace=False
        )
    return random_spec


def split_spec_and_select(
        all_spec,
        examples_divider,
        min_spec,
        spec_prob=0.7,
        example_prob=0.5
        ):
    """
    Select random specifications from a list.
    Handle the case where specifications contain examples.
    """
    final_spec = []
    random_spec = select_random_spec(
        all_spec=all_spec,
        min_spec=1,
        max_spec=len(all_spec),
        geom_prob=spec_prob
    )

    for spec in random_spec:
        spec = spec.split(examples_divider)
        if len(spec) >= 2:
            sub_spec = select_random_spec(
                all_spec=spec[1:],
                min_spec=1,
                max_spec=len(spec)-1,
                geom_prob=example_prob
            )
            spec = spec[0]
            for s in sub_spec:
                spec += s

            final_spec.append(spec)
        else:
            final_spec.append(spec[0])

    return final_spec


def build_system_prompt(language: str = "English", nb_labels: int = 10):
    """
    Build system prompt.
    """
    if language == "English":
        system_prompt = f"""
You are an expert in classification systems and synthetic data generation for \
    training coding models.

Your task is to generate realistic labels that STRICTLY correspond to a given code, \
    based on its official description.

Mandatory constraints:

1. You must produce exactly {nb_labels} labels.

2. Each label must:
    - align with the title and included domains
    - strictly comply with the criteria defined in the description
    - never fall under any excluded domains

3. The labels must be:
    - lexically diverse (no trivial rewordings)
    - structurally varied (short phrases, long descriptions, nominal forms, technical \
        wording, administrative phrasing, business-style wording, etc.)
    - realistic in a professional context

4. Never mention the code in the labels.

5. Do not explain your reasoning.

6. Output ONLY the numbered list of the {nb_labels} labels.

7. Avoid semantic duplicates.

8. Include different levels of specificity (general to highly specific cases).

Important:
If a potential label risks falling into an excluded domain, you must discard it.

Output format:
1. …
2. …
3. …
    …
{nb_labels}. …
        """

    elif language == "French":
        system_prompt = f"""
Tu es un expert en nomenclatures et en production de données synthétiques pour \
l'entraînement de modèles de codification.

Ta mission est de générer des libellés réalistes correspondant STRICTEMENT à un \
code donné, selon sa notice officielle.

Contraintes obligatoires :

1. Tu dois produire exactement {nb_labels} libellés.

2. Chaque libellé doit :
    - correspondre au titre et aux domaines inclus
    - respecter les critères définis dans la notice
    - ne jamais relever des domaines exclus

3. Les libellés doivent être :
    - variés lexicalement (pas de reformulation triviale)
    - de structures différentes (phrases courtes, longues, nominales, techniques, \
        formulations administratives, formulations métier, etc.)
    - réalistes dans un contexte professionnel

4. Ne jamais mentionner le code dans les libellés.

5. Ne jamais expliquer ton raisonnement.

6. Ne produire QUE la liste numérotée des {nb_labels} libellés.

7. Éviter les doublons sémantiques.

8. Introduire différents niveaux de précision (général / spécifique).

Important :
Si un libellé risque d'entrer dans un domaine exclu, tu dois l'écarter.

Format de sortie :

1. …
2. …
3. …
    …
{nb_labels}. …
"""
    return system_prompt


def build_user_prompt(
        code_sample: dict,
        language: str = "English",
        nb_labels: int = 10,
        includes_divider: str = "\n-",
        examples_divider: str = "\n",
        excludes_divider: str = "\n",
        ):
    """
    Build user prompt
    """
    # Extracting useful information
    if code_sample["includes"]:
        all_includes = code_sample["includes"].split(includes_divider)[1:]

        # Case with includes_also: extend the Includes
        if code_sample["includes_also"]:
            all_includes += code_sample["includes_also"].split(includes_divider)[1:]

        # Select includes randomly
        random_includes = split_spec_and_select(
            all_spec=all_includes,
            examples_divider=examples_divider,
            min_spec=1
        )
    else:
        random_includes = []

    if code_sample["excludes"]:
        all_excludes = code_sample["excludes"].split(excludes_divider)[1:]
    else:
        all_excludes = []

    # For English
    if language == "English":
        user_prompt = "I would like to generate labels corresponding to the following code:"

        user_prompt += f"\n\nCode : {code_sample["code"]}"

        if code_sample["name"]:
            user_prompt += f"\n\nTitle : {code_sample["name"]}"

        if len(random_includes) >= 1:
            user_prompt += "\n\nIncluded domains:"
            for include in random_includes:
                user_prompt += include

        if len(all_excludes) >= 1:
            user_prompt += "\n\nExcluded domains:"
            for exclude in all_excludes:
                user_prompt += "\n" + exclude

        user_prompt += f"\n\nInstruction:\nGenerate {nb_labels} different, diverse, and realistic \
            labels that strictly correspond to this code, fully complying with the official \
                description."

    # For French
    elif language == "French":
        user_prompt = "Je souhaite générer des libellés correspondant au code suivant :"

        user_prompt += f"\n\nCode : {code_sample["code"]}"

        if code_sample["name"]:
            user_prompt += f"\n\nTitre : {code_sample["name"]}"

        if len(random_includes) >= 1:
            user_prompt += "\n\n Domaines inclus :"
            for include in random_includes:
                user_prompt += include

        if len(all_excludes) >= 1:
            user_prompt += "\n\n Domaines exclus :"
            for exclude in all_excludes:
                user_prompt += "\n" + exclude

        user_prompt += f"\n\n Consigne :\nGénère {nb_labels} libellés différents, variés et \
            réalistes correspondant strictement à ce code, en respectant la notice."

    return user_prompt
