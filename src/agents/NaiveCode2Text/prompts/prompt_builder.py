import numpy.random as npr


def select_random_spec(all_spec, min_spec, max_spec, geom_prob=0.7):
    random_spec = npr.choice(
            all_spec,
            min(npr.geometric(geom_prob)-1+min_spec, max_spec),
            replace=False
        )
    return random_spec


def split_spec_and_select(all_spec, examples_divider, min_spec):
    final_spec = []
    random_spec = select_random_spec(
        all_spec=all_spec,
        min_spec=1,
        max_spec=len(all_spec),
        geom_prob=0.8
    )

    for spec in random_spec:
        spec = spec.split(examples_divider)
        if len(spec) >= 2:
            sub_spec = select_random_spec(
                all_spec=spec[1:],
                min_spec=1,
                max_spec=len(spec)-1,
                geom_prob=0.5
            )
            spec = spec[0]
            for s in sub_spec:
                spec += s

            final_spec.append(spec)
        else:
            final_spec.append(spec[0])

    return final_spec


def build_system_prompt(language: str = "English", notice: str = "NACE"):
    if notice == "NACE":
        system_prompt = "You are the representative of a company."
    else:
        system_prompt = "You are a citizen part of a household"
    return system_prompt


def build_user_prompt(
        code_sample: dict,
        language: str = "English",
        notice: str = "NACE",
        includes_divider: str = "\n-",
        examples_divider: str = "\n",
        excludes_divider: str = "\n",
        ):

    if notice == "NACE":
        user_prompt = f"Write in a short sentence in {language} the official economic activity'\
            'of your company to fill out the wording of a form."

        if code_sample["NAME"]:
            user_prompt += "\n The main activity of the company is about " + code_sample["NAME"]

        if code_sample["Includes"]:
            all_includes = code_sample["Includes"].split(includes_divider)[1:]

            # Case with IncludesAlso: extend the Includes
            if code_sample["IncludesAlso"]:
                all_includes += code_sample["IncludesAlso"].split(includes_divider)[1:]

            # Select includes randomly
            random_includes = split_spec_and_select(
                all_spec=all_includes,
                examples_divider=examples_divider,
                min_spec=1
            )
            if len(random_includes) >= 1:
                user_prompt += "\n Your company is specialized in these fields:"
                for include in random_includes:
                    user_prompt += include

        if code_sample["Excludes"]:
            all_excludes = code_sample["Excludes"].split(excludes_divider)[1:]
            user_prompt += "\n Your company cannot have an activity in these fields:"
            for exclude in all_excludes:
                user_prompt += "\n" + exclude

        user_prompt += "\n The wording must be impersonal and official, no conjugated verbs'\
            ' allowed. Only mention what what you do."

    else:
        user_prompt = f"Write in {language} the name of a product you just bought."

    return user_prompt
