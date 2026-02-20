def add_fewshot(
        fewshot: list,
        language: str = "English"
        ) -> str:

    if len(fewshot) == 0:
        return ""

    if language == "English":
        prompt = "\nHere are examples of labels for the code, copy their format and writing" \
            + " style:\n"
    elif language == "French":
        prompt = "\nVoici des exemples de libellés pour le code, copie leur format et leur style" \
            + " d'écriture:\n"
    else:
        raise "Given parameter language is not English or French"

    for example in fewshot:
        prompt += "\n- " + example

    return prompt
