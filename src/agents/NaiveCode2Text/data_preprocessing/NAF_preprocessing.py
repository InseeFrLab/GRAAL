"""
Useful if the code of the input data for sampling is different from the one
of the notice (DD.DDL).
"""


def NAF_to_NACE(
        code: str
        ) -> str:
    """
    For the case of NAF code format (DDDDL), transform it into NACE (DD.DD).

    Args:
        code (str): The code in NAF format to transform.

    Returns:
        str: The code in NACE format.
    """
    return code[0:2] + '.' + code[2:4]


def to_proper_NAF(
        code: str
        ) -> str:
    """
    For the case of bad NAF code format (DDDDL), transform it into proper NAF (DD.DDL).

    Args:
        code (str): The code in bad NAF format to transform.

    Returns:
        str: The code in proper NAF format.
    """
    return code[0:2] + '.' + code[2:]


def to_bad_NAF(
        code: str
        ) -> str:
    """
    For the case of proper NAF code format (DD.DDL), transform it into bad NAF (DDDDL).

    Args:
        code (str): The code in proper NAF format to transform.

    Returns:
        str: The code in bad NAF format.
    """
    return code[0:2] + code[3:]
