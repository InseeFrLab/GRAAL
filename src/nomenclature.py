"""Ce que les prompts doivent savoir de la nomenclature, et rien de plus.

Le graphe Neo4j porte toute la nomenclature — codes, hiérarchie, notices — et changer
de nomenclature, c'est changer de graphe (`NEO4J_URL`). Mais un prompt ne se contente
pas de citer des codes : il dit au modèle *ce qu'il classe* (une activité
d'entreprise, un produit acheté par un ménage), comment sont écrits les libellés qu'il
va lire, et à quoi ressemble un code. Rien de cela n'est dans le graphe, et le laisser
écrit en dur dans chaque agent faisait juger des tickets de caisse par un « expert
NACE/NAF » qui lit des déclarations Sirene.

Un profil regroupe donc ces quelques formulations, et `NOMENCLATURE` choisit le profil
comme `NEO4J_URL` choisit le graphe : les deux variables vont ensemble. Les phrases
sont stockées entières plutôt qu'assemblées à partir d'un nom (« activité »,
« produit ») parce que le français accorde : « cette activité », « ce produit ».

Le profil `naf2025` reproduit à l'identique les formulations d'avant ce module : les
runs NAF déjà faits restent comparables à ceux qui suivront.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(frozen=True)
class Nomenclature:
    key: str
    # « expert de la nomenclature {name} », et l'outil get_notices (« codes {name} »).
    name: str
    # En-tête du résumé de la nomenclature (cf. build_nace_summary.build_summary_text).
    summary_name: str
    # Un code tel qu'il apparaît dans le résumé, et deux formes d'entrée qu'accepte
    # l'outil get_notices (pointée ou non, si la nomenclature a les deux).
    example_code: str
    example_tool_codes: str
    # L'objet classé, décliné selon la phrase qui l'emploie.
    item_label: str  # « Activité : <libellé> »
    item_lower: str  # « Audite ce couple (activité, code) »
    item_this: str  # « correspondrait le mieux à cette activité »
    item_the: str  # « incompatible avec l'activité »
    item_described: str  # « l'activité décrite est en plein dans le champ du code »
    item_core: str  # « le coeur de l'activité est hors du champ du code »
    label_kind: str  # « couples (libellé d'activité, code) »
    # Le bloc « Comment lire un libellé » des instructions du MatchVerifier, indenté
    # comme le reste des instructions.
    reading_rules: str


NAF2025 = Nomenclature(
    key="naf2025",
    name="NACE/NAF",
    summary_name="NACE",
    example_code="47.91A",
    example_tool_codes='["47.91A", "4799A"]',
    item_label="Activité",
    item_lower="activité",
    item_this="cette activité",
    item_the="l'activité",
    item_described="l'activité décrite",
    item_core="le coeur de l'activité",
    label_kind="libellé d'activité",
    reading_rules="""\
        Comment lire un libellé — ils sont saisis par les déclarants : tronqués,
        abrégés, en majuscules, souvent moins précis que la nomenclature, parfois
        accompagnés d'un code saisi à la main qui n'engage personne.
        - Plusieurs activités **conflictuelles** (qui relèvent de codes différents et
          s'excluent) : c'est **la première citée** qui décide du code. L'ordre du
          libellé est l'ordre de l'importance, toujours.
        - Plusieurs activités qui ne se contredisent pas (une énumération, un métier
          décrit par ses tâches) : ne code pas un élément de la liste, code
          l'impression d'ensemble — ce qu'est l'activité prise comme un tout.
        - Une imprécision n'est pas une erreur : un libellé trop vague pour trancher
          finement reste correctement codé par un code dont il décrit une partie
          plausible du champ.""",
)

COICOP2018 = Nomenclature(
    key="coicop2018",
    name="COICOP 2018",
    summary_name="COICOP 2018",
    example_code="01.1.1.3.1",
    example_tool_codes='["01.1.1.3.1", "03.1.2"]',
    item_label="Produit",
    item_lower="produit",
    item_this="ce produit",
    item_the="le produit",
    item_described="le produit décrit",
    item_core="l'essentiel du produit",
    label_kind="libellé de produit",
    reading_rules="""\
        Comment lire un libellé — ce sont des dépenses de ménages, recopiées de
        tickets de caisse ou écrites à la main dans un carnet de comptes : tronqués,
        abrégés, en majuscules, avec des fautes, des marques, des quantités, des
        poids ou des prix, souvent moins précis que la nomenclature.
        - Une marque, un conditionnement, une quantité ou un prix ne changent pas le
          code : code ce qu'est le bien ou le service acheté.
        - Plusieurs articles de nature différente sur une même ligne : c'est
          l'article principal, en général **le premier cité**, qui décide du code.
        - Un libellé ambigu entre plusieurs lectures (un « café » peut être la
          boisson servie au comptoir ou le paquet acheté en magasin) : un code qui
          correspond à une lecture plausible de l'achat n'est pas une erreur.
        - Une imprécision n'est pas une erreur : un libellé trop vague pour trancher
          finement reste correctement codé par un code dont il décrit une partie
          plausible du champ.""",
)

PROFILES = {profile.key: profile for profile in (NAF2025, COICOP2018)}

DEFAULT_NOMENCLATURE = NAF2025.key


def get_nomenclature() -> Nomenclature:
    """Le profil désigné par `NOMENCLATURE` (défaut : naf2025)."""
    key = os.environ.get("NOMENCLATURE", DEFAULT_NOMENCLATURE)
    try:
        return PROFILES[key]
    except KeyError:
        raise ValueError(
            f"NOMENCLATURE={key!r} inconnue ; profils disponibles : {', '.join(PROFILES)}"
        ) from None


NOMENCLATURE = get_nomenclature()
