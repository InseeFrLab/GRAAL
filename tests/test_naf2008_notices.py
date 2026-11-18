from src.neo4j_graph.graph_builder.utils.naf2008_notices import (
    Post,
    _join_wrapped,
    attach_bodies,
    build_notices,
    split_sections,
)

LINES = [
    "Section A : sommaire, ignoré 3",
    "Section A : AGRICULTURE",
    "Cette section couvre l'exploitation des ressources.",
    "01 CULTURE ET PRODUCTION",
    "ANIMALE",
    "01.1 CULTURES NON PERMANENTES",
    "01.11 Culture de céréales",
    "01.11Z Culture de céréales",
    "Cette sous-classe comprend :",
    " - la culture de blé, de maïs",
    "et d'orge",
    "Produits associés : 01.11.11p, 01.11.12p,",
    "01.11.20p",
    "Cette sous-classe ne comprend pas :",
    " - la culture du riz (cf. 01.12Z)",
    "01.12 Culture du riz",
    "Cette classe comprend la riziculture.",
    "01.12A Riz irrigué",
    "Cette sous-classe comprend aussi :",
    " - le riz de montagne",
    "01.12B Riz pluvial",
]


def _posts():
    return [
        Post("A", "AGRICULTURE", 1),
        Post("01", "Culture et production animale", 2),
        Post("01.1", "Cultures non permanentes", 3),
        Post("01.11", "Culture de céréales", 4),
        Post("01.11Z", "Culture de céréales", 5),
        Post("01.12", "Culture du riz", 4),
        Post("01.12A", "Riz irrigué", 5),
        Post("01.12B", "Riz pluvial", 5),
    ]


def test_join_wrapped_drops_products_and_rejoins_lines():
    text = _join_wrapped(LINES[8:15])
    assert "Produits associés" not in text and "01.11.20p" not in text
    assert "- la culture de blé, de maïs et d'orge" in text


def test_split_sections_in_pdf_order():
    body = "Cette sous-classe comprend :\n- a\nCette sous-classe comprend aussi :\n- b\n"
    body += "Cette sous-classe ne comprend pas :\n- c"
    inc, also, exc = split_sections(body)
    assert inc.endswith("- a") and also.endswith("- b") and exc.endswith("- c")
    assert split_sections("Cette classe comprend x.") == ("Cette classe comprend x.", None, None)


def test_build_merges_single_subclass_classes_like_naf2025():
    posts = _posts()
    assert attach_bodies(posts, LINES) == []
    df = build_notices(posts).set_index("CODE")
    # 01.11 n'a qu'une sous-classe : un seul noeud, niveau 4, ID de la classe.
    assert "01.11" not in df.index
    assert df.loc["01.11Z", ["ID", "LEVEL", "FINAL", "PARENT_ID"]].tolist() == [
        "0111",
        4,
        1,
        "011",
    ]
    assert "riz (cf. 01.12Z)" in df.loc["01.11Z", "Excludes"]
    # 01.12 en a deux : la classe reste un noeud intermédiaire.
    assert df.loc["01.12", ["LEVEL", "FINAL"]].tolist() == [4, 0]
    assert df.loc["01.12A", ["ID", "LEVEL", "PARENT_ID"]].tolist() == ["0112A", 5, "0112"]
    assert df.loc["01.12A", "IncludesAlso"].endswith("le riz de montagne")
    assert df.loc["01", "PARENT_ID"] == "A"
