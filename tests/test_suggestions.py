"""Tests de la fusion des candidats proposés à l'annotateur.

Ne portent que sur `fuse_candidates`, la seule partie de
src.evaluation.match_verifier_suggestions qui ne demande ni Neo4j, ni l'API du modèle
supervisé : c'est aussi la seule où une erreur serait silencieuse, un mauvais classement
n'ayant l'air de rien.
"""

from src.evaluation.match_verifier_suggestions import RRF_K, SOURCE_WEIGHTS, fuse_candidates


def test_agreement_between_sources_beats_a_single_first_place():
    # 62.01Z n'est premier nulle part mais cité par deux sources ; 10.71C est premier
    # chez une seule. C'est tout l'intérêt de la fusion par rangs.
    fused = fuse_candidates(
        {
            "supervised": ["10.71C", "62.01Z"],
            "alternative": ["43.99C", "62.01Z"],
        }
    )
    assert [c["code"] for c in fused] == ["6201Z", "1071C", "4399C"]
    assert fused[0]["sources"] == ["supervised", "alternative"]


def test_codes_are_matched_across_dotted_and_undotted_forms():
    # Le modèle supervisé rend « 1071C », la recherche vectorielle « 10.71C » : sans
    # normalisation les deux sources ne se rejoindraient jamais.
    fused = fuse_candidates({"supervised": ["1071C"], "embedding": ["10.71C"]})
    assert len(fused) == 1
    assert fused[0]["code"] == "1071C"
    assert fused[0]["sources"] == ["supervised", "embedding"]
    expected = SOURCE_WEIGHTS["supervised"] + SOURCE_WEIGHTS["embedding"]
    assert fused[0]["fusion_score"] == expected / (RRF_K + 1)


def test_the_production_model_outranks_a_notice_neighbour_at_the_same_rank():
    # La recherche vectorielle rend toujours ses cinq candidats quand le modèle supervisé
    # n'en rend souvent qu'un : à poids égaux, le volume déciderait du classement.
    fused = fuse_candidates({"supervised": ["62.01Z"], "embedding": ["10.71C", "43.99C", "47.11L"]})
    assert [c["code"] for c in fused][0] == "6201Z"


def test_two_weak_sources_still_outweigh_one_strong_one():
    # L'inverse doit rester vrai : le poids départage, il ne donne pas un droit de veto.
    fused = fuse_candidates(
        {"supervised": ["62.01Z"], "embedding": ["10.71C"], "alternative": ["10.71C"]}
    )
    assert [c["code"] for c in fused][0] == "1071C"


def test_current_code_is_excluded_whatever_its_form():
    fused = fuse_candidates(
        {"supervised": ["10.71C", "62.01Z"], "alternative": ["1071C"]}, exclude="1071C"
    )
    assert [c["code"] for c in fused] == ["6201Z"]


def test_duplicates_inside_one_source_do_not_count_twice():
    # Un vote double déguisé en accord entre sources : le code doit rester noté comme
    # une seule citation, à son meilleur rang.
    fused = fuse_candidates({"embedding": ["10.71C", "1071C", "62.01Z"]})
    assert [c["code"] for c in fused] == ["1071C", "6201Z"]
    assert fused[0]["sources"] == ["embedding"]
    assert fused[0]["fusion_score"] == SOURCE_WEIGHTS["embedding"] / (RRF_K + 1)


def test_ties_are_broken_deterministically():
    # Deux sources de même poids, un candidat chacune, même rang : sans départage l'ordre
    # dépendrait du hasard des dictionnaires, et deux exécutions ne montreraient pas la
    # même liste.
    fused = fuse_candidates({"supervised": ["62.01Z"], "alternative": ["10.71C"]})
    assert [c["code"] for c in fused] == ["1071C", "6201Z"]


def test_top_k_truncates_after_fusion_not_before():
    # C est dernier partout mais cité deux fois : il doit survivre à la troncature, que
    # seul le score fusionné décide.
    fused = fuse_candidates(
        {"supervised": ["A", "B", "C"], "alternative": ["D", "E", "C"]}, top_k=2
    )
    assert [c["code"] for c in fused] == ["C", "A"]


def test_empty_and_missing_codes_are_dropped():
    fused = fuse_candidates({"alternative": [None, "", "10.71C"]})
    assert [c["code"] for c in fused] == ["1071C"]


def test_no_sources_yields_no_candidates():
    assert fuse_candidates({}) == []
