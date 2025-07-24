from typing import Optional

from clax.parameters import EmbeddingParameterConfig


def default_attraction_config(
    query_doc_pairs: Optional[int],
) -> EmbeddingParameterConfig:
    assert query_doc_pairs is not None, (
        "Either provide the number of 'query_doc_pairs' "
        "to your model or pass a custom 'attraction_config'."
    )
    return EmbeddingParameterConfig(
        use_feature="query_doc_ids",
        parameters=query_doc_pairs,
    )


def default_satisfaction_config(
    query_doc_pairs: Optional[int],
) -> EmbeddingParameterConfig:
    assert query_doc_pairs is not None, (
        "Either provide the number of 'query_doc_pairs' "
        "to your model or pass a custom 'satisfaction_config'."
    )
    return EmbeddingParameterConfig(
        use_feature="query_doc_ids", parameters=query_doc_pairs
    )


def default_examination_config(positions: Optional[int]) -> EmbeddingParameterConfig:
    assert positions is not None, (
        "Either provide the number of 'positions' "
        "to your model or pass a custom 'examination_config'."
    )
    return EmbeddingParameterConfig(
        use_feature="positions",
        parameters=positions,
    )


def default_continuation_config(positions: Optional[int]) -> EmbeddingParameterConfig:
    assert positions is not None, (
        "Either provide the number of 'positions' "
        "to your model or pass a custom 'continuation_config'."
    )
    return EmbeddingParameterConfig(
        use_feature="positions",
        parameters=positions,
    )


def default_ubm_examination_config(positions: int) -> EmbeddingParameterConfig:
    return EmbeddingParameterConfig(
        use_feature="examination_idx",
        parameters=(positions + 1) ** 2,
    )
