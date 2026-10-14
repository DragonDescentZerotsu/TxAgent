import numpy as np

from data.processing.evidence_library.shared.v1.clustered_auxiliary_mapping import cluster_values
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.data_processing.build_direct_endpoint_mapping import (
    CLUSTER_TARGET_SIZE,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    MAX_VALUES_PER_LOCAL_REQUEST,
    extraction_spec,
)


def test_direct_endpoint_inventory_is_the_only_gpt_namespace_and_model_is_pinned():
    spec = extraction_spec()
    assert (spec.source_id, spec.input_column, spec.output_field) == (
        "direct_bbb",
        "quant_metric",
        "canonical_endpoint",
    )
    assert DEFAULT_MODEL == "gpt-5.4-mini"
    assert DEFAULT_REASONING_EFFORT == "medium"
    assert CLUSTER_TARGET_SIZE == MAX_VALUES_PER_LOCAL_REQUEST == 500


def test_embedding_clusters_have_a_true_500_item_hard_bound():
    values = [f"endpoint {index}" for index in range(1201)]
    # Identical embeddings exercise the deterministic fallback used when
    # KMeans cannot meaningfully divide one oversized neighborhood.
    embeddings = np.zeros((len(values), 4), dtype=np.float32)
    clusters = cluster_values(values, embeddings, target_size=500, max_size=500)
    assert max(len(cluster.values) for cluster in clusters) <= 500
    assert {value for cluster in clusters for value in cluster.values} == set(values)
