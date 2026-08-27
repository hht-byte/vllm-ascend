from benchmarks.prometheus_delta import CACHE_COUNTERS, counter_deltas


def test_labeled_counter_samples_are_summed_by_metric_family() -> None:
    before = """
# HELP vllm:mm_cache_queries_total Encoder cache queries.
# TYPE vllm:mm_cache_queries_total counter
vllm:mm_cache_queries_total{model_name="a",worker="0"} 10
vllm:mm_cache_queries_total{model_name="a",worker="1"} 4
vllm:mm_cache_hits_total{model_name="a",worker="0"} 3
vllm:mm_cache_hits_total{model_name="a",worker="1"} 2
vllm:prefix_cache_queries_total{model_name="a"} 20
vllm:prefix_cache_hits_total{model_name="a"} 12
"""
    after = """
vllm:mm_cache_queries_total{model_name="a",worker="0"} 15
vllm:mm_cache_queries_total{model_name="a",worker="1"} 8
vllm:mm_cache_hits_total{model_name="a",worker="0"} 5
vllm:mm_cache_hits_total{model_name="a",worker="1"} 5
vllm:prefix_cache_queries_total{model_name="a"} 29
vllm:prefix_cache_hits_total{model_name="a"} 18
"""

    snapshot = counter_deltas(before, after)

    assert snapshot.values == {
        "vllm:mm_cache_queries": 9.0,
        "vllm:mm_cache_hits": 5.0,
        "vllm:prefix_cache_queries": 9.0,
        "vllm:prefix_cache_hits": 6.0,
    }
    assert snapshot.warnings == ()
    assert set(snapshot.values) == set(CACHE_COUNTERS)


def test_missing_metric_is_null_with_explicit_warning() -> None:
    before = "vllm:mm_cache_queries_total 2\n"
    after = "vllm:mm_cache_queries_total 5\n"

    snapshot = counter_deltas(before, after)

    assert snapshot.values["vllm:mm_cache_queries"] == 3.0
    assert snapshot.values["vllm:mm_cache_hits"] is None
    assert any(
        "vllm:mm_cache_hits" in warning and "missing" in warning
        for warning in snapshot.warnings
    )


def test_counter_reset_is_null_instead_of_negative_or_zero() -> None:
    before = "\n".join(f"{name}_total 10" for name in CACHE_COUNTERS)
    after = "\n".join(f"{name}_total 3" for name in CACHE_COUNTERS)

    snapshot = counter_deltas(before, after)

    assert all(value is None for value in snapshot.values.values())
    assert len(snapshot.warnings) == len(CACHE_COUNTERS)
    assert all("reset" in warning for warning in snapshot.warnings)


def test_labeled_reset_is_not_masked_by_an_increased_family_total() -> None:
    before = """
vllm:mm_cache_queries_total{worker="0"} 10
vllm:mm_cache_queries_total{worker="1"} 0
"""
    after = """
vllm:mm_cache_queries_total{worker="0"} 0
vllm:mm_cache_queries_total{worker="1"} 20
"""

    snapshot = counter_deltas(before, after, ("vllm:mm_cache_queries",))

    assert snapshot.values == {"vllm:mm_cache_queries": None}
    assert len(snapshot.warnings) == 1
    assert "reset" in snapshot.warnings[0]
    assert 'worker="0"' in snapshot.warnings[0]


def test_added_or_missing_labeled_series_is_null_instead_of_a_partial_delta() -> None:
    before = """
vllm:mm_cache_queries_total{worker="0"} 10
vllm:mm_cache_queries_total{worker="1"} 4
"""
    after = """
vllm:mm_cache_queries_total{worker="0"} 14
vllm:mm_cache_queries_total{worker="2"} 9
"""

    snapshot = counter_deltas(before, after, ("vllm:mm_cache_queries",))

    assert snapshot.values == {"vllm:mm_cache_queries": None}
    assert len(snapshot.warnings) == 1
    assert "series changed" in snapshot.warnings[0]
    assert 'worker="1"' in snapshot.warnings[0]
    assert 'worker="2"' in snapshot.warnings[0]


def test_non_counter_suffix_samples_do_not_pollute_family_sum() -> None:
    before = """
vllm:mm_cache_queries_total 1
vllm:mm_cache_queries_created 1000
"""
    after = """
vllm:mm_cache_queries_total 3
vllm:mm_cache_queries_created 2000
"""

    snapshot = counter_deltas(before, after, ("vllm:mm_cache_queries",))

    assert snapshot.values == {"vllm:mm_cache_queries": 2.0}
    assert snapshot.warnings == ()
