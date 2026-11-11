import time_inference as ti


def _snapshot(rss, private, peak):
    return {
        "proc_rss_mb": rss,
        "proc_private_mb": private,
        "os_peak_mb": peak,
        "sys_used_mb": 8000.0,
        "sys_percent": 50.0,
    }


def test_memory_snapshot_has_the_measured_fields():
    snapshot = ti.get_memory_snapshot()
    assert set(snapshot) == {"proc_rss_mb", "proc_private_mb", "os_peak_mb", "sys_used_mb", "sys_percent"}
    assert snapshot["proc_rss_mb"] > 0


def test_diff_reports_before_after_and_delta_with_prefix():
    diff = ti.diff_snapshots(_snapshot(100.0, 50.0, 120.0), _snapshot(350.5, None, 400.0), prefix="init_")
    assert diff["init_proc_rss_mb_before"] == 100.0 and diff["init_proc_rss_mb_after"] == 350.5
    assert diff["init_proc_rss_mb_delta"] == 250.5
    assert diff["init_proc_private_mb_delta"] is None  # unavailable on this platform: no made-up number


def test_invoke_measures_time_and_peak_memory(make_engine):
    engine, _, _ = make_engine()
    row = ti.invoke_with_peak_rss(engine, "q", sample_interval=0.001)
    assert row["inference_time"] > 0
    assert row["proc_rss_mb_peak_during_inference"] >= row["proc_rss_mb_before"]
    assert row["peak_source"] in {"os", "sampled"}
