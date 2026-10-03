"""End-to-end tests for the station waveform acquisition continuity check."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.conftest import make_waveform_b64, synthetic_event_payload

BASE_DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
SR = 100.0
DURATION = 120.0


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _waveform(code, start, *, duration=DURATION, arrivals=(), noise=0.02):
    b64 = make_waveform_b64(
        code, SR, _iso(start), list(arrivals), duration=duration, noise=noise
    )
    return {
        "station": code,
        "latitude": 35.0,
        "longitude": -118.0,
        "elevation_m": 0.0,
        "sampling_rate": SR,
        "start_time": _iso(start),
        "data_b64": b64,
    }


def _process(client, waveforms):
    """Ingest one batch; events are irrelevant here, only stored waveforms."""
    body = {
        "waveforms": waveforms,
        "manual_picks": [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 4},
    }
    resp = client.post("/v1/process", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _continuity(client, station, start, end):
    return client.get(
        "/v1/waveform-continuity",
        params={"station": station, "start_time": start, "end_time": end},
    )


def _t(**kwargs) -> datetime:
    return BASE_DT + timedelta(**kwargs)


# --------------------------------------------------------------------------- #
# validation: window and station
# --------------------------------------------------------------------------- #
def test_unknown_station_returns_404(client):
    resp = _continuity(client, "NOPE", "2026-09-22T00:00:00Z", "2026-09-22T01:00:00Z")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "station_not_found"


def test_empty_or_inverted_window_rejected(client):
    _process(client, [_waveform("WC01", BASE_DT)])
    for start, end in [
        ("2026-09-22T01:00:00Z", "2026-09-22T01:00:00Z"),  # empty
        ("2026-09-22T02:00:00Z", "2026-09-22T01:00:00Z"),  # inverted
    ]:
        resp = _continuity(client, "WC01", start, end)
        assert resp.status_code == 422, (start, end, resp.text)
        assert resp.json()["detail"]["code"] == "invalid_time_window"


def test_missing_or_malformed_parameters_rejected(client):
    resp = client.get(
        "/v1/waveform-continuity",
        params={"station": "WC01", "start_time": "2026-09-22T00:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "validation_error"

    resp = client.get(
        "/v1/waveform-continuity",
        params={"start_time": "2026-09-22T00:00:00Z",
                "end_time": "2026-09-22T01:00:00Z"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "validation_error"

    resp = _continuity(client, "WC01", "not-a-time", "2026-09-22T01:00:00Z")
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "validation_error"


# --------------------------------------------------------------------------- #
# coverage and gaps
# --------------------------------------------------------------------------- #
def test_window_inside_single_segment_is_fully_covered(client):
    _process(client, [_waveform("WC01", BASE_DT)])
    resp = _continuity(
        client, "WC01", "2026-09-22T00:00:30Z", "2026-09-22T00:01:30Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert rep["station"] == "WC01"
    assert rep["window"] == {
        "start": "2026-09-22T00:00:30.000Z",
        "end": "2026-09-22T00:01:30.000Z",
        "duration_s": 60.0,
    }
    assert rep["waveform_count"] == 1
    assert rep["distinct_segment_count"] == 1
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == "2026-09-22T00:00:30.000Z"
    assert cov["end"] == "2026-09-22T00:01:30.000Z"
    assert cov["duration_s"] == 60.0
    assert len(cov["ingestion_ids"]) == 1
    assert rep["gaps"] == []
    assert rep["overlaps"] == []
    assert rep["summary"] == {
        "covered_s": 60.0,
        "gap_s": 0.0,
        "overlap_s": 0.0,
        "coverage_ratio": 1.0,
    }


def test_segment_is_clipped_to_window_with_leading_and_trailing_gaps(client):
    _process(client, [_waveform("WC01", BASE_DT)])
    resp = _continuity(
        client, "WC01", "2026-09-21T23:00:00Z", "2026-09-22T03:00:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert [(c["start"], c["end"], c["duration_s"]) for c in rep["coverage"]] == [
        ("2026-09-22T00:00:00.000Z", "2026-09-22T00:02:00.000Z", 120.0)
    ]
    assert [(g["kind"], g["duration_s"]) for g in rep["gaps"]] == [
        ("leading", 3600.0),
        ("trailing", 10680.0),
    ]
    assert rep["gaps"][0]["start"] == "2026-09-21T23:00:00.000Z"
    assert rep["gaps"][0]["end"] == "2026-09-22T00:00:00.000Z"
    assert rep["gaps"][1]["start"] == "2026-09-22T00:02:00.000Z"
    assert rep["gaps"][1]["end"] == "2026-09-22T03:00:00.000Z"
    assert rep["overlaps"] == []
    assert rep["summary"]["covered_s"] == 120.0
    assert rep["summary"]["gap_s"] == 14280.0
    assert rep["summary"]["coverage_ratio"] == round(120.0 / 14400.0, 6)


def test_internal_gap_between_segments(client):
    _process(client, [_waveform("WC01", BASE_DT), _waveform("WC02", BASE_DT)])
    _process(client, [_waveform("WC01", _t(seconds=180)), _waveform("WC03", BASE_DT)])
    resp = _continuity(
        client, "WC01", "2026-09-22T00:00:00Z", "2026-09-22T00:06:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert [(c["start"], c["end"]) for c in rep["coverage"]] == [
        ("2026-09-22T00:00:00.000Z", "2026-09-22T00:02:00.000Z"),
        ("2026-09-22T00:03:00.000Z", "2026-09-22T00:05:00.000Z"),
    ]
    assert [(g["kind"], g["start"], g["end"], g["duration_s"]) for g in rep["gaps"]] == [
        ("internal", "2026-09-22T00:02:00.000Z", "2026-09-22T00:03:00.000Z", 60.0),
        ("trailing", "2026-09-22T00:05:00.000Z", "2026-09-22T00:06:00.000Z", 60.0),
    ]
    assert rep["overlaps"] == []
    # coverage segments carry their own source ingestion ids
    ids = [tuple(c["ingestion_ids"]) for c in rep["coverage"]]
    assert len(ids[0]) == 1 and len(ids[1]) == 1 and ids[0] != ids[1]


def test_touching_segments_merge_into_one_coverage(client):
    _process(client, [_waveform("WC01", BASE_DT), _waveform("WC02", BASE_DT)])
    _process(client, [_waveform("WC01", _t(seconds=120)), _waveform("WC03", BASE_DT)])
    resp = _continuity(
        client, "WC01", "2026-09-22T00:00:00Z", "2026-09-22T00:04:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["start"] == "2026-09-22T00:00:00.000Z"
    assert cov["end"] == "2026-09-22T00:04:00.000Z"
    assert cov["duration_s"] == 240.0
    assert len(cov["ingestion_ids"]) == 2
    assert rep["gaps"] == []
    assert rep["overlaps"] == []


def test_window_exactly_matching_segment_bounds(client):
    _process(client, [_waveform("WC01", BASE_DT)])
    resp = _continuity(
        client, "WC01", "2026-09-22T00:00:00Z", "2026-09-22T00:02:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()
    assert len(rep["coverage"]) == 1
    assert rep["coverage"][0]["duration_s"] == 120.0
    assert rep["gaps"] == []
    assert rep["summary"]["coverage_ratio"] == 1.0


def test_timezone_offsets_are_normalized_to_utc(client):
    _process(client, [_waveform("WC01", BASE_DT)])
    resp = _continuity(
        client, "WC01", "2026-09-22T08:00:00+08:00", "2026-09-22T08:02:00+08:00"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()
    assert rep["window"]["start"] == "2026-09-22T00:00:00.000Z"
    assert rep["window"]["end"] == "2026-09-22T00:02:00.000Z"
    assert rep["summary"]["coverage_ratio"] == 1.0


def test_known_station_without_records_in_window_returns_full_gap(client):
    _process(client, [_waveform("WC01", BASE_DT)])
    resp = _continuity(
        client, "WC01", "2026-09-23T00:00:00Z", "2026-09-23T01:00:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert rep["coverage"] == []
    assert rep["overlaps"] == []
    assert rep["waveform_count"] == 0
    assert rep["distinct_segment_count"] == 0
    assert rep["gaps"] == [
        {
            "kind": "full",
            "start": "2026-09-23T00:00:00.000Z",
            "end": "2026-09-23T01:00:00.000Z",
            "duration_s": 3600.0,
        }
    ]
    assert rep["summary"]["covered_s"] == 0.0
    assert rep["summary"]["gap_s"] == 3600.0
    assert rep["summary"]["coverage_ratio"] == 0.0


# --------------------------------------------------------------------------- #
# deduplication
# --------------------------------------------------------------------------- #
def test_duplicate_waveform_counts_once_but_keeps_all_ingestion_ids(client):
    wf = _waveform("WC01", BASE_DT)
    first = _process(client, [wf, _waveform("WC02", BASE_DT)])
    second = _process(client, [wf, _waveform("WC03", BASE_DT)])
    assert first["ingestion_id"] != second["ingestion_id"]

    resp = _continuity(
        client, "WC01", "2026-09-22T00:00:00Z", "2026-09-22T00:02:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert rep["waveform_count"] == 2
    assert rep["distinct_segment_count"] == 1
    assert len(rep["coverage"]) == 1
    cov = rep["coverage"][0]
    assert cov["duration_s"] == 120.0
    assert cov["ingestion_ids"] == sorted(
        [first["ingestion_id"], second["ingestion_id"]]
    )
    assert rep["overlaps"] == []
    assert rep["summary"]["overlap_s"] == 0.0


# --------------------------------------------------------------------------- #
# overlaps
# --------------------------------------------------------------------------- #
def test_partial_overlap_is_reported_with_source_ingestion_ids(client):
    first = _process(client, [_waveform("WC01", BASE_DT), _waveform("WC02", BASE_DT)])
    second = _process(
        client, [_waveform("WC01", _t(seconds=60)), _waveform("WC03", BASE_DT)]
    )
    resp = _continuity(
        client, "WC01", "2026-09-22T00:00:00Z", "2026-09-22T00:03:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    # the union is one merged covered interval; the overlap sits inside it
    assert [(c["start"], c["end"], c["duration_s"]) for c in rep["coverage"]] == [
        ("2026-09-22T00:00:00.000Z", "2026-09-22T00:03:00.000Z", 180.0)
    ]
    assert rep["gaps"] == []
    assert len(rep["overlaps"]) == 1
    ovl = rep["overlaps"][0]
    assert ovl["start"] == "2026-09-22T00:01:00.000Z"
    assert ovl["end"] == "2026-09-22T00:02:00.000Z"
    assert ovl["duration_s"] == 60.0
    assert ovl["ingestion_ids"] == sorted(
        [first["ingestion_id"], second["ingestion_id"]]
    )
    assert rep["summary"]["covered_s"] == 180.0
    assert rep["summary"]["overlap_s"] == 60.0


def test_same_window_different_content_is_full_overlap_not_duplicate(client):
    pulsed = _waveform("WC01", BASE_DT, arrivals=[(30.0, 4.0)])
    quiet = _waveform("WC01", BASE_DT)  # same start/sr/npts, different samples
    _process(client, [pulsed, _waveform("WC02", BASE_DT)])
    _process(client, [quiet, _waveform("WC03", BASE_DT)])

    resp = _continuity(
        client, "WC01", "2026-09-22T00:00:00Z", "2026-09-22T00:02:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert rep["waveform_count"] == 2
    assert rep["distinct_segment_count"] == 2  # different data hash: not duplicates
    assert len(rep["coverage"]) == 1
    assert rep["coverage"][0]["duration_s"] == 120.0
    assert len(rep["overlaps"]) == 1
    assert rep["overlaps"][0]["duration_s"] == 120.0
    assert len(rep["overlaps"][0]["ingestion_ids"]) == 2


def test_disjoint_overlap_regions_are_reported_separately(client):
    _process(client, [_waveform("WC01", BASE_DT), _waveform("WC02", BASE_DT)])
    _process(client, [_waveform("WC01", _t(seconds=30), duration=30.0),
                      _waveform("WC03", BASE_DT)])
    _process(client, [_waveform("WC01", _t(seconds=80), duration=30.0),
                      _waveform("WC04", BASE_DT)])
    resp = _continuity(
        client, "WC01", "2026-09-22T00:00:00Z", "2026-09-22T00:02:00Z"
    )
    assert resp.status_code == 200, resp.text
    rep = resp.json()

    assert [(o["start"], o["end"], o["duration_s"]) for o in rep["overlaps"]] == [
        ("2026-09-22T00:00:30.000Z", "2026-09-22T00:01:00.000Z", 30.0),
        ("2026-09-22T00:01:20.000Z", "2026-09-22T00:01:50.000Z", 30.0),
    ]
    assert len(rep["coverage"]) == 1
    assert rep["coverage"][0]["duration_s"] == 120.0
    assert rep["summary"]["overlap_s"] == 60.0


# --------------------------------------------------------------------------- #
# read-only: independent of runs and catalog state
# --------------------------------------------------------------------------- #
def test_result_is_independent_of_run_and_catalog_state(client):
    body, _truth = synthetic_event_payload(prefix="WC", n_stations=8)
    proc = client.post("/v1/process", json=body)
    assert proc.status_code == 200, proc.text
    ingestion_id = proc.json()["ingestion_id"]

    params = {
        "station": "WC00",
        "start_time": "2026-09-21T23:00:00Z",
        "end_time": "2026-09-22T03:00:00Z",
    }
    before = client.get("/v1/waveform-continuity", params=params)
    assert before.status_code == 200, before.text

    # New run on the same ingestion (different model version), a catalog
    # compilation with a published entry and a pick-gap assessment.
    resp = client.post(
        "/v1/models",
        json={
            "name": "default",
            "version": "2.0",
            "layers": [
                {"thickness_km": 15.0, "vp_kms": 5.2, "vs_kms": 3.0},
                {"thickness_km": 20.0, "vp_kms": 5.9, "vs_kms": 3.4},
                {"thickness_km": None, "vp_kms": 7.2, "vs_kms": 4.2},
            ],
        },
    )
    assert resp.status_code == 201, resp.text
    resp = client.post(
        "/v1/runs",
        json={
            "ingestion_id": ingestion_id,
            "model": {"name": "default", "version": "2.0"},
            "params": {"max_residual_s": 1.0, "min_stations": 4},
        },
    )
    assert resp.status_code == 201, resp.text
    resp = client.post(
        "/v1/catalog/compilations",
        json={
            "start_time": "2026-09-21T00:00:00Z",
            "end_time": "2026-09-23T00:00:00Z",
            "min_quality": "D",
            "min_shared_stations": 0,
        },
    )
    assert resp.status_code == 201, resp.text
    compilation_id = resp.json()["compilation_id"]
    for entry in resp.json()["entries"]:
        pub = client.post(
            f"/v1/catalog/entries/{entry['entry_id']}/publish",
            json={"expected_revision": 1},
        )
        assert pub.status_code == 200, pub.text
    resp = client.post(
        "/v1/pick-gap-assessments",
        json={
            "compilation_id": compilation_id,
            "arrival_tolerance_s": 2.0,
            "min_observable_events": 1,
        },
    )
    assert resp.status_code == 201, resp.text
    assessment = resp.json()

    after = client.get("/v1/waveform-continuity", params=params)
    assert after.status_code == 200, after.text
    assert after.json() == before.json()

    # The continuity check is read-only: the pick-gap report is unchanged.
    resp = client.get(f"/v1/pick-gap-assessments/{assessment['assessment_id']}")
    assert resp.status_code == 200, resp.text
    fetched = resp.json()
    assert fetched["created"] is False
    assert {k: v for k, v in fetched.items() if k != "created"} == {
        k: v for k, v in assessment.items() if k != "created"
    }


def test_repeated_calls_are_deterministic(client):
    _process(client, [_waveform("WC01", BASE_DT), _waveform("WC02", BASE_DT)])
    _process(client, [_waveform("WC01", _t(seconds=60)), _waveform("WC03", BASE_DT)])
    params = {
        "station": "WC01",
        "start_time": "2026-09-21T23:30:00Z",
        "end_time": "2026-09-22T00:30:00Z",
    }
    first = client.get("/v1/waveform-continuity", params=params)
    second = client.get("/v1/waveform-continuity", params=params)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
