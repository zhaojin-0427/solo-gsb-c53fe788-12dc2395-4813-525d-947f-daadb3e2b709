"""Station waveform acquisition continuity check.

A read-only, stateless audit over the ingested waveform metadata.  The client
submits a station code and a UTC half-open window ``[start_time, end_time)``;
the service answers "when did this station actually record data inside that
window?" directly from the ``waveforms`` table:

* Every ``Waveform`` row of the station -- across **all** ingestion batches --
  is converted into a half-open coverage interval
  ``[start_time, start_time + npts / sampling_rate)`` and clipped to the
  request window.
* Duplicate waveforms (identical station, start time, sampling rate, sample
  count and data hash) are counted **once**, but every ingestion id they were
  seen in is retained.
* A sweep over the interval endpoints classifies every elementary run:
  coverage count >= 1 is *covered*, count >= 2 is *overlap*.  Touching runs
  of the same kind are merged.  The complement inside the window is reported
  as leading / internal / trailing gaps; a known station with no record
  anywhere in the window yields one full-window gap.
* Overlap only means the acquisition ranges coincide; it does **not** infer
  anything about whether the sample contents agree.

The check reads only waveform metadata: results do not depend on runs or
catalog state, and ingestion / pick-gap reports are unaffected.  Invalid or
empty windows are rejected (422); a station that was never ingested returns
404.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from ..models import Waveform
from .pipeline import ApiError, from_epoch, iso_z, to_epoch, to_naive_utc

GAP_LEADING = "leading"
GAP_INTERNAL = "internal"
GAP_TRAILING = "trailing"
GAP_FULL = "full"


def check_continuity(
    db: Session, station: str, start_time: datetime, end_time: datetime
) -> dict:
    """Build the continuity report for one station over a half-open window."""
    win_start = to_naive_utc(start_time)
    win_end = to_naive_utc(end_time)
    if not win_end > win_start:
        raise ApiError(
            422,
            "invalid_time_window",
            "end_time must be after start_time "
            "(the window is half-open [start_time, end_time))",
        )

    rows = (
        db.query(Waveform)
        .filter(Waveform.station == station)
        .order_by(Waveform.start_time, Waveform.id)
        .all()
    )
    if not rows:
        raise ApiError(
            404,
            "station_not_found",
            f"station {station} has no ingested waveforms",
        )

    win_start_epoch = to_epoch(win_start)
    win_end_epoch = to_epoch(win_end)
    segments, waveform_count = _clipped_segments(
        rows, win_start_epoch, win_end_epoch
    )
    runs = _elementary_runs(segments, win_start_epoch, win_end_epoch)

    coverage = _merge_runs(runs, segments, lambda count: count >= 1)
    overlaps = _merge_runs(runs, segments, lambda count: count >= 2)
    gaps = _gap_runs(runs, win_start_epoch, win_end_epoch, any_coverage=bool(coverage))

    window_s = win_end_epoch - win_start_epoch
    covered_s = sum(end - start for start, end, _ids in coverage)
    overlap_s = sum(end - start for start, end, _ids in overlaps)
    return {
        "station": station,
        "window": {
            "start": iso_z(win_start),
            "end": iso_z(win_end),
            "duration_s": round(window_s, 6),
        },
        "waveform_count": waveform_count,
        "distinct_segment_count": len(segments),
        "coverage": [
            _interval_payload(start, end, ingestion_ids=ids)
            for start, end, ids in coverage
        ],
        "gaps": [
            {"kind": kind, **_interval_payload(start, end)}
            for kind, start, end in gaps
        ],
        "overlaps": [
            _interval_payload(start, end, ingestion_ids=ids)
            for start, end, ids in overlaps
        ],
        "summary": {
            "covered_s": round(covered_s, 6),
            "gap_s": round(window_s - covered_s, 6),
            "overlap_s": round(overlap_s, 6),
            "coverage_ratio": round(covered_s / window_s, 6),
        },
    }


# --------------------------------------------------------------------------- #
# interval construction
# --------------------------------------------------------------------------- #
def _clipped_segments(
    rows: list[Waveform], win_start: float, win_end: float
) -> tuple[list[dict], int]:
    """Half-open ``[start, start + npts/sr)`` clipped to the window, deduplicated.

    Returns ``(segments, waveform_count)`` where ``waveform_count`` is the
    number of waveform rows intersecting the window *before* deduplication.
    Duplicate rows (same start/sampling rate/npts/data hash) collapse into one
    segment that accumulates every source ingestion id.
    """
    by_key: dict[tuple, dict] = {}
    waveform_count = 0
    for row in rows:
        seg_start = to_epoch(row.start_time)
        seg_end = seg_start + row.npts / float(row.sampling_rate)
        lo = max(seg_start, win_start)
        hi = min(seg_end, win_end)
        if hi <= lo:  # no overlap with the (half-open) request window
            continue
        waveform_count += 1
        key = (
            row.start_time,
            float(row.sampling_rate),
            int(row.npts),
            row.data_sha256,
        )
        seg = by_key.get(key)
        if seg is None:
            by_key[key] = {
                "start": lo,
                "end": hi,
                "ingestion_ids": {row.ingestion_id},
            }
        else:
            seg["ingestion_ids"].add(row.ingestion_id)
    segments = sorted(by_key.values(), key=lambda s: (s["start"], s["end"]))
    return segments, waveform_count


def _elementary_runs(
    segments: list[dict], win_start: float, win_end: float
) -> list[tuple[float, float, frozenset]]:
    """Split ``[win_start, win_end)`` into runs with a constant active segment set."""
    events: list[tuple[float, int]] = []
    for idx, seg in enumerate(segments):
        events.append((seg["start"], idx))
        events.append((seg["end"], idx))
    events.sort(key=lambda e: e[0])

    runs: list[tuple[float, float, frozenset]] = []
    active: set[int] = set()
    prev = win_start
    i = 0
    while i < len(events):
        t = events[i][0]
        if t > prev:
            runs.append((prev, t, frozenset(active)))
        while i < len(events) and events[i][0] == t:
            idx = events[i][1]
            if idx in active:
                active.discard(idx)
            else:
                active.add(idx)
            i += 1
        prev = t
    if prev < win_end:
        runs.append((prev, win_end, frozenset(active)))
    return runs


def _merge_runs(
    runs: list[tuple[float, float, frozenset]],
    segments: list[dict],
    predicate,
) -> list[tuple[float, float, set]]:
    """Merge touching runs whose coverage count satisfies ``predicate``.

    Elementary runs are contiguous by construction, so consecutive matching
    runs touch exactly and coalesce into one interval; the source ingestion
    ids of every contributing segment are unioned.
    """
    merged: list[list] = []  # [start, end, ingestion_ids]
    for start, end, active in runs:
        if not predicate(len(active)):
            continue
        ids: set[str] = set()
        for idx in active:
            ids.update(segments[idx]["ingestion_ids"])
        if merged and merged[-1][1] == start:
            merged[-1][1] = end
            merged[-1][2].update(ids)
        else:
            merged.append([start, end, ids])
    return [(start, end, ids) for start, end, ids in merged]


def _gap_runs(
    runs: list[tuple[float, float, frozenset]],
    win_start: float,
    win_end: float,
    any_coverage: bool,
) -> list[tuple[str, float, float]]:
    """Uncovered runs, classified as leading / internal / trailing / full."""
    gaps: list[tuple[str, float, float]] = []
    for start, end, active in runs:
        if len(active) > 0:
            continue
        if not any_coverage:
            kind = GAP_FULL
        elif start == win_start:
            kind = GAP_LEADING
        elif end == win_end:
            kind = GAP_TRAILING
        else:
            kind = GAP_INTERNAL
        gaps.append((kind, start, end))
    return gaps


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _interval_payload(
    start: float, end: float, ingestion_ids: set | None = None
) -> dict:
    payload = {
        "start": iso_z(from_epoch(start)),
        "end": iso_z(from_epoch(end)),
        "duration_s": round(end - start, 6),
    }
    if ingestion_ids is not None:
        payload["ingestion_ids"] = sorted(ingestion_ids)
    return payload
