#!/usr/bin/env python3
"""Research delivery diagnostics; preserve the public helper's scoring definitions.

Request measurement is adapted from scripts/stage2/performance.py, kept unchanged
as the organizer reference. SSE content deltas are NOT tokens. These observations
are client delivery times, not device generation times or official ITL metrics.
No process-global monkeypatches: simultaneous requests own independent timelines.
"""
from __future__ import annotations
import json
import threading
import time
import urllib.request


def delivery_summary(events: list[dict]) -> dict:
    gaps = [b["offset_ms"] - a["offset_ms"] for a, b in zip(events, events[1:])]
    return {"schema": 1, "unit": "nonempty SSE content delta",
            "scope": "client-observed delivery; not per-token latency or an official gate",
            "events": events, "delta_count": len(events),
            "max_inter_delta_gap_ms": max(gaps) if gaps else None}


_ENCODED_BODIES: dict = {}


def _encoded_body(row: dict, body: dict) -> bytes:
    """Encode each case body once; the same dict object is reused per case id
    across the 16C requests of a performance replay (0.2-0.55 MB media bodies,
    ~0.4-1.5 ms of GIL-bound encode per request otherwise)."""
    key = row["id"]
    cached = _ENCODED_BODIES.get(key)
    if cached is None or cached[0] is not body:
        cached = (body, json.dumps(body).encode())
        _ENCODED_BODIES[key] = cached
    return cached[1]


def measure_request(row: dict, body: dict, request_index: int, timeout: float,
                    start: threading.Event) -> dict:
    """Send one streaming request and time it. Never raises: a failure is a failed record.

    TTFT runs from just before the request is sent to the first non-empty content delta.
    """
    import performance
    accuracy = performance.accuracy
    stream_response = performance.stream_response
    delivery = []
    req = urllib.request.Request(
        accuracy.BASE_URL + "/v1/chat/completions",
        data=_encoded_body(row, body),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    start.wait()
    dispatched = time.monotonic()
    record: dict = dict(id=row["id"], suite=row["suite"], request_index=request_index, passed=False)
    text, first_text, usage, done = [], None, None, False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("Expected a streaming text/event-stream response")
            for data in stream_response(response):
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError(f"Server error: {event['error']}")
                if event.get("usage") is not None:
                    if usage is not None and usage != event["usage"]:
                        raise ValueError("Inconsistent streaming usage")
                    usage = event["usage"]
                choices = event["choices"]
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError("Expected at most one streaming choice")
                if choices:
                    content = choices[0]["delta"].get("content")
                    if content is not None and not isinstance(content, str):
                        raise ValueError("Non-text content delta")
                    if content:
                        observed = time.monotonic()
                        delivery.append({"offset_ms": (observed - dispatched) * 1000,
                                         "utf8_bytes": len(content.encode("utf-8"))})
                        if first_text is None:
                            first_text = observed
                        text.append(content)
                    if choices[0].get("finish_reason") is not None:
                        record["finish_reason"] = choices[0]["finish_reason"]
            completed = time.monotonic()
        if not done or first_text is None or usage is None:
            raise ValueError("Incomplete stream: require text, usage, and [DONE]")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise ValueError(f"Invalid usage.{key}")
        tokens = usage["completion_tokens"]
        if tokens == 0 or usage["total_tokens"] != usage["prompt_tokens"] + tokens:
            raise ValueError("Inconsistent token counts")
        if row["suite"] in ("256", "1024", "2048") and usage["prompt_tokens"] != int(row["suite"]):
            raise ValueError(f"Expected {row['suite']} prompt tokens; got {usage['prompt_tokens']}")
        ttft = (first_text - dispatched) * 1000
        latency = (completed - dispatched) * 1000
        record.update(
            passed=True, output="".join(text), usage=usage,
            ttft_ms=ttft, latency_ms=latency,
            estimated_tpot_ms=(latency - ttft) / (tokens - 1) if tokens > 1 else None,
            dispatch_offset_ms=(dispatched - start.epoch) * 1000,
            completion_offset_ms=(completed - start.epoch) * 1000,
        )
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"
    record["delivery_timing"] = delivery_summary(delivery)
    record["seconds"] = round(time.monotonic() - dispatched, 3)
    return record

