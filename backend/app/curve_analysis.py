from collections.abc import Sequence


def _crossing_frequency(
    first: dict,
    second: dict,
    threshold_db: float,
) -> float:
    first_frequency = float(first["frequency"])
    second_frequency = float(second["frequency"])
    first_value = float(first["s11"])
    second_value = float(second["s11"])
    delta = second_value - first_value
    if delta == 0:
        return first_frequency
    ratio = (threshold_db - first_value) / delta
    return first_frequency + ratio * (second_frequency - first_frequency)


def calculate_threshold_bands(
    curve: Sequence[dict],
    threshold_db: float = -10.0,
) -> list[dict]:
    if not curve:
        return []

    ordered = sorted(curve, key=lambda point: float(point["frequency"]))
    bands: list[dict] = []
    inside = float(ordered[0]["s11"]) <= threshold_db
    start = float(ordered[0]["frequency"]) if inside else None

    for previous, current in zip(ordered, ordered[1:]):
        current_inside = float(current["s11"]) <= threshold_db
        if not inside and current_inside:
            start = _crossing_frequency(previous, current, threshold_db)
        elif inside and not current_inside and start is not None:
            end = _crossing_frequency(previous, current, threshold_db)
            bands.append(_band(start, end))
            start = None
        inside = current_inside

    if inside and start is not None:
        bands.append(_band(start, float(ordered[-1]["frequency"])))

    return bands


def _band(start: float, end: float) -> dict:
    return {
        "start_frequency": float(start),
        "end_frequency": float(end),
        "bandwidth": float(max(0.0, end - start)),
        "center_frequency": float((start + end) / 2),
    }
