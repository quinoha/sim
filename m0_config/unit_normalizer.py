"""
Unit Normalizer for QEDA M0.
Standardizes time units (ns, us, ms, s) and scalar metrics to canonical representations.
"""
from typing import Dict, Any, Optional, Tuple

TIME_CONVERSIONS_TO_NS = {
    "ns": 1.0,
    "us": 1000.0,
    "µs": 1000.0,
    "ms": 1000000.0,
    "s": 1000000000.0,
}


def normalize_time_to_ns(value: float, unit: Optional[str]) -> Tuple[float, str]:
    """Convert time value with unit to nanoseconds."""
    if not unit:
        return value, "ns"
    
    clean_unit = unit.strip().lower()
    if clean_unit not in TIME_CONVERSIONS_TO_NS:
        raise ValueError(f"Unknown time unit: '{unit}'. Supported units: {list(TIME_CONVERSIONS_TO_NS.keys())}")
    
    factor = TIME_CONVERSIONS_TO_NS[clean_unit]
    return value * factor, "ns"


def normalize_constraint(metric: str, value: float, unit: Optional[str]) -> Tuple[float, Optional[str]]:
    """Normalize constraint values based on metric type."""
    if not unit:
        return value, None
    
    clean_unit = unit.strip().lower()
    if "latency" in metric.lower() or "time" in metric.lower() or clean_unit in TIME_CONVERSIONS_TO_NS:
        norm_val, _ = normalize_time_to_ns(value, unit)
        return norm_val, "ns"
    elif clean_unit in ["%", "percent"]:
        return value / 100.0, "ratio"
    
    return value, unit
