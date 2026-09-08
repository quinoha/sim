"""
Unit tests for QEDA M0 ConfigLoader (Stage 1 Config Subsystem).
"""
import pytest
from m0_config import load_config, ConfigValidationError


def test_load_valid_experiment_yaml():
    path = "configs/experiment.yaml"
    config = load_config(path)
    
    assert config.experiment.id == "bb-vs-surface-2026-08"
    assert len(config.codes) == 2
    assert len(config.decoders) == 2
    assert len(config.compatibility.deny) == 1
    assert config.defaults.rounds == 5
    assert config.defaults.shots == 200000


def test_unit_normalization():
    path = "configs/experiment.yaml"
    config = load_config(path)
    
    # Check warmup time normalized: 2 ms -> 2,000,000 ns
    assert config.protocol.timing.warmup.normalized_ns == 2_000_000.0
    
    # Check constraint normalized: 1.0 us -> 1000.0 ns
    lat_constraint = next(c for c in config.qualification.constraints if c.id == "C-LAT")
    assert lat_constraint.normalized_value == 1000.0


def test_invalid_code_params_raises_error():
    invalid_raw = {
        "config_version": 1,
        "experiment": {"id": "test-invalid"},
        "codes": [{"id": "bad_surface", "family": "surface", "params": {}}],  # Missing distance
        "decoders": [{"id": "dec1", "kind": "mwpm"}]
    }
    with pytest.raises(ConfigValidationError) as excinfo:
        load_config(invalid_raw)
    
    assert "V-SURFACE-NO-DISTANCE" in str(excinfo.value)
