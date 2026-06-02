import pytest

from app.services.short_config import KB_MOTION_PRESETS
from app.workers.short_composer_agent import _prepare_zoompan_expr


@pytest.mark.unit
def test_all_presets_have_zoompan_key():
    for name, preset in KB_MOTION_PRESETS.items():
        assert "zoompan" in preset, f"{name} missing zoompan key"


@pytest.mark.unit
def test_all_presets_contain_duration_fps_placeholder():
    for name, preset in KB_MOTION_PRESETS.items():
        expr = preset["zoompan"]
        assert "duration*fps" in expr, f"{name} missing duration*fps"


@pytest.mark.unit
def test_prepare_zoompan_replaces_placeholder():
    expr = _prepare_zoompan_expr("zoom_in", 5.0, 30)
    assert "duration*fps" not in expr
    assert "150" in expr


@pytest.mark.unit
def test_prepare_zoompan_evaluates_different_durations():
    expr = _prepare_zoompan_expr("pan_left", 3.0, 30)
    assert "90" in expr

    expr = _prepare_zoompan_expr("zoom_out", 10.0, 30)
    assert "300" in expr


@pytest.mark.unit
def test_all_preset_names_valid():
    valid = {
        "pan_left",
        "pan_right",
        "zoom_in",
        "zoom_out",
        "static_zoom_in",
    }
    assert set(KB_MOTION_PRESETS.keys()) == valid


@pytest.mark.unit
def test_pan_left_expression_syntax():
    expr = KB_MOTION_PRESETS["pan_left"]["zoompan"]
    assert "z=" in expr
    assert "x=" in expr
    assert "y=" in expr
    assert "d=" in expr


@pytest.mark.unit
def test_zoom_in_expression_syntax():
    expr = KB_MOTION_PRESETS["zoom_in"]["zoompan"]
    assert "zoom" in expr
    assert "min(" in expr


@pytest.mark.unit
def test_static_zoom_in_is_simple():
    expr = KB_MOTION_PRESETS["static_zoom_in"]["zoompan"]
    assert "1.001" in expr
    assert "zoom+" not in expr  # no dynamic zoom increment
