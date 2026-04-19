"""Tests for config loading and run-ID management."""
import pytest
from pathlib import Path
from jr_walker.config import load_config, read_run_id


def test_load_config_defaults(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text("[warehouse]\nwidth = 60\nheight = 40\n")
    cfg = load_config(str(toml))
    assert cfg.width == 60
    assert cfg.height == 40
    assert cfg.stride == 1


def test_load_config_full(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        "[warehouse]\nwidth=60\nheight=40\n"
        "[paths]\ninput_path='x.txt'\noutput_dir='out'\nmedia_dir='med'\nrun_number_path='rn.txt'\n"
        "[limits]\nmax_ticks=8000\nmax_runtime_minutes=30.0\nstride=5\n"
        "[nests]\nx_coords=[10,20]\n"
        "[pathfinding]\nstrict_no_swap=true\nmax_idle_ticks=12\nstall_limit=12\n"
        "[progress]\norder_interval=50\ntick_interval=500\n"
    )
    cfg = load_config(str(toml))
    assert cfg.max_ticks == 8000
    assert cfg.stride == 5
    assert cfg.nest_x_coords == [10, 20]
    assert cfg.strict_no_swap is True
    assert cfg.stall_limit == 12


def test_read_run_id_creates_file(tmp_path):
    p = tmp_path / "run_number.txt"
    run_id = read_run_id(str(p))
    assert run_id == 1
    assert p.read_text().strip() == "1"


def test_read_run_id_increments(tmp_path):
    p = tmp_path / "run_number.txt"
    p.write_text("5")
    run_id = read_run_id(str(p))
    assert run_id == 6
    assert p.read_text().strip() == "6"


def test_read_run_id_handles_empty_file(tmp_path):
    p = tmp_path / "run_number.txt"
    p.write_text("")
    run_id = read_run_id(str(p))
    assert run_id == 1
