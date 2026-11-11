"""Config derivation and CLI override handling (review item 6)."""

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml

from llamlam.config import Config
from llamlam.train import (
    DEFAULT_RUNS_DIR,
    RUN_OPTIONS,
    apply_overrides,
    build_config,
    configure_logging,
    parse_args,
    resolve_run_dir,
)


def test_dim_embd_derived_from_constructor_args():
    assert Config(dim_head=8, n_heads=4).dim_embd == 32


def test_dim_embd_follows_later_changes():
    config = Config(dim_head=8, n_heads=4)
    config.n_heads = 2
    assert config.dim_embd == 16
    config.dim_head = 32
    assert config.dim_embd == 64


def test_dim_embd_cannot_be_set_inconsistently():
    config = Config()
    with pytest.raises(AttributeError):
        config.dim_embd = 123
    with pytest.raises(TypeError):
        Config(dim_embd=123)


def test_config_dump_contains_dim_embd_and_all_fields():
    config = Config(dim_head=8, n_heads=4)
    dumped = yaml.safe_load(yaml.dump(config.to_dict()))
    assert dumped["dim_embd"] == 32
    for f in dataclasses.fields(config):
        assert dumped[f.name] == getattr(config, f.name)


def test_model_shape_flags_take_effect():
    args = parse_args(["--n_layers", "3", "--n_heads", "2", "--dim_head", "16"])
    config = build_config(args)
    assert (config.n_layers, config.n_heads, config.dim_head) == (3, 2, 16)
    assert config.dim_embd == 32


def test_old_singular_flag_spellings_still_take_effect():
    config = build_config(parse_args(["--n_layer", "5", "--n_head", "4"]))
    assert (config.n_layers, config.n_heads) == (5, 4)
    assert config.dim_embd == 4 * Config().dim_head


def test_unset_flags_keep_defaults():
    assert build_config(parse_args([])) == Config()


def test_every_flag_maps_to_config_field_or_run_option():
    field_names = {f.name for f in dataclasses.fields(Config)}
    for dest in vars(parse_args([])):
        assert dest in field_names or dest in RUN_OPTIONS, dest


def test_unknown_override_raises():
    with pytest.raises(ValueError, match="n_layer"):
        apply_overrides(Config(), {"n_layer": 3})


def test_derived_override_raises():
    with pytest.raises(ValueError, match="dim_embd"):
        apply_overrides(Config(), {"dim_embd": 512})


def test_run_name_and_output_dir_take_effect(tmp_path):
    args = parse_args(["--output_dir", str(tmp_path), "--run_name", "my_run"])
    assert resolve_run_dir(args) == ("my_run", tmp_path / "my_run")


def test_run_dir_defaults():
    run_name, run_dir = resolve_run_dir(parse_args([]))
    assert run_name.startswith("test_")
    assert run_dir == Path(DEFAULT_RUNS_DIR) / run_name


def test_debug_flag_sets_debug_log_level():
    root = logging.getLogger()
    old_level = root.level
    try:
        configure_logging(debug=parse_args(["--debug"]).debug)
        assert root.level == logging.DEBUG
        configure_logging(debug=parse_args([]).debug)
        assert root.level == logging.INFO
    finally:
        root.setLevel(old_level)
