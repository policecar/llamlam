"""
Config derivation and CLI override handling (review item 6), and the
model_type / optimizer / mixed_precision / checkpoint flags.
"""

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml
from accelerate.state import AcceleratorState

from llamlam.config import MIXED_PRECISION_MODES, MODEL_TYPES, OPTIMIZERS, Config
from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel
from llamlam.train import (
    DEFAULT_RUNS_DIR,
    MODEL_CLASSES,
    RUN_OPTIONS,
    apply_overrides,
    build_accelerator,
    build_config,
    build_model,
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


##########################################
# model_type, optimizer, mixed_precision, checkpoint flags
##########################################


def test_new_choice_defaults():
    config = Config()
    assert (config.model_type, config.optimizer, config.mixed_precision) == (
        "diff",
        "adamw",
        "no",
    )
    assert config.keep_k_checkpoints == 3
    assert not hasattr(config, "bfloat16")  # replaced by mixed_precision


def test_choice_flags_take_effect():
    args = parse_args(
        [
            "--model_type", "gpt", "--optimizer", "muon",
            "--mixed_precision", "bf16", "--keep_k_checkpoints", "5",
        ]
    )  # fmt: skip
    config = build_config(args)
    assert (config.model_type, config.optimizer, config.mixed_precision) == (
        "gpt",
        "muon",
        "bf16",
    )
    assert config.keep_k_checkpoints == 5


@pytest.mark.parametrize(
    "flag, value",
    [("--model_type", "llama"), ("--optimizer", "sgd"), ("--mixed_precision", "fp8")],
)
def test_invalid_choice_flags_rejected(flag, value):
    with pytest.raises(SystemExit):
        parse_args([flag, value])


def test_choice_tuples_match_implementations():
    assert tuple(MODEL_CLASSES) == MODEL_TYPES
    assert OPTIMIZERS == ("adamw", "muon", "grokadamw")
    assert MIXED_PRECISION_MODES == ("no", "fp16", "bf16")


def test_muon_learning_rate_flag_takes_effect_with_muon():
    assert Config().muon_learning_rate == 0.02
    config = build_config(
        parse_args(["--optimizer", "muon", "--muon_learning_rate", "0.05"])
    )
    assert config.muon_learning_rate == 0.05
    assert config.learning_rate == Config().learning_rate  # untouched


@pytest.mark.parametrize("optimizer", [None, "adamw", "grokadamw"])
def test_muon_learning_rate_flag_rejected_without_muon(optimizer):
    argv = ["--muon_learning_rate", "0.05"]
    if optimizer is not None:
        argv += ["--optimizer", optimizer]
    with pytest.raises(ValueError, match="only applies to --optimizer muon"):
        build_config(parse_args(argv))


def test_muon_learning_rate_help_says_muon_only(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--muon_learning_rate" in help_text
    assert "Only with --optimizer muon" in help_text
    assert "(default: 0.02)" in help_text


def test_resume_from_is_a_run_option_not_a_config_field(tmp_path):
    args = parse_args(["--resume_from", "latest"])
    assert args.resume_from == "latest"
    assert "resume_from" not in {f.name for f in dataclasses.fields(Config)}
    assert build_config(args) == Config()


def tiny(model_type):
    return Config(
        model_type=model_type, n_layers=1, n_heads=2, dim_head=8, vocab_size=50,
        max_seq_length=16,
    )  # fmt: skip


@pytest.mark.parametrize(
    "model_type, cls", [("gpt", GPTModel), ("diff", DiffTransformer)]
)
def test_build_model_selects_class(model_type, cls):
    model = build_model(tiny(model_type))
    assert type(model) is cls


def test_build_model_unknown_type_raises():
    with pytest.raises(ValueError, match="model_type 'llama'"):
        build_model(tiny("llama"))


@pytest.fixture
def fresh_accelerator_state():
    # AcceleratorState is a process-wide singleton whose mixed_precision cannot
    # change once set; reset it so each mode gets a real Accelerator
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


@pytest.mark.parametrize("mode", ["no", "bf16"])  # bf16 autocast exists on CPU
def test_build_accelerator_passes_mixed_precision(mode, fresh_accelerator_state):
    accelerator = build_accelerator(Config(mixed_precision=mode), cpu=True)
    assert accelerator.mixed_precision == mode


@pytest.mark.parametrize("mode", ["fp8", "bfloat16", "", None, True])
def test_build_accelerator_rejects_invalid_mixed_precision(mode):
    with pytest.raises(ValueError, match="mixed_precision"):
        build_accelerator(Config(mixed_precision=mode), cpu=True)


def test_resume_from_dir_continues_in_its_run(tmp_path):
    ckpt = tmp_path / "my_run" / "last"
    args = parse_args(["--resume_from", str(ckpt)])
    assert resolve_run_dir(args) == ("my_run", tmp_path / "my_run")


def test_explicit_run_flags_win_over_resume_dir(tmp_path):
    ckpt = tmp_path / "old_run" / "last"
    args = parse_args(
        ["--resume_from", str(ckpt), "--output_dir", str(tmp_path), "--run_name", "new"]
    )
    assert resolve_run_dir(args) == ("new", tmp_path / "new")


def test_resume_latest_uses_named_run(tmp_path):
    args = parse_args(
        ["--resume_from", "latest", "--output_dir", str(tmp_path), "--run_name", "r"]
    )
    assert resolve_run_dir(args) == ("r", tmp_path / "r")


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
