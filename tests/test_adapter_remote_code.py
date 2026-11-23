"""Dataset-script opt-in reaches the loader without changing other jobs."""

import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import datasets
import pytest

import main


@pytest.fixture
def run_adapter(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("These dataset-policy tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    monkeypatch.setattr(
        main, "resolve_model_credentials", lambda: SimpleNamespace(api_key=None)
    )
    monkeypatch.setattr(main, "read_model_auth_key", lambda key: None)
    monkeypatch.setattr(main, "TaskManager", MagicMock())
    monkeypatch.setattr(
        main, "_resolve_lmeval_task", lambda benchmark, manager: benchmark
    )
    monkeypatch.setattr(main, "_build_additional_info", lambda **kwargs: {})

    def run(parameters, evaluate, benchmark="scrolls_qasper"):
        monkeypatch.setattr(main, "simple_evaluate", evaluate)
        config = SimpleNamespace(
            id="00000000-0000-0000-0000-000000000001",
            benchmark_id=benchmark,
            benchmark_index=0,
            num_examples=1,
            model=SimpleNamespace(name="test/model", url="http://localhost:8000/v1"),
            parameters=parameters,
            exports=None,
            model_dump=lambda: {},
        )
        adapter = main.LMEvalAdapter.__new__(main.LMEvalAdapter)
        adapter._settings = main.AdapterSettings(mode="k8s")
        return adapter.run_benchmark_job(config, MagicMock())

    return run


def result(benchmark):
    return {
        "results": {benchmark: {"acc,none": 1.0}},
        "samples": {benchmark: [{}]},
    }


@pytest.mark.parametrize(
    "benchmark",
    [
        "scrolls_qasper",
        "cmmlu_college_mathematics",
        "aclue_ancient_chinese_culture",
        "pile_arxiv",
    ],
)
def test_explicit_opt_in_reaches_dataset_trust_gate(
    monkeypatch, run_adapter, benchmark
):
    """Use the real datasets 3.x trust resolver, without fetching/executing scripts."""
    from datasets import load

    resolver = getattr(load, "resolve_trust_remote_code", None)
    if resolver is None:
        pytest.skip(
            "The image uses datasets 3.x; newer versions removed script support"
        )
    monkeypatch.setattr(datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False)

    def evaluate(**kwargs):
        assert resolver(None, "fixture/dataset") is True
        assert "trust_remote_code" not in kwargs["model_args"]
        return result(benchmark)

    assert (
        run_adapter({"trust_remote_code": True}, evaluate, benchmark).overall_score
        == 1.0
    )
    assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is False


def test_explicit_false_overrides_existing_benchmark_allowlist(
    monkeypatch, run_adapter
):
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", True, raising=False
    )

    def evaluate(**kwargs):
        assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is False
        return result("ethics_cm")

    run_adapter({"trust_remote_code": False}, evaluate, "ethics_cm")
    assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is True


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None, {}])
def test_invalid_opt_in_fails_before_evaluation(monkeypatch, run_adapter, value):
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False
    )
    evaluate = MagicMock(return_value=result("scrolls_qasper"))
    with pytest.raises(RuntimeError, match="trust_remote_code must be a boolean"):
        run_adapter({"trust_remote_code": value}, evaluate)
    evaluate.assert_not_called()
    assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is False


@pytest.mark.parametrize(
    ("benchmark", "previous", "expected"),
    [
        ("scrolls_qasper", False, False),
        ("scrolls_qasper", True, True),
        ("ethics_cm", False, True),
    ],
)
def test_omitted_option_preserves_existing_policy(
    monkeypatch, run_adapter, benchmark, previous, expected
):
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", previous, raising=False
    )

    def evaluate(**kwargs):
        assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is expected
        return result(benchmark)

    run_adapter({}, evaluate, benchmark)
    assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is previous


def test_opt_in_is_restored_after_failure_and_does_not_leak(monkeypatch, run_adapter):
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False
    )

    def fail(**kwargs):
        assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is True
        raise ValueError("fixture dataset failed")

    with pytest.raises(RuntimeError, match="fixture dataset failed"):
        run_adapter({"trust_remote_code": True}, fail)
    assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is False

    def evaluate(**kwargs):
        assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is False
        return result("scrolls_qasper")

    run_adapter({}, evaluate)


def test_real_local_dataset_script_loads_only_after_opt_in(
    monkeypatch, tmp_path, run_adapter
):
    """Exercise load_dataset with our own one-row script; never fetch Hub code."""
    if not hasattr(datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE"):
        pytest.skip(
            "The image uses datasets 3.x; newer versions removed script support"
        )
    monkeypatch.setattr(datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False)
    monkeypatch.setattr(datasets.config, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    script = tmp_path / "trust_fixture.py"
    script.write_text(
        "import datasets\n\n"
        "class TrustFixture(datasets.GeneratorBasedBuilder):\n"
        "    def _info(self):\n"
        "        return datasets.DatasetInfo(features=datasets.Features(\n"
        "            {'text': datasets.Value('string')}))\n"
        "    def _split_generators(self, dl_manager):\n"
        "        return [datasets.SplitGenerator(name='test')]\n"
        "    def _generate_examples(self):\n"
        "        yield 0, {'text': 'controlled local fixture'}\n"
    )

    def evaluate(**kwargs):
        loaded = datasets.load_dataset(str(script), cache_dir=str(tmp_path / "cache"))
        assert loaded["test"][0] == {"text": "controlled local fixture"}
        return result("scrolls_qasper")

    with pytest.raises(RuntimeError, match="trust_remote_code=True"):
        run_adapter({}, evaluate)
    assert (
        run_adapter({"trust_remote_code": True}, evaluate).num_examples_evaluated == 1
    )
    assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is False
