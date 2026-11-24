"""Explicit code-execution consent reaches both gates; no generated code runs."""

import os
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import datasets
import pytest

import main
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import simple_evaluate as harness_evaluate
from lm_eval.models.dummy import DummyLM


@pytest.fixture
def run_adapter(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("Code-execution policy tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False
    )
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

    def run(parameters, evaluate, benchmark="cruxeval_output"):
        monkeypatch.setattr(main, "simple_evaluate", evaluate)
        job = SimpleNamespace(
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
        return adapter.run_benchmark_job(job, MagicMock())

    return run


def result(benchmark):
    return {"results": {benchmark: {"acc,none": 1.0}}, "samples": {benchmark: [{}]}}


@pytest.mark.parametrize(
    ("benchmark", "hf_gate"),
    [
        ("cruxeval_output", "0"),
        ("humaneval", "1"),
        ("humaneval_instruct", "1"),
        ("mbpp", "1"),
    ],
)
def test_explicit_opt_in_reaches_harness_and_required_hf_gate(
    monkeypatch, run_adapter, benchmark, hf_gate
):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "previous")

    def evaluate(**kwargs):
        assert kwargs["confirm_run_unsafe_code"] is True
        assert "confirm_run_unsafe_code" not in kwargs["model_args"]
        assert os.environ["HF_ALLOW_CODE_EVAL"] == hf_gate
        return result(benchmark)

    assert (
        run_adapter(
            {"confirm_run_unsafe_code": True}, evaluate, benchmark
        ).overall_score
        == 1.0
    )
    assert os.environ["HF_ALLOW_CODE_EVAL"] == "previous"


@pytest.mark.parametrize("parameters", [{}, {"confirm_run_unsafe_code": False}])
def test_no_opt_in_disables_both_gates_even_with_inherited_env(
    monkeypatch, run_adapter, parameters
):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")

    def evaluate(**kwargs):
        assert kwargs["confirm_run_unsafe_code"] is False
        assert os.environ["HF_ALLOW_CODE_EVAL"] == "0"
        return result("humaneval")

    run_adapter(parameters, evaluate, "humaneval")
    assert os.environ["HF_ALLOW_CODE_EVAL"] == "1"


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None, {}])
def test_non_boolean_consent_is_rejected_before_evaluation(run_adapter, value):
    evaluate = MagicMock(return_value=result("cruxeval_output"))
    with pytest.raises(RuntimeError, match="confirm_run_unsafe_code must be a boolean"):
        run_adapter({"confirm_run_unsafe_code": value}, evaluate)
    evaluate.assert_not_called()


@pytest.mark.parametrize(
    "benchmark", ["arc_easy", "mbpp_plus", "cruxeval_input", "unknown"]
)
def test_opt_in_does_not_expand_the_supported_benchmark_list(run_adapter, benchmark):
    evaluate = MagicMock(return_value=result(benchmark))
    with pytest.raises(RuntimeError, match="not supported for benchmark"):
        run_adapter({"confirm_run_unsafe_code": True}, evaluate, benchmark)
    evaluate.assert_not_called()


@pytest.mark.parametrize("previous", [None, "0", "1"])
def test_environment_is_restored_after_evaluation_error(
    monkeypatch, run_adapter, previous
):
    if previous is None:
        monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    else:
        monkeypatch.setenv("HF_ALLOW_CODE_EVAL", previous)

    def evaluate(**kwargs):
        assert kwargs["confirm_run_unsafe_code"] is True
        assert os.environ["HF_ALLOW_CODE_EVAL"] == "1"
        raise ValueError("fixture evaluation failed")

    with pytest.raises(RuntimeError, match="fixture evaluation failed"):
        run_adapter({"confirm_run_unsafe_code": True}, evaluate, "humaneval")
    assert os.environ.get("HF_ALLOW_CODE_EVAL") == previous


@pytest.mark.parametrize(
    "parameters",
    [{}, {"confirm_run_unsafe_code": False}, {"confirm_run_unsafe_code": True}],
)
def test_real_harness_unsafe_gate_with_non_executing_synthetic_task(
    monkeypatch, run_adapter, parameters
):
    """Exercise the real Harness gate, but use a scorer that never evaluates code."""
    data = datasets.DatasetDict(
        {
            "test": datasets.Dataset.from_list(
                [{"question": "Return a fixed string", "answer": "fixed"}]
            )
        }
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: data)
    task = ConfigurableTask(
        config={
            "task": "cruxeval_output",
            "dataset_path": "fixture",
            "test_split": "test",
            "unsafe_code": True,
            "output_type": "generate_until",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "generation_kwargs": {"until": ["\n"], "max_gen_toks": 1},
            "process_results": lambda doc, results: {
                "acc": float(results[0] == "fixed")
            },
            "metric_list": [
                {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
            ],
        }
    )

    class RecordingLM(DummyLM):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def generate_until(self, requests, disable_tqdm=False):
            self.calls += len(requests)
            return ["fixed"] * len(requests)

    model = RecordingLM()

    def evaluate(**kwargs):
        return harness_evaluate(
            model=model,
            tasks=[task],
            confirm_run_unsafe_code=kwargs.get("confirm_run_unsafe_code", False),
            limit=1,
            bootstrap_iters=0,
            log_samples=True,
        )

    if parameters.get("confirm_run_unsafe_code") is True:
        results = run_adapter(parameters, evaluate)
        assert results.overall_score == 1.0
        assert results.num_examples_evaluated == 1
        assert model.calls == 1
    else:
        with pytest.raises(RuntimeError, match="marked as unsafe"):
            run_adapter(parameters, evaluate)
        assert model.calls == 0
