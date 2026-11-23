# EvalHub dataset loading scripts

For the EvalHub adapter with Hugging Face Datasets 3.x, the benchmark's top-level
`parameters.trust_remote_code` controls whether the dataset loader may execute
custom loading scripts. For example:

```json
{
  "provider_id": "lm_evaluation_harness",
  "id": "scrolls_qasper",
  "parameters": {
    "trust_remote_code": true
  }
}
```

The value must be a JSON boolean. An explicit `false` disables dataset-script
trust for that evaluation, including for a benchmark on the adapter's existing
allowlist. If omitted, the adapter preserves its existing policy: the
`ethics_cm` exception and the runtime's existing Datasets configuration.
The previous Datasets setting is restored after evaluation, including on errors,
so one job's explicit choice does not carry over to a later job.

This parameter applies to **dataset loading**, not model loading. It does not set
`HF_ALLOW_CODE_EVAL` or `confirm_run_unsafe_code` for tasks that execute generated
model output.

Enabling this option executes the dataset's Python loading script inside the
adapter process. It does not provide a sandbox or add isolation. Use it only for
approved dataset code and deployment policies; a script-free dataset is an
alternative where available. This change does not make script-based datasets
compatible with Datasets versions that have removed loading-script support, and
it does not resolve separate metric, model, network or dataset errors.

See [Hugging Face's Datasets 3.1 loading guide](https://huggingface.co/docs/datasets/v3.1.0/loading).
