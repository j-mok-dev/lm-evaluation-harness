# EvalHub code-execution opt-in

The LM adapter accepts an explicit boolean
`parameters.confirm_run_unsafe_code` for these benchmark IDs:

- `cruxeval_output`
- `humaneval`
- `humaneval_instruct`
- `mbpp`

For example, a benchmark entry can contain:

```json
{
  "id": "cruxeval_output",
  "provider_id": "lm_evaluation_harness",
  "parameters": {
    "confirm_run_unsafe_code": true
  }
}
```

The adapter passes this setting to the Harness `simple_evaluate` entry point.
When the parameter is omitted or false, the Harness continues to reject tasks
marked unsafe. Strings, numbers and null are rejected. Opt-in for other benchmark
IDs is rejected instead of expanding code execution to arbitrary tasks or groups.

HumanEval and MBPP also use the Hugging Face `code_eval` metric. Their existing
`HF_ALLOW_CODE_EVAL` gate is enabled only during a confirmed run, then restored
even when evaluation fails. An inherited environment variable cannot replace
the explicit job parameter. CRUXEval uses its own execution path and does not
need the HF metric gate; that gate remains disabled during a CRUXEval run.

This option permits execution in the existing adapter runtime. It does not add
a security sandbox. CRUXEval's existing reliability guard is not a sandbox
either. Deployment and release policy must be reviewed before enabling this in
shared environments or default collection/smoke definitions. The implementation
does not change those definitions, dataset trust or model-loading trust.

The local regression tests exercise the real Harness unsafe-task gate with a
synthetic task and a fixed-output model. Its scorer compares strings and never
executes generated code; the tests do not establish safe execution of actual
CRUXEval, HumanEval or MBPP submissions.
