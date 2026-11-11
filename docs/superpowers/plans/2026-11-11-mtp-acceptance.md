# MTP-only acceptance instrumentation

User approved implementing the described acceptance metrics and H=1/2/3/5
comparison. Keep device execution, target verification, cache and graph layout
unchanged. No push or service restart.

## Design

Opt-in `acceptance_stats=False`. Track each actual MTP CPU proposal with request
lifetime and expected drive-token anchor. Observe the next raw sampler result
before EOS/budget early returns. Require matching scheduled prefix, anchor and
full verification coverage. Use vLLM's len(sampled)-1 accepted count and account
for invalid scheduled tail tokens. Unknown observations are excluded explicitly.

Keep zero-acceptance rounds, actual position exposure, acceptance histogram,
cold/warm first/subsequent proposal counters and recomputed position counts.
Cancelled/unverified proposals do not enter acceptance denominators. Request
cleanup emits one INFO JSON summary, per-round detail is DEBUG only. No extra
device reads or clocks. Analyzer aggregates weighted counts and rejects duplicate
or corrupt summaries. Final summaries do not cover still-active/crashed requests.

## Tasks

- [x] RED pure counters and integration, implement source and lifecycle tracking.
- [x] Cover zero/full/partial, external-source isolation, truncation/invalid tail,
  EOS, cancellation, ID reuse, analysis weighting and malformed/missing logs.
- [x] Refresh six deployment routes, installed tests and configuration/docs.
- [x] Related and full regression, source/installed parity, CLI smoke, lint,
  independent checks against upstream scheduler's raw verifier accounting.

## Verification

- Pure tracker tests failed before implementation; worker integration failed
  for missing `acceptance_stats` constructor support before wiring.
- Related suite: 207 run, 193 passed, 14 device skips. Installed suite: 131 run,
  117 passed, 14 device skips. Logs: `mtp-acceptance-related.log` and
  `mtp-acceptance-installed.log` under `.tmp/mtp-test-env`.
- Full research: 373 run, 19 errors and one failure, identities matching the
  prior baseline. Log: `.tmp/mtp-test-env/mtp-acceptance-full.log`.
- Six patch routes apply/check, compile and match all 13 source modules;
  empty re-export, reverse-check and 10 mirrored test AST checks pass.
- CLI test executes the stdlib-only analyzer as a subprocess, writes JSON,
  checks weighted acceptance; missing, duplicate and malformed logs fail.
- Enabled worker instrumentation preserves exactly one Torch `tolist()` per
  batch. Counts are observed before EOS early returns; cancellation cleanup
  emits pending proposals without treating them as verifier rejections.
- Source check: upstream v0.23 scheduler `update_from_output()` computes
  accepted drafts as `len(generated_token_ids)-1` before `_update_request_with_output`;
  `make_spec_decoding_stats()` subtracts invalid scheduled tail tokens. The new
  tracker uses the same accounting with added MTP provenance/position guards.
- All normal INFO summaries require completed worker cleanup; active/crashed
  lifetimes are explicitly absent. Device latency and log overhead require
  910B measurement. No NPU forward or synchronization has been added.
