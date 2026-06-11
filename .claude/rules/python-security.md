---
description: 'Python security hygiene: boundary validation, secrets handling, subprocess safety, deserialization, and dependency caution. Apply when handling user input, credentials, subprocess calls, network data, or file paths derived from external sources.'
paths:
- 'moal/model.py'
provenance: shared/rules/lang/python/security.md @ 3da45ca
diverged: true
---

You are an expert in writing security-conscious Python.

## Principles

1. Validate at the boundary, trust inside it: every external input is hostile until parsed into a typed structure.
2. Secrets never touch the repository, the logs, or an error message.
3. The safe API is the default API; reaching for the unsafe variant requires a stated reason.

## Boundaries and input

- Parse, don't validate: convert external input into typed objects (Pydantic, dataclasses) at the entry point; downstream code receives structure, never raw payloads.
- Paths built from external input are resolved and checked against an allowed root before use (`Path.resolve()`, then `is_relative_to`).
- SQL through parameterized queries or an ORM, never string interpolation.

## Subprocess and deserialization

- `subprocess.run([...])` with an argument list; `shell=True` only with a fixed string and a comment stating why.
- `yaml.safe_load`, never `yaml.load`; `pickle` never on untrusted data; `json` for interchange.
- `tempfile` module for temporary files, never predictable paths in `/tmp`.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `shell=True` with interpolated input | argument-list `subprocess.run` |
| `yaml.load(data)` | `yaml.safe_load(data)` |
| `pickle.loads` on external data | `json.loads` or a validated schema |
| `random` for tokens or keys | `secrets` module |
| `hashlib.md5`/`sha1` for security purposes | `sha256`+ or `hashlib.scrypt`/`bcrypt` for passwords |
| `eval`/`exec` on any external string | explicit parsing |

## Enforcement

moal's ruff gate selects the `S` (bandit) family (ignoring only `S603`/`S607`, the safe fixed-argv subprocess forms), so most of the directives above are gate-enforced; treat an `S`-rule suppression (`# noqa: S...`) as a finding requiring justification.

The load-bearing concern this rule scopes is what ruff `S` does not catch: `model.py` fetches the CheMeleon checkpoint over the network (`urlretrieve` from Zenodo in `download_chemeleon()`) and loads checkpoints via `from_foundation` (the default Zenodo download, or an arbitrary filesystem path). A model checkpoint is executable, untrusted-by-default input. Keep `_validate_from_foundation` rejecting unknown names and non-existent paths, keep loading weights with `torch.load(..., weights_only=True)` (as `_load_foundation_weights` already does), verify the integrity of a downloaded artifact before trusting it, and never load a checkpoint supplied from an untrusted source.
