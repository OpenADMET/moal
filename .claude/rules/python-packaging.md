---
description: 'Python project layout and dependency management: environment managers (uv default; pixi or conda where the repo declares them), pyproject.toml as single source, flat layout, lockfile policy, and the ruff/pyright standing gate. Apply when creating or modifying pyproject.toml, lockfiles, environment.yml, requirements files, or project scaffolding.'
paths:
- '**/pyproject.toml'
- '**/uv.lock'
- '**/requirements*.txt'
- '**/setup.py'
- '**/setup.cfg'
- '**/environment.yml'
- '**/pixi.toml'
provenance: shared/rules/lang/python/packaging.md @ 3da45ca
diverged: true
---

You are an expert in modern Python packaging and project structure.

## Principles

1. One file owns project metadata: `pyproject.toml`. Anything duplicating it is drift waiting to happen.
2. Environments are disposable; lockfiles make them reproducible.
3. The lint and type gate is part of the project, never a personal preference.

## Environment and dependencies

- This project builds with setuptools (the `[build-system]` table in `pyproject.toml`) and installs editable with `pip install -e .[dev]`; there is no uv project or committed lockfile. The catalog default is uv, hence this rule's `diverged` flag: follow the discipline below under setuptools, do not migrate the build backend.
- As a library, moal declares ranged dependencies (sensible lower and upper bounds, e.g. `lightning>=2.0,<2.6.2`) in `[project.dependencies]` rather than pinning through a lockfile; tighten a bound only with a stated reason.
- Dev tooling (pytest, ruff, pyright, pre-commit) lives in `[project.optional-dependencies].dev`, never mixed into runtime dependencies.
- Declare a new dependency in `pyproject.toml` first, then reinstall; never let an ad-hoc `pip install` into the active environment stand in for the manifest.

## Layout

- Flat layout: the importable package lives at `<package>/` in the repo root beside `pyproject.toml`, with tests in `tests/` at the root, never inside the package.
- The package is installed editable (`pip install -e .`); imports resolve through the environment, never by relying on the current working directory.
- All project metadata lives in `pyproject.toml` (setuptools reads the PEP 621 `[project]` table); do not reintroduce `setup.py` or `setup.cfg`.
- Entry points declared in `[project.scripts]` (moal exposes `moal = "moal.cli:main"`), never as loose top-level scripts.

## Standing gate

- ruff (lint + format) and pyright configured in `pyproject.toml`, run via pre-commit and CI.
- Gate configuration changes are reviewed like code: loosening a rule needs a reason in the commit message.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `requirements.txt` as the source of truth | `[project.dependencies]` in `pyproject.toml` |
| ad-hoc `pip install X` into the env as the dependency record | declare in `pyproject.toml`, then reinstall |
| reintroducing `setup.py` / `setup.cfg` | keep all metadata in the PEP 621 `[project]` table (setuptools reads it) |
| dev tools in runtime `[project.dependencies]` | the `[project.optional-dependencies].dev` group |
| `[tool.poetry]` sections | PEP 621 `[project]` table |
