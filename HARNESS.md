# Test harness

A fork of [emarco177/langchain-course](https://github.com/emarco177/langchain-course). Upstream
ships no tests; this branch adds a pytest + coverage + Sonar harness and nothing else.

## Why it is structured this way

The course keeps one project per branch, and **those branches have unrelated histories** —
`git merge-base main project/hello-world` returns nothing. So the harness cannot be merged in.
Instead every harness file is purely additive and configuration lives outside `pyproject.toml`
(which each project branch owns):

| File | Purpose |
|---|---|
| `pytest.ini` | testpaths, asyncio mode, the 90% coverage gate |
| `.coveragerc` | what gets measured, and what is excluded |
| `tests/conftest.py` | strips provider API keys from the env for every test |
| `.github/workflows/ci.yml` | lint-free test run on `uv sync --locked`, then Sonar |
| `sonar-project.properties` | points Sonar at `coverage.xml` |

That means `git cherry-pick` lands the whole harness on any project branch with no conflicts.

## Starting a project branch

```bash
git checkout -b my/hello-world upstream/project/hello-world
git cherry-pick <harness commit on main>      # git log main --oneline
uv add --dev pytest pytest-cov pytest-asyncio  # updates that branch's uv.lock
uv sync --locked
uv run pytest
```

Work on `my/*` branches rather than the `project/*` ones, so upstream stays pullable:

```bash
git fetch upstream
```

## The coverage gate

`--cov-fail-under=90` is in `addopts`, so plain `pytest` enforces it. It will fail on a freshly
cherry-picked branch — upstream's scripts have no tests. That is the intended pressure: the way
to satisfy it is to pull logic out of the `main.py` glue and into functions worth testing.

While exploring, override it rather than editing the config:

```bash
uv run pytest --cov-fail-under=0
```

`if __name__ == "__main__":` blocks are excluded in `.coveragerc` — upstream's scripts fire live
API calls there, and testing the instructor's glue code is not the point of the gate.

## Sonar

Two one-time interactive steps:

1. Import the repo at [sonarcloud.io](https://sonarcloud.io) → *Analyze new project*, and check
   the key it assigns matches `sonar.projectKey`.
2. Generate a token (*My Account → Security*) and add it:
   ```bash
   gh secret set SONAR_TOKEN --repo jakequicksilver9/langchain-course
   ```

Until that secret exists the Sonar step skips and CI still reports green.

Local CLI — `sonar.exe` ships in `%LOCALAPPDATA%\sonarqube-cli\bin` and is not on PATH:

```bash
export PATH="$PATH:$LOCALAPPDATA/sonarqube-cli/bin"   # Git Bash
sonar auth login                                      # interactive, once
sonar link langchain-course                           # writes .sonar-config.json
uv run pytest && sonar scan                           # coverage first, then analyse
```

Add `.scannerwork/` and `.sonar-config.json` to `.git/info/exclude` — they are local artifacts,
and upstream's `.gitignore` is left untouched so it keeps merging cleanly.

## Version note

Upstream pins `langchain>=0.3.27` with no upper bound, so a fresh resolve pulls 1.x while some
lecture code is 0.3-era. LangChain 1.0 removed `LLMChain`, `RetrievalQA`, `AgentExecutor` and
`initialize_agent` — they live in `langchain-classic` now, and agents are built with
`create_agent` on the LangGraph runtime. Import errors from video code are probably that.

`uv sync --locked` is the defence: it installs the versions the videos were recorded against
rather than re-resolving.
