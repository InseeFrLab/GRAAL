"""Identify an eval run by what produced it: a prompt revision and a model.

An LLM eval measures a prompt *run through a model*, and both can change under a
fixed script. The prompt lives in the repo (MatchVerifier's
`get_instructions`/`build_prompt`, the field descriptions of its output model, …);
the model is whatever `GENERATION_MODEL` named at the time (cf. base_agent.py).
Writing every run to one flat output directory therefore silently overwrites the
numbers of the previous prompt/model with those of the current one, and nothing in
the result says which produced which figure. `run_subpath()` gives a run a
`<commit>/<model>` directory naming both, and `assert_clean()` refuses to start
when the files defining the prompt carry uncommitted edits — which would make the
commit half of that name a lie.

The commit tag names HEAD, i.e. the whole tree, so it is only as meaningful as the
working tree is clean; `assert_clean()` enforces that for the files that matter
most, not for every file, and `revision_tag(allow_dirty=True)` deliberately
suffixes the name with `-dirty` rather than pretending a work-in-progress prompt
is the committed one.

The model half carries no such guard, and can't: it names the model the run
actually asked for, read from the same environment variable the agents read, so
it is accurate by construction rather than by promise.
"""

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Run git against the repo containing this file, not the caller's cwd: an eval
# launched from anywhere must still pin itself to this checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]

DIRTY_SUFFIX = "-dirty"


class DirtyWorkingTreeError(RuntimeError):
    """Raised when prompt-defining files have uncommitted (or untracked) changes."""


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed, cannot pin this run to a commit") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {REPO_ROOT}: {exc.stderr.strip()}"
        ) from exc
    return completed.stdout.strip()


def revision_sha() -> str:
    """Full SHA of HEAD."""
    return _git("rev-parse", "HEAD")


def revision_tag(allow_dirty: bool = False) -> str:
    """Directory-safe name for the current commit.

    The tag pointing at HEAD when there is one, `<nearest-tag>-<n>-g<sha>` when
    HEAD is past a tag, and the abbreviated SHA when the repo has no tags at all —
    `git describe --tags --always`, in other words, which degrades gracefully in a
    repo that doesn't tag its releases.

    `allow_dirty=True` appends `-dirty`, so a run made against uncommitted prompt
    edits gets its own directory instead of colonising the one that belongs to the
    committed prompt. Successive dirty runs do overwrite each other: that's the
    point of the suffix — dirty results are scratch, only a committed run is a
    result you can come back to.
    """
    tag = _git("describe", "--tags", "--always", "HEAD")
    return f"{tag}{DIRTY_SUFFIX}" if allow_dirty else tag


def dirty_paths(paths: list[str]) -> list[str]:
    """Which of `paths` (repo-relative) are modified, staged, or untracked."""
    # --untracked-files=all so a prompt file that was never committed counts as
    # dirty rather than silently passing: an uncommitted file is exactly the case
    # where the commit tag would say nothing about what actually ran.
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *paths)
    # Porcelain v1 lines are "XY <path>". Split on the first run of whitespace
    # rather than slicing a fixed width: _git() strips its output, so an unstaged
    # line (" M path") loses its leading space and would be misaligned by one.
    # Renames read "orig -> new"; the destination is the path that matters.
    dirty = set()
    for line in status.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            dirty.add(fields[1].split(" -> ")[-1].strip('"'))
    return sorted(dirty)


def assert_clean(paths: list[str], allow_dirty: bool = False) -> None:
    """Refuse to run when any of `paths` has uncommitted changes.

    `allow_dirty=True` downgrades the refusal to a warning — the caller is then
    expected to pass the same flag to `revision_tag()`, so the run's output lands
    under a `-dirty` name.
    """
    dirty = dirty_paths(paths)
    if not dirty:
        return
    listed = ", ".join(dirty)
    if allow_dirty:
        logger.warning(
            f"Running against uncommitted changes in {listed}; results will be written "
            f"under a '{DIRTY_SUFFIX}' directory and are not reproducible from a commit"
        )
        return
    raise DirtyWorkingTreeError(
        f"Uncommitted changes in {listed}. This eval measures the prompt these files "
        "define, and its results are filed under the current commit, so running now "
        "would file results for a prompt that commit doesn't contain. Commit (or stash) "
        "them first, or pass --allow-dirty to write to a scratch "
        f"'{revision_tag(allow_dirty=True)}' directory instead."
    )


# Model ids routinely carry a vendor prefix ("Qwen/Qwen3-32B") or a tag
# ("llama3:70b"); anything outside this set would either nest an unintended
# subdirectory or need quoting on the command line.
_UNSAFE_IN_PATH = re.compile(r"[^A-Za-z0-9._-]+")


def model_slug(model: str | None = None) -> str:
    """Directory-safe name of the generation model, from GENERATION_MODEL by default.

    Read from the same environment variable `BaseAgent` passes to the Agent, so the
    directory names the model that actually answered rather than one a flag claimed:
    there is deliberately no way to label a run with a model it didn't use.
    """
    raw = model if model is not None else os.environ.get("GENERATION_MODEL")
    if not raw:
        raise RuntimeError(
            "GENERATION_MODEL is not set, so this run cannot be filed under the model "
            "that produced it. Set it in the environment (cf. .env), as the agents do."
        )
    return _UNSAFE_IN_PATH.sub("-", raw).strip("-")


def run_subpath(allow_dirty: bool = False, model: str | None = None) -> str:
    """`<commit>/<model>` — the directory a run's outputs belong under.

    Commit first, model second: a prompt revision is the coarser change (it can
    invalidate every model's numbers at once), so grouping by it keeps the models
    of one prompt revision side by side, which is the comparison usually wanted.
    """
    return os.path.join(revision_tag(allow_dirty=allow_dirty), model_slug(model))
