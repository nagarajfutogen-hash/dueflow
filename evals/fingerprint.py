"""Reproducibility fingerprints stored on every eval run.

config_hash: what the agent was (models, rules, prompts, scripts, rubric).
dataset_hash: exactly which scenario files ran, even with uncommitted edits.
git_sha: the commit, when the project is a git repository (read from .git; no git binary).
"""

import hashlib
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that define the agent's behavior. Rubric is added by the judge (Phase 5).
CONFIG_FILES = (
    "config/models.yaml",
    "app/policy/rules.yaml",
    "app/agent/nodes.py",
    "app/agent/scripts.py",
)
# The simulator is not part of the agent's config: it is versioned on its own
# (eval_runs.simulator_version), so a simulator fix never looks like an agent change.


def files_hash(paths: Iterable[Path], root: Path = REPO_ROOT) -> str:
    """sha256 over (relative path, content) of each file, independent of listing order."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.resolve().relative_to(root.resolve()).as_posix()):
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def config_hash(extra: Iterable[Path] = (), root: Path = REPO_ROOT) -> str:
    return files_hash([root / f for f in CONFIG_FILES] + list(extra), root)


def git_sha(root: Path = REPO_ROOT) -> str | None:
    """HEAD commit read from .git (loose or packed ref). None if not a git repository."""
    git = root / ".git"
    head = git / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text(encoding="utf-8").strip()
    if not ref.startswith("ref: "):
        return ref or None  # detached HEAD holds the sha itself
    name = ref.removeprefix("ref: ")
    loose = git / name
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip() or None
    packed = git / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            sha, _, ref_name = line.partition(" ")
            if ref_name == name:
                return sha
    return None  # unborn branch: no commits yet
