#!/usr/bin/env python3
"""Canonical Git tag to Debian version and APT suite mapping for vBuf-ML."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


_BASE = r"(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)\.(?P<patch>0|[1-9][0-9]*)"
_SEQUENCE = r"(?:[2-9]|[1-9][0-9]+)"
_PRERELEASE = re.compile(rf"(?P<channel>alpha|beta)-{_BASE}(?:\.(?P<build>{_SEQUENCE}))?\Z")
_RELEASE = re.compile(rf"release-{_BASE}(?:-r(?P<revision>{_SEQUENCE}))?\Z")


@dataclass(frozen=True)
class ReleaseIdentity:
    tag: str
    channel: str
    apt_suite: str
    upstream_version: str
    debian_revision: int
    debian_version: str

    def as_json(self) -> dict[str, str | int]:
        return asdict(self)


def parse_release_tag(tag: str) -> ReleaseIdentity:
    """Parse only canonical release tags; reject aliases and malformed versions.

    alpha-0.1.0[.N] and beta-0.1.0[.N] map to Debian prereleases.
    release-0.1.0[-rN] maps to the final Debian version/revision. N starts at
    2 so that the unsuffixed spelling has one canonical representation.
    """
    match = _PRERELEASE.fullmatch(tag)
    if match:
        channel = match.group("channel")
        base = ".".join(match.group(part) for part in ("major", "minor", "patch"))
        build = match.group("build")
        upstream = f"{base}~{channel}" + (f".{build}" if build else "")
        revision = 1
        return ReleaseIdentity(
            tag=tag,
            channel=channel,
            apt_suite=channel,
            upstream_version=upstream,
            debian_revision=revision,
            debian_version=f"{upstream}-{revision}",
        )

    match = _RELEASE.fullmatch(tag)
    if match:
        base = ".".join(match.group(part) for part in ("major", "minor", "patch"))
        revision = int(match.group("revision") or "1")
        return ReleaseIdentity(
            tag=tag,
            channel="release",
            apt_suite="release",
            upstream_version=base,
            debian_revision=revision,
            debian_version=f"{base}-{revision}",
        )

    raise ValueError(
        f"unsupported or non-canonical release tag {tag!r}; expected "
        "alpha-X.Y.Z[.N], beta-X.Y.Z[.N], or release-X.Y.Z[-rN]"
    )


def is_release_namespace(tag: str) -> bool:
    """Whether a tag claims a managed namespace and must be strictly parsed."""
    return tag.startswith(("alpha-", "beta-", "release-"))
