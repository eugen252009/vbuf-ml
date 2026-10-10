#!/usr/bin/env python3
"""Build a clean, reproducible vBuf-ML Debian binary package from a Git commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

from version import ReleaseIdentity, is_release_namespace, parse_release_tag


REPOSITORY_URL = "https://github.com/eugen252009/vbuf-ml"
GGML_URL = "https://github.com/ggml-org/ggml.git"
RUST_TOOLCHAIN = "1.95.0"
EXPECTED_ARCHITECTURE = "amd64"
ARCHIVE_PATHS = (
    "debian",
    "docs/vbuf-ml/debian-alpha.md",
    "integrations/ggml",
    "rust-toolchain.toml",
    "rust/.cargo",
    "rust/Cargo.lock",
    "rust/Cargo.toml",
    "rust/src",
    "rust/vbuf-layout",
    "rust/vbuf-ml",
    "rust/vbuf-runtime/Cargo.toml",
    "rust/vbuf-runtime/build.rs",
    "rust/vbuf-runtime/cuda",
    "rust/vbuf-runtime/src",
)
GGML_PIN_RE = re.compile(
    r'^set\(VBUF_GGML_COMMIT "([0-9a-f]{40})" CACHE STRING', re.MULTILINE
)


class BuildError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise BuildError(f"required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = ""
        if capture:
            detail = f"\n{exc.stdout or ''}{exc.stderr or ''}"
        raise BuildError(f"command failed ({exc.returncode}): {' '.join(command)}{detail}") from exc
    return result.stdout.strip() if capture else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args], capture=True)


def canonical_repository(repo: Path) -> str:
    remote = git(repo, "remote", "get-url", "origin")
    accepted = {
        "git@github.com:eugen252009/vbuf-ml.git",
        "git@github.com:eugen252009/vbuf-ml",
        "https://github.com/eugen252009/vbuf-ml.git",
        "https://github.com/eugen252009/vbuf-ml",
        "ssh://git@github.com/eugen252009/vbuf-ml.git",
    }
    if remote not in accepted:
        raise BuildError(
            f"origin must identify {REPOSITORY_URL}; found {remote!r}"
        )
    return REPOSITORY_URL


def ensure_clean(repo: Path) -> None:
    status = git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise BuildError("release builds require a clean worktree and index:\n" + status)


def tagged_commit(repo: Path, tag: str, candidate: bool) -> tuple[str, str]:
    ensure_clean(repo)
    head = git(repo, "rev-parse", "HEAD^{commit}")
    tag_object = git(repo, "cat-file", "-t", f"refs/tags/{tag}") if tag in git(repo, "tag", "--list").splitlines() else ""
    if candidate:
        if tag_object:
            raise BuildError(f"candidate tag already exists locally: {tag}")
        return head, "candidate"
    if tag_object != "tag":
        raise BuildError(f"--tag requires an existing annotated Git tag: {tag}")
    target = git(repo, "rev-parse", f"refs/tags/{tag}^{{commit}}")
    if target != head:
        raise BuildError(
            f"release mode requires HEAD to equal annotated tag {tag} ({target}); found {head}"
        )
    return target, "annotated-local-tag"


def validate_tag_catalog(repo: Path, identity: ReleaseIdentity, requested_tag: str) -> None:
    versions: dict[str, str] = {}
    for existing in git(repo, "tag", "--list").splitlines():
        if not is_release_namespace(existing):
            continue
        try:
            parsed = parse_release_tag(existing)
        except ValueError as exc:
            raise BuildError(f"malformed tag in managed release namespace: {existing}: {exc}") from exc
        prior = versions.get(parsed.debian_version)
        if prior is not None and prior != existing:
            raise BuildError(
                f"existing tags {prior} and {existing} map to the same Debian version "
                f"{parsed.debian_version}"
            )
        versions[parsed.debian_version] = existing
        if parsed.debian_version == identity.debian_version and existing != requested_tag:
            raise BuildError(
                f"Debian version collision: {existing} already maps to {identity.debian_version}"
            )


def read_ggml_pin(source: Path) -> str:
    cmake = (source / "integrations/ggml/CMakeLists.txt").read_text(encoding="utf-8")
    match = GGML_PIN_RE.search(cmake)
    if not match:
        raise BuildError("could not extract the immutable GGML pin from CMakeLists.txt")
    return match.group(1)


def tool_version(command: list[str]) -> str:
    return run(command, capture=True).splitlines()[0]


def build_environment(architecture: str) -> dict[str, Any]:
    os_release: dict[str, str] = {}
    release_file = Path("/etc/os-release")
    if release_file.exists():
        for line in release_file.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')
    versions = {
        "c_compiler": tool_version(["cc", "--version"]),
        "cxx_compiler": tool_version(["c++", "--version"]),
        "cmake": tool_version(["cmake", "--version"]),
        "cargo": tool_version(["cargo", "--version"]),
        "rustc": tool_version(["rustc", "--version"]),
        "dpkg_buildpackage": tool_version(["dpkg-buildpackage", "--version"]),
        "dpkg_deb": tool_version(["dpkg-deb", "--version"]),
        "debhelper": tool_version(["dh", "--version"]),
        "python": tool_version([sys.executable, "--version"]),
    }
    rustup = shutil.which("rustup")
    if rustup:
        versions["rustup"] = tool_version([rustup, "--version"])
    package_output = run(
        ["dpkg-query", "-W", "-f=${db:Status-Abbrev} ${binary:Package}=${Version}\\n"],
        capture=True,
    )
    installed_packages = sorted(
        line[4:] for line in package_output.splitlines() if line.startswith("ii ")
    )
    return {
        "architecture": architecture,
        "distribution": os_release.get("PRETTY_NAME", "unknown"),
        "versions": versions,
        "installed_packages": installed_packages,
        "container_image": os.environ.get("VBUF_BUILD_IMAGE", "unspecified"),
    }


def archive_source(repo: Path, commit: str, archive: Path) -> str:
    with archive.open("wb") as output:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", "--prefix=vbuf-ml/", commit, "--", *ARCHIVE_PATHS],
            stdout=output,
            check=True,
        )
    return sha256_file(archive)


def extract_source(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:") as stream:
        members = stream.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "vbuf-ml":
                raise BuildError(f"unsafe path in Git source archive: {member.name}")
            if member.issym() or member.islnk():
                raise BuildError(f"unexpected link in Git source archive: {member.name}")
        try:
            stream.extractall(destination, filter="data")
        except TypeError:  # Python < 3.12; member names were checked above.
            stream.extractall(destination)
    source = destination / "vbuf-ml"
    if not (source / "debian/rules").is_file() or not (source / "integrations/ggml/CMakeLists.txt").is_file():
        raise BuildError("Git archive is missing required Debian or native build inputs")
    return source


def fetch_ggml(destination: Path, commit: str, env: dict[str, str]) -> Path:
    checkout = destination / "ggml"
    checkout.mkdir()
    run(["git", "init", "--quiet"], cwd=checkout, env=env)
    run(["git", "remote", "add", "origin", GGML_URL], cwd=checkout, env=env)
    network_env = {**env, "GIT_TERMINAL_PROMPT": "0"}
    run(["git", "fetch", "--quiet", "--depth=1", "--no-tags", "origin", commit], cwd=checkout, env=network_env)
    run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=checkout, env=env)
    actual = git(checkout, "rev-parse", "HEAD")
    if actual != commit:
        raise BuildError(f"fetched GGML revision mismatch: wanted {commit}, found {actual}")
    if git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise BuildError("fetched GGML checkout is not clean")
    return checkout


def write_changelog(source: Path, identity: ReleaseIdentity, epoch: int) -> None:
    changelog = source / "debian/changelog"
    import datetime

    date = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )
    stanza = (
        f"vbuf-ml ({identity.debian_version}) {identity.apt_suite}; urgency=low\n\n"
        "  * Build the tagged vBuf-ML Alpha/Beta/Release source snapshot.\n\n"
        f" -- Eugen <eugen252009@gmail.com>  {date}\n\n"
    )
    changelog.write_text(stanza, encoding="utf-8")


def write_build_provenance(
    source: Path,
    *,
    identity: ReleaseIdentity,
    commit: str,
    tree: str,
    epoch: int,
    archive_sha256: str,
    ggml_commit: str,
    environment: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    lock = source / "rust/Cargo.lock"
    patch = source / "integrations/ggml/patches/ggml-native-attention-av.patch"
    doc: dict[str, Any] = {
        "format": "vbuf-ml-debian-build-provenance",
        "format_version": 1,
        "repository": REPOSITORY_URL,
        "git_tag": identity.tag,
        "git_commit": commit,
        "git_tree": tree,
        "source_date_epoch": epoch,
        "source_archive_sha256": archive_sha256,
        "package": {
            "name": "vbuf-ml",
            "version": identity.debian_version,
            "architecture": EXPECTED_ARCHITECTURE,
        },
        "channel": identity.channel,
        "apt_suite": identity.apt_suite,
        "external_dependencies": {
            "ggml": {
                "repository": GGML_URL,
                "commit": ggml_commit,
                "native_attention_patch_sha256": sha256_file(patch),
            },
            "rust_cargo_lock_sha256": sha256_file(lock),
        },
        "build_environment": environment,
        "build_configuration": {
            "cmake_build_type": "Release",
            "build_jobs": 2,
            "rust_toolchain": RUST_TOOLCHAIN,
            "rust_target_cpu": "x86-64",
            "ggml_shared_libraries": False,
            "private_library_runpath": "$ORIGIN/lib",
            "ggml_native": False,
            "cuda": False,
            "cpu_repack": False,
            "ggml_openmp": False,
            "ggml_blas": False,
            "qualification_faults": False,
            "model_payloads_included": False,
        },
        "qualification_boundary": (
            "Alpha package build and package smoke checks only; no model inference, "
            "generation parity, coding-agent qualification, CUDA, or Native AV qualification is implied."
        ),
    }
    output_path.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return doc


def package_metadata(package: Path) -> dict[str, str]:
    raw = run(["dpkg-deb", "-f", str(package)], capture=True)
    result: dict[str, str] = {}
    key = ""
    for line in raw.splitlines():
        if line[:1].isspace() and key:
            result[key] += " " + line.strip()
        elif ": " in line:
            key, value = line.split(": ", 1)
            result[key] = value
    return result


def verify_package(
    package: Path, identity: ReleaseIdentity, source_commit: str, build_root: Path
) -> dict[str, Any]:
    fields = package_metadata(package)
    expected = {
        "Package": "vbuf-ml",
        "Version": identity.debian_version,
        "Architecture": EXPECTED_ARCHITECTURE,
    }
    for field, value in expected.items():
        if fields.get(field) != value:
            raise BuildError(f"package metadata {field} mismatch: {fields.get(field)!r} != {value!r}")
    depends = fields.get("Depends", "")
    if not depends or "libvbuf_ml.so" in depends or "ggml" in depends.lower():
        raise BuildError(f"unexpected or missing runtime dependencies: {depends!r}")

    with tempfile.TemporaryDirectory(prefix="vbuf-deb-check-") as temp:
        root = Path(temp)
        extract = root / "root"
        control = root / "control"
        extract.mkdir()
        control.mkdir()
        run(["dpkg-deb", "-x", str(package), str(extract)])
        run(["dpkg-deb", "-e", str(package), str(control)])
        executable = extract / "usr/lib/vbuf-ml/vbuf_compat_server"
        library = extract / "usr/lib/vbuf-ml/lib/libvbuf_ml.so"
        launcher = extract / "usr/bin/vbuf_compat_server"
        provenance_path = extract / "usr/share/vbuf-ml/vbuf-ml-build-provenance.json"
        for path in (executable, library, launcher, provenance_path):
            if not path.is_file():
                raise BuildError(f"package is missing required payload: {path.relative_to(extract)}")
        scripts = {"preinst", "postinst", "prerm", "postrm", "config", "triggers"}
        present_scripts = scripts.intersection(path.name for path in control.iterdir())
        if present_scripts:
            raise BuildError(f"unexpected package maintainer scripts: {sorted(present_scripts)}")

        expected_launcher = '#!/bin/sh\nexec /usr/lib/vbuf-ml/vbuf_compat_server "$@"\n'
        if launcher.read_text(encoding="utf-8") != expected_launcher:
            raise BuildError("launcher does not exec the packaged runtime binary")
        run([str(executable), "--help"], env={**os.environ, "LD_LIBRARY_PATH": ""})
        readelf = run(["readelf", "-d", str(executable)], capture=True)
        if "RUNPATH" not in readelf or "$ORIGIN/lib" not in readelf:
            raise BuildError("runtime binary must use only the relative private-library RUNPATH $ORIGIN/lib")
        if str(source_commit) in readelf or "/tmp/" in readelf or "/home/" in readelf:
            raise BuildError("runtime binary contains a build/source path in its dynamic metadata")
        ldd = run(["ldd", str(executable)], capture=True)
        if "not found" in ldd:
            raise BuildError(f"package executable has unresolved shared libraries:\n{ldd}")
        binary_strings = run(["strings", "-a", str(executable)], capture=True)
        forbidden_paths = (str(build_root), "/home/", "/workspace/")
        if any(path in binary_strings for path in forbidden_paths):
            raise BuildError("runtime binary contains a temporary, home, or workspace build path")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("git_commit") != source_commit:
            raise BuildError("installed package provenance does not identify the selected Git commit")
        if provenance.get("package", {}).get("version") != identity.debian_version:
            raise BuildError("installed package provenance version does not match the package")
        contents = run(["dpkg-deb", "-c", str(package)], capture=True)
        forbidden = ("/etc/systemd/", ".service", "/var/lib/", "/etc/vbuf")
        if any(value in contents for value in forbidden):
            raise BuildError("package unexpectedly installs a service or persistent application state")

    return {
        "metadata": fields,
        "package_sha256": sha256_file(package),
        "package_size_bytes": package.stat().st_size,
        "runtime_dependencies": depends,
        "runtime_smoke": "vbuf_compat_server --help passed; no model was loaded",
        "maintainer_scripts": [],
    }


def build_once(
    *,
    source_archive: Path,
    build_root: Path,
    identity: ReleaseIdentity,
    commit: str,
    tree: str,
    epoch: int,
    archive_sha256: str,
    ggml_commit: str,
    environment: dict[str, Any],
    cargo_home: Path,
    fetch_cargo: bool,
) -> Path:
    build_root.mkdir(parents=True)
    source = extract_source(source_archive, build_root)
    ggml = fetch_ggml(build_root, ggml_commit, os.environ.copy())
    write_changelog(source, identity, epoch)
    provenance_path = build_root / "build-provenance.json"
    write_build_provenance(
        source,
        identity=identity,
        commit=commit,
        tree=tree,
        epoch=epoch,
        archive_sha256=archive_sha256,
        ggml_commit=ggml_commit,
        environment=environment,
        output_path=provenance_path,
    )

    env = os.environ.copy()
    env.update(
        {
            "CARGO_HOME": str(cargo_home),
            "CARGO_NET_OFFLINE": "true",
            "CARGO_INCREMENTAL": "0",
            "CARGO_BUILD_JOBS": "2",
            "CARGO_PROFILE_RELEASE_DEBUG": "0",
            "RUSTUP_TOOLCHAIN": RUST_TOOLCHAIN,
            "RUSTFLAGS": f"-C target-cpu=x86-64 --remap-path-prefix={source}=.",
            "SOURCE_DATE_EPOCH": str(epoch),
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "VBUF_GGML_SOURCE_DIR": str(ggml),
            "VBUF_BUILD_PROVENANCE_FILE": str(provenance_path),
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    if fetch_cargo:
        run(
            [
                "cargo",
                "fetch",
                "--locked",
                "--target",
                "x86_64-unknown-linux-gnu",
                "--manifest-path",
                str(source / "rust/Cargo.toml"),
            ],
            cwd=source,
            env={**env, "CARGO_NET_OFFLINE": "false"},
        )
    run(
        [
            "dpkg-buildpackage",
            "-b",
            "-uc",
            "-us",
            "-aamd64",
            "-j2",
        ],
        cwd=source,
        env=env,
    )
    packages = list(build_root.glob("vbuf-ml_*.deb"))
    if len(packages) != 1:
        raise BuildError(f"expected exactly one vbuf-ml .deb from dpkg-buildpackage; found {packages}")
    return packages[0]


def command_build(args: argparse.Namespace, repo: Path) -> None:
    candidate = args.candidate_tag is not None
    tag = args.candidate_tag if candidate else args.tag
    if not tag:
        raise BuildError("choose exactly one of --candidate-tag or --tag")
    identity = parse_release_tag(tag)
    repository = canonical_repository(repo)
    commit, tag_kind = tagged_commit(repo, tag, candidate)
    validate_tag_catalog(repo, identity, tag)
    if run(["dpkg", "--print-architecture"], capture=True) != EXPECTED_ARCHITECTURE:
        raise BuildError("this first package is qualified only for Debian amd64")

    commit_time = git(repo, "show", "-s", "--format=%ct", commit)
    epoch = int(commit_time)
    tree = git(repo, "rev-parse", f"{commit}^{{tree}}")
    ggml_commit = read_ggml_pin(repo)
    environment = build_environment(EXPECTED_ARCHITECTURE)
    if environment["versions"]["rustc"].split()[1] != RUST_TOOLCHAIN:
        raise BuildError(f"rustc {RUST_TOOLCHAIN} is required; found {environment['versions']['rustc']}")
    if environment["versions"]["cargo"].split()[1] != RUST_TOOLCHAIN:
        raise BuildError(f"cargo {RUST_TOOLCHAIN} is required; found {environment['versions']['cargo']}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package_name = f"vbuf-ml_{identity.debian_version}_{EXPECTED_ARCHITECTURE}.deb"
    package_path = output / package_name
    manifest_path = output / f"vbuf-ml_{identity.debian_version}_{EXPECTED_ARCHITECTURE}.build-manifest.json"
    sums_path = output / f"vbuf-ml_{identity.debian_version}_{EXPECTED_ARCHITECTURE}.SHA256SUMS"
    for path in (package_path, manifest_path, sums_path):
        if path.exists():
            raise BuildError(f"refusing to overwrite an existing release artifact: {path}")

    with tempfile.TemporaryDirectory(prefix="vbuf-ml-debian-build-") as temp:
        temp_root = Path(temp)
        archive = temp_root / "source.tar"
        archive_sha256 = archive_source(repo, commit, archive)
        cargo_home = temp_root / "cargo-home"
        cargo_home.mkdir()
        results: list[dict[str, Any]] = []
        build_count = 2 if args.check_reproducibility else 1
        built_packages: list[Path] = []
        for index in range(build_count):
            build_root = temp_root / f"build-{index + 1}"
            package = build_once(
                source_archive=archive,
                build_root=build_root,
                identity=identity,
                commit=commit,
                tree=tree,
                epoch=epoch,
                archive_sha256=archive_sha256,
                ggml_commit=ggml_commit,
                environment=environment,
                cargo_home=cargo_home,
                fetch_cargo=index == 0,
            )
            result = verify_package(package, identity, commit, build_root)
            results.append(result)
            built_packages.append(package)

        if len(results) == 2 and results[0]["package_sha256"] != results[1]["package_sha256"]:
            raise BuildError(
                "independent package builds differ: "
                f"{results[0]['package_sha256']} != {results[1]['package_sha256']}"
            )
        shutil.copyfile(built_packages[0], package_path)

    manifest = {
        "format": "vbuf-ml-debian-build-manifest",
        "format_version": 1,
        "repository": repository,
        "git_tag": tag,
        "git_tag_kind": tag_kind,
        "git_commit": commit,
        "git_tree": tree,
        "source_date_epoch": epoch,
        "source_archive_sha256": archive_sha256,
        "package": {
            "name": "vbuf-ml",
            "version": identity.debian_version,
            "architecture": EXPECTED_ARCHITECTURE,
            "apt_suite": identity.apt_suite,
            "filename": package_name,
            "sha256": sha256_file(package_path),
            "size_bytes": package_path.stat().st_size,
        },
        "external_dependencies": {
            "ggml": {
                "repository": GGML_URL,
                "commit": ggml_commit,
                "native_attention_patch_sha256": sha256_file(
                    repo / "integrations/ggml/patches/ggml-native-attention-av.patch"
                ),
            },
            "rust_cargo_lock_sha256": sha256_file(repo / "rust/Cargo.lock"),
        },
        "build_environment": environment,
        "independent_builds": results,
        "byte_identical_reproducible": len(results) == 2
        and results[0]["package_sha256"] == results[1]["package_sha256"],
        "publication": "not attempted; no TPA.run repository was changed",
        "model_inference": "not run",
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    sums_path.write_text(f"{manifest['package']['sha256']}  {package_name}\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True, indent=2))


def command_version(args: argparse.Namespace) -> None:
    print(json.dumps(parse_release_tag(args.tag).as_json(), sort_keys=True, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    version_parser = subparsers.add_parser("version", help="show the canonical tag mapping")
    version_parser.add_argument("--tag", required=True)
    build_parser = subparsers.add_parser("build", help="build a package from a clean Git snapshot")
    choice = build_parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--candidate-tag", help="validate a not-yet-created candidate tag against clean HEAD")
    choice.add_argument("--tag", help="build the exact clean HEAD named by an annotated local Git tag")
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument(
        "--check-reproducibility",
        action="store_true",
        help="build twice in independent temporary source/build directories and compare .deb bytes",
    )
    args = parser.parse_args()
    try:
        if args.command == "version":
            command_version(args)
        else:
            command_build(args, Path(__file__).resolve().parents[2])
    except (BuildError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"debian package build error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
