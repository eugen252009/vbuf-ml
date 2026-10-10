---
date: 2026-10-10
summary: First reproducible Debian 13 Alpha package passed local build, package, TPA signature, and APT lifecycle checks; hosted publication remains blocked.
---

# Debian Alpha package: local build and lifecycle qualification

**Status:** qualified for the narrow package/install scope below; unpublished.
This is not model-inference, coding-agent, or backend qualification.

## Scope and identity

The package bundles the existing CPU-only `integrations/ggml` target
`vbuf_compat_server` with its private Rust `vbuf-ml` consumer cdylib. It is
Debian 13 (Trixie), `amd64` only. No model, payload, system service, persistent
application state, CUDA path, or new inference behavior is included. The
separate `vbuf-runtime` cdylib is not linked into this target.

The build used clean candidate source commit
`2c8dc80288f3fcfbb53ce905d62cdc0ae1c38f3c` (tree
`7ecfbab7d5eac762ba4143efda4e2d5730d371ea`). Candidate mode validated the
prospective identity `alpha-0.1.0`; **no Git tag was created**. The immutable
GGML commit was `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, with the checked-in
native-attention patch and Rust `Cargo.lock` hashes recorded in the package
manifest. CUDA, CPU repacking, OpenMP, BLAS, native CPU tuning, and qualification
fault injection were disabled. Rust used the x86-64 baseline and pinned 1.95.0
toolchain. The full builder image and tool versions are in
[`raw/environment.txt`](vbuf-ml-debian-alpha-qualification-20261010/raw/environment.txt).

## Build and package checks

Two builds from independent temporary source and build directories produced
byte-identical `.deb` files for `0.1.0~alpha-1`:

- Package SHA-256:
  `ba6b5cc2f0227bf9e46535e58de075253a0975e1a3a365676c6732f9674772d2`
- Size: 1,608,876 bytes
- APT suite: `alpha`; package architecture: `amd64`
- Source archive SHA-256:
  `c3b4b02d116e898a8656334a0598eed6475ce9d9b1a4c9fab84be3150ec9d8d4`

The package metadata resolved runtime dependencies to `libc6 (>= 2.38)`,
`libgcc-s1 (>= 4.2)`, and `libstdc++6 (>= 14)`. The package verifier checked
the metadata and payload inventory, no maintainer scripts, the private
`$ORIGIN/lib` RUNPATH, resolved shared libraries, absence of build/workspace
paths, and the embedded source provenance. The only runtime smoke was
`vbuf_compat_server --help`; it loaded no model.

A second candidate version, `alpha-0.1.0.2` → `0.1.0~alpha.2-1`, was built once
for upgrade testing (not as a second reproducibility claim). Its package
SHA-256 is
`7559de429e5f40cd4a864501119f17565098c9277b8f76e794cc6a3c0017f4c7`.
Both complete package build manifests, checksum files, build logs, and package
metadata checks are preserved under [`raw/`](vbuf-ml-debian-alpha-qualification-20261010/raw/).
The `.deb` artifacts remain local under `/tmp`; they are not committed.

`lintian --fail-on error` passed for both packages. Lintian emitted the expected
initial-upload-without-bug-reference warning and a `no-manual-page` warning;
its invocation as root also emitted its root-privilege warning. Build logs retain
compiler, Python tar-extraction deprecation, CMake, `dwz`, and dpkg-shlibdeps
warnings; none prevented package creation or verification. The detailed
iteration failures and their fixes are recorded in
[`raw/iteration-failures.txt`](vbuf-ml-debian-alpha-qualification-20261010/raw/iteration-failures.txt).
Committed text logs normalize carriage-return progress updates to line endings
and trim trailing whitespace; original container captures remain local outside
the repository.

The five package-version tests passed, workflow YAML passed `actionlint`, and
the lifecycle script passed `sh -n`. The workflow's build, lintian, and package
lifecycle steps were exercised locally in the same pinned Debian image. No
hosted GitHub Actions run was triggered because no push was made.

## APT package lifecycle

The Debian APT lifecycle smoke test passed in the Trixie `amd64` container. It
installed `0.1.0~alpha-1` from a local `.deb`, reinstalled it, upgraded to
`0.1.0~alpha.2-1`, removed it, reinstalled the newer version, and purged it.
It checked installed versions, executable `--help`, and removal of package
files. No model inference was performed.

## Local TPA and signed APT integration

The locally built standalone TPA binary was from source commit
`ddcdf2ec9f7e15d985393b20ca8b493e358b5775`, SHA-256
`ab6ca5ca05d4cf70f89dd7d80bcdf5a065d5d3b4306c45550d43ed9aabf59219`.
Its capability record reported repository format `tpa-apt-repository` v1 and
`repository.pack.v1` / `repository.verify.v1` support. TPA packed the two local
debs into a fresh, locally signed `alpha` / `amd64` / `main` APT repository.
Independent TPA verification returned `valid=true`, `signed=true`,
`signatureVerified=true`, with both package identities and artifact hashes
matching the package manifests. The ephemeral local signing-key fingerprint
was `B6382752339ABB717A3E6F681B8D02EFD5D61D57`; its private key was not
committed or used outside this local test.

A disposable Debian APT client trusted only the exported test public key through
`Signed-By` and `file:/out/local-apt`. `apt-get update` accepted the signed
`alpha` Release metadata, selected `0.1.0~alpha.2-1` as the candidate, installed
the older version, and upgraded it to the newer version. The client then purged
the package and removed the temporary source and keyring. This establishes the
local standalone TPA/APT contract only, not hosted repository behavior.

## Hosted publication and qualification boundaries

No package was uploaded, no repository was created on TPA.run, and no hosted key
or public repository URL was selected. The checked-out TPA.run source examined
at `b615d7e0309e629784f52a718b26b578abb6c300` has a candidate verifier requiring
`dists/stable`, `Suite: stable`, and `Codename: stable`; this does not establish
the deployed service's behavior. Hosted `alpha`, `beta`, and `release` suite
support remains unverified and blocked. The local standalone TPA's custom-suite
support is not evidence of hosted TPA.run support.

The package lifecycle and `--help` checks do not qualify any model, numerical
contract, generation, CUDA, Qwen3-8B coding-agent behavior, or Qwen3-14B Native
AV. Existing experimental and fail-closed model-admission boundaries remain
unchanged. The stopped Native AV matrix was not run or backfilled.
