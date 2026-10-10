# Debian Alpha package

The first package is a narrow CPU-only `amd64` distribution of the existing
`integrations/ggml` `vbuf_compat_server` target and its required private
`vbuf-ml` Rust consumer library. It is not a generic development SDK and does
not replace the canonical vBuf-ML source/materialization/runtime boundaries.
It does not link llama.cpp, install a service, open a listener during package
installation, or include model weights, model payloads, credentials, or a
signing key. The separate Rust `vbuf-runtime` cdylib is not linked by this
server target and is not included.

## Release identity and Debian ordering

Only these canonical Git tag forms are package releases:

| Git tag | Debian version | Intended APT suite |
|---|---|---|
| `alpha-X.Y.Z` | `X.Y.Z~alpha-1` | `alpha` |
| `alpha-X.Y.Z.N` (`N >= 2`) | `X.Y.Z~alpha.N-1` | `alpha` |
| `beta-X.Y.Z` | `X.Y.Z~beta-1` | `beta` |
| `beta-X.Y.Z.N` (`N >= 2`) | `X.Y.Z~beta.N-1` | `beta` |
| `release-X.Y.Z` | `X.Y.Z-1` | `release` |
| `release-X.Y.Z-rN` (`N >= 2`) | `X.Y.Z-N` | `release` |

Version components are canonical decimal numbers (no leading zeros). Suffix
`.1`/`-r1` is rejected because the unsuffixed form already means revision 1.
Other and historical tags do not produce packages. Debian ordering is checked
with `dpkg --compare-versions`; for one upstream version Alpha sorts before
Beta, which sorts before Release, and increasing sequence/revision numbers sort
upward within a channel. APT still compares versions, not channel intent: a
newer Alpha version can outrank an older Release if both suites are configured.

The mapping tool is read-only:

```sh
python3 packaging/debian/build.py version --tag alpha-0.1.0
python3 packaging/debian/build.py version --tag beta-0.1.0.2
python3 packaging/debian/build.py version --tag release-0.1.0
```

The build tool rejects dirty source, noncanonical or colliding managed tags,
wrong repository identity, non-amd64 builders, mutable/unexpected dependency
revisions, and attempts to overwrite an existing artifact. Candidate mode
builds clean `HEAD` without creating a Git tag. Release mode requires an
annotated local tag whose commit is exactly `HEAD`. The tool never creates,
rewrites, or pushes tags and never publishes packages.

## Build and provenance

The first qualification target is Debian 13 (Trixie), `amd64`, CPU-only. The
CMake target uses the immutable GGML commit in
`integrations/ggml/CMakeLists.txt`, the checked-in `rust/Cargo.lock`, and the
pinned Rust 1.95.0 toolchain. Build configuration disables CUDA, CPU repacking,
OpenMP, BLAS, native-CPU tuning, and qualification fault injection. Rust is
compiled for the x86-64 baseline, not `target-cpu=native`.

Install Debian build prerequisites (`build-essential`, `ca-certificates`,
`cargo`, `cmake`, `debhelper`, `dpkg-dev`, `git`, `pkg-config`, `python3`, and
`rustc`) and make the
pinned Rust toolchain available first in `PATH`:

```sh
rustup toolchain install 1.95.0 --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version  # must report 1.95.0
cargo --version  # must report 1.95.0

python3 packaging/debian/build.py build \
  --candidate-tag alpha-0.1.0 \
  --output /tmp/vbuf-ml-debs \
  --check-reproducibility
```

The build extracts an allow-listed source archive from the exact Git commit,
fetches the pinned GGML commit and Cargo.lock dependencies, then performs the
Debian build offline. `SOURCE_DATE_EPOCH` is the commit timestamp. The
reproducibility option builds twice from independent temporary source/build
directories and requires byte-identical `.deb` files. The adjacent JSON
manifest records the Git commit/tree/tag, source-archive and package SHA-256,
Debian version/suite/architecture, GGML and patch revisions, Cargo.lock hash,
compiler/build-tool versions, effective build configuration, and qualification
boundary. It contains no build-host home-directory paths.

Run the version tests with:

```sh
python3 -m unittest discover -s scripts/tests -p 'test_debian_package_versions.py' -v
```

## Install and run

A locally built package can be installed without configuring an APT repository:

```sh
sudo apt install ./vbuf-ml_0.1.0~alpha-1_amd64.deb
vbuf_compat_server --help
```

The command requires a semantic model bootstrap and an external payload source
at run time; neither is supplied by the package. The server defaults to
`127.0.0.1:8080` and does not provide authentication. Keep it loopback-bound
unless an independently secured network boundary is provided. Installing the
package does not start it. There is no systemd unit or package-owned persistent
model/application state. Removal and purge remove package files only.

The current executable is a thin OpenAI-compatible subset, not universal model
support. This package build only smoke-tests `--help`; it does not run model
inference, generation, parity, or performance qualification. Qwen3-8B remains
identity-gated and experimental, and is not qualified as a coding agent.
Qwen3-14B Native AV remains `Candidate_NOT_Valid`; no stopped qualification
matrix is changed or backfilled. No Numerical Contract, execution default,
model admission rule, or inference path is changed by this packaging work.

## TPA.run / APT channel status

The intended repository mapping is suite/codename `alpha`, `beta`, and
`release`, component `main`, architecture `amd64`. This is a packaging contract,
not evidence that the hosted TPA.run service accepts those suites.

The inspected TPA.run CLI has no safe publication dry-run: `publish` starts a
remote publication session before its interactive confirmation and can upload
and finalize artifacts. It was not invoked. The checked-out TPA.run source at
`b615d7e0309e629784f52a718b26b578abb6c300` validates a candidate specifically
under `dists/stable` and requires `Suite: stable` and `Codename: stable`.
TPA's own integration contract says compatibility with any deployed TPA.run
build is not verified. Therefore support for the proposed three hosted suites
is **unverified and currently blocked**; the local source check must not be
mistaken for a claim about the live service.

No TPA.run repository was created or selected, no public repository URL or
signing key for this project was verified, and no APT source was installed.
When the hosted service has a separately verified multi-suite contract and an
operator provisions a repository and key, the intended deb822 source will be:

```deb822
Types: deb
URIs: <verified TPA.run repository URL>
Suites: alpha
Components: main
Architectures: amd64
Signed-By: <verified repository keyring path>
```

This is deliberately a non-installable template until the URL, suite support,
repository key, and signing policy are verified. Do not substitute a guessed
`/r/...` URL or a generic TPA.run key. The corresponding `beta` and `release`
source stanzas should be added only after the same checks. No Git tag triggers
TPA.run or GitHub publication.

The standalone TPA repository builder supports suite/codename selection and
can validate a local offline candidate tree. That validates local APT metadata
only; it does not qualify TPA.run authorization, hosted candidate validation,
signing, publication, or activation.
