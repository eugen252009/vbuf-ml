---
date: 2026-10-10
summary: Local TPA.run APT-suite support and vBuf-ML Alpha install/upgrade passed; hosted suite support remains unverified.
---

# TPA.run APT-suite support and vBuf-ML Alpha qualification

**Date:** 2026-10-10

**Decision:** local implementation and vBuf-ML Alpha install/upgrade qualified; hosted TPA.run suite support remains unverified and is not ready for use.

**Deployment/publication:** none.

## Scope and implementation

TPA.run's local feature branch is `feature/apt-suite-support-20261011`, based on
`b615d7e0309e629784f52a718b26b578abb6c300`, with implementation commit
`ace7281506860c6677e38f57a8e484b0f75a7402`. It is committed locally and the
worktree is clean; the commit has **not** been pushed or deployed.

The implementation adds exactly four repository-scoped APT suites:
`stable`, `alpha`, `beta`, and `release`. Stable remains the omitted-field
default. Schema 19 adds a constrained `repositories.suite` column; migration
from existing schemas preserves repositories as Stable. Suite configuration is
immutable. Repository creation, API/CLI views, local/SSH publication, signing
and candidate validation, package lifecycle, browser indexes, prebuilt
manifests/client uploads, and GitHub-import publication use the configured
suite. Per-request suite overrides and cross-suite candidates are rejected.
Existing Stable prebuilt manifests remain accepted. The client fills a missing
suite only from immutable repository metadata for older TPA manifest producers;
an explicit mismatch is not rewritten.

The contract is documented in the TPA.run branch at `docs/apt-suites.md` and
`docs/prebuilt-generation-publication.md`. This is separate-repository,
one-suite-per-repository configuration; it does not copy package inventories
or change the generic vBuf wire format.

## Local verification

- `go test ./... -count=1` passed for every Go package.
- `go vet ./...` passed with no diagnostics.
- `git diff --check` and `gofmt` checks passed before the local commit.
- The opt-in `TestVBufMLAlphaAPTInstallAndUpgrade` passed against the actual
  TPA.run local publication worker and a disposable Debian 13 APT client.
  It checked the exact vBuf-ML package identities and hashes, signed Alpha
  `Release`/`InRelease` metadata, an Alpha-only `dists/alpha`, APT metadata
  refresh, installation, upgrade policy selection, upgrade, installed package
  versions, and a packaged `vbuf_compat_server --help` smoke check.

Artifacts and results:

| Package | Debian version | SHA-256 |
|---|---|---|
| `vbuf-ml_0.1.0~alpha-1_amd64.deb` | `0.1.0~alpha-1` | `ba6b5cc2f0227bf9e46535e58de075253a0975e1a3a365676c6732f9674772d2` |
| `vbuf-ml_0.1.0~alpha.2-1_amd64.deb` | `0.1.0~alpha.2-1` | `7559de429e5f40cd4a864501119f17565098c9277b8f76e794cc6a3c0017f4c7` |

The Debian 13 client image was pinned to
`debian@sha256:a99cfc517144bc59b1978475ec53b46ecabec7e43635402ee5b77cc54cd1b20a`.
APT selected `0.1.0~alpha.2-1` over the installed `0.1.0~alpha-1` candidate
from `TPA-Repo:alpha`; both installs completed and `dpkg-query` confirmed the
versions. This qualified the local server-managed signing/publication path.
The `--help` invocation is only a package smoke test, not inference or model
qualification. The test contacted its local `httptest` service, not the hosted
TPA.run deployment.

The suite/migration/API/client/SSH tests also passed as part of the complete Go
suite. The opt-in qualification and full Go test output are preserved under
[`raw/`](vbuf-ml-tparun-apt-suite-20261010/raw/).

## Hosted workflow and readiness

A read-only `GET https://tpa.run/version` on 2026-10-10 returned revision
`044e0d18a562e1d6974500bf8c957cebb9e5aba6`, version `0.1.1`, and schema `11`
(started 2026-09-26). This is not the local feature commit and gives no evidence
that the deployed service accepts or publishes non-Stable suites. The separate
LAN Beta `InRelease` is not evidence of TPA.run multi-suite support.

The inspected local deployment hook, `deploy/deploy-tpahub`, takes an exact
`refs/heads/staging` commit. It currently guards the live baseline at Schema 11,
backs up the database and state, stops the service, and requires
`TPARUN_ALLOW_SCHEMA_MIGRATION=true` for a schema transition before running the
candidate's explicit `tparun migrate`; health and public-artifact checks follow,
with rollback handling. The hook's public-artifact probe checks the existing
Stable path. No GitHub Actions deployment workflow was present in this source
checkout. The hook was inspected only; it was not invoked. The feature branch
was not pushed to `staging`, and neither a production migration nor a hosted
Alpha install was attempted.

**Readiness distinction:**

1. **Local code:** implementation committed and Go-qualified.
2. **vBuf-ML Alpha package path:** signed local APT install and upgrade passed
   with the two exact, previously qualified package artifacts.
3. **Hosted TPA.run:** **not ready / unverified**. The deployed revision and
   Schema 11 marker predate this feature. Deployment, Schema 11-to-19 migration,
   Stable regression checks, and a hosted signed Alpha APT-client qualification
   require a separate authorized rollout and verification.

No vBuf-ML package was published, no release tag was created, and no TPA.run
push or deployment occurred. This work does not promote Qwen3-8B or qualify
inference.

## Qualification boundary and remaining work

The local signed Debian 13 install/upgrade result is measured evidence for the
local server-managed path only. It does not establish hosted availability,
production migration safety, Beta/Release end-to-end APT behavior, or a full
client-signed Alpha generation publication. Prebuilt suite binding, client
manifest normalization, suite-aware validation, SSH propagation, migration,
and candidate isolation have unit/contract coverage, but client-signed Alpha
publication through APT remains a separate qualification item.

Before any hosted use, review and authorize the deployment plan; take and verify
a coherent production backup; deploy the exact feature commit through the
staging-based workflow; explicitly qualify the schema migration and Stable
regression; then test signed Alpha metadata and install/upgrade using the hosted
service. Until those steps occur, consumers should not be told that hosted
TPA.run supports these suites.
