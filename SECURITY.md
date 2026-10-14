# Security and sensitive-information handling

The competition brief requires the submission to protect sensitive information
(约束 c / 3f). This prototype is deliberately **offline and self-contained**, which
removes most exposure. The points below document what the code does and the rules
to follow when running it on real (possibly export-controlled or proprietary)
electromagnetic matrices.

## What the code does

- **No run-time network access.** The solver, benchmark, and CLI perform *no*
  network calls. Every computation is local and deterministic.
- **No telemetry / analytics.** Nothing is phoned home; no usage data leaves the
  machine.
- **No hidden persistence.** The only files written are the ones you explicitly
  name: the result `.npz` (eigenvalues/eigenvectors/residuals), the diagnostics
  `.json` beside it, and any `--output` / `results/` artefact you request. The
  solver never copies the input matrix anywhere.
- **Only numeric arrays are read.** `load_matrix` accepts a SciPy sparse `.npz`
  (three arrays: `data`, `indices`, `indptr`) or a dense `.npy`. Any embedded
  metadata, comments, or auxiliary fields in a container format are ignored — they
  are never parsed, logged, or re-emitted.

## Rules when handling real matrices

1. **Keep data out of version control.** `data/raw/`, `results/`, and `*.npz` are
   `.gitignore`d. Never commit a matrix that is proprietary, export-controlled, or
   personally identifying.
2. **Run offline.** Because the tool needs no network, run it on an air-gapped or
   firewalled host when the matrices are sensitive. The only component that touches
   the network is `tools/fetch_matrices.py`, which is an *opt-in* downloader for
   **public** SuiteSparse matrices — do not point it at private sources.
3. **Minimise outputs.** Prefer writing results to a local scratch path and delete
   them after use. Eigenvectors can be as sensitive as the matrix (they reveal
   modal shapes), so treat the `.npz` output with the same classification as the
   input.
4. **Filesystem permissions.** Create output directories with restrictive
   permissions (e.g. `umask 077`) so the `.npz`/`.json` are not world-readable.
5. **Redaction / field filtering.** If a matrix container carries extra fields
   (labels, coordinates, material tags), extract only the numeric matrix into a
   plain `.npz` first; the solver then sees nothing else. There is no field that
   the solver reads besides the matrix itself.

## Supply-chain hygiene

- Dependencies are limited to NumPy and SciPy (plus pytest for tests). See
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licenses and roles.
- Pin exact versions for a frozen submission: `python -m pip freeze >
  requirements.lock.txt`. Review the resolved set before submission.
- Prefer official wheels from PyPI or a mirror you trust; verify hashes if your
  process requires it.

## Reporting

This is a competition prototype, not a hardened service. For any concern about a
specific deployment, contact the maintainers listed in the submission cover sheet
rather than opening a public issue with sensitive details.
