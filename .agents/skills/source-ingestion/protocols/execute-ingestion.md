# Protocol: execute ingestion

Context: the strict config validates and the user-authorized local selections
are ready.

## Mission

Run one bounded ingestion through a supported public surface and retain only
sanitized lifecycle evidence.

## Steps

1. Read `../references/supported-surfaces.md` and
   `../references/lifecycle-and-diagnostics.md`.
2. Choose a private project or standalone engine root. Ensure the output root is
   ignored or otherwise protected before processing private source material.
3. Run the checked-in CLI from the repository root:

   ```text
   python tuner.py ingest --config CONFIG --select ALIAS=PATH [--select ALIAS=PATH ...] --json
   ```

4. Capture stdout as exactly one JSON document and keep stderr empty. Do not add
   debug printing, parser exception prose, absolute paths, or source excerpts.
5. Run
   `python .skills/source-ingestion/scripts/validate_ingestion_result.py RESULT --require-verified [--expected-source-count N]`.
6. If the command or validator does not report a verified result, stop the
   success path and run `diagnose-ingestion.md`. Do not bypass preflight or call
   private runtime modules.
7. Stop when the public result is mechanically valid and reports `verified`.

## Guidelines

- Pattern: preserve the emitted IDs and digests as evidence while keeping
  private content and paths out of the record.
- Anti-pattern: interpret a zero process exit without validating the JSON
  result, or treat a bundle path as proof of verification.

## Next

On a verified result, run `verify-and-handoff.md`. On any other result, run
`diagnose-ingestion.md` first.
