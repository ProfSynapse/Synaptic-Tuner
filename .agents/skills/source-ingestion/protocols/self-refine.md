# Protocol: self-refine

Context: run after a session that actually exercised this skill and revealed
evidence about what helped or caused friction.

## Mission

Apply one small, evidence-backed improvement and record it without expanding the
skill spec or recipe catalog speculatively.

## Steps

1. Review the session for one concrete success, friction point, or user
   correction.
2. Ask for user feedback when available; their correction outranks an inferred
   preference.
3. Choose the smallest durable change within the approved V1 scope. Do not add a
   recipe until a concrete use case is separately aligned and approved.
4. Edit the focused protocol, reference, template, or validator; keep
   `SKILL.md` a slim router.
5. Add a newest-first entry to `../references/refinement-log.md` with date,
   evidence, change, and files touched. If no change is justified, record that.
6. Run the checks in `validate-and-sync.md`. A refinement is incomplete until
   canonical and mirror trees agree.
7. Stop after reporting the evidence, the small change, and validation result.

## Guidelines

- Pattern: prefer a sharper existing instruction or stable mechanical check.
- Anti-pattern: use self-refinement to bypass user approval for a new format,
  recipe, runtime capability, or source mutation.

## Next

Run `validate-and-sync.md`; this project-local skill is never packaged as a
standalone `.skill` artifact.
