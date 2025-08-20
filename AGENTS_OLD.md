# Agents for <repo-name>

## 0. Scope
Purpose: …
Stack: …
Out of scope: …

## 1. Agents
| Agent | Mandate | Triggers | Deliverables | Autonomy | Stop |
|---|---|---|---|---|---|
| Planner | … | … | Plan.md | Low | Plan created |
| Coder | … | … | Code + Docs | Medium | Tests pass |
| Tester | … | … | Tests + Report | Medium | Coverage ≥ X% |
| Linter | … | … | Lint report | Low | 0 errors |
| Reviewer | … | … | Review.md | Low | Risks resolved |

## 2. Conventions
- …

## 3. Tools
### shell
Use, examples, limits…
### file_io
…
### http
…

## 4. Context Rules
Must read…, ignore…

## 5. I/O Contracts
### Plan.md (YAML)
```yaml
task: ""
objectives: []
acceptance: []
changes: []
risks: []
rollback: ""

Review.md

Sections: Summary, Risks, Security, Open Questions.

6. Protocol
	1.	Plan → 2) Code → 3) Tests → 4) Lint → 5) Test → 6) Review → 7) Stop.

7. Guardrails

Secrets, licenses, PII, migrations.

8. Quality Gates

Lint=0; Coverage ≥ X; Typecheck=0; Security clean; Docs updated.

9. Error/Retry Discipline

Backoff policy, bail-out rules.

10. CI Integration

Artifacts; required checks.

11. Local Runbook

Setup, lint, test, typecheck commands.

12. Examples
	•	Example A: <one paragraph + file list>
	•	Example B: <…>

13. Version

Agents.md v, updated .

---

# Common failure points to avoid
- Vague deliverables. Fix with explicit file names and schemas.
- Tool drift. Fix with a pinned runbook and example invocations.
- Secret leakage. Fix with `.env.example` and hard “no inline secrets” rule.
- Non-reproducible tests. Fix with seed setting and deterministic fixtures.
- Excess autonomy. Keep agents narrow; require stop conditions.

---

# Acceptance checklist (keep at bottom of file)
Copy/paste and use in PRs.

- [ ] Plan.md exists and lists acceptance criteria.
- [ ] All modified files have tests covering changed lines.
- [ ] Lint/type/security checks pass locally and in CI.
- [ ] Docs updated: README usage, examples runnable.
- [ ] Review.md present with risks and mitigations.
- [ ] No secrets, no license violations, no large context dumps.

---
