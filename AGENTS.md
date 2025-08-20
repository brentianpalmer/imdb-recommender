# Agents for imdb-recommender

## 0. Scope
**Purpose**: IMDb personal recommendation system with SVD collaborative filtering and ElasticNet feature engineering approaches. Learns from user ratings/watchlist to generate personalized movie/TV recommendations with scientific validation.

**Stack**: Python 3.10+, scikit-learn, pandas, numpy, typer CLI, pytest, GitHub Actions CI/CD

**Out of scope**: Real-time inference, web UI, production deployment, external API integrations, user authentication

## 1. Agents
| Agent | Mandate | Triggers | Deliverables | Autonomy | Stop |
|---|---|---|---|---|---|
| Planner | Analyze task requirements, create implementation plan with acceptance criteria | Task assignment, feature request | Plan.md (YAML schema) | Low | Plan validated & approved |
| Coder | Implement features/fixes per plan, maintain code quality standards | Plan approved, bug report | Source code + docstrings | Medium | All tests pass locally |
| Tester | Write/update tests, ensure coverage, validate functionality | New code merged, bug fix | Test files + coverage report | Medium | Coverage ≥ 85% |
| Linter | Code quality checks, style enforcement (warnings only) | Code changes pushed | Lint report in CI | Low | Report generated |
| Reviewer | Security audit, risk assessment, code review | Pull request opened | Review.md | Low | Security/performance risks documented |

## 2. Conventions
- **File naming**: snake_case for Python files, PascalCase for classes
- **Branch naming**: `feature/description`, `bugfix/description`, `hotfix/description`
- **Commit messages**: Conventional commits format (`feat:`, `fix:`, `docs:`, `test:`)
- **Documentation**: Docstrings for all public functions/classes, README updates for user-facing changes
- **Error handling**: Graceful degradation, informative error messages, proper exception types

## 3. Tools
### shell
**Use**: Package management, git operations, test execution, data processing scripts
**Examples**: 
```bash
pip install -e .
pytest tests/ -q --cov=imdb_recommender
git commit -m "feat: add ElasticNet cross-validation"
python elasticnet_recommender.py --ratings_file data/raw/ratings.csv --watchlist_file data/raw/watchlist.xlsx --topk 10
```
**Limits**: No sudo commands, no system modifications, max 30s runtime for CI commands

### file_io
**Use**: Source code editing, configuration updates, data file processing, report generation
**Examples**: Reading CSV files, updating pyproject.toml, creating test fixtures, generating documentation
**Limits**: No binary file editing, max 10MB file size, preserve file permissions

### http
**Use**: Package downloads via pip, GitHub API for CI status (read-only)
**Examples**: `pip install scikit-learn`, fetching GitHub release info
**Limits**: No external API calls requiring auth, no data uploads, PyPI/GitHub only

## 4. Context Rules
**Must read**: README.md, pyproject.toml, src test files, CI configuration, Plan.md if exists
**Ignore**: Cache directories (__pycache__, .ruff_cache), build artifacts, .git/, data/raw/ (PII)
**Priority**: Core modules (cli.py, recommender_*.py, data_io.py) > utilities > examples

## 5. I/O Contracts
### Plan.md (YAML)
```yaml
task: "string - Brief task description"
objectives: 
  - "Specific measurable goal 1"
  - "Specific measurable goal 2"
acceptance:
  - "Test case 1 passes"
  - "Performance metric X achieved"
  - "Documentation updated"
changes:
  - path: "file/path.py"
    ops: ["create", "modify", "delete"]
  - path: "tests/test_file.py" 
    ops: ["create", "modify"]
risks:
  - "Performance degradation risk"
  - "Breaking change impact"
rollback: "Specific rollback procedure"
```

### Review.md
**Sections**: 
- **Summary**: Change overview, files modified, test results
- **Diff risks**: Breaking changes, performance impacts, edge cases
- **Security**: Input validation, data handling, dependency updates
- **Open questions**: Unresolved decisions, future considerations

### Test Reports
**Location**: `/reports/coverage.xml`, `/reports/pytest.html`
**Format**: JUnit XML for CI integration, HTML for detailed review

## 6. Protocol
1. **Plan** → Create Plan.md with objectives, risks, rollback
2. **Code** → Implement changes per plan, add docstrings
3. **Test** → Write/update tests, ensure coverage ≥ 85%
4. **Lint** → Run quality checks (warnings acceptable)
5. **Full Test** → Execute complete test suite locally
6. **Review** → Generate Review.md, document risks
7. **Stop** → Merge when all quality gates pass

**Stop conditions**: Test failures, security risks unresolved, coverage below threshold
**Bail-out rules**: After 3 failed iterations, escalate to human review

## 7. Guardrails
- **Secrets**: No API keys, credentials, or PII in code. Use environment variables with .env.example template
- **Licenses**: MIT license only, no GPL dependencies, document all third-party code
- **PII**: Never log/store user data from IMDb exports, use fixtures in tests
- **Migrations**: Backwards-compatible data schema changes, migration scripts in data/migrations/

## 8. Quality Gates
- **Lint**: 0 errors required (ruff check), warnings acceptable
- **Coverage**: ≥ 85% test coverage (pytest-cov)
- **Typecheck**: Not enforced (no mypy requirement)
- **Security**: Clean bandit scan or documented waiver
- **Docs**: README.md updated for user-facing changes, docstrings for new public APIs

## 9. Error/Retry Discipline
**Backoff policy**: 1s, 5s, 15s delays for CI retries
**Bail-out rules**: Max 3 attempts for flaky tests, immediate fail for security issues
**Error categories**: 
- Transient (network, CI environment) → retry
- Logic (test failures, lint errors) → fix required
- Blocking (security, data corruption) → immediate escalation

## 10. CI Integration
**Required checks**: 
- `test` job (pytest on Python 3.12)
- `lint` job (ruff + black, continue-on-error)

**Artifacts**: 
- coverage.xml → Codecov
- Test results → GitHub Actions summary

**Branch protection**: All checks must pass for main branch

## 11. Local Runbook
```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -e .
pip install scikit-learn pytest pytest-cov ruff black

# Development
pytest tests/ -q                    # Run tests
pytest --cov=imdb_recommender     # Coverage
ruff check .                       # Lint
black .                           # Format
pre-commit run --all-files        # Full check

# Usage
imdbrec recommend --config config.toml --topk 10
python elasticnet_recommender.py --ratings_file data/raw/ratings.csv --watchlist_file data/raw/watchlist.xlsx
```

## 12. Examples
**Example A**: Add new recommendation algorithm
- Plan.md: Add SVD++ algorithm with matrix factorization improvements
- Files: `imdb_recommender/recommender_svdpp.py`, `tests/test_svdpp.py`, update CLI
- Tests: Cross-validation accuracy ≥ current SVD, performance benchmarks
- Review: Memory usage impact, backwards compatibility

**Example B**: Bug fix for missing IMDb ratings
- Plan.md: Handle null ratings in watchlist processing 
- Files: `imdb_recommender/data_io.py`, `tests/test_data_io.py`
- Tests: Edge case coverage for missing/invalid ratings
- Review: Data integrity validation, graceful degradation

## 13. Version
**Agents.md v1.0.0**, updated 2025-08-20

---

# Common failure points to avoid
- **Vague deliverables**: Fix with explicit test cases and performance metrics in Plan.md
- **Tool drift**: Pin exact commands in runbook, validate against pyproject.toml
- **Secret leakage**: Use fixtures for test data, never commit real IMDb exports
- **Non-reproducible tests**: Set random seeds, use deterministic test fixtures
- **Excess autonomy**: Require human approval for breaking changes, security issues

---

# Acceptance checklist (keep at bottom of file)
Copy/paste and use in PRs.

- [ ] Plan.md exists with measurable acceptance criteria
- [ ] All modified files have tests covering ≥ 85% of changed lines
- [ ] Lint/security checks pass locally and in CI (warnings acceptable)
- [ ] Docs updated: README usage examples, API docstrings
- [ ] Review.md present with security/performance risk assessment
- [ ] No secrets, PII, or large data files committed
- [ ] Cross-validation results documented for ML changes
