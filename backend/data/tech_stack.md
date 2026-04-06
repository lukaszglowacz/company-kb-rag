# Tech Stack & Engineering Standards — NovaTech Solutions

## Backend

| Technology | Version | Usage |
|---|---|---|
| Python | 3.11 | Primary backend language |
| FastAPI | 0.110+ | REST API framework |
| PostgreSQL | 15 | Primary relational database |
| Redis | 7 | Caching and background job queues |
| Celery | 5 | Async task processing |
| SQLAlchemy | 2.0 | ORM |

## Frontend

| Technology | Version | Usage |
|---|---|---|
| TypeScript | 5 | Primary frontend language (strict mode enforced) |
| Next.js | 14 | Frontend framework (App Router) |
| Tailwind CSS | 3 | Styling |
| Vitest | 1 | Unit and integration tests |

## Infrastructure

| Technology | Usage |
|---|---|
| AWS | Primary cloud provider |
| ECS Fargate | Container orchestration |
| RDS | Managed PostgreSQL |
| ElastiCache | Managed Redis |
| CloudFront | CDN |
| GitHub Actions | CI/CD pipeline |
| Docker Compose | Local development |
| Terraform | Infrastructure as code |

## Code Standards

### Python
- Formatting: `black` (line length 88)
- Linting: `flake8`
- Type checking: `mypy --strict` (required for all modules)
- Tests: `pytest` with minimum 80% coverage per module
- Docstrings: required for public functions and classes

### TypeScript
- Strict mode enabled in `tsconfig.json`
- ESLint with Next.js recommended rules, zero warnings policy
- All API responses must be typed — no `any`
- Tests: Vitest + @testing-library/react for hooks and components

## Git Workflow

- Branch naming: `feat/`, `fix/`, `chore/`, `docs/`
- All changes via Pull Requests — no direct pushes to `main`
- CI must pass before merge
- At least one approval required

## Security Standards

- API keys stored in environment variables only — never committed to git
- `.env` files excluded from version control via `.gitignore`
- Dependencies audited monthly via `npm audit` and `pip-audit`
- All HTTP endpoints require input validation via Pydantic (backend) or Zod (frontend)
