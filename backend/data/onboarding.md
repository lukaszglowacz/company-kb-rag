# Onboarding Guide — NovaTech Solutions

## Welcome

Welcome to NovaTech! This guide covers everything you need for your first two weeks.
If anything is unclear, ask your buddy — every new hire is assigned one on Day 1.

## Day 1 Checklist

- [ ] Collect laptop from IT (Room 102, Warsaw HQ or via courier for remote)
- [ ] Set up 1Password (company password manager) — IT will send an invite
- [ ] Activate Google Workspace account (email, Calendar, Drive)
- [ ] Join Slack — key channels: `#general`, `#engineering`, `#deployments`, `#random`
- [ ] Book intro calls with your team lead and buddy (use Calendly links in their Slack profiles)
- [ ] Read this guide fully

## Access & Tools

| Tool | Purpose | How to get access |
|---|---|---|
| GitHub | Code repositories | Ask IT, added to `novatech-solutions` org |
| Linear | Issue tracker | Auto-provisioned with Google login |
| Notion | Internal documentation | Auto-provisioned with Google login |
| Datadog | Monitoring & logs | Ask your team lead |
| AWS Console | Infrastructure (engineers only) | Requires separate approval from CTO |
| Figma | Design files | Ask the Product team |

## Development Environment Setup

```bash
# 1. Clone the main monorepo
git clone https://github.com/novatech-solutions/platform.git
cd platform

# 2. Install dependencies
cp .env.example .env.local   # fill in values from 1Password vault "Dev Secrets"
docker-compose up -d         # starts PostgreSQL, Redis, and all services

# 3. Verify everything works
curl http://localhost:8000/health   # should return {"status": "ok"}
open http://localhost:3000          # frontend dev server
```

Detailed setup docs are in Notion → Engineering → Local Development Setup.

## First Two Weeks

### Week 1
- Shadow your buddy on their daily workflow
- Complete the LogiCore onboarding walkthrough in the staging environment
- Read the Architecture Decision Records (ADRs) in Notion
- Attend the weekly Engineering All-Hands (Tuesdays 10:00 CET)

### Week 2
- Pick up your first ticket from the `good first issue` column in Linear
- Pair with your buddy on the first PR
- Attend your first Sprint Review

## Communication

- **Slack** — default async channel. Respond within 4 hours during working hours.
- **Email** — for external communication and formal matters only.
- **Linear** — all engineering tasks tracked here. Update ticket status daily.
- **1:1s** — weekly with your team lead. Agenda lives in Notion.

## Working Hours & Remote Policy

- Core hours: 10:00–15:00 CET (overlap required for meetings)
- Outside core hours: flexible
- Remote-first: full remote allowed. Office days not mandatory, but Warsaw and Kraków
  offices are open Monday–Friday.
- Announce planned days off in `#general` and block your calendar.

## Probation Period

- Duration: 3 months
- Mid-probation check-in at 6 weeks with your team lead
- End-of-probation review — covers technical skills, communication, and team fit

## Who to Ask

| Topic | Person / Channel |
|---|---|
| HR, contracts, benefits | hr@novatech.io |
| IT equipment, accounts | it@novatech.io or `#it-support` on Slack |
| Engineering questions | Your buddy or `#engineering` |
| Product questions | Your team's PM |
| Urgent production issues | `#incidents` on Slack, then page on-call via PagerDuty |
