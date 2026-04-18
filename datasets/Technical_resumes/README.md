---
license: mit
language: en
task_categories:
- token-classification
tags:
- resumes
- NLP
- synthetic
- real-world
- recruitment
- job matching
pretty_name: "Master Resumes"
size_categories:
- 1K<n<10K
---

# Technical Resumes (Master Resumes)

A merged collection of **real (anonymized) + synthetic** resumes for technical roles, normalized into a single nested JSON schema. Designed for training NLP models on resume parsing and candidate–job matching.

## File

| File | Records | Size | Format |
|---|---|---|---|
| `master_resumes.jsonl` | 4,817 | 16 MB | JSON Lines (one resume per line) |

## Schema (per resume)

Each line is a JSON object with these top-level sections:

| Section | Description |
|---|---|
| `personal_info` | `name`, `email`, `phone`, `location` (city/country/remote_preference), `summary`, `linkedin`, `github` |
| `experience[]` | List of jobs — each with `company`, `company_info` (industry, size), `title`, `level`, `employment_type`, `dates` (start/end/duration), `responsibilities[]`, `technical_environment` (technologies, methodologies, tools) |
| `education[]` | List of degrees — each with `degree` (level/field/major), `institution` (name/location/accreditation), `dates`, `achievements` (gpa, honors, coursework) |
| `skills.technical` | Categorized skills with proficiency levels: `programming_languages`, `frameworks`, `databases`, `cloud` |
| `skills.languages` | Spoken languages with fluency level |
| `projects[]` | List of projects — each with `name`, `description`, `technologies[]`, `role`, `url`, `impact` |
| `certifications` | Certifications (often empty in real resumes) |

## Data composition

- **Real resumes:** anonymized CV submissions, normalized into the schema above. PII has been stripped — many fields appear as `"Unknown"` or `"Not Provided"`.
- **Synthetic resumes:** generated with Python's `Faker` library using role-specific templates (Java Developer, Data Scientist, DevOps Engineer, etc.).
- **Domain:** ~97% Technology — 79 unique tech job titles, fairly evenly distributed (~200 each).
- **Career length:** 1–4 jobs per resume (avg 2) — uniformly distributed, reflecting synthetic generation.

## Suggested uses

- **Resume parsing models** — train extractors that pull structured fields from unstructured CVs.
- **Candidate–job matching** — embed resume + job description, score similarity.
- **Data augmentation** — supplement smaller real datasets with the synthetic portion.
- **EDA** — analyze skill, experience, and education patterns across tech roles.

## Caveats

- **Tech-only** — won't generalize to non-technical industries (use `../Tech_Designer/` for designer category).
- **Sparse fields** — certifications, GPAs, project URLs often empty.
- **Real vs. synthetic mix** is not labeled per record — keep this in mind when validating models.
- **Short career histories** (max 4 jobs) — not representative of senior/long-tenure candidates.
- Synthetic names/emails are Faker placeholders, not real PII.

## Source

Originally published on HuggingFace by `datasetmaster`. License: MIT.

## Quick start

```python
import json

with open('master_resumes.jsonl') as f:
    resumes = [json.loads(line) for line in f]

print(f"Loaded {len(resumes)} resumes")
print(resumes[0]['personal_info'])
```
