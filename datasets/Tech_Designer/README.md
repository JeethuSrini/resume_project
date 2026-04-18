# Tech & Designer Resumes

Two resume CSVs split by job category from the public **Kaggle Resume Dataset** (originally a single 2,484-row file covering 24 categories). We extracted just the two categories relevant to our project.

## Files

| File | Rows | Size | Category |
|---|---|---|---|
| `Resume_INFORMATION-TECHNOLOGY.csv` | 120 | 2.8 MB | INFORMATION-TECHNOLOGY |
| `Resume_DESIGNER.csv` | 107 | 2.2 MB | DESIGNER |

## Schema (4 columns)

| Column | Type | Description |
|---|---|---|
| `ID` | int | Unique resume identifier (e.g., `16852973`) |
| `Resume_str` | text | Full resume content as plain text — avg ~6,300 chars, max ~39K |
| `Resume_html` | text | Same resume rendered as styled HTML — avg ~15,700 chars, max ~62K |
| `Category` | string | Job/industry label — either `INFORMATION-TECHNOLOGY` or `DESIGNER` |

## What's in the data

- **Real resumes** (not synthetic) — raw, unstructured text scraped from public CV listings.
- **No nested fields** — everything (name, experience, education, skills) is embedded inside `Resume_str` / `Resume_html` as free-form text.
- **No PII filtering** — names, emails, phone numbers may appear; handle accordingly.
- **English only.**

## Suggested uses

- Resume **classification** (predict category from text).
- **Text extraction** experiments (parse skills, roles, education out of unstructured strings).
- Comparison/benchmark against the structured JSON dataset in `../Technical_resumes/`.

## Caveats

- Small sample size (~100 per category) — fine for prototyping, not for production training.
- HTML version contains styling/layout noise; prefer `Resume_str` for NLP tasks.
- No ground-truth structured fields — you'll need to parse or annotate yourself if you want skills/experience as separate features.

## Source

Subset of the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). Originally split from `Resume.csv` by filtering on the `Category` column.
