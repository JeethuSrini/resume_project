"""
For each resume file, creates a new dataset where every resume version
(original + paraphrase_1..4) is paired with a name drawn randomly from
each race/gender subgroup in race_gender_names.json.

Within each (resume_id, subgroup) block:
  - All 5 names in the subgroup are used exactly once.
  - Names are assigned to versions (original, paraphrase_1…4) at random.

Output: one JSON file per input resume file, saved alongside the inputs.
Each output is a list of flat records:
  {resume_id, category, race, gender, name, version, text}
"""

import json
import random
from pathlib import Path

SEED = 42  # set to None for non-reproducible randomness

DATA_DIR = Path(__file__).parent
NAMES_FILE = DATA_DIR / "race_gender_names.json"

INPUT_FILES = [
    DATA_DIR / "designer_resume_paraphrases_sample_10.json",
    DATA_DIR / "it_resume_paraphrases_sample_10.json",
]

VERSIONS = ["original", "paraphrase_1", "paraphrase_2", "paraphrase_3", "paraphrase_4"]


def assign_names(resumes: list[dict], subgroups: dict, rng: random.Random) -> list[dict]:
    records = []
    for resume in resumes:
        resume_id = resume["resume_id"]
        category = resume["category"]
        for race, genders in subgroups.items():
            for gender, names in genders.items():
                if len(names) != len(VERSIONS):
                    raise ValueError(
                        f"Subgroup {race}/{gender} has {len(names)} names "
                        f"but there are {len(VERSIONS)} versions."
                    )
                shuffled = names.copy()
                rng.shuffle(shuffled)
                for version, name in zip(VERSIONS, shuffled):
                    records.append(
                        {
                            "resume_id": resume_id,
                            "category": category,
                            "race": race,
                            "gender": gender,
                            "name": name,
                            "version": version,
                            "text": resume[version],
                        }
                    )
    return records


def main():
    rng = random.Random(SEED)

    with open(NAMES_FILE) as f:
        subgroups = json.load(f)

    for input_path in INPUT_FILES:
        with open(input_path) as f:
            resumes = json.load(f)

        records = assign_names(resumes, subgroups, rng)

        stem = input_path.stem  # e.g. "designer_resume_paraphrases_sample_10"
        output_path = DATA_DIR / f"{stem}_named.json"
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

        n_resumes = len(resumes)
        n_subgroups = sum(len(g) for g in subgroups.values())
        print(
            f"{input_path.name} → {output_path.name} "
            f"({n_resumes} resumes × {n_subgroups} subgroups × {len(VERSIONS)} names "
            f"= {len(records)} records)"
        )


if __name__ == "__main__":
    main()
