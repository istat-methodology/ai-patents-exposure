from openai import OpenAI
import pandas as pd
import json
import os
from dotenv import load_dotenv
from time import sleep

# -----------------------------------------------------------
# Config: control output size/cost
# -----------------------------------------------------------
INCLUDE_JUSTIFICATIONS = False  # ← toggle this to True when you want explanations
MAX_SECONDARIES = 3  # cap secondary suggestions to keep JSON small

MODEL = "gpt-5-mini"  # Currently available models "gpt-5", "gpt-5-mini", "gpt-5-nano"
SAMPLE_SIZE = None  # For quick testing, set to None to use the full dataset

# -----------------------------------------------------------
# File paths and configuration
# -----------------------------------------------------------
ISCO_PATH = "resources/classification/ISCO-08_structure_and_definitions.xlsx"
PATENTS_SAMPLE_PATH = "sample/patents_sample_part3.xlsx"
OUTPUT_PATH = "output/patents_classified_chatgpt_part3.csv"


# -----------------------------------------------------------
# Debug printing
# -----------------------------------------------------------
PRINT_DEBUG = False  # master switch
PRINT_PROMPT = True  # show prompt sent to the model
PRINT_OUTPUT = True  # show raw JSON output from the model
PRINT_TRUNCATE = 1200  # max chars to print for prompt/output (None = full)


def _truncate(s: str, n: int | None) -> str:
    if s is None or n is None or len(s) <= n:
        return s
    return s[:n] + f"... [truncated {len(s)-n} chars]"


# -----------------------------------------------------------
# Load ISCO classification files
# -----------------------------------------------------------


def load_isco_sub_major(path: str) -> pd.DataFrame:
    """Load ISCO-08 Sub-Major Groups (Level 2) from Excel."""
    df = pd.read_excel(path, sheet_name="ISCO-08 EN Struct and defin")
    sub_major = df[df["Level"] == 2][["ISCO 08 Code", "Title EN"]]
    sub_major.columns = ["Code", "Title"]
    return sub_major


def load_isco_minor(path: str) -> pd.DataFrame:
    """Load ISCO-08 Minor Groups (Level 3) from Excel."""
    df = pd.read_excel(path, sheet_name="ISCO-08 EN Struct and defin")
    sub_major = df[df["Level"] == 3][["ISCO 08 Code", "Title EN"]]
    sub_major.columns = ["Code", "Title"]
    return sub_major


# -----------------------------------------------------------
# Prompt generation for JSON output (justifications optional)
# -----------------------------------------------------------


def generate_prompt_json(
    title: str,
    abstract: str,
    isco_df: pd.DataFrame,
    include_justifications: bool,
    max_secondaries: int | None = None,
) -> str:
    """Create a JSON-friendly prompt for the OpenAI API."""
    entries = [
        {"code": str(row.Code), "title": row.Title} for _, row in isco_df.iterrows()
    ]

    secondary_codes = ""
    if max_secondaries is not None and max_secondaries > 0:
        secondary_codes = f"(at most {max_secondaries}, ordered by relevance)"
    else:
        secondary_codes = "(if any)"

    if include_justifications:
        fields = f"""
Return ONLY a JSON object with the following fields:
- primary_code: string, the best matching Minor Group code
- primary_justification: short string explaining the choice
- secondary_codes: array of strings {secondary_codes}
- secondary_justifications: object mapping each secondary code to a short justification
"""
    else:
        fields = f"""
Return ONLY a JSON object with the following fields:
- primary_code: string, the best matching Minor Group code
- secondary_codes: array of strings {secondary_codes}
Do NOT include any justification fields. Keep the JSON minimal.
"""

    return f"""
You are a labor market expert. Given the following patent abstract, classify it into the most appropriate ISCO-08 Minor Group.

Patent title:
\"\"\"{title}\"\"\"

Patent abstract:
\"\"\"{abstract}\"\"\"

List of available ISCO-08 Minor Groups:
{json.dumps(entries, ensure_ascii=False)}
{fields}
"""


# -----------------------------------------------------------
# Validate model output against ISCO list
# -----------------------------------------------------------


def validate_codes(
    classification: dict, valid_codes: set, include_justifications: bool
) -> dict:
    """
    Check if primary and secondary codes exist in the ISCO list.
    If justifications are disabled, do not add any justification fields.
    """
    # Primary
    if classification.get("primary_code") not in valid_codes:
        classification["primary_code"] = "INVALID"
        if include_justifications:
            classification["primary_justification"] = (
                classification.get("primary_justification", "")
                + " [⚠ Invalid ISCO code suggested]"
            )
        else:
            # ensure we don't accidentally keep a justification
            classification.pop("primary_justification", None)

    # Secondary
    valid_secondary = []
    updated_justifications = {}
    for code in classification.get("secondary_codes", []):
        if code not in valid_codes:
            valid_secondary.append("INVALID")
            if include_justifications:
                updated_justifications["INVALID"] = (
                    classification.get("secondary_justifications", {}).get(code, "")
                    + " [⚠ Invalid ISCO code suggested]"
                )
        else:
            valid_secondary.append(code)
            if include_justifications:
                updated_justifications[code] = classification.get(
                    "secondary_justifications", {}
                ).get(code, "")

    classification["secondary_codes"] = valid_secondary

    # Only attach secondary_justifications if requested
    if include_justifications:
        classification["secondary_justifications"] = updated_justifications
    else:
        classification.pop("secondary_justifications", None)

    # If justifications are disabled, also strip any stray fields
    if not include_justifications:
        classification.pop("primary_justification", None)

    return classification


# -----------------------------------------------------------
# Classification function using OpenAI API with enforced JSON output
# -----------------------------------------------------------


def classify_abstract_json(
    client,
    title,
    abstract,
    isco_df,
    model="gpt-5",
    include_justifications=True,
    max_secondaries: int | None = None,
    debug_label: str = "",
):
    """Send patent abstract to OpenAI API for classification (with optional prints)."""
    prompt = generate_prompt_json(
        title, abstract, isco_df, include_justifications, max_secondaries
    )

    if PRINT_DEBUG and PRINT_PROMPT:
        print(f"\n===== PROMPT {debug_label} =====")
        print(_truncate(prompt, PRINT_TRUNCATE))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content

        if PRINT_DEBUG and PRINT_OUTPUT:
            print(f"----- OUTPUT {debug_label} -----")
            print(_truncate(content, PRINT_TRUNCATE))

        return json.loads(content)

    except Exception as e:
        print(f"❌ OpenAI API error {debug_label}: {e}")
        return {}


# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load ISCO codes and patents
isco_df = load_isco_minor(ISCO_PATH)
patents_df = pd.read_excel(PATENTS_SAMPLE_PATH)
# Drop unnecessary columns and reset index
if "description" in patents_df.columns:
    patents_df = patents_df.drop("description", axis=1)

# Remove semicolon from the abstracts (semicolon breaks prompt parsing)
patents_df["title"] = patents_df["title"].str.replace(";", "", regex=False)
patents_df["abstract"] = patents_df["abstract"].str.replace(";", "", regex=False)

patents_df = patents_df.reset_index(drop=True)
valid_codes = set(str(code) for code in isco_df["Code"])

# -----------------------------------------------------------
# (Optional) quick test subset
# -----------------------------------------------------------
patents_df = patents_df.head(SAMPLE_SIZE) if SAMPLE_SIZE else patents_df

# -----------------------------------------------------------
# Main loop: classify and validate
# -----------------------------------------------------------

results = []

for idx, row in patents_df.iterrows():
    label = f"[{idx+1}/{len(patents_df)} ID={row.get('id','?')}]"
    print(f"🔍 Processing {label}")
    classification = classify_abstract_json(
        client,
        row["title"],
        row["abstract"],
        isco_df,
        model=MODEL,
        include_justifications=INCLUDE_JUSTIFICATIONS,
        max_secondaries=MAX_SECONDARIES,
    )
    classification = validate_codes(classification, valid_codes, INCLUDE_JUSTIFICATIONS)
    results.append(classification)
    sleep(0.2)  # small pause to be gentle on rate limits

# -----------------------------------------------------------
# Save results
# -----------------------------------------------------------

results_df = pd.DataFrame(results)

# If justifications are off, make sure we don't accidentally create empty columns
if not INCLUDE_JUSTIFICATIONS:
    for col in ["primary_justification", "secondary_justifications"]:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])

final_df = pd.concat([patents_df, results_df], axis=1)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ File generated: {OUTPUT_PATH}")
