from openai import OpenAI
import pandas as pd
import json
import os
from dotenv import load_dotenv
from time import sleep

# -----------------------------------------------------------
# CONFIGURAZIONE
# -----------------------------------------------------------

SPLIT = 1
MODEL = "gpt-5-mini"
INCLUDE_JUSTIFICATIONS = False
MAX_SECONDARIES = 3

CHECKPOINT_EVERY = 10
RESUME_IF_EXISTS = True

CP_PATH = "resources/classification/CP2021_3digit.xlsx"
PATENTS_SAMPLE_PATH = (
    f"sample/update/split_files/patents_sample_update_part_{SPLIT}.csv"
)
OUTPUT_PATH = f"output/update/patents_classified_chatgpt_part_{SPLIT}.csv"

# ordine colonne centralizzato
COLUMN_ORDER = ["row_id", "id", "primary_code", "secondary_codes"]

# -----------------------------------------------------------
# FUNZIONI
# -----------------------------------------------------------


def load_cp(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="3_digit")
    df.columns = ["code", "title", "description"]
    df["code"] = df["code"].astype(str).str.strip()
    return df


def generate_prompt(title, abstract, cp_df, include_justifications, max_secondary):
    entries = [{"code": c, "title": t} for c, t in zip(cp_df["code"], cp_df["title"])]
    secondary_hint = f"(at most {max_secondary})" if max_secondary else "(if any)"

    if include_justifications:
        fields = """
Return ONLY a JSON object with:
- primary_code
- primary_justification
- secondary_codes {secondary_hint}
- secondary_justifications (mapping code→justification)
"""
    else:
        fields = f"""
Return ONLY a JSON object with:
- primary_code
- secondary_codes {secondary_hint}
(no justification fields)
"""

    return f"""
You are a labor market expert. Classify the following patent into the appropriate CP2021 3-digit occupation group.

Patent title:
\"\"\"{title}\"\"\"

Patent abstract:
\"\"\"{abstract}\"\"\"

Available codes and titles:
{json.dumps(entries, ensure_ascii=False)}

{fields}
"""


def classify(client, title, abstract, cp_df):
    prompt = generate_prompt(
        title, abstract, cp_df, INCLUDE_JUSTIFICATIONS, MAX_SECONDARIES
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ API error: {e}")
        return {}


def validate_codes(result, valid_set):
    if result.get("primary_code") not in valid_set:
        result["primary_code"] = "INVALID"
    result["secondary_codes"] = [
        c if c in valid_set else "INVALID" for c in result.get("secondary_codes", [])
    ]
    return result


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Forza l’ordine colonne globale."""
    return df.reindex(columns=COLUMN_ORDER)


def load_existing_results(path: str):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"🟡 Resuming from {len(df)} rows.")
            return reorder_columns(df)
        except:
            print("⚠️ Error loading existing output. Starting fresh.")
    return None


def save_checkpoint(df, path):
    df = reorder_columns(df)
    df.to_csv(path, index=False)
    print(f"💾 Checkpoint saved ({len(df)} rows)")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cp_df = load_cp(CP_PATH)
patents_df = pd.read_csv(PATENTS_SAMPLE_PATH)
valid_codes = set(cp_df["code"])

existing_df = load_existing_results(OUTPUT_PATH) if RESUME_IF_EXISTS else None
done_ids = set(existing_df["row_id"]) if existing_df is not None else set()

results = []

for _, row in patents_df.iterrows():
    rid = row["row_id"]

    # Skippa già classificati
    if rid in done_ids:
        continue

    print(f"🔍 Processing row_id={rid}")

    result = classify(client, row["title"], row["abstract"], cp_df)
    result = validate_codes(result, valid_codes)

    # assegna id informazioni esterne
    result["row_id"] = rid
    result["id"] = row.get("id", "")

    results.append(result)

    # checkpoint
    if len(results) % CHECKPOINT_EVERY == 0:
        temp = pd.DataFrame(results)
        if existing_df is not None:
            temp = pd.concat([existing_df, temp], ignore_index=True)
        existing_df = reorder_columns(temp)
        save_checkpoint(existing_df, OUTPUT_PATH)

    sleep(0.2)

# ===== Salvataggio finale =====
results_df = pd.DataFrame(results)

final_df = (
    pd.concat([existing_df, results_df], ignore_index=True)
    if existing_df is not None
    else results_df
)

final_df = reorder_columns(final_df)
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Completed. File saved to {OUTPUT_PATH}")
