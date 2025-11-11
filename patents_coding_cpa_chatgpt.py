from openai import OpenAI
import pandas as pd
import json
import os
from dotenv import load_dotenv
from time import sleep

# -----------------------------------------------------------
# CONFIGURAZIONE
# -----------------------------------------------------------
SPLIT = 1  # numero del file di input da processare
MODEL = "gpt-5-mini"
INCLUDE_JUSTIFICATIONS = False
MAX_SECONDARIES = 3

# Checkpoint e resume
CHECKPOINT_EVERY = 10
RESUME_IF_EXISTS = True

# File di input/output
CP_PATH = "resources/classification/CP2021_3digit.xlsx"
PATENTS_SAMPLE_PATH = (
    f"sample/update/split_files/patents_sample_update_part_{SPLIT}.csv"
)
OUTPUT_PATH = f"output/update/patents_classified_chatgpt_part_{SPLIT}.csv"

# -----------------------------------------------------------
# FUNZIONI DI SUPPORTO
# -----------------------------------------------------------


def load_cp(path: str) -> pd.DataFrame:
    """Carica la classificazione CP2021 (3-digit)"""
    df = pd.read_excel(path, sheet_name="3_digit")
    df.columns = ["code", "title", "description"]
    df["code"] = df["code"].astype(str).str.strip()
    return df


def generate_prompt(title, abstract, cp_df, include_justifications, max_secondaries):
    """Genera il prompt JSON-friendly per il modello"""
    entries = [{"code": c, "title": t} for c, t in zip(cp_df["code"], cp_df["title"])]
    secondary_hint = f"(at most {max_secondaries})" if max_secondaries else "(if any)"

    if include_justifications:
        fields = f"""
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
You are a labor market expert. Classify the following patent into the most appropriate CP2021 3-digit occupation group.

Patent title:
\"\"\"{title}\"\"\"

Patent abstract:
\"\"\"{abstract}\"\"\"

Available codes and titles:
{json.dumps(entries, ensure_ascii=False)}

{fields}
"""


def classify(client, title, abstract, cp_df):
    """Chiama il modello con output JSON"""
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
        print(f"❌ Errore API: {e}")
        return {}


def validate_codes(result, valid_codes):
    """Controlla la validità dei codici CP suggeriti"""
    if result.get("primary_code") not in valid_codes:
        result["primary_code"] = "INVALID"
    valids = []
    for c in result.get("secondary_codes", []):
        valids.append(c if c in valid_codes else "INVALID")
    result["secondary_codes"] = valids
    return result


def load_existing_results(path):
    """Se esiste un file di output, lo ricarica"""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"🟡 Found existing output with {len(df)} rows — resuming.")
            return df
        except Exception:
            print("⚠️ Error reading existing output, starting fresh.")
    return None


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Carica classificazione CP e sample brevetti
cp_df = load_cp(CP_PATH)
patents_df = pd.read_csv(PATENTS_SAMPLE_PATH)
valid_codes = set(cp_df["code"])

# Controlla output esistente
existing_df = load_existing_results(OUTPUT_PATH) if RESUME_IF_EXISTS else None
done_ids = (
    set(existing_df["row_id"])
    if existing_df is not None and "row_id" in existing_df.columns
    else set()
)

results = []

# Loop principale
for _, row in patents_df.iterrows():
    rid = row["row_id"]
    if rid in done_ids:
        continue

    print(f"🔍 Processing row_id={rid}")
    result = classify(client, row["title"], row["abstract"], cp_df)
    result = validate_codes(result, valid_codes)
    result["row_id"] = rid
    result["id"] = row.get("id", "")
    results.append(result)

    # Checkpoint
    if len(results) % CHECKPOINT_EVERY == 0:
        temp = pd.DataFrame(results)
        if existing_df is not None:
            temp = pd.concat([existing_df, temp], ignore_index=True)
        temp.to_csv(OUTPUT_PATH, index=False)
        existing_df = temp
        print(f"💾 Checkpoint saved ({len(temp)} rows)")

    sleep(0.2)

# Salvataggio finale
results_df = pd.DataFrame(results)
final_df = (
    pd.concat([existing_df, results_df], ignore_index=True)
    if existing_df is not None
    else results_df
)

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Completed. File saved to {OUTPUT_PATH}")
