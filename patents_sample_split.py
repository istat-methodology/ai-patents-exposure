import os
import pandas as pd

# === VARIABILI GLOBALI ===
INPUT_FILE = "sample/update/patents_sample_7k.csv"
OUTPUT_FOLDER = "sample/update/split_files"
ROWS_PER_FILE = 500  # <-- numero di righe per file

# === CREA LA CARTELLA DI OUTPUT SE NON ESISTE ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === LEGGI IL FILE DI INPUT ===
df = pd.read_csv(INPUT_FILE, sep="|")

# === AGGIUNGI UN PROGRESSIVO ===
df = df.reset_index(drop=True)
df.insert(0, "row_id", df.index + 1)  # 👈 row_id come PRIMA colonna

# === CALCOLA IL NUMERO DI FILE DA CREARE ===
num_files = (len(df) + ROWS_PER_FILE - 1) // ROWS_PER_FILE

print(f"📄 File totale: {len(df)} righe")
print(f"🔹 Verranno creati {num_files} file da {ROWS_PER_FILE} righe ciascuno")

# === SPLIT E SALVATAGGIO ===
for i in range(num_files):
    start_row = i * ROWS_PER_FILE
    end_row = start_row + ROWS_PER_FILE
    chunk = df.iloc[start_row:end_row]

    output_path = os.path.join(OUTPUT_FOLDER, f"patents_sample_update_part_{i+1}.csv")
    chunk.to_csv(output_path, index=False)

    print(f"✅ Creato: {output_path}")

print("🎉 Operazione completata con successo!")
