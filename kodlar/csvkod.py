import os
import csv

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

DATASET_PATH = r"C:\Users\kucuk\Desktop\ses_verileri"

rows = []

print("Klasör taranıyor:", DATASET_PATH)

# os.walk ile alt klasörleri de dolaşıyoruz
for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if not file.endswith(".wav"):
            continue

        print("İşleniyor:", file)

        try:
            parts = file.replace(".wav", "").split("-")
            if len(parts) != 7:
                print("❌ Hatalı format:", file)
                continue

            emotion = parts[2]
            actor = parts[6]

            emotion_name = emotion_labels.get(emotion, "unknown")

            # filename yerine tam yolu da kaydedebiliriz
            filepath = os.path.join(root, file)
            rows.append([filepath, emotion, emotion_name, actor])

        except Exception as e:
            print("Hata:", file, e)

output_path = os.path.join(DATASET_PATH, "dataset.csv")

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "emotion_id", "emotion_name", "actor"])
    writer.writerows(rows)

print("✔ CSV oluşturuldu:", output_path)
print("Toplam satır:", len(rows))
