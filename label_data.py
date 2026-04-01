import os
import pandas as pd

data = []

base_path = "dataset"

for label_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, label_name)

    if os.path.isdir(folder_path):

        # Assign label
        label = 0 if label_name == "real" else 1

        for file in os.listdir(folder_path):
            file_path = os.path.join(label_name, file)

            data.append({
                "filename": file_path,
                "label": label
            })

df = pd.DataFrame(data)
df.to_csv("labels.csv", index=False)

print("CSV file created: labels.csv")