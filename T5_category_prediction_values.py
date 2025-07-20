import pandas as pd

# Define label categorization
def categorize_label(label):
    label = str(label).strip()
    if label.lower() == "yes":
        return "yes"
    elif label.lower() == "no":
        return "no"
    elif label.replace('.', '', 1).isdigit():
        return "integer"
    else:
        return "string"

# Load CSV
csv_path = 'categorized_output.csv'
df = pd.read_csv(csv_path)

# Create label category column
df['Label Category'] = df['Ground Truth'].apply(categorize_label)

# Normalize Match column
df['Match'] = df['Match'].astype(str).str.strip().str.upper()

# Calculate TRUE % for each Label Category
summary = (
    df.groupby('Label Category')['Match']
    .apply(lambda x: (x == "TRUE").mean() * 100)
    .reset_index(name='True %')
)

# Display result
print(summary)
