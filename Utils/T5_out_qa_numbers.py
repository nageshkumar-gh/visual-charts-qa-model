import pandas as pd

# Define the function to categorize each value
def categorize_label(label):
    label = str(label).strip()
    if label.lower() == "yes":
        return "yes"
    elif label.lower() == "no":
        return "no"
    elif label.replace('.', '', 1).isdigit():  # handles integers and floats
        return "integer"
    else:
        return "string"

# Load your CSV
csv_path = 't5_predictions.csv'  #

df = pd.read_csv(csv_path)

# Apply the categorization to the 'Ground Truth' column
df['Label Category'] = df['Ground Truth'].apply(categorize_label)

# Optional: Save the updated DataFrame to a new CSV
df.to_csv('categorized_output.csv', index=False)

# Show a preview
print(df[['Ground Truth', 'Label Category']].head())

