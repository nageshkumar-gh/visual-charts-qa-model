import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the CSV file
df = pd.read_csv('Utils/output/output.csv')  # Replace with your path

# Count and plot chart types in full dataset
type_counts = df['type'].value_counts()
type_percent = df['type'].value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    'Count': type_counts,
    'Percentage': type_percent.round(2)
})
print(" Full Dataset Chart Type Distribution:")
print(summary_df)

# Plot full dataset distribution
plt.figure(figsize=(8, 5))
bars = plt.bar(summary_df.index, summary_df['Count'])
for bar, pct in zip(bars, summary_df['Percentage']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{pct:.1f}%', ha='center', va='bottom')
plt.title('Full Dataset Chart Type Distribution')
plt.xlabel('Chart Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# -----------------------------
# Split the data stratified by 'type'
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['type'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['type'], random_state=42)

# Save filenames only to CSV
train_df[['filename','type']].to_csv('Utils/output/train_split_files.csv', index=False)
val_df[['filename','type']].to_csv('Utils/output/val_split_files.csv', index=False)
test_df[['filename','type']].to_csv('Utils/output/test_split_files.csv', index=False)

print(f"\n Split Sizes:")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# -----------------------------
# Define reusable distribution function
def show_distribution(df, name):
    type_counts = df['type'].value_counts()
    type_percent = df['type'].value_counts(normalize=True) * 100

    summary = pd.DataFrame({
        'Count': type_counts,
        'Percentage': type_percent.round(2)
    })

    print(f"\n {name} Set Chart Type Distribution:")
    print(summary)

    # Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(summary.index, summary['Count'])
    for bar, pct in zip(bars, summary['Percentage']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{pct:.1f}%', ha='center', va='bottom')
    plt.title(f'{name} Set Chart Type Distribution')
    plt.xlabel('Chart Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Run for all sets
show_distribution(train_df, "Train")
show_distribution(val_df, "Validation")
show_distribution(test_df, "Test")
