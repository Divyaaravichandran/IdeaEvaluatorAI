import pandas as pd

# Load dataset
file_path = "ideas.xlsx"
df = pd.read_excel(file_path)

# Display basic information
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Check advanced vs normal ideas
print("\nAdvance distribution:")
print(df["advance"].value_counts())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display first 5 ideas
print("\nFirst 5 ideas:")
print(df.head())

# Separate advanced and normal ideas
advanced_ideas = df[df["advance"] == 1]
normal_ideas = df[df["advance"] == 0]

print("\nAdvanced ideas:", len(advanced_ideas))
print("Normal ideas:", len(normal_ideas))