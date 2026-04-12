import pandas as pd
import os

# Step 1: Create initial DataFrame
data = {
    "Name": ["Alice", "Bob"],
    "Age": [25, 30],
    "City": ["Delhi", "Mumbai"]
}

df = pd.DataFrame(data)

# # Step 2: Add more rows using .loc
df.loc[len(df)] = ["Charlie", 28, "Bangalore"]
# df.loc[len(df)] = ["David", 35, "Hyderabad"]

# Step 3: Create 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Step 4: Save DataFrame to CSV
file_path = os.path.join("data", "people_data.csv")
df.to_csv(file_path, index=False)

print("DataFrame saved to:", file_path)
print("\nFinal DataFrame:")
# print(df)