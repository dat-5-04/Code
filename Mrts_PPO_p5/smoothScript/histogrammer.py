import pandas as pd
import math

# Function to categorize values into specified ranges
def categorize_range(value):
    if 0 <= value <= 2000:
        range_start = math.floor(value / 200) * 200
        range_end = range_start + 200
        return f'{range_start}-{range_end}'
    else:
        return 'Other'

# Read the input CSV file
input_file = 'wins_2k_model.csv'
output_file = 'value_counts_200_range_wins.csv'  # Output file name

data = pd.read_csv(input_file)

# Apply categorization to a new column 'Range' and count occurrences
data['Range'] = data['Value'].apply(categorize_range)
value_counts = data['Range'].value_counts().reset_index()
value_counts.columns = ['Range', 'Count']

# Save the value counts to a new CSV file
value_counts.to_csv(output_file, index=False)

print("Value counts within ranges (200 increments) have been saved to", output_file)

