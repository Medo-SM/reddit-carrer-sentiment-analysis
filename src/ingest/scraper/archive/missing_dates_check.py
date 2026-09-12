import pandas as pd

# The script goess through the ['created_utc'] column to count the days that have been scraped.
# It iterate through the Series in chunks for RAM sake.

file_path = "/content/drive/MyDrive/reddit_project/cscareerquestions_comments.jsonl"
chunks = pd.read_json(file_path, lines=True, chunksize = 10000)

target_column = 'created_utc'
data_count = 0
desired_count = 2223
all_date_list = []

for chunk in chunks:
  chunk[target_column] = pd.to_datetime(chunk[target_column], unit='s', utc=True)
  chunk[target_column] = chunk[target_column].dt.date
  unique_dates = chunk[target_column].drop_duplicates().dropna()
  all_date_list.append(unique_dates)

filtered_df = pd.concat(all_date_list, ignore_index=True)
filtered_df = filtered_df.drop_duplicates()

data_count = len(filtered_df)
difference_count = desired_count - data_count

if (data_count == desired_count):
  print("No missing date")
elif (data_count > desired_count):
  print(f"Something went wrong! Data count is larger than the desired count by : {-difference_count}")
else:
    print(f"Missing dates = {difference_count}")


