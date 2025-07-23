import pandas as pd
import json
import os

# --- Configuration ---
# 1. The name of the input CSV file that contains all predictions.
INPUT_CSV_PATH = 'all_predictions_debug.csv'

# 2. The name of the final JSON file you want to create.
OUTPUT_JSON_PATH = 'output_headings_and_titles.json'
# --- End Configuration ---


def main():
    """
    Reads a CSV file with predictions, filters for headings and titles,
    and saves the result as a JSON file.
    """
    print("--- Starting CSV to JSON Conversion ---")

    # 1. Check if the input CSV file exists
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"ERROR: Input file not found at '{INPUT_CSV_PATH}'")
        print("Please make sure you have run the main prediction script first to generate this file.")
        return

    # 2. Load the CSV data into a DataFrame
    print(f"Reading data from '{INPUT_CSV_PATH}'...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 3. Filter for only 'heading' and 'title'
    print("Filtering for 'heading' and 'title'...")
    # Ensure the column 'predicted_label' exists
    if 'predicted_label' not in df.columns:
        print(f"ERROR: The CSV file must contain a column named 'predicted_label'.")
        return

    filtered_df = df[df['predicted_label'].isin(['H1', 'H2', 'H3', 'Title'])]

    if filtered_df.empty:
        print("\nWARNING: No 'heading' or 'title' labels were found in the CSV.")
        print("The output JSON file will be empty.")

    # 4. Prepare the data for JSON output
    # Select and rename columns for the final JSON
    output_df = filtered_df[['text', 'predicted_label', 'page_number']].copy()
    output_df.rename(columns={'predicted_label': 'type'}, inplace=True)

    # Convert the filtered data to a list of dictionaries
    output_json_data = output_df.to_dict('records')

    # 5. Save the results to a JSON file
    print(f"Saving filtered results to '{OUTPUT_JSON_PATH}'...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_json_data, f, indent=4, ensure_ascii=False)

    print("\n--- SUCCESS ---")
    print(f"Process complete. {len(output_json_data)} headings/titles found and saved to '{OUTPUT_JSON_PATH}'.")


if __name__ == "__main__":
    main()
