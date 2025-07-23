import pandas as pd
import joblib
import fitz  # PyMuPDF
import os

# --- Configuration ---
MODEL_PATH = 'models\pdf_text_classifier_model.pkl'
PDF_FILE_PATH = 'sample_dataset/pdfs/file02.pdf'  # <--- Set your PDF path here
DEBUG_CSV_PATH = 'all_predictions_debug.csv'
# --- End Configuration ---


def extract_pdf_data(pdf_path):
    """Extracts text and basic metadata from the PDF."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return None

    extracted_data = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            extracted_data.append({
                                "text": text,
                                "font_size": span["size"],
                                "is_bold": "bold" in span["font"].lower(),
                                "font_name": span.get("font", "default"),
                                "page_number": page_num + 1
                            })
    return extracted_data


def main():
    print("--- Starting PDF Classification ---")

    # 1. Check file existence
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        return
    if not os.path.exists(PDF_FILE_PATH):
        print(f"ERROR: PDF file not found at '{PDF_FILE_PATH}'")
        return

    # 2. Load model
    print(f"Loading model from '{MODEL_PATH}'...")
    try:
        model_pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("Model loaded successfully.")

    # 3. Extract features
    print(f"Extracting text and features from '{PDF_FILE_PATH}'...")
    pdf_data = extract_pdf_data(PDF_FILE_PATH)

    if not pdf_data:
        print("No text could be extracted from the PDF.")
        return

    df = pd.DataFrame(pdf_data)
    df['is_bold'] = df['is_bold'].astype(int)

    # --- Add Missing Features (must match training features) ---
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['ends_with_period'] = df['text'].str.endswith('.')
    df['is_all_caps'] = df['text'].apply(lambda x: str(x).isupper())
    df['capitalized_word_ratio'] = df['text'].apply(
        lambda x: sum(1 for w in str(x).split() if w.istitle()) / (len(str(x).split()) + 1e-5)
    )
    df['digit_ratio'] = df['text'].apply(
        lambda x: sum(c.isdigit() for c in str(x)) / (len(str(x)) + 1e-5)
    )
    df['font_size_ratio'] = df['font_size'] / (df['font_size'].max() + 1e-5)
    df['block_index_on_page'] = df.groupby('page_number').cumcount()
    df['y_coordinate_normalized'] = df['block_index_on_page'] / (df['block_index_on_page'].max() + 1e-5)
    df['contains_chapter_keyword'] = df['text'].str.lower().str.contains('chapter|section|part')
    df['is_italic'] = False
    df['is_centered'] = False
    if 'font_name' not in df.columns:
        df['font_name'] = 'default'

    # --- Optional dummy column if your pipeline expects it ---
    if 'a' not in df.columns:
        df['a'] = pd.NA

    print(f"Extracted {len(df)} text blocks.")

    # 4. Predict
    print("Predicting labels...")
    predictions = model_pipeline.predict(df)
    df['predicted_label'] = predictions

    # 5. Save and print results
    print(f"\nSaving ALL predictions to '{DEBUG_CSV_PATH}'...")
    df.to_csv(DEBUG_CSV_PATH, index=False, encoding='utf-8-sig')
    print("Debug CSV file created.")

    print("\n--- Prediction Summary ---")
    print(f"Unique labels predicted: {df['predicted_label'].unique()}")

    print("\n--- Full Output ---")
    print(df[['text', 'predicted_label', 'page_number']].to_string())

    print("\n--- DONE ---")


if __name__ == "__main__":
    main()
