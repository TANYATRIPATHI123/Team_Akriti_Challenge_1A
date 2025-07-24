import pandas as pd
import joblib
import fitz  # PyMuPDF
import os

def extract_pdf_data(pdf_path):
    """Extracts text and basic metadata from the PDF."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Error opening PDF file: {e}")

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

def predict_from_pdf(pdf_path, model_path='model\pdf_text_classifier_model.pkl'):
    
    """Extracts features, runs model prediction, returns DataFrame (no CSV writing)."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at '{pdf_path}'")

    # Load model
    model_pipeline = joblib.load(model_path)

    # Extract text blocks
    pdf_data = extract_pdf_data(pdf_path)
    if not pdf_data:
        raise ValueError("No text could be extracted from the PDF.")

    # Create DataFrame
    df = pd.DataFrame(pdf_data)
    df['is_bold'] = df['is_bold'].astype(int)

    # Add features (must match training)
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
    if 'a' not in df.columns:
        df['a'] = pd.NA

    # Predict
    df['predicted_label'] = model_pipeline.predict(df)

    return df  # Return the DataFrame to be used elsewhere