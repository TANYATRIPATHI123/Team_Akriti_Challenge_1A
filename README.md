# adobe-text-extractor
## PDF Heading Classifier

This project extracts and classifies structural headings (Title, H1, H2, H3) from a PDF using machine learning. It leverages layout, text style, and formatting features to predict heading types and outputs a structured JSON that conforms to the schema defined in sample_dataset/schema/output_schema.json.


##  Approach

1. PDF Parsing:
    - Utilizes PyMuPDF (fitz) to extract text spans and visual metadata like font size, boldness, font name, and position.
2. Feature Engineering:
    - Derived features include:
        - font_size
        - is_bold
        - is_all_caps
        - char_count, word_count
        - capitalized_word_ratio
        - digit_ratio
        - ends_with_period
3. Prediction:
    - A pre-trained scikit-learn pipeline (serialized with joblib) is used to classify each text block.
4. Post-processing:
    - Only the heading predictions (Title, H1, H2, H3) are filtered and converted to a structured JSON list that adheres to a defined schema.



##  Libraries & Tools Used

- [PyMuPDF](https://pymupdf.readthedocs.io/) – PDF parsing
- pandas – Data handling
- scikit-learn – ML model and pipeline
- joblib – Model loading
- Docker – Containerized environment



##  Running the Project

##  Local Python Execution

1. Install dependencies :
    
    
    pip install -r requirements.txt
    
    
2. *Add PDFs to app/input/ folder :* 

Place your input PDFs here:



app/input/
├── file1.pdf
├── file2.pdf


3. Run the pipeline :


python process.py


4. Outputs will be saved in :


app/outputs/
├── file1.json
├── file2.json


##  Running with Docker

1. Build the Docker image (on AMD64 architecture)


docker run -it adobe-india-hackathon25-main:round1a


2. Prepare input/output folders on host

Place your input PDFs inside a local input/ folder:


input
├── doc1.pdf
├── doc2.pdf


Create an empty output/ folder (if not present):


mkdir -p output



3. Run the container


docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-heading-classifier


This will automatically process all PDFs inside /app/input, and generate a corresponding filename.json inside /app/output.