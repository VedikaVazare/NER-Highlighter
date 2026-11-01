# NER Highlighter

A Streamlit application that detects and highlights named entities in text, TXT files, and PDF files using spaCy. It highlights people, organizations, places, dates, and money.

## Requirements
- Python 3.9 or later (use a Python version supported by your installed spaCy release)

## Setup

1. Create and activate a virtual environment:
   - Windows: `py -m venv venv` then `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

The app downloads the spaCy `en_core_web_sm` model automatically the first time it is needed.
