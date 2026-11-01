"""
Named Entity Recognition (NER) Highlighter  -  "Marginalia"
Paste text or upload a PDF / TXT file; entities get marked up like a
highlighter pen: people, organizations, places, dates and money.

Run:  streamlit run app.py
"""
import html
import re

import pandas as pd
import spacy
import streamlit as st

# ---------------------------------------------------------------
# 1. CATEGORY SETUP  (spaCy label -> friendly category)
# ---------------------------------------------------------------
LABEL_TO_CATEGORY = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Places",
    "LOC": "Places",
    "FAC": "Places",
    "DATE": "Dates",
    "TIME": "Dates",
    "MONEY": "Money"
}

# (marker color, short tag shown next to the highlight)
CATEGORY_STYLE = {
    "People":        ("#ffd08a", "person"),
    "Organizations": ("#9ec5fe", "org"),
    "Places":        ("#a8e6b0", "place"),
    "Dates":         ("#f7b2c4", "date"),
    "Money":         ("#ffe066", "money")
}

# Extra rule for rupee amounts, which the small model often misses
RUPEE_PATTERN = re.compile(
    r"(?:₹|Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?"
    r"(?:\s?(?:crore|lakh|million|billion|thousand|cr|k)\b)?",
    re.IGNORECASE,
)

MAX_CHARS = 900_000

SAMPLE_TEXT = (
    "Elon Musk said on Tuesday that Boeing and NASA will launch a new mission "
    "from Florida in March 2027, backed by a $2.4 billion contract. Meanwhile, "
    "Sundar Pichai of Google met officials in Paris on 14 July during the AI "
    "Action Summit to discuss regulation across Europe. In Mumbai, Tata Motors "
    "unveiled a new electric SUV priced at ₹14.5 lakh on 5 January 2027."
)


# ---------------------------------------------------------------
# 2. PAGE STYLING  (forces its own colors, so it looks the same
#    whether your OS / browser is in light or dark mode)
# ---------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

:root { --ink:#1f1b16; --muted:#6f6659; --paper:#fffdf9; --bg:#f4f1ea; --line:#e2dccf; --accent:#c2410c; }

html, body, .stApp { background: var(--bg) !important; color: var(--ink) !important;
                     font-family: 'Inter', sans-serif; }
.stApp p, .stApp label, .stApp span, .stApp li, .stApp div { color: inherit; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent !important; height: 0; }
.block-container { max-width: 1280px; padding: 0 2.2rem 3rem !important; }

/* ---------- top bar ---------- */
.nav { display:flex; align-items:center; justify-content:space-between;
       padding: 1.1rem 0; border-bottom: 1px solid var(--line); margin-bottom: 1.8rem; }
.brand { display:flex; align-items:center; gap:.7rem; }
.logo { width:34px; height:34px; border-radius:9px; background:var(--ink); color:#ffe066;
        display:grid; place-items:center; font-weight:700; font-family:'Fraunces',serif; font-size:1.2rem; }
.brand b { font-family:'Fraunces',serif; font-size:1.3rem; letter-spacing:.01em; }
.pills-top { display:flex; gap:.5rem; }
.pills-top span { font-family:'JetBrains Mono',monospace; font-size:.72rem; padding:.28rem .7rem;
                  border:1px solid var(--line); border-radius:99px; background:var(--paper); color:var(--muted); }

.hero h1 { font-family:'Fraunces',serif; font-size:2.9rem; line-height:1.08; margin:0 0 .5rem; letter-spacing:-.01em; }
.hero h1 em { font-style:normal; background:linear-gradient(180deg,transparent 58%,#ffd966 58%); padding:0 .1em; }
.hero p { color:var(--muted); font-size:1.02rem; max-width: 640px; margin:0 0 1.6rem; }

/* ---------- cards ---------- */
.card-title { font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
              color:var(--muted); margin: 0 0 .6rem; }
.paper { background:var(--paper); border:1px solid var(--line); border-radius:14px;
         padding:1.5rem 1.8rem; line-height:2.35; font-size:1.02rem;
         box-shadow:0 1px 0 var(--line), 0 12px 30px -18px rgba(60,40,10,.25);
         max-height:560px; overflow-y:auto; color:var(--ink); }

/* entity highlight: colored chip with a dark tag, like displaCy */
.ent { padding:.14em .5em; border-radius:7px; color:#1f1b16 !important; font-weight:500;
       box-decoration-break:clone; -webkit-box-decoration-break:clone; white-space:normal; }
.ent b { font: 700 .58rem 'Inter',sans-serif; letter-spacing:.09em; text-transform:uppercase;
         background:#1f1b16; color:#fff !important; padding:.2em .5em; border-radius:5px;
         margin-left:.5em; vertical-align:.12em; }

.legend { display:flex; flex-wrap:wrap; gap:.4rem; margin:.2rem 0 1rem; }
.legend span { font-size:.78rem; font-weight:600; padding:.22rem .75rem; border-radius:99px; color:#1f1b16; }

.stats { display:grid; grid-template-columns:1fr 1fr; gap:.65rem; margin-bottom:1.2rem; }
.stat { background:var(--paper); border:1px solid var(--line); border-radius:12px;
        padding:.7rem .9rem; display:flex; align-items:center; gap:.75rem; }
.stat .dot { width:12px; height:12px; border-radius:50%; flex:none; }
.stat .n { font-family:'Fraunces',serif; font-size:1.55rem; font-weight:700; line-height:1; color:var(--ink); }
.stat .l { font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:600; }
.stat.zero { opacity:.45; }

/* ---------- streamlit widgets, restyled ---------- */
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height:2.6rem; padding:0 1.1rem; font-weight:600; color:var(--muted) !important;
                               background:transparent; border-radius:8px 8px 0 0; }
.stTabs [aria-selected="true"] { color:var(--ink) !important; background:var(--paper); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--accent) !important; height:3px; }

textarea { background:var(--paper) !important; color:var(--ink) !important; border:1px solid var(--line) !important;
           border-radius:12px !important; font-size:1rem !important; line-height:1.6 !important; }
textarea::placeholder { color:#9a9082 !important; }

[data-testid="stFileUploaderDropzone"] { background:var(--paper) !important; border:2px dashed #cfc6b3 !important;
                                         border-radius:14px !important; padding:2.2rem 1rem !important; }
[data-testid="stFileUploaderDropzone"] * { color:var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] small { color:var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { background:var(--ink) !important; color:#fff !important;
                                                border:none !important; border-radius:8px !important; }
[data-testid="stFileUploaderDropzone"] button * { color:#fff !important; }

div.stButton > button, div.stDownloadButton > button { border-radius:10px; font-weight:600; padding:.55rem 1.3rem;
        border:1px solid var(--ink); background:var(--paper); color:var(--ink) !important; }
div.stButton > button:hover, div.stDownloadButton > button:hover { background:#ece6d8; border-color:var(--ink); }
div.stButton > button[kind="primary"] { background:var(--ink); color:#fff !important; border-color:var(--ink); }
div.stButton > button[kind="primary"] p { color:#fff !important; }
div.stButton > button[kind="primary"]:hover { background:var(--accent); border-color:var(--accent); }

[data-testid="stPills"] button, [data-testid="stPills"] [role="option"] { border-radius:99px !important; }
[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:var(--line) !important; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------
# 3. MODEL + FILE READING
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading language model...")
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def read_uploaded_file(uploaded):
    """Return plain text from a .txt or .pdf upload."""
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(uploaded)
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages).strip()
    raw = uploaded.read()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return ""


# ---------------------------------------------------------------
# 4. ENTITY EXTRACTION
# ---------------------------------------------------------------
def extract_entities(nlp, text):
    doc = nlp(text)
    ents = [
        {"text": e.text.strip(), "label": e.label_,
         "category": LABEL_TO_CATEGORY[e.label_],
         "start": e.start_char, "end": e.end_char}
        for e in doc.ents if e.label_ in LABEL_TO_CATEGORY
    ]

    # rupee amounts: add them, or widen a shorter MONEY entity spaCy found
    for m in RUPEE_PATTERN.finditer(text):
        s, e = m.span()
        overlap = [x for x in ents if s < x["end"] and e > x["start"]]
        if all(x["category"] == "Money" and (x["end"] - x["start"]) < (e - s) for x in overlap):
            ents = [x for x in ents if x not in overlap]
            ents.append({"text": m.group().strip(), "label": "MONEY (rule)",
                         "category": "Money", "start": s, "end": e})

    return sorted(ents, key=lambda x: x["start"])


# ---------------------------------------------------------------
# 5. HIGHLIGHT RENDERING
# ---------------------------------------------------------------
def highlight_text(text, entities):
    out, cursor = [], 0
    for e in entities:
        out.append(html.escape(text[cursor:e["start"]]))
        color, tag = CATEGORY_STYLE[e["category"]]
        out.append(f'<span class="ent" style="background:{color}">'
                   f'{html.escape(e["text"])}<b>{tag}</b></span>')
        cursor = e["end"]
    out.append(html.escape(text[cursor:]))
    body = "".join(out)
    return re.sub(r"\n{2,}", "<br><br>", body).replace("\n", " ")


def legend_html(categories):
    chips = "".join(f'<span style="background:{CATEGORY_STYLE[c][0]}">{c}</span>' for c in categories)
    return f'<div class="legend">{chips}</div>'


def stats_html(categories, counts):
    cards = "".join(
        f'<div class="stat {"zero" if counts.get(c, 0) == 0 else ""}">'
        f'<span class="dot" style="background:{CATEGORY_STYLE[c][0]}"></span>'
        f'<div><div class="n">{counts.get(c, 0)}</div><div class="l">{c}</div></div></div>'
        for c in categories
    )
    return f'<div class="stats">{cards}</div>'


def build_download_html(body):
    return (
        "<!doctype html><meta charset='utf-8'><title>NER Highlights</title>"
        "<style>body{max-width:780px;margin:2.5rem auto;font:17px/2.3 Georgia,serif;"
        "color:#1f1b16;background:#fffdf9;padding:0 1.2rem}"
        ".ent{padding:.14em .5em;border-radius:7px;font-weight:500}"
        ".ent b{font:700 .55em sans-serif;letter-spacing:.09em;text-transform:uppercase;"
        "background:#1f1b16;color:#fff;padding:.2em .5em;border-radius:5px;margin-left:.5em}"
        f"</style>{body}"
    )


# ---------------------------------------------------------------
# 6. APP
# ---------------------------------------------------------------
def load_sample():
    st.session_state.text_input = SAMPLE_TEXT


def analyze(nlp, text):
    if not text.strip():
        st.warning("Add some text or a file first.")
        return
    if len(text) > MAX_CHARS:
        st.info(f"Long document: analysing the first {MAX_CHARS:,} characters.")
        text = text[:MAX_CHARS]
    nlp.max_length = len(text) + 1000
    with st.spinner("Analysing..."):
        st.session_state.result = (text, extract_entities(nlp, text))


def main():
    st.set_page_config(page_title="Marginalia · NER Highlighter", page_icon="🖍️", layout="wide")
    inject_css()
    nlp = load_model()

    st.markdown(
        """
        <div class="nav">
          <div class="brand"><div class="logo">M</div><b>Marginalia</b></div>
          <div class="pills-top"><span>spaCy · en_core_web_sm</span><span>5 entity types</span></div>
        </div>
        <div class="hero">
          <h1>Read less. <em>Spot everything.</em></h1>
          <p>Paste an article or upload a PDF. People, organizations, places, dates and money
         are highlighted and indexed automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- input ----------------
    st.markdown('<div class="card-title">1 · Your document</div>', unsafe_allow_html=True)
    tab_paste, tab_upload = st.tabs(["✍️  Paste text", "📎  Upload PDF / TXT"])

    with tab_paste:
        st.text_area("Text", key="text_input", height=200, label_visibility="collapsed",
                     placeholder="Paste a news article or any paragraph here...")
        b1, b2, _ = st.columns([1.3, 1, 5])
        if b1.button("Highlight entities", type="primary", key="go_text"):
            analyze(nlp, st.session_state.get("text_input", ""))
        b2.button("Try a sample", on_click=load_sample)

    with tab_upload:
        up = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt", "md"],
                              label_visibility="collapsed")
        if up is not None:
            with st.spinner("Reading file..."):
                file_text = read_uploaded_file(up)
            if file_text:
                st.success(f"Read **{up.name}** · {len(file_text):,} characters")
                if st.button("Highlight entities", type="primary", key="go_file"):
                    analyze(nlp, file_text)
            else:
                st.error("No readable text found. Scanned PDFs need OCR first.")

    # ---------------- results ----------------
    if "result" not in st.session_state:
        return
    text, all_entities = st.session_state.result

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">2 · Results</div>', unsafe_allow_html=True)
    chosen = st.pills("Show categories", list(CATEGORY_STYLE), selection_mode="multi",
                      default=list(CATEGORY_STYLE), label_visibility="collapsed") or []
    entities = [e for e in all_entities if e["category"] in chosen]
    counts = pd.Series([e["category"] for e in all_entities]).value_counts().to_dict()
    body = highlight_text(text, entities)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(legend_html(chosen), unsafe_allow_html=True)
        st.markdown(f'<div class="paper">{body}</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f'<div class="card-title">Entities found · {len(entities)}</div>', unsafe_allow_html=True)
        st.markdown(stats_html(list(CATEGORY_STYLE), counts), unsafe_allow_html=True)

        if not entities:
            st.info("No entities for the selected categories.")
            return

        table = (pd.DataFrame(entities).groupby(["text", "category"]).size().reset_index(name="mentions")
                   .sort_values(["category", "mentions"], ascending=[True, False]))
        table.columns = ["Entity", "Category", "Mentions"]
        st.dataframe(table, hide_index=True, height=300)

        d1, d2 = st.columns(2)
        d1.download_button("⬇ CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name="entities.csv", mime="text/csv")
        d2.download_button("⬇ HTML", build_download_html(body).encode("utf-8"),
                           file_name="highlighted.html", mime="text/html")


if __name__ == "__main__":
    main()