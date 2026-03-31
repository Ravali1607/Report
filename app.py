from flask import Flask, render_template, request, jsonify
import os
import tempfile
import base64
import tiktoken
from db import dbConnection
from openai import OpenAI
import pytesseract
import cv2
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
app = Flask(__name__)

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL      = "gpt-oss:120b-cloud"

 
client = OpenAI(os.getenv("API_KEY"))
 
MODEL = "gpt-5.4-mini"
try:
    encoding = tiktoken.encoding_for_model(MODEL)
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")
reports = []


# ──────────────────────────────────────────────────────────────
#  OCR ROUTE
#  Uses easyocr — pure Python, NO sudo / system install needed.
#
#  Install (one time):
#    pip install easyocr Pillow pdfplumber
#
#  easyocr downloads its own model files (~100 MB) on first run.
#  They are cached in ~/.EasyOCR/ automatically.
#
#  Supported: jpg, png, webp, bmp, tiff, PDF
# ──────────────────────────────────────────────────────────────

# Load easyocr reader once at startup (heavy — do NOT load per request)
# _ocr_reader = None

# def get_ocr_reader():
#     """Lazy-load the easyocr reader (Telugu + English)."""
#     global _ocr_reader
#     if _ocr_reader is None:
#         import easyocr
#         # 'te' = Telugu,  'en' = English
#         # gpu=False → works on any machine without CUDA
#         _ocr_reader = easyocr.Reader(['te', 'en'], gpu=False)
#         print("[INFO] easyocr reader loaded (te + en)")
#     return _ocr_reader


@app.route("/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "ఫైల్ అందలేదు"}), 400

    file     = request.files["file"]
    filename = file.filename or "upload"
    ext      = os.path.splitext(filename)[1].lower()

    suffix = ext if ext else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── PDF: selectable text first, OCR fallback ───────────
        if ext == ".pdf":
            text = extract_pdf_text(tmp_path)
            if not text.strip():
                text = ocr_pdf_pages_easyocr(tmp_path)
            return jsonify({"text": text.strip()})

        # ── Image ──────────────────────────────────────────────
        # text = ocr_image_easyocr(tmp_path)
        # return jsonify({"text": text.strip()})
        text = ocr_image_tesseract(tmp_path).strip()

        # 🧠 Auto detect type
        if len(text) > 30:   # has readable text
            return jsonify({
                "mode": "ocr",
                "text": text
            })
        else:
            # 🔥 Call OpenAI for scene description
            scene_prompt = "ఈ చిత్రంలో ఏమి జరుగుతోంది వివరంగా తెలుగులో వివరించండి."

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": scene_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(tmp_path)}"
                        }}
                    ]}
                ]
            )

            description = response.choices[0].message.content.strip()

            return jsonify({
                "mode": "scene",
                "description": description
            })

    except ImportError as e:
        pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
        return jsonify({
            "error": f"'{pkg}' ఇన్‌స్టాల్ కాలేదు. "
                     f"Run: pip install easyocr Pillow pdfplumber"
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
# def ocr_image_easyocr(image_path: str) -> str:
#     """
#     OCR using easyocr (Telugu + English).
#     No system dependencies — pure pip install.
#     Preprocesses image to improve accuracy on phone photos.
#     """
#     from PIL import Image, ImageFilter, ImageEnhance
#     import numpy as np

#     # ── Preprocess for better accuracy ────────────────────────
#     img = Image.open(image_path)

#     # Normalise mode
#     if img.mode not in ("RGB", "L"):
#         img = img.convert("RGB")

#     # Scale up small images — OCR accuracy drops on tiny images
#     w, h = img.size
#     if max(w, h) < 1500:
#         scale = 1500 / max(w, h)
#         img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

#     # Sharpen + contrast boost (helps with blurry phone photos)
#     img = img.filter(ImageFilter.SHARPEN)
#     img = ImageEnhance.Contrast(img).enhance(1.4)

#     # easyocr accepts a numpy array or file path
#     img_array = np.array(img)

#     reader  = get_ocr_reader()
#     results = reader.readtext(img_array, detail=0, paragraph=True)
#     return "\n".join(results)



def ocr_image_tesseract(image_path: str) -> str:
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    gray = cv2.medianBlur(gray, 3)

    # Threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Convert to PIL
    pil_img = Image.fromarray(thresh)

    # Telugu + English
    text = pytesseract.image_to_string(pil_img, lang='eng+tel')

    return text

def extract_pdf_text(pdf_path: str) -> str:
    """Extract selectable (non-scanned) text from PDF using pdfplumber."""
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n\n".join(pages)


def ocr_pdf_pages_easyocr(pdf_path: str) -> str:
    """
    OCR a scanned PDF: convert each page to image, then run easyocr.
    Requires: pip install pdf2image
    No sudo needed — pdf2image uses its own bundled poppler on Windows/Mac.
    On Linux without sudo: pip install pdf2image and
      download poppler binaries to a local path (see pdf2image docs).
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        return (
            "[స్కాన్ PDF చదవలేకపోయాను. "
            "pip install pdf2image అమలు చేయండి]"
        )

    try:
        pages = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        return f"[PDF పేజీలు తెరవడంలో లోపం: {e}]"

    results = []
    for i, page_img in enumerate(pages):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            page_img.save(tmp.name)
            page_path = tmp.name
        try:
            text = ocr_image_tesseract(page_path)
            if text.strip():
                results.append(f"[పేజి {i+1}]\n{text.strip()}")
        finally:
            try:
                os.unlink(page_path)
            except Exception:
                pass

    return "\n\n".join(results)


@app.route("/")
def home():
    return render_template("smart_report.html")


@app.route("/submit", methods=["POST"])
def submit():
    # data        = request.json
    # user_prompt = data.get("prompt")
    user_prompt = request.form.get("text","")
    image = request.files.get("image")
#     full_prompt = f"""
# You are a report generator.

# IMPORTANT RULES:
# - Output MUST be completely in Telugu
# - Do NOT use English words
# - Base the report strictly on the user input
# - You may include logical and general real-world reasons (like supply issues, economic conditions, or current affairs) ONLY if they are commonly known and relevant
# - Do NOT add specific details like exact dates, names, or locations unless mentioned
# - Do NOT create imaginary or false information
# - Keep the explanation realistic and meaningful

# TASK:
# 1. Expand the given input into a detailed report (max 500 words)
# 2. Include possible general reasons if relevant (like shortages, global issues, etc.)
# 3. Provide exactly 3 suggestions

# FORMAT:

# REPORT:
# <detailed Telugu report>

# SUGGESTIONS:
# 1. ...
# 2. ...
# 3. ...

# USER INPUT:
# {user_prompt}
# """
    full_prompt = f"""
నువ్వు సీనియర్ వార్తా సంపాదకుడివి.

ఇన్‌పుట్: ఒక చిన్న, అసంపూర్తి రిపోర్టర్ నోట్ + అవసరమైతే ఫోటో వివరణ

పని:
- తెలుగులో మాత్రమే రాయండి
- ఇచ్చిన సమాచారాన్ని స్పష్టమైన, వాస్తవిక వార్తగా (400–500 పదాలు) మార్చండి
- తటస్థంగా, సక్రమంగా, సులభంగా చదవగలిగేలా ఉండాలి
- ఊహాజనిత లేదా తప్పుడు వివరాలు జోడించవద్దు
- ఫోటో వివరణ ఉంటే దాన్ని వార్తలో సహజంగా చేర్చండి
- ఫోటోలో కనిపించిన విషయాలు ముఖ్యమైతే అవి ప్రధానంగా ఉండాలి
- అవసరమైతే "ఫోటోలో కనిపించిన విధంగా..." వంటి సహజమైన వాక్య నిర్మాణం ఉపయోగించండి
- తర్వాత లోపించిన సమాచారాన్ని గరిష్టంగా 3 పాయింట్లుగా ఇవ్వండి
- ప్రతి పాయింట్ ఒక స్పష్టమైన ప్రశ్న లేదా లోపమైన సమాచారం కావాలి

ఫార్మాట్:
NEWS:
<వార్త వివరణ>

SUGGESTIONS:
- <లోపించిన సమాచారం 1>
- <లోపించిన సమాచారం 2>
- <లోపించిన సమాచారం 3>

ఇన్‌పుట్:
{user_prompt}
"""

    # response = requests.post(, json={
    #     "model": MODEL,
    #     "prompt": full_prompt,
    #     "stream": False
    # })
    input_tokens = len(encoding.encode(full_prompt))
    
    response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": full_prompt}]
                )
    text = response.choices[0].message.content.strip()
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    print(f"Prompt tokens: {usage}")
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    # result     = response.json()
    # output     = result.get("response", "")
    # text       = output.strip()
    text_upper = text.upper()

    if "SUGGESTIONS:" in text_upper:
        split_index  = text_upper.find("SUGGESTIONS:")
        label_length = len("SUGGESTIONS:")
    elif "సూచనలు:" in text:
        split_index  = text.find("సూచనలు:")
        label_length = len("సూచనలు:")
    else:
        split_index = -1

    if split_index != -1:
        report      = text[:split_index].replace("REPORT:", "").replace("నివేదిక:", "").strip()
        suggestions = text[split_index + label_length:].strip()
    else:
        report      = text
        suggestions = "No suggestions"
    print(f"Input tokens: {input_tokens}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
    return jsonify({"report": report, "suggestions": suggestions})


# @app.route("/save_report", methods=["POST"])
# def save_report():
#     data   = request.json
#     name = "User"
#     description = data.get("report")
#     report_date = data.get("date")
#     report_time = data.get("time")
#     if report_date:
#         report_date = datetime.strptime(report_date, "%d-%m-%Y").date()

#     if report_time:
#         report_time = datetime.strptime(report_time, "%H:%M:%S").time()

#     conn   = dbConnection.get_connection()
#     cursor = conn.cursor()
#     # query  = "INSERT INTO reports (name, date, time, report) VALUES (%s, %s, %s, %s)"
#     cursor.execute(
#     "INSERT INTO ANDHRAPRABHA_REPORT (NAME, DESCRIPTION, REPORT_DATE, REPORT_TIME) VALUES (?, ?, ?, ?)",
#         (name, description, report_date, report_time)
#     )
#     conn.commit()
#     # values = ("User", data.get("date"), data.get("time"), data.get("report"))
#     # cursor.execute(query, values)
#     cursor.close()
#     conn.close()
#     return jsonify({"status": "saved"})



from datetime import datetime

@app.route("/save_report", methods=["POST"])
def save_report():
    data = request.json

    name = "User"
    description = data.get("report")

    report_date_str = data.get("date")
    report_time_str = data.get("time")

    print("RAW DATE:", report_date_str)
    print("RAW TIME:", report_time_str)

    # ✅ Convert to proper Python types
    report_date = None
    report_time = None

    try:
        if report_date_str:
            report_date = datetime.strptime(report_date_str, "%Y-%m-%d").date()
    except Exception as e:
        print("DATE ERROR:", e)

    try:
        if report_time_str:
            report_time = datetime.strptime(report_time_str, "%H:%M:%S").time()
    except Exception as e:
        print("TIME ERROR:", e)

    conn = dbConnection.get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO ANDHRAPRABHA_REPORT (NAME, DESCRIPTION, REPORT_DATE, REPORT_TIME) VALUES (?, ?, ?, ?)",
        (name, description, report_date, report_time)
    )

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"status": "saved"})

@app.route("/get_reports")
def get_reports():
    conn   = dbConnection.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ANDHRAPRABHA_REPORT ORDER BY id DESC")
    # data   = cursor.fetchall()
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    # ✅ Convert to list of dicts
    data = []
    for row in rows:
        data.append(dict(zip(columns, row)))

    cursor.close()
    conn.close()
    return jsonify(data)


@app.route("/reports")
def reports_page():
    return render_template("view.html")

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5020))  # Use CF-provided port, fallback to 5020
    # app.run(host="0.0.0.0", port=5020, debug=True)
    app.run(host="0.0.0.0", port=8080, debug=True)
