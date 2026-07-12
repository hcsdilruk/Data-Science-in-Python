import glob
import io
import os
import re
import sys

import pandas as pd
import pdfplumber
from pypdf import PdfReader, PdfWriter

DISTRICTS = [
    "COLOMBO", "GAMPAHA", "KALUTARA", "MATALE", "KANDY", "NUWARA ELIYA",
    "GALLE", "MATARA", "HAMBANTOTA", "JAFFNA", "KILINOCHCHI", "MANNAR",
    "MULLAITIVU", "VAVUNIYA", "TRINCOMALEE", "BATTICALOA", "AMPARA",
    "PUTTALAM", "KURUNEGALA", "ANURADHAPURA", "POLONNARUWA", "BADULLA",
    "MONARAGALA", "KEGALLE", "RATNAPURA",
]
DISTRICT_FIRST_WORDS = {d.split()[0] for d in DISTRICTS}

NUM_RE = re.compile(r"^\d\.\d{4}$")
ANY_NUM_RE = re.compile(r"\d\.\d{4}|\d{4}\.\d")
CODE_RE = re.compile(r"^\d{3}[A-Z]$")
YEAR_RE = re.compile(r"ACADEMIC YEAR\s*[-:–]?\s*(\d{4})\s*/\s*(\d{4})")

VOCAB = re.compile(
    r"UNIVERSITY|SCIENCE|MEDICINE|ENGINEERING|TECHNOLOGY|MANAGEMENT|"
    r"SURGERY|NURSING|PHARMACY|AGRICULTURE|DENTAL|ARTS|LAW|COMMERCE|"
    r"STUDIES|LANKA|COLOMBO|PERADENIYA|JAFFNA|RUHUNA|KELANIYA|MORATUWA|"
    r"INSTITUTE|AESTHETIC|DRAMA|THEATRE|VISUAL|PERFORMING|MUSIC|DANCE|"
    r"EDUCATION|INFORMATION|BUSINESS|FINANCE|STATISTICS|DESIGN",
    re.IGNORECASE,
)

def _rotated_page_words(src_path, page_idx, deg):
    
    reader = PdfReader(src_path)
    page = reader.pages[page_idx]
    if deg:
        page.rotate(deg)
    writer = PdfWriter()
    writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    with pdfplumber.open(buf) as pdf:
        p = pdf.pages[0]
        words = p.extract_words(extra_attrs=["upright"])
        text = p.extract_text() or ""
    return words, text


def _candidate_pages(src_path):
    
    out = []
    with pdfplumber.open(src_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            if len(ANY_NUM_RE.findall(text)) >= 40:
                out.append(i)
    return out


def _grid_rotation(src_path, page_idx):
    
    for deg in (0, 90, 180, 270):
        words, text = _rotated_page_words(src_path, page_idx, deg)
        upright = [w["text"] for w in words if w.get("upright", True)]
        district_hits = sum(1 for d in DISTRICT_FIRST_WORDS if d in upright)
        numbers = sum(1 for t in upright if NUM_RE.match(t))
        if district_hits >= 15 and numbers >= 50:
            return deg, words, text
    return None, None, None


def _cluster_lines(words, tol=3.0):
    
    lines = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def _district_rows(words):
    
    rows = []
    for line in _cluster_lines([w for w in words if w.get("upright", True)]):
        tokens = [w["text"] for w in line]
        for d in DISTRICTS:
            parts = d.split()
            if tokens[: len(parts)] == parts:
                rows.append((d, line[len(parts):]))
                break
    return rows


def _column_centers(rows, gap=6.0):
    
    centers = sorted(
        (w["x0"] + w["x1"]) / 2
        for _, vw in rows for w in vw
        if NUM_RE.match(w["text"]) or w["text"] == "NQC"
    )
    if not centers:
        return []
    clusters = [[centers[0]]]
    for c in centers[1:]:
        if c - clusters[-1][-1] <= gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return [sum(cl) / len(cl) for cl in clusters]


def _read_stack(stack_words):
    
    lines = []
    for w in sorted(stack_words, key=lambda w: (round(w["x0"]), w["top"])):
        if lines and abs(w["x0"] - lines[-1][0]["x0"]) <= 3:
            lines[-1].append(w)
        else:
            lines.append([w])

    def render(reverse):
        parts = []
        for line in lines:
            ws = [w["text"] for w in sorted(line, key=lambda w: w["top"])]
            if reverse:
                ws = [t[::-1] for t in reversed(ws)]
            parts.append(" ".join(ws))
        return " ".join(parts)

    def score(s):
        pts = 2 * len(VOCAB.findall(s))
       
        if "(" in s and (")" not in s or s.index("(") < s.index(")")):
            pts += 1
        elif ")" in s:
            pts -= 1
        return pts

    fwd, rev = render(False), render(True)
    return fwd if score(fwd) >= score(rev) else rev


def _column_headers(words, col_centers):
    
    if not col_centers:
        return {}
    gaps = [b - a for a, b in zip(col_centers, col_centers[1:])]
    half = (min(gaps) / 2 + 1) if gaps else 12.0

    code_at = {}
    stacks = {c: [] for c in col_centers}
    for w in words:
        cx = (w["x0"] + w["x1"]) / 2
        c = min(col_centers, key=lambda cc: abs(cc - cx))
        if abs(c - cx) > half:
            continue
        t = w["text"]
        if CODE_RE.match(t) or CODE_RE.match(t[::-1]):
            code_at[c] = t if CODE_RE.match(t) else t[::-1]
        elif not w.get("upright", True):
            stacks[c].append(w)

    headers = {}
    for i, c in enumerate(col_centers):
        text = _read_stack(stacks[c]) if stacks[c] else ""
        m = re.search(r"\(([^)]*)\)?", text)
        university = m.group(1).strip().rstrip(")") if m else ""
        course = re.sub(r"\([^)]*\)?", " ", text)
        course = " ".join(t for t in course.split() if t not in ("#", "*")).strip(" #*")
        headers[c] = (code_at.get(c, f"C{i + 1:02d}"), course, university)
    return headers


def extract_pdf(src_path):
   
    out = []
    for page_idx in _candidate_pages(src_path):
        deg, words, text = _grid_rotation(src_path, page_idx)
        if deg is None:
            continue
        flat = " ".join(text.split())

        years = YEAR_RE.findall(flat)
        data_year = None
        if years:
            
            y = min(years, key=lambda y: int(y[0]))
            data_year = f"{y[0]}/{y[1]}"

        syllabus = ""
        if re.search(r"New Syllabus", flat, re.I):
            syllabus = "new"
        elif re.search(r"Old Syllabus", flat, re.I):
            syllabus = "old"

        rows = _district_rows(words)
        col_centers = _column_centers(rows)
        if not col_centers:
            continue
        headers = _column_headers(words, col_centers)

        for district, value_words in rows:
            for w in value_words:
                t = w["text"]
                if not (NUM_RE.match(t) or t == "NQC"):
                    continue
                cx = (w["x0"] + w["x1"]) / 2
                c = min(col_centers, key=lambda cc: abs(cc - cx))
                code, course, university = headers[c]
                out.append({
                    "data_year": data_year,
                    "syllabus": syllabus,
                    "district": district,
                    "uni_code": code,
                    "course": course,
                    "university": university,
                    "zscore": float(t) if t != "NQC" else None,
                    "status": "OK" if t != "NQC" else "NQC",
                    "source": os.path.basename(src_path),
                    "page": page_idx + 1,
                })
    return out


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdf_dirs = [os.path.join(base, "data", "pdfs"),
                os.path.join(base, "data", "cutoffs_raw")]
    out_dir = os.path.join(base, "data", "cutoffs")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for d in pdf_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.pdf"))):
            print(f"Processing {os.path.basename(path)} ...")
            rows = extract_pdf(path)
            print(f"  extracted {len(rows)} cut-off values")
            all_rows.extend(rows)

    if not all_rows:
        sys.exit("No cut-off data extracted")

    df = pd.DataFrame(all_rows)
   
    df = df.drop_duplicates(
        subset=["data_year", "syllabus", "district", "course", "university"])
    out = os.path.join(out_dir, "cutoffs.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")
    print(df.groupby(["data_year", "syllabus"]).size().to_string())


if __name__ == "__main__":
    main()
