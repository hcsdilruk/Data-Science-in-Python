import glob
import os
import re

import pandas as pd
import pdfplumber

HEADING_RE = re.compile(r"^\s*\d\.\d[\d.]*\s+([A-Za-z][A-Za-z &,'()/-]{2,70})\s*$")
CODE_RE = re.compile(r"Course Code\s*[-–:]\s*(\d+)", re.I)
INTAKE_RE = re.compile(r"Proposed Intake\s*[-–:]\s*([\d,]+)", re.I)
MULTI_INTAKE_RE = re.compile(
    r"Proposed Intakes?\s*[:\-–]\s*((?:[A-Za-z &,'/-]+\s*[-–]\s*[\d,]+\s*;?\s*)+)",
    re.I)
EDITION_RE = re.compile(r"\((\d{4})[-–]\d{4}\)")


def extract_pdf(path):
    m = EDITION_RE.search(os.path.basename(path))
    year = int(m.group(1)) if m else None

    rows = []
    heading = None
    code = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            for line in text.splitlines():
                h = HEADING_RE.match(line)
                if h:
                    name = h.group(1).strip()
                    
                    if not re.search(r"\b(the|and of|for the)\b", name.lower()):
                        heading = " ".join(name.split())
                        code = None
                c = CODE_RE.search(line)
                if c:
                    code = c.group(1)

                mm = MULTI_INTAKE_RE.search(line)
                if mm and ";" in mm.group(1):
                    for part in mm.group(1).split(";"):
                        pm = re.match(r"\s*([A-Za-z &,'/-]+?)\s*[-–]\s*([\d,]+)",
                                      part)
                        if pm:
                            rows.append({
                                "year": year,
                                "course": pm.group(1).strip().upper(),
                                "course_code": code,
                                "intake": int(pm.group(2).replace(",", "")),
                                "source": os.path.basename(path),
                            })
                    continue

                i = INTAKE_RE.search(line)
                if i and heading:
                    rows.append({
                        "year": year,
                        "course": heading.upper(),
                        "course_code": code,
                        "intake": int(i.group(1).replace(",", "")),
                        "source": os.path.basename(path),
                    })
    return rows


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(base, "data", "cutoffs")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for path in sorted(glob.glob(os.path.join(base, "data", "pdfs", "*.pdf"))):
        print(f"Processing {os.path.basename(path)} ...")
        rows = extract_pdf(path)
        print(f"  extracted {len(rows)} intake figures")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["year", "course"])
    out = os.path.join(out_dir, "intakes.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")
    print(df.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
