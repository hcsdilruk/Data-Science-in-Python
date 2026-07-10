
import re

PREDICT_RE = re.compile(
    r"predict|forecast|will\s+(the|i|be)|next\s+year|chance|chances|"
    r"can\s+i\s+get|could\s+i\s+get|what\s+can\s+i|trend|rising|falling|"
    r"going\s+(up|down)|increase|decrease|expect|likely|my\s+z[\s-]?score|"
    r"i\s+(got|have|scored)",
    re.IGNORECASE,
)
TREND_RE = re.compile(r"trend|rising|falling|going\s+(up|down)|increase|decrease",
                      re.IGNORECASE)
INTAKE_RE = re.compile(r"intake|how\s+many\s+students|seats|places|vacanc|demand",
                       re.IGNORECASE)
CHANCE_RE = re.compile(r"chance|can\s+i|could\s+i|what\s+can\s+i|i\s+(got|have|scored)|my\s+z",
                       re.IGNORECASE)
Z_RE = re.compile(r"(\d\.\d{1,4})")
YEAR_RE = re.compile(r"(20\d{2})")


class QuestionRouter:

    def __init__(self, predictor):
        self.predictor = predictor

    def classify(self, question):
        return "prediction" if PREDICT_RE.search(question) else "factual"



    def parse(self, question):
        q_upper = question.upper()

        district = None
        for d in self.predictor.districts:
            if d in q_upper:
                district = d
                break
        if district is None:
            for token in re.findall(r"[A-Za-z]{4,}", question):
                district = self.predictor.match_district(token)
                if district:
                    break

        course = None
        
        for c in sorted(self.predictor.courses, key=len, reverse=True):
            if c in q_upper:
                course = c
                break
        if course is None:
            for m in re.finditer(r"[A-Za-z][A-Za-z &]{3,}", question):
                cand = self.predictor.match_course(m.group())
                if cand:
                    course = cand
                    break

        z = None
        m = Z_RE.search(question)
        if m:
            z = float(m.group(1))

        target_year = None
        m = YEAR_RE.search(question)
        if m:
            target_year = int(m.group(1))

        if INTAKE_RE.search(question):
            intent = "intake"
        elif z is not None or CHANCE_RE.search(question):
            intent = "chance"
        elif TREND_RE.search(question):
            intent = "trend"
        else:
            intent = "forecast"

        return {"intent": intent, "course": course, "district": district,
                "zscore": z, "target_year": target_year}

    def _intake_course_from(self, question):
        q = question.upper()
        for c in sorted(self.predictor.intake_courses, key=len, reverse=True):
            if c in q:
                return c
        for m in re.finditer(r"[A-Za-z][A-Za-z &]{3,}", question):
            cand = self.predictor.match_intake_course(m.group())
            if cand:
                return cand
        return None

    

    def answer(self, question):
        p = self.parse(question)
        pred = self.predictor

        if p["intent"] == "intake":
            
            course = (pred.match_intake_course(p["course"])
                      or self._intake_course_from(question))
            if course is None:
                return ("Please mention the course of study whose intake "
                        "you are asking about.")
            f = pred.intake_forecast(course)
            if not f:
                return "No intake history found for that course."
            hist = ", ".join(f"{y}: {n}" for y, n in f["history"])
            return (f"Predicted proposed intake for {f['course']} in "
                    f"{f['target_year']}: about {f['prediction']} students "
                    f"({f['direction']}, {f['method']}).\n"
                    f"History — {hist}\n"
                    f"Note: this is a trend estimate, not an official figure.")

        if p["district"] is None:
            return ("Please mention your district (e.g. Colombo, Gampaha) — "
                    "cut-offs differ by district.")

        if p["intent"] == "chance":
            if p["zscore"] is None:
                return ("Please include the Z-score you obtained, e.g. "
                        "'I got 1.85 from Gampaha — what can I get into?'")
            results = pred.admission_chances(p["zscore"], p["district"],
                                             course=p["course"])
            if not results:
                return "No historical cut-off data found for that selection."
            lines = [f"With a Z-score of {p['zscore']} from {p['district']} "
                     f"district (based on trends up to "
                     f"{results[0]['target_year'] - 1}):"]
            for r in results[:12]:
                band = f" ±{r['band']}" if r["band"] else ""
                lines.append(f"  [{r['verdict'].upper():10s}] {r['course']}"
                             f" — predicted cut-off {r['prediction']}{band}"
                             f" (your margin {r['gap']:+.3f})")
            lines.append("Note: predictions are trend estimates from past "
                         "cut-offs; actual values vary between years.")
            return "\n".join(lines)

        if p["course"] is None:
            return ("Please mention the course of study (e.g. Medicine, "
                    "Engineering, Law).")

        if p["intent"] == "trend":
            t = pred.trend(p["course"], p["district"])
            if not t:
                return "No historical cut-off data found for that selection."
            hist = ", ".join(f"{y}: {z}" for y, z in t["history"])
            return (f"The cut-off for {t['course']} ({p['district']} district) "
                    f"is {t['direction']} (slope {t['slope']:+.4f}/year).\n"
                    f"History — {hist}")

        f = pred.forecast(p["course"], p["district"],
                          target_year=p["target_year"])
        if not f:
            return "No historical cut-off data found for that selection."
        hist = ", ".join(f"{y}: {z}" for y, z in f["history"])
        band = f" (±{f['band']})" if f["band"] else ""
        return (f"Predicted cut-off for {f['course']} ({p['district']} "
                f"district) in {f['target_year']}: {f['prediction']}{band}\n"
                f"Method: {f['method']}. History — {hist}\n"
                f"Note: this is a trend estimate, not an official figure.")
