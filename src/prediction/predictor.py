import difflib
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(BASE, "data", "cutoffs", "cutoffs.csv")
INTAKE_CSV = os.path.join(BASE, "data", "cutoffs", "intakes.csv")


BORDERLINE_MARGIN = 0.05


def _start_year(data_year):
    m = re.match(r"(\d{4})", str(data_year))
    return int(m.group(1)) if m else None


class CutoffPredictor:

    def __init__(self, csv_path=CSV, intake_csv=INTAKE_CSV):
        df = pd.read_csv(csv_path)
        df = df[df["syllabus"] != "old"].copy()
        df["year"] = df["data_year"].map(_start_year)
        df = df.dropna(subset=["year", "course", "district"])
        df["year"] = df["year"].astype(int)
        self.df = df
        self.courses = sorted(df["course"].dropna().unique())
        self.universities = sorted(df["university"].dropna().unique())
        self.districts = sorted(df["district"].dropna().unique())

        self.intakes = None
        self.intake_courses = []
        if os.path.exists(intake_csv):
            self.intakes = pd.read_csv(intake_csv)
            self.intake_courses = sorted(self.intakes["course"].dropna().unique())

    
    def match_course(self, name):
        if not name:
            return None
        name = name.upper().strip()
        if name in self.courses:
            return name
        hits = [c for c in self.courses if name in c]
        if len(hits) == 1:
            return hits[0]
        close = difflib.get_close_matches(name, self.courses, n=1, cutoff=0.75)
        return close[0] if close else (hits[0] if hits else None)

    def match_university(self, name):
        if not name:
            return None
        low = name.lower().strip()
        hits = [u for u in self.universities if low in u.lower()]
        if hits:
            return min(hits, key=len)
        close = difflib.get_close_matches(name, self.universities, n=1, cutoff=0.6)
        return close[0] if close else None

    def match_district(self, name):
        if not name:
            return None
        name = name.upper().strip()
        if name in self.districts:
            return name
        close = difflib.get_close_matches(name, self.districts, n=1, cutoff=0.7)
        return close[0] if close else None

    
    def series(self, course, district, university=None):
        
        d = self.df[(self.df["course"] == course)
                    & (self.df["district"] == district)]
        if university:
            d = d[d["university"] == university]
        d = d.dropna(subset=["zscore"])
        
        agg = d.groupby("year")["zscore"].min().reset_index()
        return agg.sort_values("year")

    

    def forecast(self, course, district, university=None, target_year=None):
        
        s = self.series(course, district, university)
        if s.empty:
            return None
        years = s["year"].to_numpy().reshape(-1, 1)
        z = s["zscore"].to_numpy()
        last_year = int(s["year"].max())
        target_year = target_year or last_year + 1

        history = list(zip(s["year"].tolist(), s["zscore"].round(4).tolist()))
        if len(s) < 3:
            return {
                "course": course, "district": district,
                "university": university, "target_year": target_year,
                "prediction": round(float(z[-1]), 4),
                "band": None, "method": "last value (too few points)",
                "history": history, "slope": None,
            }

        model = LinearRegression().fit(years, z)
        pred = float(model.predict([[target_year]])[0])
        resid = z - model.predict(years)
        band = float(max(np.std(resid) * 2, 0.02))
        return {
            "course": course, "district": district,
            "university": university, "target_year": target_year,
            "prediction": round(pred, 4), "band": round(band, 4),
            "method": f"linear regression on {len(s)} years",
            "history": history, "slope": round(float(model.coef_[0]), 4),
        }

    def trend(self, course, district, university=None):
        
        f = self.forecast(course, district, university)
        if not f:
            return None
        slope = f["slope"]
        if slope is None:
            values = [z for _, z in f["history"]]
            slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
        if slope > 0.01:
            direction = "rising"
        elif slope < -0.01:
            direction = "falling"
        else:
            direction = "stable"
        f["direction"] = direction
        return f

    def admission_chances(self, zscore, district, course=None, top=15):
        
        courses = [course] if course else self.courses
        results = []
        for c in courses:
            f = self.forecast(c, district)
            if not f:
                continue
            margin = (f["band"] or BORDERLINE_MARGIN) / 2 + BORDERLINE_MARGIN
            gap = zscore - f["prediction"]
            if gap >= margin:
                verdict = "likely"
            elif gap >= -margin:
                verdict = "borderline"
            else:
                verdict = "unlikely"
            results.append({**f, "student_z": zscore, "gap": round(gap, 4),
                            "verdict": verdict})
        
        order = {"likely": 0, "borderline": 1, "unlikely": 2}
        results.sort(key=lambda r: (order[r["verdict"]], -r["prediction"]))
        if course:
            return results
        picks = [r for r in results if r["verdict"] != "unlikely"]
        return picks[:top] if picks else results[:5]

    

    def match_intake_course(self, name):
        if not name or not self.intake_courses:
            return None
        name = name.upper().strip()
        if name in self.intake_courses:
            return name
        hits = [c for c in self.intake_courses if name in c]
        if hits:
            return min(hits, key=len)
        close = difflib.get_close_matches(name, self.intake_courses, n=1,
                                          cutoff=0.75)
        return close[0] if close else None

    def intake_forecast(self, course):
        
        if self.intakes is None:
            return None
        s = (self.intakes[self.intakes["course"] == course]
             .dropna(subset=["intake"]).sort_values("year"))
        if s.empty:
            return None
        years = s["year"].to_numpy().reshape(-1, 1)
        n = s["intake"].to_numpy().astype(float)
        last_year = int(s["year"].max())
        history = list(zip(s["year"].tolist(), s["intake"].astype(int).tolist()))
        if len(s) < 3:
            return {"course": course, "target_year": last_year + 1,
                    "prediction": int(n[-1]), "slope": None,
                    "method": "last value (too few points)",
                    "history": history, "direction": "unknown"}
        model = LinearRegression().fit(years, n)
        pred = float(model.predict([[last_year + 1]])[0])
        slope = float(model.coef_[0])
        direction = ("growing" if slope > 1 else
                     "shrinking" if slope < -1 else "stable")
        return {"course": course, "target_year": last_year + 1,
                "prediction": int(round(pred)), "slope": round(slope, 2),
                "method": f"linear regression on {len(s)} years",
                "history": history, "direction": direction}

    
 
    def evaluate(self):
        
        newest = int(self.df["year"].max())
        y_true, y_pred, rows = [], [], []
        keys = self.df[["course", "district"]].drop_duplicates()
        for course, district in keys.itertuples(index=False):
            s = self.series(course, district)
            if len(s) < 3 or int(s["year"].max()) != newest:
                continue
            train = s[s["year"] < newest]
            if len(train) < 2:
                continue
            model = LinearRegression().fit(
                train["year"].to_numpy().reshape(-1, 1),
                train["zscore"].to_numpy())
            pred = float(model.predict([[newest]])[0])
            actual = float(s[s["year"] == newest]["zscore"].iloc[0])
            y_true.append(actual)
            y_pred.append(pred)
            rows.append({"course": course, "district": district,
                         "actual": actual, "predicted": round(pred, 4),
                         "abs_error": round(abs(actual - pred), 4)})
        mae = mean_absolute_error(y_true, y_pred) if y_true else None
        return {"n_series": len(y_true), "target_year": newest,
                "mae": round(mae, 4) if mae is not None else None,
                "details": pd.DataFrame(rows)}




    
