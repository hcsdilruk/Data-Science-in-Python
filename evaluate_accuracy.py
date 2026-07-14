###Accuracy evaluation for the cut-off prediction feature.
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.prediction.predictor import CutoffPredictor


def run():
    p = CutoffPredictor()
    newest = int(p.df["year"].max())

    truth, persistence, trend, mean2 = [], [], [], []
    for course, district in p.df[["course", "district"]].drop_duplicates().itertuples(index=False):
        s = p.series(course, district)
        if len(s) < 3 or int(s["year"].max()) != newest:
            continue
        train = s[s["year"] < newest]
        if len(train) < 2:
            continue
        yrs = train["year"].to_numpy().reshape(-1, 1)
        z = train["zscore"].to_numpy()
        truth.append(float(s[s["year"] == newest]["zscore"].iloc[0]))
        persistence.append(float(z[-1]))
        trend.append(float(LinearRegression().fit(yrs, z).predict([[newest]])[0]))
        mean2.append(float(z[-2:].mean()))

    truth = np.array(truth)

    def metrics(pred):
        pred = np.array(pred)
        err = np.abs(pred - truth)
        return (mean_absolute_error(truth, pred),
                float(np.sqrt(mean_squared_error(truth, pred))),
                100 * float(np.mean(err <= 0.05)),
                100 * float(np.mean(err <= 0.10)),
                100 * float(np.mean(err <= 0.20)))

    print("=" * 60)
    print(" PREDICTION ACCURACY  (leave-last-year-out backtest)")
    print("=" * 60)
    print(f" Series evaluated : {len(truth)}      Target year : {newest}/{newest + 1}")
    print("-" * 60)
    print(f" {'method':16}{'MAE':>7}{'RMSE':>7}{'Acc.10':>8}{'Acc.20':>8}")
    for name, pred in [("persistence", persistence),
                       ("linear trend", trend),
                       ("2-year mean *", mean2)]:
        mae, rmse, a05, a10, a20 = metrics(pred)
        print(f" {name:16}{mae:7.3f}{rmse:7.3f}{a10:7.1f}%{a20:7.1f}%")
    print("-" * 60)
    mae, rmse, a05, a10, a20 = metrics(mean2)
    print(f" ADOPTED = 2-year mean :")
    print(f"   MAE {mae:.3f} | RMSE {rmse:.3f} | within 0.05: {a05:.1f}% |"
          f" within 0.10: {a10:.1f}% | within 0.20: {a20:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run()
