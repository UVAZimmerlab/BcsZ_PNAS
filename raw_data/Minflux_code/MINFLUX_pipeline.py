from pathlib import Path
from itertools import combinations
import hashlib
import os
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

INPUT = Path(os.environ.get("MINFLUX_INPUT_DIR", Path.home() / "Desktop" / "MINFLUX_FIN"))
OUTPUT = Path(os.environ.get("MINFLUX_OUTPUT_DIR", INPUT / "MINFLUX_results"))
FILES = {"Positive": "Positive_control.xlsx",
         "Experiment": "Experiment.xlsx",
         "Negative": "Negative.xlsx"}
PREFIX = {"Positive": "AB", "Experiment": "AZ", "Negative": "AL"}
BATCHES = {"Positive": {1: (1, 6), 2: (7, 11), 3: (12, 14)},
           "Experiment": {1: (1, 4), 2: (5, 6), 3: (7, 10)},
           "Negative": {1: (1, 6), 2: (7, 11), 3: (12, 16), 4: (17, 24)}}
COLUMNS = {"loc x": "x", "loc y": "y", "loc z": "z", "TRC id": "trc_id",
           "DCR (itr 7)": "dcr7", "DCR (itr 9)": "dcr9", "EFO": "efo"}
DISTANCES = np.arange(50, 301, 25)
NN_COLUMNS = [f"NN_{d}nm_pct" for d in DISTANCES]
MIN_LOCS, MIN_CENTROIDS, MIN_CELLS = 5, 4, 2
REPEATS, SEED = 2000, 42


def batch_id(condition, sheet):
    match = re.fullmatch(r"([A-Za-z]+)_(\d+)", sheet)
    if match is None or match.group(1).upper() != PREFIX[condition]:
        return None
    number = int(match.group(2))
    return next((r for r, (lo, hi) in BATCHES[condition].items() if lo <= number <= hi), None)


def read_cell(raw, condition, sheet):
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{condition}/{sheet}: missing columns {sorted(missing)}")
    data = raw[list(COLUMNS)].rename(columns=COLUMNS).copy()
    for col in ("x", "y", "z", "dcr7", "dcr9", "efo"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=list(COLUMNS.values()))
    n_nonfinite = len(raw) - len(data)
    before = len(data)
    data = data[(data.efo > 0) & (data.efo <= 100_000)].copy()
    n_efo = before - len(data)
    before = len(data)
    data = data[data.groupby("trc_id").trc_id.transform("size") >= MIN_LOCS].copy()
    qc = {"condition": condition, "sample_id": sheet,
          "biological_replicate": batch_id(condition, sheet), "raw_localizations": len(raw),
          "missing_values_removed": n_nonfinite, "efo_removed": n_efo,
          "short_trace_localizations_removed": before - len(data), "retained_localizations": len(data)}
    if data.empty:
        return pd.DataFrame(), qc
    data[["x", "y", "z"]] *= 1e9
    centers = data.groupby("trc_id", sort=False)[["x", "y", "z", "dcr7", "dcr9", "efo"]].median().reset_index()
    centers["n_localizations"] = data.groupby("trc_id", sort=False).size().to_numpy()
    low = centers.dcr7.between(.10, .40) & centers.dcr9.between(.10, .40)
    high = centers.dcr7.between(.55, .85) & centers.dcr9.between(.55, .85)
    centers["dcr_class"] = np.select([low, high], ["low", "high"], default="other")
    centers.insert(0, "condition", condition)
    centers.insert(1, "sample_id", sheet)
    centers.insert(2, "biological_replicate", qc["biological_replicate"])
    return centers, qc


def cell_nn(centers, condition, sheet):
    counts = centers.dcr_class.value_counts() if not centers.empty else pd.Series(dtype=int)
    nl, nh = int(counts.get("low", 0)), int(counts.get("high", 0))
    result = {"low_centroids": nl, "high_centroids": nh,
              "other_centroids": int(counts.get("other", 0)),
              "qualifies_spatial_qc": min(nl, nh) >= MIN_CENTROIDS}
    result.update({col: np.nan for col in NN_COLUMNS})
    result.update(NN200_low_to_high_pct=np.nan, NN200_high_to_low_pct=np.nan)
    if not result["qualifies_spatial_qc"]:
        return result
    low = centers.loc[centers.dcr_class == "low", ["x", "y", "z"]].to_numpy(float)
    high = centers.loc[centers.dcr_class == "high", ["x", "y", "z"]].to_numpy(float)
    size = min(nl, nh)
    token = f"{SEED}_{condition}_{sheet}_COMMON_2D_low_0.10_0.40_high_0.55_0.85_label_shuffle"
    rng = np.random.default_rng(int(hashlib.sha256(token.encode()).hexdigest()[:8], 16))
    forward, backward = np.zeros(len(DISTANCES)), np.zeros(len(DISTANCES))
    n_runs = 1 if nl == nh else REPEATS
    for _ in range(n_runs):
        a = low[rng.choice(nl, size, replace=False)] if nl > size else low
        b = high[rng.choice(nh, size, replace=False)] if nh > size else high
        da = cKDTree(b).query(a, k=1)[0]
        db = cKDTree(a).query(b, k=1)[0]
        forward += np.mean(da[:, None] <= DISTANCES[None, :], axis=0)
        backward += np.mean(db[:, None] <= DISTANCES[None, :], axis=0)
    forward, backward = forward / n_runs, backward / n_runs
    result.update({col: 50 * (forward[i] + backward[i]) for i, col in enumerate(NN_COLUMNS)})
    i = np.flatnonzero(DISTANCES == 200)[0]
    result["NN200_low_to_high_pct"] = 100 * forward[i]
    result["NN200_high_to_low_pct"] = 100 * backward[i]
    return result


def mean_sem(values):
    values = np.asarray(values, float)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return float(values.mean()), sd, sd / np.sqrt(len(values)) if len(values) > 1 else np.nan


def exact_permutation(x, y):
    pooled = np.r_[x, y]
    observed = float(np.mean(x) - np.mean(y))
    diffs = []
    for inds in combinations(range(len(pooled)), len(x)):
        selected = np.zeros(len(pooled), dtype=bool)
        selected[list(inds)] = True
        diffs.append(float(pooled[selected].mean() - pooled[~selected].mean()))
    return observed, sum(abs(v) >= abs(observed) - 1e-12 for v in diffs) / len(diffs), len(diffs)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_cells, all_centers = [], []
    for condition, filename in FILES.items():
        with pd.ExcelFile(INPUT / filename) as book:
            for sheet in book.sheet_names:
                centers, qc = read_cell(pd.read_excel(book, sheet_name=sheet), condition, sheet)
                qc.update(cell_nn(centers, condition, sheet))
                all_cells.append(qc)
                if not centers.empty:
                    all_centers.append(centers)
    cells = pd.DataFrame(all_cells)
    eligible = cells[cells.qualifies_spatial_qc & cells.biological_replicate.notna()]
    batch_counts = eligible.groupby(["condition", "biological_replicate"]).size()
    cells["included_in_paper"] = cells.apply(
        lambda r: bool(r.qualifies_spatial_qc and pd.notna(r.biological_replicate)
                       and batch_counts.get((r.condition, r.biological_replicate), 0) >= MIN_CELLS), axis=1)
    cells["exclusion_reason"] = np.select(
        [cells.biological_replicate.isna(), ~cells.qualifies_spatial_qc, ~cells.included_in_paper],
        ["unmapped batch", "insufficient low/high centroids", "fewer than 2 qualifying cells in batch"], default="")
    included = cells[cells.included_in_paper]
    if included.empty:
        raise ValueError("No cells pass the spatial and batch QC rules")
    replicate = included.groupby(["condition", "biological_replicate"], as_index=False).agg(
        cells=("sample_id", "size"), cell_ids=("sample_id", ", ".join),
        **{col: (col, "mean") for col in NN_COLUMNS +
           ["NN200_low_to_high_pct", "NN200_high_to_low_pct"]})
    replicate["NN200_pct"] = replicate["NN_200nm_pct"]
    replicate["biological_replicate"] = replicate.biological_replicate.astype(int)
    conditions, curves = [], []
    for condition, group in replicate.groupby("condition", sort=False):
        mean, sd, sem = mean_sem(group.NN200_pct)
        conditions.append({"condition": condition, "biological_replicates": len(group),
                           "included_cells": int(group.cells.sum()), "mean_NN200_pct": mean,
                           "SD_NN200_pct": sd, "SEM_NN200_pct": sem})
        for distance, col in zip(DISTANCES, NN_COLUMNS):
            avg, _, err = mean_sem(group[col])
            curves.append({"condition": condition, "threshold_nm": int(distance),
                           "mean_NN_pct": avg, "SEM_NN_pct": err})
    summary, curves = pd.DataFrame(conditions), pd.DataFrame(curves)
    az = replicate.loc[replicate.condition == "Experiment", "NN200_pct"].to_numpy()
    al = replicate.loc[replicate.condition == "Negative", "NN200_pct"].to_numpy()
    tests = pd.DataFrame(columns=["comparison", "difference_pp", "p_two_sided_exact", "permutations"])
    if len(az) and len(al):
        effect, p, n = exact_permutation(az, al)
        tests.loc[0] = ["Experiment vs Negative", effect, p, n]
    for name, table in {"cells": cells, "centroids": pd.concat(all_centers, ignore_index=True),
                        "biological_replicates": replicate, "condition_summary": summary,
                        "nn_curves": curves, "primary_test": tests}.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    print(summary.to_string(index=False))
    print(tests.to_string(index=False))
    print(f"CSV files saved in: {OUTPUT}")


if __name__ == "__main__":
    main()
