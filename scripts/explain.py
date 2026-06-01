"""Train and produce a feature-importance report (figure + JSON)."""
import argparse, json, os, sys
sys.path.insert(0, "src")
from sklearn.model_selection import train_test_split
from churn.data import FEATURES, TARGET, load_data
from churn.pipeline import build_pipeline
from churn.explain import permutation_importances, plot_importances

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Telco CSV (optional; synthetic if omitted).")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.csv)
    X, y = df[FEATURES], (df[TARGET] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline().fit(Xtr, ytr)

    rows = permutation_importances(pipe, Xte, yte)
    plot_importances(rows, f"{args.out_dir}/feature_importance.png")
    json.dump([{"feature": f, "importance": m, "std": s} for f, m, s in rows],
              open(f"{args.out_dir}/feature_importance.json", "w"), indent=2)
    print("Top features by permutation importance:")
    for f, m, s in rows[:8]:
        print(f"  {f:20s} {m:.4f} +/- {s:.4f}")
