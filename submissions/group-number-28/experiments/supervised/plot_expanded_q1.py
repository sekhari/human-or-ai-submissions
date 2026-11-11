"""Plot recorded expanded Q1 results without generating report prose or PDFs."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "q1_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "q1_xdg_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from human_ai.data import ROOT, read_csv

SHORT={"lr":"Logistic regression","nb":"Multinomial NB","svm":"Linear SVM + sigmoid",
       "tree":"Decision tree","forest":"Random forest","boost":"Gradient boosting",
       "mlp":"Neural network (MLP)","lr_svd":"LR (SVD control)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report/figures")
    args = parser.parse_args()
    study = ROOT / "results/supervised" / f"{args.prefix}_study"
    rows = read_csv(study / "comparison.csv")
    cv = json.loads((study / "cv_results.json").read_text())
    plan = json.loads((study / "q1_expanded_plan.json").read_text())
    uncertainty = json.loads((study / "bootstrap.json").read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    out=args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    for i,row in enumerate(rows):
        f=row["model"];value=float(row["validation_auroc"])
        low,high=uncertainty["auroc_95_intervals"][f]
        color="#147d92" if row["representation"]=="sparse" else "#b16c30"
        ax.errorbar(value,i,xerr=[[value-low],[high-value]],fmt="o",capsize=3,color=color)
        ax.scatter(float(row["train_auroc"]),i,marker="^",color="#9099a1",s=30)
        ax.annotate(f"{value:.6f}",(value,i),xytext=(0,10),textcoords="offset points",ha="center",fontsize=8)
    ax.set_yticks(range(len(rows)),[SHORT[r["model"]] for r in rows])
    ax.set_ylim(len(rows)-0.4,-0.65)
    ax.set_xlim(min(uncertainty["auroc_95_intervals"][r["model"]][0] for r in rows)-0.03,1.01)
    ax.set_xlabel("AUROC");ax.grid(axis="x",alpha=.2)
    ax.set_title("Expanded Q1: fixed 8,000 / 2,000 development split",loc="left",fontweight="bold",pad=15)
    fig.suptitle("Dots/whiskers: validation + conditional 95% bootstrap CI; triangles: training\nBlue: sparse TF-IDF; orange: SVD representations; intervals are not selection-adjusted",y=-.02,fontsize=9)
    fig.savefig(out/f"q1_{args.prefix}_comparison.png",dpi=180,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    for i,family in enumerate(plan["families"]):
        candidates=[r for r in cv if r["model"]==family]
        for stage,color in ((1,"#a9b5c2"),(2,"#cf7b36")):
            values=[r["mean_cv_auroc"] for r in candidates if r["stage"]==stage]
            ax.scatter(values,[i]*len(values),color=color,s=27,alpha=.8,label=f"Stage {stage}" if i==0 else None)
        bestcv=max(candidates,key=lambda r:r["mean_cv_auroc"])
        ax.scatter(bestcv["mean_cv_auroc"],i,s=110,facecolors="none",edgecolors="#17364d",lw=1.5)
    ax.set_yticks(range(len(plan["families"])),[SHORT[f] for f in plan["families"]]);ax.invert_yaxis()
    ax.set_xlabel("Mean training-only CV AUROC (3 shared folds)");ax.grid(axis="x",alpha=.2)
    ax.set_title(f"{len(cv)} distinct configurations; rings mark the selected settings",loc="left",fontweight="bold")
    ax.legend(frameon=False,fontsize=9)
    fig.savefig(out/f"q1_{args.prefix}_search.png",dpi=180,bbox_inches="tight");plt.close(fig)
    print(f"Saved expanded Q1 experiment figures to {out}")


if __name__ == "__main__":
    main()
