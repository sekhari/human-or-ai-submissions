import argparse, os, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="Reproduce the Nov 9 experiments and new Kaggle submissions")
    p.add_argument("--data",required=True)
    p.add_argument("--output",default="results")
    p.add_argument("--skip-prior",action="store_true",help="Reuse an already generated prior/ result folder")
    a=p.parse_args();root=Path(__file__).resolve().parent;out=Path(a.output).resolve();data=Path(a.data).expanduser().resolve()
    if not data.is_file():p.error(f"Data ZIP not found: {data}")
    env=os.environ.copy();env["COMPETITION_ZIP"]=str(data);env["EXPERIMENT_DIR"]=str(out);env["PRIOR_RESULTS"]=str(out/"prior")
    if not a.skip_prior:
        subprocess.run([sys.executable,str(root.parent/"improved_ensemble/run.py"),"--data",str(data),"--output",str(out/"prior")],env=env,check=True)
    # Training is local. These scripts do not submit to Kaggle or call an inference API.
    for script in ["build_features.py","trees.py","char_cnn.py","kernel.py","pseudo.py","minibert.py","finalize.py"]:
        subprocess.run([sys.executable,str(root/script)],env=env,check=True)
    print(f"Verified CSVs: {out/'deliverables'}")

if __name__=="__main__":main()
