"""Prepare a small final handoff without competition data or checkpoints."""
import json,shutil
from zipfile import ZipFile,ZIP_DEFLATED
from train import ROOT


def main():
    best=json.loads((ROOT/'results/optimization/current_best.json').read_text())
    meta=json.loads((ROOT/'report_metadata.json').read_text())
    destination=ROOT/'output/final_delivery';destination.mkdir(exist_ok=True,parents=True)
    files={'submission.csv':ROOT/'output/submission.csv','Group_Number_11_report.pdf':ROOT/'report/main.pdf',
           'Group_Number_11_overleaf.zip':ROOT/'output/group11_overleaf.zip'}
    for name,source in files.items():shutil.copy2(source,destination/name)
    note=f'''Group Number 11 - final materials

Team: Ansh Raj Bhargava (ab3396), Yilin Gao (yg695), Yingzi Gu (yg667)
Branch: {meta['submission_branch']}
Repository: {meta['repository_url']}/tree/{meta['submission_branch']}
Course PR: {meta['pr_url']}

Final candidate: {best['candidate']}
User-reported Kaggle public AUROC: {best['public_auc']:.5f}
Private result: not revealed
Scored CSV SHA256: {best['sha256']}

submission.csv is byte-identical to the scored final version.
The PDF contains the PR link, team roles, measured results and AI/pretrained-model disclosure.
Upload the Overleaf ZIP as a new project to edit the paper.
Source code, exact commands, dependencies and experiment history are in the team directory on GitHub.
No competition input data or model checkpoints are included in this delivery ZIP.
Team members should review the paper/code and push contributions to {meta['submission_branch']} in the final team repository.
'''
    (destination/'README.txt').write_text(note)
    archive=ROOT/'output/Group_Number_11_final_materials.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for file in sorted(destination.iterdir()):z.write(file,arcname=file.name)
    print(archive)


if __name__=='__main__':main()
