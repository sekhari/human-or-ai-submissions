"""Package editable Overleaf sources and verify the compiled PDF is present."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re
from train import ROOT


def main():
    report=ROOT/'report';output=ROOT/'output';output.mkdir(exist_ok=True)
    figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',(report/'main.tex').read_text())
    sources=['main.tex']+list(dict.fromkeys(figures))
    required=sources+['main.pdf']
    for name in required:
        if not (report/name).is_file():raise FileNotFoundError(report/name)
    with ZipFile(output/'group11_overleaf.zip','w',ZIP_DEFLATED) as z:
        for name in sources:
            z.write(report/name,arcname=name)
    print(output/'group11_overleaf.zip')


if __name__=='__main__':main()
