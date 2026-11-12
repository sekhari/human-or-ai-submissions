"""Execute plain-Python notebook cells, preserving outputs without Jupyter dependencies.

Usage: python run_notebook.py Alma_RF_Improvements.ipynb
Optional --extra-site-packages appends an existing compatible dependency directory.
Notebook magics are not supported. Use PyCharm/Jupyter for interactive execution.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import traceback

# Windows consoles may default to GBK; passage snippets contain Unicode.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=Path)
parser.add_argument("--extra-site-packages", type=Path)
args = parser.parse_args()
if args.extra_site_packages:
    sys.path.append(str(args.extra_site_packages.resolve()))
path = args.notebook.resolve()
os.chdir(path.parent)
original = path.read_bytes()
notebook = json.loads(original)
namespace = {"__name__": "__main__"}
execution_count = 0
failed = False
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    execution_count += 1
    cell["execution_count"] = execution_count
    buffer = io.StringIO()
    terminal = sys.stdout

    class Tee:
        def write(self, text):
            buffer.write(text)
            terminal.write(text)
            terminal.flush()
            return len(text)

        def flush(self):
            terminal.flush()

    def flush_output():
        if buffer.tell():
            cell["outputs"].append(
                {"output_type": "stream", "name": "stdout", "text": buffer.getvalue()}
            )
            buffer.seek(0)
            buffer.truncate(0)

    def display(value):
        flush_output()
        data = {"text/plain": repr(value)}
        if hasattr(value, "_repr_html_"):
            html = value._repr_html_()
            if html:
                data["text/html"] = html
        cell["outputs"].append(
            {"output_type": "display_data", "data": data, "metadata": {}}
        )
        terminal.write(repr(value) + "\n")
        terminal.flush()

    namespace["display"] = display
    print("Running", cell.get("id"), flush=True)
    try:
        with contextlib.redirect_stdout(Tee()), contextlib.redirect_stderr(Tee()):
            exec(compile("".join(cell["source"]), f"{path.name}:{cell.get('id')}", "exec"), namespace)
        flush_output()
    except Exception as error:
        flush_output()
        failed = True
        cell["outputs"].append({
            "output_type": "error",
            "ename": type(error).__name__,
            "evalue": str(error),
            "traceback": traceback.format_exc().splitlines(),
        })
        print(traceback.format_exc(), flush=True)
        break

notebook["metadata"]["language_info"]["version"] = sys.version.split()[0]
notebook["metadata"]["execution_environment"] = {
    "python": sys.executable,
    "extra_site_packages": str(args.extra_site_packages) if args.extra_site_packages else None,
    "runner": "run_notebook.py",
}
if path.read_bytes() != original:
    path = path.with_name(path.stem + "_Executed.ipynb")
    print("Original edited during execution; saving a separate executed copy.")
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", path, flush=True)
raise SystemExit(1 if failed else 0)
