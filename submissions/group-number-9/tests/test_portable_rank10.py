"""Integration check against the actual uploaded CSV, without datasets/weights."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[1]


class PortableRank10Tests(unittest.TestCase):
    def test_exact_uploaded_csv_reproduction(self):
        expected=ROOT/'submissions/all_blend_roberta_rank10_10000.csv'
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'submission.csv'
            subprocess.run([sys.executable,'-m','experiments.all_blend_roberta_rank',
                '--verify-original','--out',str(output)],cwd=ROOT,check=True,capture_output=True)
            self.assertEqual(output.read_bytes(),expected.read_bytes())
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),
                'c398ae317ba46233e5a90e8867cbff6c242e9bf49274001bf757cef39255700a')


if __name__=='__main__':unittest.main()
