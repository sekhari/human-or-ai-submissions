import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


class PackagePortabilityTests(unittest.TestCase):
    def test_package_imports_do_not_import_legacy_root_modules_or_change_path(self):
        script = '''
import sys
before=list(sys.path)
from models.multinomial_nb import multinomial_nb_blend, multinomial_nb_submit
assert before == sys.path
assert not any(name in sys.modules for name in ["multinomial_nb", "multinomial_nb_blend", "multinomial_nb_real", "multinomial_nb_ensemble", "multinomial_nb_submit"])
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')

    def test_histories_and_submission_reference_are_bundled_without_outputs(self):
        module = importlib.import_module('models.multinomial_nb.multinomial_nb_blend')
        submit = importlib.import_module('models.multinomial_nb.multinomial_nb_submit')
        package = Path(module.__file__).resolve().parent
        for path in [module.SELECTION_HISTORY, module.SELECTION_METADATA, submit.REFERENCE]:
            self.assertTrue(path.is_file(), str(path))
            self.assertTrue(path.is_relative_to(package / 'references'))
            self.assertNotIn('outputs', path.parts)
        _, _, selected = module.load_selection_history(module.NBScoreBlend())
        self.assertAlmostEqual(selected['mean_cv_auroc'], .9357384375, places=12)
        reference = json.loads(submit.REFERENCE.read_text())
        self.assertAlmostEqual(reference['models']['ensemble']['validation']['auroc'], .939743, places=12)
        self.assertAlmostEqual(reference['models']['single']['validation']['auroc'], .900036, places=12)


if __name__ == '__main__':
    unittest.main()
