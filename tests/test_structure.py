import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from check_structure import check_structure, StructureError


class TestStructureCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.project_root = os.path.join(self.tmpdir, "SS_models_training")
        os.makedirs(self.project_root)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_valid_project(self):
        root = self.project_root
        for f in ["CONSTITUTION.md", "AGENTS.md", "MEMORY.md", "README.md"]:
            open(os.path.join(root, f), "w").close()
        for d in ["interactions", "inspired_codes", "CPP_test", "specs"]:
            os.makedirs(os.path.join(root, d))
        interaction = os.path.join(root, "interactions", "ss_cube-wall")
        os.makedirs(interaction)
        for f in ["README.md"]:
            open(os.path.join(interaction, f), "w").close()
        for d in ["python", "data", "output", "figures"]:
            os.makedirs(os.path.join(interaction, d))
        spec = os.path.join(root, "specs", "00-project-structure-and-workflow")
        os.makedirs(spec)
        for f in ["requirements.md", "design.md", "tasks.md"]:
            open(os.path.join(spec, f), "w").close()

    def test_valid_project_passes(self):
        self._create_valid_project()
        result = check_structure(self.project_root)
        self.assertTrue(result.is_valid)

    def test_missing_top_level_file_fails(self):
        self._create_valid_project()
        os.remove(os.path.join(self.project_root, "CONSTITUTION.md"))
        result = check_structure(self.project_root)
        self.assertFalse(result.is_valid)
        self.assertIn("CONSTITUTION.md", result.errors[0])

    def test_missing_top_level_dir_fails(self):
        self._create_valid_project()
        import shutil
        shutil.rmtree(os.path.join(self.project_root, "interactions"))
        result = check_structure(self.project_root)
        self.assertFalse(result.is_valid)
        self.assertIn("interactions", result.errors[0])

    def test_missing_interaction_subdir_fails(self):
        self._create_valid_project()
        import shutil
        shutil.rmtree(os.path.join(self.project_root, "interactions", "ss_cube-wall", "python"))
        result = check_structure(self.project_root)
        self.assertFalse(result.is_valid)
        self.assertIn("python", result.errors[0])

    def test_missing_interaction_readme_fails(self):
        self._create_valid_project()
        os.remove(os.path.join(self.project_root, "interactions", "ss_cube-wall", "README.md"))
        result = check_structure(self.project_root)
        self.assertFalse(result.is_valid)
        self.assertIn("README.md", result.errors[0])

    def test_missing_spec_file_fails(self):
        self._create_valid_project()
        os.remove(os.path.join(self.project_root, "specs", "00-project-structure-and-workflow", "design.md"))
        result = check_structure(self.project_root)
        self.assertFalse(result.is_valid)
        self.assertIn("design.md", result.errors[0])

    def test_invalid_interaction_name_fails(self):
        self._create_valid_project()
        os.rename(
            os.path.join(self.project_root, "interactions", "ss_cube-wall"),
            os.path.join(self.project_root, "interactions", "cube-wall"),
        )
        result = check_structure(self.project_root)
        self.assertFalse(result.is_valid)
        self.assertIn("cube-wall", result.errors[0])

    def test_shape_shape_interaction_passes(self):
        self._create_valid_project()
        os.rename(
            os.path.join(self.project_root, "interactions", "ss_cube-wall"),
            os.path.join(self.project_root, "interactions", "ss_cube-ss_cube"),
        )
        result = check_structure(self.project_root)
        self.assertTrue(result.is_valid)

    def test_empty_subfolder_omitted_passes(self):
        self._create_valid_project()
        import shutil
        shutil.rmtree(os.path.join(self.project_root, "interactions", "ss_cube-wall", "figures"))
        result = check_structure(self.project_root)
        self.assertTrue(result.is_valid)


if __name__ == "__main__":
    unittest.main()
