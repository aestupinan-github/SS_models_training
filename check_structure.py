#!/usr/bin/env python3
"""Structure-check script for SS_models_training.

Validates that the project folder structure conforms to
specs/00-project-structure-and-workflow/requirements.md.
"""

import os
import re
import sys
from dataclasses import dataclass, field


class StructureError(Exception):
    pass


@dataclass
class StructureResult:
    is_valid: bool
    errors: list = field(default_factory=list)


def check_structure(project_root: str) -> StructureResult:
    errors = []

    top_level_files = ["constitution.md", "AGENTS.md", "MEMORY.md", "README.md"]
    top_level_dirs = ["interactions", "inspired_codes", "CPP_test", "specs"]

    for f in top_level_files:
        if not os.path.isfile(os.path.join(project_root, f)):
            errors.append(f"Missing top-level file: {f}")

    for d in top_level_dirs:
        if not os.path.isdir(os.path.join(project_root, d)):
            errors.append(f"Missing top-level directory: {d}")

    interactions_dir = os.path.join(project_root, "interactions")
    if os.path.isdir(interactions_dir):
        interaction_pattern = re.compile(r"^ss_[a-zA-Z0-9_]+-wall$|^ss_[a-zA-Z0-9_]+-ss_[a-zA-Z0-9_]+$")
        for entry in sorted(os.listdir(interactions_dir)):
            entry_path = os.path.join(interactions_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if not interaction_pattern.match(entry):
                errors.append(f"Invalid interaction folder name: {entry}")
                continue
            required_files = ["README.md"]
            required_dirs = ["python", "data", "output"]
            optional_dirs = ["figures"]
            for f in required_files:
                if not os.path.isfile(os.path.join(entry_path, f)):
                    errors.append(f"Missing file in {entry}: {f}")
            for d in required_dirs:
                if not os.path.isdir(os.path.join(entry_path, d)):
                    errors.append(f"Missing directory in {entry}: {d}")
            for d in optional_dirs:
                if not os.path.isdir(os.path.join(entry_path, d)):
                    continue

    specs_dir = os.path.join(project_root, "specs")
    if os.path.isdir(specs_dir):
        for entry in sorted(os.listdir(specs_dir)):
            entry_path = os.path.join(specs_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            for f in ["requirements.md", "design.md", "tasks.md"]:
                if not os.path.isfile(os.path.join(entry_path, f)):
                    errors.append(f"Missing spec file in {entry}: {f}")

    return StructureResult(is_valid=len(errors) == 0, errors=errors)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    result = check_structure(project_root)
    if result.is_valid:
        print("Structure check PASSED")
        return 0
    else:
        print("Structure check FAILED")
        for err in result.errors:
            print(f"  - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
