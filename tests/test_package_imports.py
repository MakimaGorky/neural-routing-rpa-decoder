import importlib
import os
import subprocess
import sys
from pathlib import Path


def test_routing_rpa_imports_from_src_package():
    package = importlib.import_module("routing_rpa")

    package_path = Path(package.__file__).resolve()
    assert package_path.parts[-3:] == ("src", "routing_rpa", "__init__.py")
    assert "src_old" not in package_path.parts


def test_main_subpackages_import_without_legacy_runtime():
    subpackages = [
        "codes",
        "projections",
        "decoders",
        "channels",
        "data",
        "training",
        "eval",
        "experiments",
        "utils",
    ]

    for name in subpackages:
        module = importlib.import_module(f"routing_rpa.{name}")
        module_path = Path(module.__file__).resolve()
        assert module_path.parts[-4:] == ("src", "routing_rpa", name, "__init__.py")
        assert "src_old" not in module_path.parts

    assert not any(module_name == "src_old" for module_name in sys.modules)


def test_fresh_process_can_import_evaluator_first():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    command = [
        sys.executable,
        "-c",
        "from routing_rpa.eval.evaluator import Evaluator; print(Evaluator.__name__)",
    ]

    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Evaluator"
