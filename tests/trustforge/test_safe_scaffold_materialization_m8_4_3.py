from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "materialize_safe_scaffolds_m8_4_3.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m8_4_3_scaffold", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m8 = load_module()


def test_authority_anchor():
    assert m8.M8_4_2_ACCEPTANCE_COMMIT == "954ad5f"


def test_exactly_four_authorized_scaffolds():
    assert set(m8.SCAFFOLDS) == {
        "hpc/README.md",
        "studies/paper02_conditioning_audit/README.md",
        "studies/paper03_augmentation_regimes/README.md",
        "studies/paper04_selective_policies/README.md",
    }


def test_historical_authority_paths_are_preserved():
    assert set(m8.HISTORICAL_AUTHORITIES) == {
        "slurm",
        "papers/paper2_conditional_generation_done_right",
        "papers/paper3_when_does_synth_help",
        "papers/paper4_selective_synth_policies",
    }


def test_readmes_mark_scaffold_only():
    for text in m8.SCAFFOLDS.values():
        assert "**Status:** `SCAFFOLD_ONLY`" in text
        assert "SCAFFOLD_CREATION != MIGRATION" in text
        assert "DESTINATION_EXISTENCE != AUTHORITY_TRANSFER" in text
        assert "954ad5f" in text


def test_paper2_readme_preserves_historical_authority():
    text = m8.SCAFFOLDS["studies/paper02_conditioning_audit/README.md"]
    assert "papers/paper2_conditional_generation_done_right/" in text
    assert "not" in text.lower()
    assert "authoritative" in text.lower()


def test_paper3_readme_preserves_historical_authority():
    text = m8.SCAFFOLDS["studies/paper03_augmentation_regimes/README.md"]
    assert "papers/paper3_when_does_synth_help/" in text
    assert "minority-class intervention" in text


def test_paper4_readme_preserves_historical_authority():
    text = m8.SCAFFOLDS["studies/paper04_selective_policies/README.md"]
    assert "papers/paper4_selective_synth_policies/" in text
    assert "class-repair" in text


def test_hpc_readme_preserves_slurm_authority():
    text = m8.SCAFFOLDS["hpc/README.md"]
    assert "slurm/" in text
    assert "run_gcs.slurm" in text
    assert "run_suite.slurm" in text
    assert "run_tuning.slurm" in text
    assert "No historical Slurm file has been moved or copied here." in text


def test_materialized_scaffolds_are_exact_and_contain_only_readme():
    states = m8.check(REPO_ROOT)
    for relative_file, state in states.items():
        if state == "MATERIALIZED":
            path = REPO_ROOT / relative_file
            entries = sorted(p.name for p in path.parent.iterdir())
            assert entries == ["README.md"]
            assert path.read_text(encoding="utf-8") == m8.normalized(
                m8.SCAFFOLDS[relative_file]
            )


def test_no_historical_source_is_inside_scaffold_map():
    for relative_file in m8.SCAFFOLDS:
        assert not relative_file.startswith("papers/")
        assert not relative_file.startswith("slurm/")


def test_tool_does_not_contain_move_copy_or_destructive_operations():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "shutil.move",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copytree",
        "os.rename",
        "Path.rename",
        "git mv",
        "git clean",
        "git reset",
        "rm -rf",
        ".unlink(",
    ]
    for token in forbidden:
        assert token not in source


def test_normalized_content_has_single_eof_newline():
    for text in m8.SCAFFOLDS.values():
        normalized = m8.normalized(text)
        assert normalized.endswith("\n")
        assert not normalized.endswith("\n\n")
