from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "accept_rename_readiness_m8_6.py"
SCHEMA = (
    REPO_ROOT
    / "studies"
    / "repository_migration"
    / "audits"
    / "m8_6"
    / "rename_readiness_acceptance.schema.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m8_6_readiness", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m8 = load_module()


def data():
    return m8.build_acceptance()


def test_schema_loads():
    parsed = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    assert parsed["schema_type"] == "m8_6_rename_readiness_acceptance"


def test_acceptance_validates():
    m8.validate(data(), m8.load_schema(REPO_ROOT))


def test_anchor_commits():
    src = data()["source_acceptance"]
    assert src["m8_5_acceptance_commit"] == "ff5f617"
    assert src["m8_4_acceptance_commit"] == "a11445a"


def test_transaction_does_not_hardcode_parent_commit_as_runtime_head():
    transaction_text = "\n".join(
        data()["transaction"]["preflight"] + data()["transaction"]["postflight"]
    )
    assert "ff5f617" not in transaction_text
    assert "RENAME_PREFLIGHT_HEAD" in transaction_text


def test_status_is_ready_for_controlled_rename():
    assert data()["status"] == "READY_FOR_CONTROLLED_RENAME"


def test_identity_transition_exact():
    ident = data()["identity_transition"]
    assert ident["old_repository_name"] == "GenCyberSynth_Phase1_Scaffold"
    assert ident["new_repository_name"] == "TrustForge"
    assert ident["old_remote"] == "git@github.com:edgemindstudio/GenCyberSynth_Phase1_Scaffold.git"
    assert ident["new_remote"] == "git@github.com:edgemindstudio/TrustForge.git"


def test_exactly_one_direct_rename_blocker():
    findings = data()["readiness_findings"]
    assert findings["direct_rename_blocker_count"] == 1
    assert findings["direct_rename_blocker"] == "Git remote origin"
    assert findings["blocker_resolvable_inside_transaction"] is True


def test_historical_rewrite_not_required():
    findings = data()["readiness_findings"]
    assert findings["historical_rewrite_required"] is False
    assert findings["paper_migration_required_before_rename"] is False
    assert findings["gcs_core_rename_required"] is False
    assert findings["protected_gencys_storage_rename_required"] is False


def test_only_repository_rename_transaction_is_authorized():
    authority = data()["authority_boundary"]
    assert authority["repository_rename_transaction_authorized"] is True
    for key, value in authority.items():
        if key != "repository_rename_transaction_authorized":
            assert value is False


def test_ordered_transaction_is_content_neutral():
    steps = data()["transaction"]["ordered_actions"]
    assert [step["step"] for step in steps] == [1, 2, 3, 4]
    assert all(step["content_mutation"] is False for step in steps)


def test_remote_update_is_inside_transaction():
    steps = data()["transaction"]["ordered_actions"]
    assert any(
        "git@github.com:edgemindstudio/TrustForge.git" in step["action"]
        for step in steps
    )


def test_local_directory_rename_is_inside_transaction():
    steps = data()["transaction"]["ordered_actions"]
    assert any(
        "GenCyberSynth_Phase1_Scaffold to TrustForge" in step["action"]
        for step in steps
    )


def test_historical_paper1_identity_is_preserved():
    preserved = "\n".join(data()["preserve_without_rewrite"])
    assert "Paper 1 repository identity" in preserved
    assert "Paper 1 deterministic generator historical repository constant" in preserved


def test_paper2_historical_paths_are_preserved():
    preserved = "\n".join(data()["preserve_without_rewrite"])
    assert "Paper 2 historical result/evidence absolute paths" in preserved


def test_protected_storage_is_preserved():
    preserved = "\n".join(data()["preserve_without_rewrite"])
    assert "/home/bruno.fonkeng/gencys" in preserved


def test_gcs_core_and_aliases_are_preserved():
    preserved = "\n".join(data()["preserve_without_rewrite"])
    assert "gcs-core" in preserved
    assert "GCS_*" in preserved


def test_transaction_forbids_paper_migration_and_historical_rewrite():
    forbidden = set(data()["transaction"]["forbidden_during_transaction"])
    assert "Do not rewrite Paper 1 repository/provenance fields." in forbidden
    assert "Do not rewrite Paper 2 historical result/evidence absolute paths." in forbidden
    assert "Do not migrate Papers 2–4." in forbidden


def test_head_and_tree_stability_are_required():
    pre = set(data()["transaction"]["preflight"])
    post = set(data()["transaction"]["postflight"])
    assert "Capture git rev-parse HEAD and treat it as RENAME_PREFLIGHT_HEAD." in pre
    assert (
        "Capture git rev-parse HEAD^{tree} and treat it as RENAME_PREFLIGHT_TREE."
        in pre
    )
    assert "Verify git rev-parse HEAD matches RENAME_PREFLIGHT_HEAD." in post
    assert (
        "Verify git rev-parse HEAD^{tree} matches RENAME_PREFLIGHT_TREE."
        in post
    )


def test_postflight_validates_new_top_level_and_remote():
    post = "\n".join(data()["transaction"]["postflight"])
    assert "/home/bruno.fonkeng/ProbabilisticModels/TrustForge" in post
    assert "git@github.com:edgemindstudio/TrustForge.git" in post


def test_current_branding_is_separate_followup():
    followup = data()["post_rename_followup"]
    branding = next(x for x in followup if x["item"] == "Current-facing repository branding")
    assert branding["timing"] == "separate post-rename commit"


def test_paper2_portability_debt_is_deferred_to_paper2_migration():
    followup = data()["post_rename_followup"]
    p2 = next(x for x in followup if x["item"] == "Paper 2 portability debt")
    assert p2["timing"] == "Paper 2 migration"


def test_principles_distinguish_identity_and_history():
    p = set(data()["principles"])
    assert "REPOSITORY_RENAME != HISTORICAL_REWRITE" in p
    assert "CURRENT_IDENTITY != HISTORICAL_EXECUTION_IDENTITY" in p
    assert "TREE_CONTENT_MUST_REMAIN_STABLE_DURING_IDENTITY_RENAME" in p


def test_markdown_deterministic():
    first = m8.render_markdown(data())
    second = m8.render_markdown(data())
    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


def test_tool_does_not_perform_rename():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "git remote set-url",
        "git mv",
        "os.rename(",
        "Path.rename(",
        "shutil.move(",
        "subprocess.run(",
        "subprocess.check_call(",
    ]
    for token in forbidden:
        assert token not in source
