"""Task 4 validation: campaign config and docs reflect staged generic-resource dispatch.

Checks that:
- Campaign YAML contains staged dispatch knobs and promotion gate
- Campaign YAML no longer implies a single undifferentiated trial per iteration
- Docs (README.md, AGENTS.md) document generic-resource dispatch
- No project-dispatch surface references named custom resources like "aorus": 1
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def campaign() -> dict:
    with open(ROOT / "ml_metaopt_campaign.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def readme_text() -> str:
    return (ROOT / "README.md").read_text()


@pytest.fixture(scope="module")
def agents_text() -> str:
    return (ROOT / "AGENTS.md").read_text()


@pytest.fixture(scope="module")
def claude_text() -> str:
    return (ROOT / "CLAUDE.md").read_text()


# ---------------------------------------------------------------------------
# Campaign YAML: staged dispatch knobs
# ---------------------------------------------------------------------------


class TestCampaignDispatchKnobs:
    REQUIRED_KEYS = [
        "cpu_screen_max_concurrent",
        "cpu_screen_cpus_per_trial",
        "gpu_promotion_max_concurrent",
        "gpu_promotion_gpus_per_trial",
        "cpu_eval_max_concurrent",
    ]

    def test_dispatch_policy_has_staged_knobs(self, campaign):
        dp = campaign["dispatch_policy"]
        for key in self.REQUIRED_KEYS:
            assert key in dp, f"Missing dispatch_policy.{key}"
            assert isinstance(dp[key], int) and dp[key] >= 1

    def test_dispatch_policy_matches_code_defaults(self, campaign):
        from scripts.dispatch_policy import DispatchPolicy

        dp = campaign["dispatch_policy"]
        defaults = DispatchPolicy()
        assert dp["cpu_screen_max_concurrent"] == defaults.cpu_screen_max_concurrent
        assert dp["cpu_screen_cpus_per_trial"] == defaults.cpu_screen_cpus_per_trial
        assert dp["gpu_promotion_max_concurrent"] == defaults.gpu_promotion_max_concurrent
        assert dp["gpu_promotion_gpus_per_trial"] == defaults.gpu_promotion_gpus_per_trial
        assert dp["cpu_eval_max_concurrent"] == defaults.cpu_eval_max_concurrent

    def test_promotion_gate_present(self, campaign):
        dp = campaign["dispatch_policy"]
        assert "promotion_min_score" in dp
        assert "promotion_top_k" in dp
        assert 0.0 < dp["promotion_min_score"] < 1.0
        assert dp["promotion_top_k"] >= 1

    def test_promotion_gate_matches_trial_budget(self, campaign):
        dp = campaign["dispatch_policy"]
        tb = campaign["execution"]["trial_budget"]
        assert dp["promotion_top_k"] == tb["gpu_promotions_per_iteration"]

    def test_trial_budget_is_staged(self, campaign):
        tb = campaign["execution"]["trial_budget"]
        assert tb["kind"] == "staged", f"Expected staged trial budget, got {tb['kind']}"
        assert "screen_trials_per_iteration" in tb
        assert "gpu_promotions_per_iteration" in tb

    def test_trial_budget_not_single_fixed(self, campaign):
        tb = campaign["execution"]["trial_budget"]
        assert tb["kind"] != "fixed_trials" or tb.get("value", 0) != 1, (
            "Trial budget still implies single undifferentiated trial"
        )


# ---------------------------------------------------------------------------
# No named-resource assumptions in project dispatch surfaces
# ---------------------------------------------------------------------------

# Surfaces that form the project dispatch path (code + config)
_DISPATCH_SURFACE_GLOBS = [
    "scripts/dispatch_policy.py",
    "scripts/dispatch_trial.py",
    "scripts/build_batch_manifest.py",
    "ml_metaopt_campaign.yaml",
]


class TestNoNamedResources:
    @pytest.fixture(scope="class")
    def dispatch_surface_text(self) -> str:
        found = []
        for pattern in _DISPATCH_SURFACE_GLOBS:
            found.extend(ROOT.glob(pattern))
        assert len(found) == len(_DISPATCH_SURFACE_GLOBS), (
            f"Missing dispatch surface files: expected {len(_DISPATCH_SURFACE_GLOBS)}, "
            f"found {len(found)}"
        )
        return "\n".join(path.read_text() for path in found)

    def test_dispatch_surfaces_no_aorus_resource(self, dispatch_surface_text):
        # Check for the resource dict pattern, not just the host name
        assert '"aorus"' not in dispatch_surface_text, (
            'Project dispatch surface contains named resource "aorus"'
        )
        assert "'aorus'" not in dispatch_surface_text

    def test_readme_no_named_resource_dispatch(self, readme_text):
        # README may mention Aorus as infrastructure, but should not show
        # resources={"aorus": 1} as the project dispatch pattern
        assert 'resources={"aorus"' not in readme_text
        assert "resources={\"aorus\"" not in readme_text

    def test_agents_no_named_resource_dispatch(self, agents_text):
        # Allowed: mentioning the pattern in negation context ("No named...")
        # Forbidden: presenting it as the active dispatch contract
        for line in agents_text.splitlines():
            if 'resources={"aorus"' in line or "resources={\"aorus\"" in line:
                lower = line.lower()
                assert "no " in lower or "never" in lower or "e.g." in lower, (
                    f"AGENTS.md uses named-resource dispatch as active pattern:\n  {line}"
                )

    def test_claude_no_named_resource_dispatch(self, claude_text):
        assert 'resources={"aorus"' not in claude_text
        assert "resources={\"aorus\"" not in claude_text


# ---------------------------------------------------------------------------
# Docs mention generic-resource dispatch
# ---------------------------------------------------------------------------


class TestDocsGenericResources:
    def test_readme_mentions_generic_resources(self, readme_text):
        lower = readme_text.lower()
        assert "generic" in lower and "resource" in lower, (
            "README should document generic-resource dispatch model"
        )

    def test_readme_mentions_num_gpus(self, readme_text):
        assert "num_gpus" in readme_text

    def test_readme_mentions_delegated_execution(self, readme_text):
        lower = readme_text.lower()
        assert "delegat" in lower, (
            "README should mention delegated execution"
        )

    def test_agents_mentions_generic_resources(self, agents_text):
        lower = agents_text.lower()
        assert "generic" in lower and "resource" in lower

    def test_agents_mentions_dispatch_contract(self, agents_text):
        lower = agents_text.lower()
        assert "dispatch contract" in lower or "dispatch path" in lower

    def test_agents_preserves_torch_caution(self, agents_text):
        assert "NEVER" in agents_text
        assert "pip install torch" in agents_text or "pip-install" in agents_text

    def test_agents_preserves_add_self_loops_warning(self, agents_text):
        assert "add_self_loops=False" in agents_text

    def test_agents_preserves_gfx1010_note(self, agents_text):
        assert "gfx1010" in agents_text
