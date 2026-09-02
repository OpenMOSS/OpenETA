from __future__ import annotations

import inspect

from scripts.univtac import resume_curobo_loader_validation as runner
from scripts.univtac import run_official_curobo_example


def test_official_example_wrapper_runs_unmodified_path() -> None:
    source = inspect.getsource(run_official_curobo_example.main)
    assert 'runpy.run_path(str(example), run_name="__main__")' in source
    assert "args.example.write_text" not in source
    assert 'parser.add_argument("--expected-root"' in source
    assert 'payload["extension_provenance_ok"]' in source


def test_smoke_order_precedes_libuipc_regression() -> None:
    source = inspect.getsource(runner.main)
    assert source.index("kinematics_example.py") < source.index("ik_example.py") < source.index("collision_check_example.py") < source.index("probe_libuipc_core.py")


def test_packaging_failure_does_not_claim_kernel_or_author_bundle() -> None:
    source = inspect.getsource(runner.main)
    assert 'manifest["author_bundle_type"] = "none_example_packaging_failure"' in source
    assert 'manifest["gates"][gate] = manifest["classification"]' in source
