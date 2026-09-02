from __future__ import annotations
import inspect
from scripts.univtac import apply_isaaclab_build_tool_bridge as runner

def test_three_mutations_are_fixed_and_each_counted_once():
 source=inspect.getsource(runner.main)
 assert source.index('"pip","install","--no-index","--no-deps",str(scm)') < source.index('"pip","uninstall","--yes","vcs-versioning"') < source.index('"pip","install","--no-index","--no-deps",str(packaging)')
 assert 'manifest["mutation_invocations"][step]=1' in source
 assert "--upgrade" not in source and "--force-reinstall" not in source

def test_nbt_follows_all_mutations_and_d6():
 source=inspect.getsource(runner.main)
 assert source.index('manifest["stages"]["D6"]="passed"') < source.index('stage="NBT"')
 assert '"isaac_actual_install_invocations":0' in source
 assert "AppLauncher" not in source and "scripts/install.sh" not in source

def test_wheel_dependencies_are_parsed_not_compared_by_formatting():
 from scripts.univtac import prepare_isaaclab_build_tool_wheels as prepare
 source=inspect.getsource(prepare.main)
 assert "Requirement(item)" in source
 assert "active_contract" in source
