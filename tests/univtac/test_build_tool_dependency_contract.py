from __future__ import annotations
from pathlib import Path
import copy, pytest, yaml
from sim.envs.univtac.build_tool_dependency_contract import dependency_graph, validate_config, validate_proposed_graph

ROOT=Path(__file__).resolve().parents[2]; CONFIG=ROOT/"configs/univtac/isaaclab_build_tool_bridge.yaml"
def cfg(): return yaml.safe_load(CONFIG.read_text())

def test_fixed_bridge_config():
 validate_config(cfg()); assert cfg()["after"]["setuptools-scm"]=="8.1.0"; assert cfg()["after"]["vcs-versioning"] is None; assert cfg()["after"]["packaging"]=="23.0"

def test_graph_allows_only_setuptools_scm_reverse_dependency():
 graph=dependency_graph([{"name":"setuptools-scm","version":"10.2.2","requires":["vcs-versioning<3,>=2.3.2.dev0","packaging>=20","setuptools"]},{"name":"vcs-versioning","version":"2.3.2","requires":["packaging>=26.2"]},{"name":"setuptools","version":"75.8.2","requires":[]},{"name":"packaging","version":"26.3","requires":[]}])
 assert validate_proposed_graph(graph)["success"] is True
 graph["reverse_dependencies"]["vcs-versioning"].append({"distribution":"other","requirement":"vcs-versioning"})
 assert validate_proposed_graph(graph)["success"] is False

def test_mutation_order_cannot_change():
 changed=copy.deepcopy(cfg()); changed["mutation_order"].reverse()
 with pytest.raises(ValueError,match="order"): validate_config(changed)
