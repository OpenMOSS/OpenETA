#!/usr/bin/env python3
"""Serialize the installed and proposed R0.9 build-tool dependency graph."""

import argparse, importlib.metadata, importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "sim/envs/univtac/build_tool_dependency_contract.py"
spec = importlib.util.spec_from_file_location("univtac_build_tool_contract", path)
if spec is None or spec.loader is None: raise ImportError(path)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    distributions=[{"name":d.metadata.get("Name"),"version":d.version,"requires":list(d.requires or [])} for d in importlib.metadata.distributions() if d.metadata.get("Name")]
    graph=helper.dependency_graph(distributions); proposed=helper.validate_proposed_graph(graph)
    payload={"schema_version":"openeta.univtac.build_tool_dependency_graph.v1","graph":graph,"proposed":proposed,"success":proposed["success"]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    raise SystemExit(0 if payload["success"] else 1)
if __name__=="__main__": main()
