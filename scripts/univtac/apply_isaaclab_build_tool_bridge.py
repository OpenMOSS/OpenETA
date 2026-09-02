#!/usr/bin/env python3
"""Apply the three authorized R0.9 build-tool mutations and run NBT."""

from __future__ import annotations

import argparse, json, subprocess, sys, traceback
from pathlib import Path
from typing import Any
import yaml

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.univtac.install_isaac51_blackwell_runtime import preflight, run_native_suite
from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.resume_isaac51_blackwell_install import package_probe, protected_binary_state, compare_protected_binaries
from scripts.univtac.run_isaac51_runtime_gates import fingerprint_environment, compare_fingerprints
from sim.envs.univtac.build_tool_dependency_contract import APPLIED_LABEL, EXPECTED_ORDER, validate_config
from sim.envs.univtac.isaac51_blackwell_runtime import clean_runtime_environment, load_json, managed, require_process
from sim.envs.univtac.resource_sanitation import utc_now, write_json, collect_inotify_inventory, collect_gpu_inventory, collect_process_inventory, attach_gpu_usage, build_restore_ready
from sim.envs.univtac.validated_dependency_baseline import audit_changes_against_plan

STAGES=("R0","D0","D1","D2","D3","D4","D5_STEP1","D5_STEP2","D5_STEP3","D6","NBT","P0A","P0B","I0A","I0B","I1","I2","N0","S0","S1","G0","L0","C0","H0")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--runtime-config",type=Path,required=True); p.add_argument("--bridge-config",type=Path,required=True); p.add_argument("--source-checkout",type=Path,required=True); p.add_argument("--curobo-checkout",type=Path,required=True); p.add_argument("--r092-output",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--conda-exe",type=Path,required=True); a=p.parse_args()
 runtime=yaml.safe_load(a.runtime_config.read_text()); cfg=yaml.safe_load(a.bridge_config.read_text()); validate_config(cfg)
 source=a.source_checkout.resolve(); curobo=a.curobo_checkout.resolve(); previous=a.r092_output.resolve(); output=a.output_root.resolve()
 if output.exists(): raise FileExistsError(output)
 output.mkdir(parents=True); base=Path(subprocess.run([str(a.conda_exe),"info","--base"],check=True,capture_output=True,text=True).stdout.strip()); source_env=runtime["environment"]["clone_from"]; prefix=base/"envs"/cfg["environment"]; python=prefix/"bin/python"; env,_=clean_runtime_environment(prefix,source,runtime["runtime"]["gpu"])
 manifest={"schema_version":"openeta.univtac.r093_build_tool_bridge.v1","runtime_variant":runtime["runtime_variant"]+"+"+APPLIED_LABEL,"superseded_planned_label":cfg["superseded_label"],"superseding_applied_label":APPLIED_LABEL,**runtime["claims"],"status":"running","classification":None,"started_at":utc_now(),"stages":{s:"not_run_due_to_gate" for s in STAGES},"mutation_invocations":{s:0 for s in EXPECTED_ORDER},"isaac_actual_install_invocations":0,"isaac_started":False,"agent_started":False}
 write_json(output/"run_manifest.json",manifest); r08_before={}; before_binaries={}
 try:
  resource=preflight(runtime,(ROOT,source,curobo,previous,output),output); write_json(output/"resume_preflight/resources.json",resource)
  prior=load_json(previous/"run_manifest.json"); packages=package_probe(python=python,source=source,root=output/"resume_preflight",name="packages",environment=env,timeout=600); r08_before=fingerprint_environment(a.conda_exe,source_env); before_binaries=protected_binary_state(curobo,prefix)
  required={"setuptools":"75.8.2","setuptools-scm":"10.2.2","vcs-versioning":"2.3.2","wheel":"0.42.0","packaging":"26.3","filelock":"3.32.3"}
  conditions={"resources":resource["passed"],"r092_classification":prior.get("classification")=="existing_environment_requires_packaging24_or_newer","r092_no_mutation":prior.get("packaging_install_invocations")==0 and prior.get("isaac_actual_install_invocations")==0,"versions":all(packages["all_distributions"].get(k)==v for k,v in required.items()),"flatdict_absent":packages["distributions"].get("flatdict") is None,"isaac_absent":packages["distributions"].get("isaaclab") is None and packages["distributions"].get("isaacsim") is None,"tacex_absent":packages["distributions"].get("tacex") is None and packages["distributions"].get("tacex-assets") is None,"curobo_source":packages.get("nvidia_curobo_editable_source")==str(curobo),"source_clean":not subprocess.run(["git","status","--short","--untracked-files=no"],cwd=source,capture_output=True,text=True).stdout.strip()}
  write_json(output/"resume_preflight/summary.json",{"passed":all(conditions.values()),"conditions":conditions,"packages":packages,"protected_binaries":before_binaries})
  if not all(conditions.values()): manifest["classification"]="r093_resume_precondition_failed"; raise RuntimeError("R0 failed")
  manifest["stages"]["R0"]="passed"
  graph_path=output/"dependency_graph_before/dependency_graph_before.json"; graph_process=managed([str(python),str(ROOT/"scripts/univtac/audit_build_tool_dependency_graph.py"),"--output",str(graph_path)],cwd=source,output_root=output/"dependency_graph_before",name="audit",timeout_seconds=600,environment=env); graph=load_json(graph_path)
  if graph["proposed"]["other_vcs_versioning_reverse_dependencies"]: manifest["classification"]="vcs_versioning_has_other_active_reverse_dependency"; manifest["stages"]["D0"]="failed"; raise RuntimeError("other vcs reverse dependency")
  if graph_process.get("returncode") or not graph["success"]: manifest["classification"]="proposed_build_tool_graph_inconsistent"; manifest["stages"]["D0"]="failed"; raise RuntimeError("proposed graph inconsistent")
  manifest["stages"]["D0"]="passed"
  wheel_root=output/"build_tool_wheels"; wheel_process=managed([str(python),str(ROOT/"scripts/univtac/prepare_isaaclab_build_tool_wheels.py"),"--config",str(a.bridge_config.resolve()),"--target-python",str(python),"--curobo",str(curobo),"--univtac",str(source),"--output-root",str(wheel_root)],cwd=source,output_root=output/"isolated_build_tool_smoke",name="prepare",timeout_seconds=1800,environment=env)
  require_process("D1-D3 wheel preparation",wheel_process); wheels=load_json(wheel_root/"summary.json"); manifest["stages"].update({"D1":"passed","D2":"passed","D3":"passed"})
  scm=Path(wheels["records"]["setuptools-scm"]["path"]); packaging=Path(wheels["records"]["packaging"]["path"])
  plan={"order":list(EXPECTED_ORDER),"commands":[[str(python),"-m","pip","install","--no-index","--no-deps",str(scm)],[str(python),"-m","pip","uninstall","--yes","vcs-versioning"],[str(python),"-m","pip","install","--no-index","--no-deps",str(packaging)]]}; write_json(output/"mutation_plan/mutation_plan.json",plan); manifest["stages"]["D4"]="passed"
  snapshots=[packages]; allowed=[{"setuptools-scm":"8.1.0"},{"vcs-versioning":None},{"packaging":"23.0"}]
  for index,(step,command) in enumerate(zip(EXPECTED_ORDER,plan["commands"],strict=True),1):
   manifest["mutation_invocations"][step]=1; write_json(output/"run_manifest.json",manifest); result=managed(command,cwd=source,output_root=output/f"build_tool_bridge/step{index}",name="mutation",timeout_seconds=600,environment=env); require_process(step,result); after=package_probe(python=python,source=source,root=output/f"build_tool_bridge/step{index}",name="packages_after",environment=env,timeout=600)
   before=snapshots[-1]["all_distributions"]; expected=allowed[index-1]; audit=audit_changes_against_plan(before,after["all_distributions"],expected); write_json(output/f"build_tool_bridge/step{index}/diff.json",audit)
   if not audit["success"]: manifest["classification"]="build_tool_bridge_unexpected_environment_change"; manifest["stages"][f"D5_STEP{index}"]="failed"; raise RuntimeError(f"step {index} unexpected diff")
   snapshots.append(after); manifest["stages"][f"D5_STEP{index}"]="passed"
  check=managed([str(python),"-m","pip","check"],cwd=source,output_root=output/"dependency_graph_after",name="pip_check",timeout_seconds=600,environment=env); require_process("D6 pip check",check); final=snapshots[-1]; final_expected={"setuptools":"75.8.2","setuptools-scm":"8.1.0","packaging":"23.0","wheel":"0.42.0","filelock":"3.32.3"}; ok=all(final["all_distributions"].get(k)==v for k,v in final_expected.items()) and final["all_distributions"].get("vcs-versioning") is None and compare_protected_binaries(before_binaries,protected_binary_state(curobo,prefix))["success"]
  write_json(output/"dependency_graph_after/summary.json",{"success":ok,"packages":final,"pip_check":check})
  if not ok: manifest["classification"]="build_tool_bridge_post_install_check_failed"; manifest["stages"]["D6"]="failed"; raise RuntimeError("D6 failed")
  manifest["stages"]["D6"]="passed"
  run_native_suite(stage="NBT",python=python,source=source,curobo=curobo,output_root=output/"native_regression",environment=env,timeouts=runtime["timeouts_seconds"]); manifest["stages"]["NBT"]="passed"; manifest["status"]="build_tool_bridge_validated"; manifest["classification"]="build_tool_bridge_native_validated"
 except BaseException as exc:
  manifest["status"]="failed"; manifest["classification"]=manifest["classification"] or "build_tool_bridge_mutation_failed"; manifest["failure"]={"class":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}; write_json(output/"author_bundle/build_tool_bridge/summary.json",{"classification":manifest["classification"],"failure":manifest["failure"]}); raise
 finally:
  errors=[]
  try:
   if r08_before:
    after=fingerprint_environment(a.conda_exe,source_env); comp=compare_fingerprints(r08_before,after); write_json(output/"environment_fingerprints/r08_comparison.json",comp); manifest["r08_unchanged"]=comp["unchanged"]
  except BaseException as exc: errors.append({"stage":"r08", "error":str(exc)})
  try:
   cleanup=managed_cleanup_summary(output); manifest["managed_cleanup_complete"]=cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"]; write_json(output/"final_resources/managed_cleanup.json",cleanup); ino=collect_inotify_inventory(); gpu=collect_gpu_inventory(); procs=attach_gpu_usage(collect_process_inventory(related_roots=(ROOT,source,curobo,output)),gpu); restore=build_restore_ready(original_instances=128,original_watches=65536,inotify_inventory=ino,process_inventory=procs,gpu_inventory=gpu); restore["sysctl_restore_performed_by_runner"]=False; write_json(output/"restore_ready.json",restore)
  except BaseException as exc: errors.append({"stage":"cleanup","error":str(exc)})
  manifest["finalization_errors"]=errors; manifest["ended_at"]=utc_now(); write_json(output/"run_manifest.json",manifest); write_json(output/"summary.json",manifest)
if __name__=="__main__": main()
