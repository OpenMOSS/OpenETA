#!/usr/bin/env python3
"""Continue a validated R0.9.3 build-tool bridge through N0."""

from __future__ import annotations
import argparse, subprocess, sys, traceback
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.univtac.install_isaac51_blackwell_runtime import run_native_suite
from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.resume_isaac51_blackwell_install import package_probe, protected_binary_state, compare_protected_binaries, dry_run
from sim.envs.univtac.flatdict_build_contract import audit_report_uses_wheel, sha256_file
from sim.envs.univtac.isaac51_blackwell_runtime import clean_runtime_environment, load_json, managed, process_ok, require_process
from sim.envs.univtac.resource_sanitation import utc_now, write_json
from sim.envs.univtac.simulator_install_contract import audit_installed_versions, audit_pip_report, constraints_text, load_pip_report, required_runtime_versions, validate_config
from sim.envs.univtac.validated_dependency_baseline import audit_changes_against_plan, package_diff, report_install_versions

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--source-checkout",type=Path,required=True); p.add_argument("--curobo-checkout",type=Path,required=True); p.add_argument("--r091-output",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--conda-exe",type=Path,required=True); a=p.parse_args()
 cfg=yaml.safe_load(a.config.read_text()); validate_config(cfg); source=a.source_checkout.resolve(); curobo=a.curobo_checkout.resolve(); r091=a.r091_output.resolve(); output=a.output_root.resolve(); manifest=load_json(output/"run_manifest.json")
 if manifest.get("status")!="build_tool_bridge_validated" or manifest.get("stages",{}).get("NBT")!="passed": raise RuntimeError("R0.9.3 build-tool bridge/NBT not validated")
 base=Path(subprocess.run([str(a.conda_exe),"info","--base"],check=True,capture_output=True,text=True).stdout.strip()); prefix=base/"envs"/cfg["environment"]["conda_name"]; python=prefix/"bin/python"; env,_=clean_runtime_environment(prefix,source,cfg["runtime"]["gpu"]); before_binaries=protected_binary_state(curobo,prefix); protected=list(required_runtime_versions(cfg)); wheel=Path(load_json(r091/"flatdict_bridge/run_manifest.json")["wheel"])
 if sha256_file(wheel)!="94614060d175f1acbff62a4bca3051e5145f4ed9c880a36861e47533f82ba212": raise RuntimeError("R0.9.1 flatdict wheel identity mismatch")
 constraints=output/"dependency_plan/final_protected_constraints.txt"; constraints.parent.mkdir(parents=True,exist_ok=True); constraints.write_text(constraints_text(cfg)); plans={"P0A":([cfg["install"]["isaaclab_requirement"]],cfg["install"]["isaac_extra_index_url"],"isaac"),"P0B":(list(cfg["install"]["tacex_dependencies"]),None,"tacex")}; reports={}
 try:
  current=package_probe(python=python,source=source,root=output/"dependency_plan",name="packages_before",environment=env,timeout=600); expected=required_runtime_versions(cfg); audit=audit_installed_versions(current["distributions"],expected)
  if not audit["success"] or current["all_distributions"].get("vcs-versioning") is not None: manifest["classification"]="r093_resume_precondition_failed"; raise RuntimeError("aligned build-tool baseline missing")
  for stage,(requirements,index,name) in plans.items():
   report_path=output/f"dependency_plan/{name}_report.json"; process,report=dry_run(name=name,python=python,source=source,root=output/"dependency_plan",constraints=constraints,wheelhouse=wheel.parent,report=report_path,requirements=requirements,extra_index=index,environment=env,timeout=1800)
   if not process_ok(process) or not report_path.is_file(): manifest["stages"][stage]="failed"; manifest["classification"]="build_tool_aligned_isaac_resolution_conflict" if stage=="P0A" else "additional_legacy_sdist_resolution_failure"; raise RuntimeError(f"{stage} failed")
   report=load_pip_report(report_path); pa=audit_pip_report(report,protected).to_dict(); fw=audit_report_uses_wheel(report,wheel); vcs=any((item.get("metadata",{}).get("name","").lower().replace("_","-")=="vcs-versioning") for item in report.get("install",[])); summary={"passed":pa["success"] and fw["success"] and not vcs,"protected":pa,"flatdict":fw,"vcs_versioning_planned":vcs}; write_json(output/f"dependency_plan/{name}_summary.json",summary)
   if not summary["passed"]: manifest["stages"][stage]="failed"; manifest["classification"]="build_tool_aligned_isaac_resolution_conflict"; raise RuntimeError(f"{stage} unsafe plan")
   reports[stage]=report; manifest["stages"][stage]="passed"
  before=current; install=managed([str(python),"-m","pip","install","--no-index","--no-deps",str(wheel)],cwd=source,output_root=output/"flatdict_install",name="install",timeout_seconds=600,environment=env); require_process("I0A flatdict",install); after=package_probe(python=python,source=source,root=output/"flatdict_install",name="packages_after",environment=env,timeout=600); diff=package_diff(before["all_distributions"],after["all_distributions"],allowed_additions={"flatdict":"4.0.1"}); write_json(output/"flatdict_install/diff.json",diff)
  if not diff["success"]: manifest["classification"]="protected_dependency_changed_during_actual_install"; raise RuntimeError("flatdict diff")
  manifest["stages"]["I0A"]="passed"; manifest["isaac_actual_install_invocations"]=1; write_json(output/"run_manifest.json",manifest); actual=output/"install_isaac/actual_report.json"; command=[str(python),"-m","pip","install","--report",str(actual),"--constraint",str(constraints),"--find-links",str(wheel.parent),"--only-binary=flatdict",cfg["install"]["isaaclab_requirement"],"--extra-index-url",cfg["install"]["isaac_extra_index_url"]]; proc=managed(command,cwd=source,output_root=output/"install_isaac",name="install",timeout_seconds=7200,environment=env)
  if not process_ok(proc): manifest["classification"]="isaac51_package_install_failed"; raise RuntimeError("Isaac install failed")
  actual_report=load_pip_report(actual); protected_actual=audit_pip_report(actual_report,protected).to_dict(); installed=package_probe(python=python,source=source,root=output/"install_isaac",name="packages_after",environment=env,timeout=600); binaries=compare_protected_binaries(before_binaries,protected_binary_state(curobo,prefix)); versions=audit_installed_versions(installed["distributions"],expected)
  if not protected_actual["success"] or not binaries["success"] or not versions["success"]: manifest["classification"]="protected_dependency_changed_during_actual_install"; raise RuntimeError("protected change during Isaac install")
  manifest["stages"]["I0B"]="passed"
  before_i1=installed; deps_report=output/"install_tacex/dependencies_report.json"; deps=managed([str(python),"-m","pip","install","--report",str(deps_report),"--constraint",str(constraints),"--find-links",str(wheel.parent),"--only-binary=flatdict",*cfg["install"]["tacex_dependencies"]],cwd=source,output_root=output/"install_tacex/dependencies",name="install",timeout_seconds=1800,environment=env); require_process("I1 deps",deps); edit=[str(source/item) for item in cfg["install"]["vendored_editables"]]; cmd=[str(python),"-m","pip","install","--no-deps"]; [cmd.extend(["-e",item]) for item in edit]; er=managed(cmd,cwd=source,output_root=output/"install_tacex/editables",name="install",timeout_seconds=1800,environment=env); require_process("I1 editables",er); after_i1=package_probe(python=python,source=source,root=output/"install_tacex",name="packages_after",environment=env,timeout=600); i1diff=audit_changes_against_plan(before_i1["all_distributions"],after_i1["all_distributions"],report_install_versions(reports["P0B"]),additional_allowed={"tacex":"0.1.0","tacex-assets":"0.1.0"}); write_json(output/"install_tacex/diff.json",i1diff)
  if not i1diff["success"]: manifest["classification"]="protected_dependency_changed_during_actual_install"; raise RuntimeError("I1 diff")
  manifest["stages"]["I1"]="passed"; check=managed([str(python),"-m","pip","check"],cwd=source,output_root=output/"package_provenance",name="pip_check",timeout_seconds=600,environment=env); require_process("I2 pip check",check); final=package_probe(python=python,source=source,root=output/"package_provenance",name="installed",environment=env,timeout=600); origins=final["module_origins"]; origin_ok=all([Path(origins["tacex"]).is_relative_to(source),Path(origins["tacex_assets"]).is_relative_to(source),Path(origins["tacex_uipc"]).is_relative_to(source),Path(origins["uipc"]).is_relative_to(prefix),Path(origins["curobo"]).is_relative_to(curobo),Path(origins["isaacsim"]).is_relative_to(prefix),Path(origins["isaaclab"]).is_relative_to(prefix)])
  if not origin_ok or not compare_protected_binaries(before_binaries,protected_binary_state(curobo,prefix))["success"]: manifest["classification"]="isaac51_blackwell_source_package_mix"; raise RuntimeError("I2 provenance")
  manifest["stages"]["I2"]="passed"; run_native_suite(stage="N0",python=python,source=source,curobo=curobo,output_root=output/"post_install_native_regression",environment=env,timeouts=cfg["timeouts_seconds"]); manifest["stages"]["N0"]="passed"; manifest["status"]="install_validated"; manifest["classification"]="native_and_package_layer_ready_for_isaac_startup"
 except BaseException as exc:
  manifest["status"]="failed"; manifest["failure"]={"class":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}; raise
 finally:
  cleanup=managed_cleanup_summary(output); write_json(output/"final_resources/continuation_cleanup.json",cleanup); manifest["managed_cleanup_complete"]=cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"]; manifest["ended_at"]=utc_now(); write_json(output/"run_manifest.json",manifest); write_json(output/"summary.json",manifest)
if __name__=="__main__": main()
