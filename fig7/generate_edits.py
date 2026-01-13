from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def _iter_dict_paths(d: Any, prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    """Yield (path, value) for every key in nested dicts."""
    if isinstance(d, dict):
        for k, v in d.items():
            if not isinstance(k, str):
                continue
            p = prefix + (k,)
            yield (p, v)
            yield from _iter_dict_paths(v, p)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _iter_dict_paths(v, prefix + (str(i),))


def _get_by_path(root: Any, path: Tuple[str, ...]) -> Any:
    cur = root
    for p in path:
        if isinstance(cur, dict):
            cur = cur[p]
        elif isinstance(cur, list):
            cur = cur[int(p)]
        else:
            raise KeyError(path)
    return cur


def _set_by_path(root: Any, path: Tuple[str, ...], value: Any) -> None:
    cur = root
    for p in path[:-1]:
        if isinstance(cur, dict):
            cur = cur[p]
        elif isinstance(cur, list):
            cur = cur[int(p)]
        else:
            raise KeyError(path)
    last = path[-1]
    if isinstance(cur, dict):
        cur[last] = value
    elif isinstance(cur, list):
        cur[int(last)] = value
    else:
        raise KeyError(path)


def set_any(cfg: Dict[str, Any], key_candidates: List[str], value: Any) -> None:

    candidates_lc = {k.lower() for k in key_candidates}

    # Try recursive match first
    for path, _ in _iter_dict_paths(cfg):
        last = path[-1]
        if last.lower() in candidates_lc:
            _set_by_path(cfg, path, value)
            return

    # Otherwise set at top-level (best-effort)
    cfg[key_candidates[0]] = value


def load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: Dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def sd3_sweep() -> List[Dict[str, Any]]:

    runs: List[Dict[str, Any]] = []

    # ODE inversion (3 target CFG)
    for j, cfg_tgt in enumerate([13.5, 16.5, 19.5]):
        runs.append({
            "model": "sd3",
            "method": "ode_inv",
            "label": f"ODEInv_cfg{cfg_tgt:g}",
            "order_idx": j,
            "overrides": {
                "T": 50,
                "n_max": 33,
                "cfg_src": 3.5,
                "cfg_tgt": cfg_tgt,
            }
        })

    # iRFDS (official impl / hypers): we just set method; your run_script.py must support it
    runs.append({
        "model": "sd3",
        "method": "irfds",
        "label": "iRFDS_official",
        "order_idx": 0,
        "overrides": {
            "T": 50,
        }
    })

    # FlowEdit (3 target CFG)
    for j, cfg_tgt in enumerate([13.5, 16.5, 19.5]):
        runs.append({
            "model": "sd3",
            "method": "flowedit",
            "label": f"Ours_cfg{cfg_tgt:g}",
            "order_idx": j,
            "overrides": {
                "T": 50,
                "n_max": 33,
                "cfg_src": 3.5,
                "cfg_tgt": cfg_tgt,
            }
        })

    # # SDEdit
    # nmax_list = [10, 15, 20, 25, 30, 35, 40]
    # for i, nmax in enumerate(nmax_list):
    #     strength = 0.1 * (i + 2)  # 0.2..0.8 (for labeling only)
    #     runs.append({
    #         "model": "sd3",
    #         "method": "sdedit",
    #         "label": f"SDEdit_{strength:.1f}",
    #         "order_idx": i,
    #         "overrides": {
    #             "T": 50,
    #             "n_max": nmax,
    #             "cfg_tgt": 13.5,
    #         }
    #     })

    return runs


def flux_sweep(include_extra_figS3: bool = False) -> List[Dict[str, Any]]:

    # runs: List[Dict[str, Any]] = []

    # # SDEdit
    # nmax_list = [7, 14, 21]
    # strength_list = [0.25, 0.50, 0.75]
    # for i, (nmax, s) in enumerate(zip(nmax_list, strength_list)):
    #     runs.append({
    #         "model": "flux",
    #         "method": "sdedit",
    #         "label": f"SDEdit_{s:g}",
    #         "order_idx": i,
    #         "overrides": {
    #             "T": 28,
    #             "n_max": nmax,
    #             "cfg_tgt": 5.5,
    #         }
    #     })

    # # ODE inv
    # ode_nmax_values = [24] + ([20] if include_extra_figS3 else [])
    # for ode_nmax in ode_nmax_values:
    #     for j, cfg_tgt in enumerate([3.5, 5.5]):
    #         runs.append({
    #             "model": "flux",
    #             "method": "ode_inv",
    #             "label": f"ODEInv_n{ode_nmax}_cfg{cfg_tgt:g}",
    #             "order_idx": j,
    #             "overrides": {
    #                 "T": 28,
    #                 "n_max": ode_nmax,
    #                 "cfg_src": 1.5,
    #                 "cfg_tgt": cfg_tgt,
    #             }
    #         })

    # # FlowEdit
    # for j, cfg_tgt in enumerate([3.5, 5.5]):
    #     runs.append({
    #         "model": "flux",
    #         "method": "flowedit",
    #         "label": f"Ours_cfg{cfg_tgt:g}",
    #         "order_idx": j,
    #         "overrides": {
    #             "T": 28,
    #             "n_max": 24,
    #             "cfg_src": 1.5,
    #             "cfg_tgt": cfg_tgt,
    #         }
    #     })

    # # RF-Inversion
    # etas = [0.9] + ([1.0] if include_extra_figS3 else [])
    # taus = [8, 7, 6]
    # idx = 0
    # for eta in etas:
    #     for tau in taus:
    #         runs.append({
    #             "model": "flux",
    #             "method": "rf_inv",
    #             "label": f"RFInv_eta{eta:g}_tau{tau}",
    #             "order_idx": idx,
    #             "overrides": {
    #                 "T": 28,
    #                 "rf_s": 0,          # starting time
    #                 "rf_tau": tau,      # stopping time
    #                 "rf_eta": eta,      # strength
    #             }
    #         })
    #         idx += 1

    # # RF Edit
    # for k, inj in enumerate([2, 3, 5]):
    #     runs.append({
    #         "model": "flux",
    #         "method": "rf_edit",
    #         "label": f"RFEdit_inj{inj}",
    #         "order_idx": k,
    #         "overrides": {
    #             "rfedit_steps": 30,
    #             "rfedit_guidance": 2,
    #             "rfedit_injection": inj,
    #         }
    #     })

    pass


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd3_template", type=str, default="SD3_exp.yaml")
    ap.add_argument("--flux_template", type=str, default="FLUX_exp.yaml")
    ap.add_argument("--out_root", type=str, default="outputs_fig7")
    ap.add_argument("--exp_out_dir", type=str, default="scripts/fig7/exp_generated")
    ap.add_argument("--include_extra_figS3", action="store_true",
                    help="Also run extra hyperparams mentioned for Fig S3 (ODE nmax=20, RFInv eta=1.0).")
    ap.add_argument("--skip_irfds", action="store_true",
                    help="Skip iRFDS run (if your run_script.py doesn't support it).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only generate YAMLs + manifest, do not call run_script.py.")
    ap.add_argument("--model", type=str, choices=["sd3", "flux", "both"], default="both",
                    help="Which model to run: 'sd3', 'flux', or 'both' (default: both)")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    sd3_template = (repo_root / args.sd3_template).resolve()
    flux_template = (repo_root / args.flux_template).resolve()
    out_root = (repo_root / args.out_root).resolve()
    exp_out_dir = (repo_root / args.exp_out_dir).resolve()

    if not sd3_template.exists():
        raise FileNotFoundError(f"Missing SD3 template: {sd3_template}")
    if not flux_template.exists():
        raise FileNotFoundError(f"Missing FLUX template: {flux_template}")
    if not (repo_root / "run_script.py").exists():
        raise FileNotFoundError("You must run this from the FlowEdit repo root (run_script.py not found).")

    # Load templates and generate runs based on selected model
    runs = []
    sd3_base = None
    flux_base = None
    
    if args.model in ["sd3", "both"]:
        sd3_base = load_yaml(sd3_template)
        # Handle case where YAML contains a list with single dict
        if isinstance(sd3_base, list) and len(sd3_base) > 0:
            sd3_base = sd3_base[0]
        runs.extend(sd3_sweep())
    
    if args.model in ["flux", "both"]:
        flux_base = load_yaml(flux_template)
        # Handle case where YAML contains a list with single dict
        if isinstance(flux_base, list) and len(flux_base) > 0:
            flux_base = flux_base[0]
        runs.extend(flux_sweep(include_extra_figS3=args.include_extra_figS3))
    
    if args.skip_irfds:
        runs = [r for r in runs if r["method"] != "irfds"]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = out_root / "runs_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write manifest
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "model", "method", "label", "order_idx", "exp_yaml", "output_dir", "template_used", "timestamp"])

        for idx, r in enumerate(runs):
            model = r["model"]
            method = r["method"]
            label = r["label"]
            order_idx = r["order_idx"]
            overrides = r["overrides"]

            base = sd3_base if model == "sd3" else flux_base
            template_used = str(sd3_template if model == "sd3" else flux_template)

            cfg = copy.deepcopy(base)

            # Output directory per run
            run_id = f"{model}_{method}_{idx:03d}_{label}".replace(" ", "_")
            run_dir = out_root / model / method / run_id
            set_any(cfg, ["output_dir", "out_dir", "save_dir", "results_dir"], str(run_dir))

            # Method / algorithm selector
            set_any(cfg, ["method", "mode", "edit_method", "algorithm"], method)

            # Standard overrides
            if "T" in overrides:
                set_any(cfg, ["T", "t_steps", "num_steps", "num_inference_steps", "T_steps"], overrides["T"])
            if "n_max" in overrides:
                set_any(cfg, ["n_max", "nmax"], overrides["n_max"])
            if "cfg_src" in overrides:
                set_any(cfg, ["cfg_src", "cfg_source", "cfg@source", "guidance_scale_src", "source_cfg", "src_guidance_scale"], overrides["cfg_src"])
            if "cfg_tgt" in overrides:
                set_any(cfg, ["cfg_tgt", "cfg_target", "cfg@target", "guidance_scale_tgt", "target_cfg", "tar_guidance_scale"], overrides["cfg_tgt"])

            # RF-Inversion overrides
            if "rf_s" in overrides:
                set_any(cfg, ["rf_s", "s_start", "rf_start", "start_time"], overrides["rf_s"])
            if "rf_tau" in overrides:
                set_any(cfg, ["rf_tau", "tau", "stop_time", "stopping_time"], overrides["rf_tau"])
            if "rf_eta" in overrides:
                set_any(cfg, ["rf_eta", "eta", "strength"], overrides["rf_eta"])

            # RF Edit overrides
            if "rfedit_steps" in overrides:
                set_any(cfg, ["rfedit_steps", "steps", "rf_steps"], overrides["rfedit_steps"])
            if "rfedit_guidance" in overrides:
                set_any(cfg, ["rfedit_guidance", "guidance", "guidance_scale"], overrides["rfedit_guidance"])
            if "rfedit_injection" in overrides:
                set_any(cfg, ["rfedit_injection", "injection", "injection_scale", "inject"], overrides["rfedit_injection"])

            # Save exp yaml (wrap in list to match run_script.py's expected format)
            exp_path = exp_out_dir / model / method / f"{run_id}.yaml"
            dump_yaml([cfg], exp_path)

            w.writerow([run_id, model, method, label, order_idx, str(exp_path), str(run_dir), template_used, timestamp])

            # Run editing
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n=== Running {run_id} ===")
                cmd = [sys.executable, "run_script.py", "--exp_yaml", str(exp_path)]
                subprocess.run(cmd, check=True)

    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
