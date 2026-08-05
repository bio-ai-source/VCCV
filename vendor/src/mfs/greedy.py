from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference.decision import assign_decision_state
from src.mfs.coverage import coverage_regularized_order
from src.utils.io import ensure_dir, load_yaml


def _nearest_lookup(
    fused_idx: pd.DataFrame,
    *,
    target_key: str,
    context_key: str,
    pert_time: float,
    pert_dose: float,
    mode: str,
) -> pd.Series | None:
    idx = fused_idx[
        (fused_idx["target_key"] == target_key)
        & (fused_idx["context_key"] == context_key)
        & (fused_idx["mode"] == mode)
    ].copy()
    if idx.empty:
        return None
    d = np.abs(idx["pert_time"].astype(float) - float(pert_time)) + np.abs(
        np.log10(idx["pert_dose"].astype(float) + 1e-6) - np.log10(float(pert_dose) + 1e-6)
    )
    return idx.iloc[int(np.argmin(d.to_numpy(dtype=float)))]


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    v = np.clip(np.diag(cov), 1e-8, None)
    d = np.sqrt(v)
    r = cov / (d[:, None] * d[None, :])
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    r = 0.5 * (r + r.T)
    np.fill_diagonal(r, 1.0)
    # PSD projection
    eig, q = np.linalg.eigh(r)
    eig = np.clip(eig, 1e-6, None)
    r_psd = (q * eig) @ q.T
    d2 = np.sqrt(np.clip(np.diag(r_psd), 1e-8, None))
    r_psd = r_psd / (d2[:, None] * d2[None, :])
    np.fill_diagonal(r_psd, 1.0)
    return r_psd


def _logdet_objective(L: np.ndarray, idx: list[int], stab_lambda: float) -> float:
    if len(idx) == 0:
        return 0.0
    sub = L[np.ix_(idx, idx)]
    a = np.eye(len(idx), dtype=float) + sub + np.eye(len(idx), dtype=float) * stab_lambda
    sign, logdet = np.linalg.slogdet(a)
    if sign <= 0:
        return -1e9
    return float(logdet)


def run_mfs(repo_root: Path) -> None:
    cfg = load_yaml(repo_root / "configs/mfs.yaml")
    fused = pd.read_parquet(repo_root / "data/processed/do_fused_mu_var.parquet")
    sig = pd.read_parquet(repo_root / "data/processed/signatures_drug.parquet")
    gcols = sorted([c for c in sig.columns if c.startswith("G") and c[1:].isdigit()], key=lambda x: int(x[1:]))
    vcols = sorted([c for c in fused.columns if c.startswith("V") and c[1:].isdigit()], key=lambda x: int(x[1:]))
    g2i = {g: i for i, g in enumerate(gcols)}

    fused_idx = fused.copy()
    pred_dir = repo_root / "results/predictions_json/mechanism_instances"
    panel_dir = ensure_dir(repo_root / "results/mfs_panels")
    budgets = [int(x) for x in cfg["budgets"]]
    main_budget = budgets[0]
    design_top_m = int(cfg.get("design_top_m", 10))
    rho_max = float(cfg.get("max_collinearity", 0.95))
    eps_w = float(cfg.get("epsilon_w", 1e-6))
    stab_lambda = float(cfg.get("stability_lambda", 1e-6))
    max_instances = int(cfg.get("max_instances", 500))
    max_exhaustive_instances = int(cfg.get("max_exhaustive_instances", 120))
    exhaustive_pool_size = int(cfg.get("exhaustive_pool_size", 12))
    relax_grid_cfg = cfg.get("relax_collinearity_grid", [0.97, 0.99, 0.995, 0.999, 1.0])
    relax_grid = []
    for x in relax_grid_cfg:
        try:
            xv = float(x)
        except Exception:
            continue
        if xv > rho_max:
            relax_grid.append(xv)
    relax_grid = sorted(set(relax_grid))

    # Global noise correlation for DPP redundancy modeling.
    cov_global = np.cov(fused[gcols].to_numpy(dtype=float), rowvar=False)
    global_reference_variance = np.clip(np.diag(cov_global), 0.0, None)
    R_global = _cov_to_corr(cov_global)
    sigma_null = np.var(sig[gcols].to_numpy(dtype=float), axis=0) + 0.1

    opt_gap_lines = ["# MFS greedy approximation gap (log-det objective)"]
    all_ratios = []
    js_paths = sorted(pred_dir.glob("*.json"))[:max_instances]
    for js_path in js_paths:
        obj = json.loads(js_path.read_text(encoding="utf-8"))
        inst = obj["instance_id"]
        info = obj["input_summary"]
        decision = obj.get("decision") or assign_decision_state(
            obj["posterior_distribution"]
        )
        if decision["state"] != "panel":
            continue
        hyps = sorted(obj["posterior_distribution"], key=lambda x: -float(x.get("posterior", 0.0)))[:design_top_m]
        if len(hyps) < 2:
            continue

        means = []
        vars_diag = []
        post = []
        valid_h = []
        for h in hyps:
            ht = h["type"]
            post.append(float(h.get("posterior", 0.0)))
            if ht == "null":
                means.append(np.zeros(len(gcols), dtype=float))
                vars_diag.append(sigma_null.copy())
                valid_h.append(h)
                continue
            if ht == "single":
                rr = _nearest_lookup(
                    fused_idx,
                    target_key=h["target_key"],
                    context_key=info["context_key"],
                    pert_time=float(info["pert_time"]),
                    pert_dose=float(info["pert_dose"]),
                    mode=h["mode"],
                )
                if rr is None:
                    continue
                means.append(rr[gcols].to_numpy(dtype=float))
                vars_diag.append(np.maximum(rr[vcols].to_numpy(dtype=float), 1e-6))
                valid_h.append(h)
                continue
            if ht == "poly":
                pm = h.get("poly_meta", {})
                tg = pm.get("targets", [])
                ww = np.asarray(pm.get("weights", []), dtype=float)
                if len(tg) < 2 or len(ww) != len(tg):
                    continue
                ww = np.clip(ww, 1e-6, None)
                ww = ww / ww.sum()
                mu_mix = np.zeros(len(gcols), dtype=float)
                var_mix = np.zeros(len(gcols), dtype=float)
                ok = True
                for t, w in zip(tg, ww):
                    rr = _nearest_lookup(
                        fused_idx,
                        target_key=t,
                        context_key=info["context_key"],
                        pert_time=float(info["pert_time"]),
                        pert_dose=float(info["pert_dose"]),
                        mode=h["mode"],
                    )
                    if rr is None:
                        ok = False
                        break
                    mu = rr[gcols].to_numpy(dtype=float)
                    var = np.maximum(rr[vcols].to_numpy(dtype=float), 1e-6)
                    mu_mix += w * mu
                    var_mix += (w**2) * var
                if not ok:
                    continue
                means.append(mu_mix)
                vars_diag.append(var_mix + sigma_null * 0.05)
                valid_h.append(h)

        if len(means) < 2:
            continue
        p = np.asarray(post[: len(means)], dtype=float)
        p = np.clip(p, 1e-8, None)
        p = p / p.sum()
        H = np.stack(means, axis=0)
        V = np.stack(vars_diag, axis=0)

        mu_bar = np.sum(p[:, None] * H, axis=0)
        b = np.sum(p[:, None] * (H - mu_bar[None, :]) ** 2, axis=0)
        w_noise = np.sum(p[:, None] * V, axis=0)
        s = np.clip(b / (w_noise + eps_w), 0.0, None)
        L = (np.sqrt(s)[:, None] * R_global) * np.sqrt(s)[None, :]
        L = 0.5 * (L + L.T)
        L = L + np.eye(len(gcols), dtype=float) * stab_lambda

        if decision["panel_policy"] == "coverage":
            selected = coverage_regularized_order(
                global_reference_variance,
                R_global,
                budget=main_budget,
                redundancy_coefficient=float(cfg.get("coverage_redundancy_coefficient", 1.0)),
            )
            panel = [gcols[i] for i in selected]
            out = {
                "instance_id": inst,
                "budget": main_budget,
                "panel": panel,
                "decision": decision,
                "objective": "coverage_regularized_variance",
                "reference_variance": "empirical variance across the frozen fused dictionary",
                "redundancy_coefficient": float(cfg.get("coverage_redundancy_coefficient", 1.0)),
            }
            (panel_dir / f"{inst}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
            continue

        selected: list[int] = []
        rejected: list[dict[str, object]] = []
        gain_curve: list[float] = []
        relax_used_steps = 0
        hk = [g for g in cfg["housekeeping_genes"] if g in g2i]
        for g in hk:
            gi = g2i[g]
            if gi not in selected:
                selected.append(gi)

        for _ in range(main_budget - len(selected)):
            base_obj = _logdet_objective(L, selected, stab_lambda=stab_lambda)
            gain_cache: list[tuple[int, float, float]] = []
            for gi in range(len(gcols)):
                if gi in selected:
                    continue
                max_corr = max(abs(float(R_global[gi, sj])) for sj in selected) if selected else 0.0
                gain = _logdet_objective(L, selected + [gi], stab_lambda=stab_lambda) - base_obj
                gain_cache.append((gi, float(gain), float(max_corr)))

            best = None
            best_thr = rho_max
            for thr in [rho_max] + relax_grid:
                cand = [x for x in gain_cache if x[2] <= thr]
                if not cand:
                    continue
                cand_best = max(cand, key=lambda t: t[1])
                best = (cand_best[0], cand_best[1], cand_best[2])
                best_thr = thr
                break
            if best is None:
                # No feasible candidate even at relaxed thresholds.
                for gi, _, corr in gain_cache:
                    rejected.append({"gene": gcols[gi], "reason": f"collinearity>{rho_max:.3f}", "max_corr": corr})
                break
            selected.append(best[0])
            gain_curve.append(float(max(best[1], 0.0)))
            if best_thr > rho_max + 1e-12:
                relax_used_steps += 1
                rejected.append(
                    {
                        "gene": gcols[best[0]],
                        "reason": "selected_with_relaxed_collinearity",
                        "max_corr": float(best[2]),
                        "effective_threshold": float(best_thr),
                    }
                )
        panel = [gcols[i] for i in selected]
        out = {
            "instance_id": inst,
            "budget": main_budget,
            "panel": panel,
            "decision": decision,
            "logdet_gain_curve": gain_curve,
            "entropy_curve_proxy": gain_curve,
            "objective": "logdet_dpp",
            "rho_base": rho_max,
            "rho_relax_grid": relax_grid,
            "n_relaxed_steps": int(relax_used_steps),
            "rejected": rejected,
        }
        (panel_dir / f"{inst}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

        # Approximation quality (B<=10) by exhaustive on a bounded subset for speed.
        if len(all_ratios) >= max_exhaustive_instances:
            continue
        b_small = min(10, main_budget)
        if b_small <= 0:
            continue
        rank_genes = np.argsort(-s)[:exhaustive_pool_size].tolist()
        cand_pool = list(dict.fromkeys(selected[:exhaustive_pool_size] + rank_genes))
        if len(cand_pool) < b_small:
            continue
        greedy_score = _logdet_objective(L, selected[:b_small], stab_lambda=stab_lambda)
        best_score = -1e9
        for comb in combinations(cand_pool, b_small):
            s_obj = _logdet_objective(L, list(comb), stab_lambda=stab_lambda)
            if s_obj > best_score:
                best_score = s_obj
        ratio = greedy_score / max(best_score, 1e-8)
        all_ratios.append(ratio)

    opt_gap_lines.append(f"- evaluated_instances: {len(js_paths)}")
    if all_ratios:
        opt_gap_lines.append(f"- n_instances_with_exhaustive: {len(all_ratios)}")
        opt_gap_lines.append(f"- mean_ratio: {float(np.mean(all_ratios)):.4f}")
        opt_gap_lines.append(f"- min_ratio: {float(np.min(all_ratios)):.4f}")
    else:
        opt_gap_lines.append("- no instance available for exhaustive check")
    (repo_root / "results/logs/mfs_greedy_opt_gap.md").write_text("\n".join(opt_gap_lines), encoding="utf-8")
