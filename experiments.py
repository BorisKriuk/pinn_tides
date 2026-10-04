#!/usr/bin/env python3
"""
experiments.py — every experiment in the paper (cached per fit) + figures (PDF/PNG + pgfplots .dat) + LaTeX tables.

  python experiments.py all [--quick] [--skip-res]
  python experiments.py check      E0 energy identity + adjoint gradcheck (PHYS, PHYS-fc, NN-8, POLY2, NN-8+energy)
  python experiments.py cv         E1 5-fold hash CV: capacity ladder + feature ablation  (Table: main)
  python experiments.py block      E2 spatial-block CV (k-means blocks on the sphere)       [H2]
  python experiments.py reg        E3 lambda x width sweep
  python experiments.py transfer   E4 fit on M2 only, predict S2,K1,O1 (frequency transfer) [H1]
  python experiments.py energy     E5 energy-constrained fits, skill/admissibility Pareto   [H4]
  python experiments.py distill    E6 distil NN-8 into polynomial laws + partial dependence [H3]
  python experiments.py slr        E7 spring-range sensitivity to +1 m sea level (fold ensemble) [H5]
  python experiments.py res        E8 2° vs 1° on a common gauge set                         [H6]
  python experiments.py figures    figs/*.pdf|png|dat, tables/*.tex from results/summary_*.json
Each fit is cached in results/runs/<hash>.json (+ fields .npz); reruns skip finished fits.
"""
import argparse, hashlib, json, math, time
import numpy as np
import torch
import tide as T

RUNS, FIELDS = T.RESULTS / "runs", T.RESULTS / "fields"
FIGS, TABS = T.ROOT / "figs", T.ROOT / "tables"
MAIN = ["phys", "poly:2", "nn:8", "nn:32", "nn:128", "nn:32:2:H+slope", "nn:32:2:H+slope+abslat"]
CORE = ["phys", "nn:8", "nn:32", "nn:128", "nn:32:2:H+slope"]
FAMILY = ["nn:8", "nn:32", "nn:128"]
TRANSFER = ["phys", "physfc", "nn:8", "nn:8:2:H+slope+abslat", "nn:8:2:H+slope+fw"]
LAMBDAS, MUS = [1e-3, 1e-4, 1e-6], [0.0, 0.03, 0.1, 0.3, 1.0]
W1, W2 = 3.5, 7.16                                # IEEE column / text width [in]


# ============================================================================ infrastructure
class Ctx:
    def __init__(self, a, res, keep_ids=None):
        self.res = res
        self.lat, self.lon, self.Hraw = T.load_bathymetry(res)
        self.m = T.Mesh(self.lat, self.lon, self.Hraw)
        g = T.build_gauges(self.m, a.year, a.max_gauges)
        if keep_ids is not None:
            g = [r for r in g if r["id"] in keep_ids]
        self.g, self.w = g, T.density_weights(g)
        self.gdig = T.digest([r["id"] for r in g])
        print(f"[ctx] {res:g}°: {self.m.N} ocean cells, {self.m.Nu + self.m.Nv + self.m.N} unknowns, {len(g)} gauges")


_CTX = {}


def get_ctx(a, res=None, keep=None):
    key = (res or a.res, keep)
    if key not in _CTX:
        _CTX[key] = Ctx(a, res or a.res, keep)
    return _CTX[key]


def folds(a):
    return [0, 1] if a.quick else list(range(a.kfolds))


def base_cfg(a, **kw):
    c = dict(res=a.res, year=a.year, seed=a.seed, k=a.kfolds, iters=a.iters, scheme="hash", buffer_km=0.0,
             names=a.constituents.split(","), names_eval=None, l2_nn=a.l2_nn, l2_phys=a.l2_phys, energy=None)
    c.update(kw)
    return c


def run_one(ctx, spec, fold, cfg):
    key = hashlib.sha256(json.dumps(dict(cfg, spec=spec, fold=fold, gauges=ctx.gdig, v=3),
                                    sort_keys=True).encode()).hexdigest()[:16]
    f = RUNS / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    C = T.make_closure(spec)
    tr, te = T.split_gauges(ctx.g, cfg["seed"], fold, cfg["k"], cfg["scheme"], cfg["buffer_km"], verbose=False)
    if len(te) == 0 or len(tr) < 10:
        return None
    l2 = cfg["l2_phys"] if isinstance(C, T.PhysClosure) else cfg["l2_nn"]
    print(f"[run] {C.label:36s} fold {fold} {cfg['scheme']:5s} {cfg['res']:g}° l2={l2:g} "
          f"energy={cfg['energy']} fit={cfg['names']} ({len(tr)}/{len(te)})", flush=True)
    fit = T.fit_theta(ctx.m, C, ctx.g, tr, cfg["names"], ctx.w, C.theta0(cfg["seed"]), l2, cfg["iters"],
                      energy=cfg["energy"], verbose=False)
    sc, fields = T.score_split(ctx.m, C, fit["theta"], ctx.g, tr, te, cfg["names"], ctx.w, cfg["names_eval"],
                               keep_fields=True)
    np.savez_compressed(FIELDS / f"{key}.npz", **fields)
    rec = dict(key=key, spec=spec, closure=C.spec, label=C.label, n_params=C.n, fold=fold, cfg=cfg, l2=l2,
               theta=fit["theta"], train_loss=fit["loss"], n_evals=fit["n_evals"], seconds=fit["seconds"],
               history=fit["history"], **sc)
    rec = T._jsonable(rec)
    f.write_text(json.dumps(rec))
    print(f"      in {sc['in_sample']:.1f} | out {sc['held_out']:.1f} | GP {sc['gp']:.1f} | D_M2 {sc['D_TW']:.2f} TW "
          f"({100 * sc['shallow_frac']:.0f}% shallow) | {fit['seconds']:.0f} s, {fit['n_evals']} evals", flush=True)
    return rec


def run_cv(ctx, specs, cfg, fl):
    out = {}
    for s in specs:
        recs = [r for r in (run_one(ctx, s, f, cfg) for f in fl) if r is not None]
        if recs:
            out[s] = recs
    return out


def kendall(a, b):
    n, s = len(a), 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / max(n * (n - 1) / 2, 1)


def summarise(res, ref="phys"):
    keys = ("test_ids", "test_w", "e_model", "e_gp", "e_nearest", "e_eq", "test_lat", "test_lon")

    def pooled(recs):
        cat = {k: np.concatenate([np.asarray(r[k], float) for r in recs]) for k in keys}
        o = np.argsort(cat["test_ids"])
        return {k: v[o] for k, v in cat.items()}

    P = {s: pooled(r) for s, r in res.items()}
    rms = lambda p, e: 100 * math.sqrt(np.sum(p["test_w"] * p[e]) / np.sum(p["test_w"]) / 2)
    rows = []
    for s, recs in res.items():
        A = lambda k: np.array([r[k] for r in recs], float)
        p = P[s]
        nm = list(recs[0]["held_out_row"])
        row = dict(spec=s, label=recs[0]["label"], n_params=recs[0]["n_params"], n_folds=len(recs),
                   in_mean=A("in_sample").mean(), in_sd=A("in_sample").std(),
                   out_mean=A("held_out").mean(), out_sd=A("held_out").std(),
                   gap_mean=(A("held_out") - A("in_sample")).mean(), gap_sd=(A("held_out") - A("in_sample")).std(),
                   gp_mean=A("gp").mean(), gp_sd=A("gp").std(), nearest_mean=A("nearest").mean(),
                   nearest_sd=A("nearest").std(),
                   pooled=rms(p, "e_model"), pooled_gp=rms(p, "e_gp"), pooled_nearest=rms(p, "e_nearest"),
                   pooled_eq=rms(p, "e_eq"), vs_gp=T.paired_bootstrap(p["e_model"], p["e_gp"], p["test_w"]),
                   p_gp_foldmean=A("p_vs_gp").mean(), D=A("D_TW").mean(), D_sd=A("D_TW").std(),
                   D_all={n: float(np.mean([r["D_TW_all"][n] for r in recs])) for n in recs[0]["D_TW_all"]},
                   shallow=A("shallow_frac").mean(), polar=A("polar_frac").mean(), beta=A("beta").mean(),
                   r_eff_q=np.mean([r["r_eff_q"] for r in recs], 0), seconds=A("seconds").mean(),
                   energy_closure_max=float(max(abs(v) for r in recs for v in r["energy_closure"].values())),
                   out_row={n: float(np.mean([r["held_out_row"][n] for r in recs])) for n in nm},
                   out_row_sd={n: float(np.std([r["held_out_row"][n] for r in recs])) for n in nm},
                   gp_row={n: float(np.mean([r["gp_row"][n] for r in recs])) for n in nm},
                   D_row=np.mean([r["D_row_M2"] for r in recs], 0), keys=[r["key"] for r in recs],
                   history0=recs[0]["history"], pooled_arrays={k: p[k] for k in keys})
        if ref in P and s != ref and np.array_equal(P[s]["test_ids"], P[ref]["test_ids"]):
            row["vs_ref"] = T.paired_bootstrap(p["e_model"], P[ref]["e_model"], p["test_w"])
            row["wins_vs_ref"] = int(sum(x["held_out"] < y["held_out"] for x, y in zip(recs, res[ref])))
        rows.append(row)
    fam = [r for r in rows if r["spec"] in FAMILY]
    tau = kendall([r["in_mean"] for r in fam], [r["out_mean"] for r in fam]) if len(fam) > 1 else None
    return T._jsonable(dict(ref=ref, rows=rows, kendall_in_vs_out=tau))


def save(name, obj):
    (T.RESULTS / name).write_text(json.dumps(T._jsonable(obj), indent=1))


def load(name):
    f = T.RESULTS / name
    return json.loads(f.read_text()) if f.exists() else None


# ============================================================================ experiments
def exp_check(a):
    ctx = get_ctx(a)
    out = []
    for spec, en in [("phys", None), ("physfc", None), ("poly:2", None), ("nn:8", None),
                     ("nn:8", dict(mu_D=0.1, mu_sh=0.1))]:
        r = T.run_check(ctx.m, T.make_closure(spec), ctx.g, ("M2", "K1"), en, a.seed)
        r["spec_str"] = spec
        out.append(r)
        print(f"[check] {r['label']:30s} energy " + ", ".join(f"{n}:{e['rel_mismatch']:.1e}" for n, e in r["energy"].items())
              + " | grad " + ", ".join(f"{d['rel']:.1e}" for d in r["grad"]))
    save("check.json", out)


def exp_cv(a):
    ctx = get_ctx(a)
    S = summarise(run_cv(ctx, MAIN, base_cfg(a), folds(a)))
    S["lat"] = ctx.m.lat_np
    save("summary_cv_hash.json", S)
    _print_summary(S, "hash CV")


def exp_block(a):
    ctx = get_ctx(a)
    labs = T.spatial_blocks(ctx.g, a.kfolds, a.seed)
    S = summarise(run_cv(ctx, CORE, base_cfg(a, scheme="block", buffer_km=a.buffer_km), folds(a)))
    S["block_sizes"] = np.bincount(labs, minlength=a.kfolds)
    S["block_of"] = {int(r["id"]): int(l) for r, l in zip(ctx.g, labs)}
    S["gauge_lat"], S["gauge_lon"] = [r["lat"] for r in ctx.g], [r["lon"] for r in ctx.g]
    save("summary_cv_block.json", S)
    _print_summary(S, "spatial-block CV")


def exp_reg(a):
    ctx = get_ctx(a)
    rows = []
    for lam in LAMBDAS:
        S = summarise(run_cv(ctx, FAMILY, base_cfg(a, l2_nn=lam), folds(a)))
        for r in S["rows"]:
            r["lam"] = lam
            rows.append(r)
    save("summary_reg.json", dict(rows=rows))


def exp_transfer(a):
    ctx = get_ctx(a)
    cfg = base_cfg(a, names=["M2"], names_eval=["M2", "S2", "K1", "O1"])
    S = summarise(run_cv(ctx, TRANSFER, cfg, folds(a)))
    save("summary_transfer.json", S)
    for r in S["rows"]:
        print(f"[transfer] {r['label']:34s} " + " ".join(f"{n}:{v:5.1f}" for n, v in r["out_row"].items())
              + "   GP " + " ".join(f"{n}:{v:5.1f}" for n, v in r["gp_row"].items()))


def exp_energy(a):
    ctx = get_ctx(a)
    rows = []
    for mu in MUS:
        cfg = base_cfg(a, energy=None if mu == 0 else dict(mu_D=mu, mu_sh=mu))
        for r in summarise(run_cv(ctx, ["phys", "nn:8"], cfg, folds(a)))["rows"]:
            r["mu"] = mu
            rows.append(r)
            print(f"[energy] mu={mu:<5g} {r['label']:28s} out {r['out_mean']:.1f} D {r['D']:.2f} TW shallow {100 * r['shallow']:.0f}%")
    save("summary_energy.json", dict(rows=rows))


def _feat_points(feats, H, slope, s, om):
    H, slope, s = np.broadcast_arrays(np.asarray(H, float), np.asarray(slope, float), np.asarray(s, float))
    cols = dict(H=0.5 * np.log(np.maximum(H, T.H_MIN) / 1000.0), slope=np.log1p(slope / 0.005), lat=s,
                abslat=np.abs(s), fw=np.minimum(np.abs(2 * T.OMEGA_E * s) / om, 3.0))
    return torch.as_tensor(np.stack([cols[f] for f in feats], -1))


def exp_distill(a):
    ctx, m = get_ctx(a), get_ctx(a).m
    cfg = base_cfg(a)
    recs = run_cv(ctx, ["nn:8"], cfg, folds(a))["nn:8"]
    C = T.make_closure("nn:8")
    omM2 = T.omega("M2")
    X = T.features(m, C.feats, omM2)
    mask = m.maskb
    wc = np.broadcast_to(m.area_np[:, None], mask.shape)[mask]
    slope_deep = float(np.median(m.slope_np[(m.H_np > 3000) & mask]))
    slope_rough = float(np.percentile(m.slope_np[(m.H_np > 3000) & mask], 90))
    s_grid, H_grid = np.linspace(-0.97, 0.97, 97), np.logspace(1.3, 3.8, 80)
    out, pd = [], dict(s=s_grid, H=H_grid, lat_smooth=[], lat_rough=[], H_curve=[], slope_deep=slope_deep,
                       slope_rough=slope_rough)
    for rec in recs:
        th = torch.tensor(rec["theta"])
        with torch.no_grad():
            h = C.pre(th, X).numpy()[mask]
            pd["lat_smooth"].append(T.r_from_h(C.pre(th, _feat_points(C.feats, 4000.0, slope_deep, s_grid, omM2))).numpy())
            pd["lat_rough"].append(T.r_from_h(C.pre(th, _feat_points(C.feats, 4000.0, slope_rough, s_grid, omM2))).numpy())
            pd["H_curve"].append(T.r_from_h(C.pre(th, _feat_points(C.feats, H_grid, slope_deep, 0.5, omM2))).numpy())
        tr, te = T.split_gauges(ctx.g, cfg["seed"], rec["fold"], cfg["k"], verbose=False)
        for deg in (1, 2, 3):
            Pc = T.PolyClosure(deg, C.feats)
            with torch.no_grad():
                Phi = Pc.design(X).numpy()[mask]
            sw = np.sqrt(wc)
            coef, *_ = np.linalg.lstsq(Phi * sw[:, None], h * sw, rcond=None)
            yhat = Phi @ coef
            ybar = np.sum(wc * h) / wc.sum()
            R2 = 1 - np.sum(wc * (h - yhat) ** 2) / np.sum(wc * (h - ybar) ** 2)
            theta_p = np.concatenate([coef, [rec["theta"][-1]]])
            sc, _ = T.score_split(m, Pc, theta_p, ctx.g, tr, te, cfg["names"], ctx.w)
            out.append(T._jsonable(dict(deg=deg, fold=rec["fold"], R2=R2, n_params=Pc.n, coef=coef, terms=Pc.names[:-1],
                                        held_out=sc["held_out"], in_sample=sc["in_sample"], nn_held_out=rec["held_out"],
                                        D_TW=sc["D_TW"], shallow=sc["shallow_frac"], e_model=sc["e_model"],
                                        test_w=sc["test_w"], test_ids=sc["test_ids"])))
            print(f"[distill] fold {rec['fold']} deg {deg}: R2={R2:.3f} held-out {sc['held_out']:.1f} (NN {rec['held_out']:.1f})")
    summ = []
    for deg in (1, 2, 3):
        D = [o for o in out if o["deg"] == deg]
        e = np.concatenate([o["e_model"] for o in D]); w = np.concatenate([o["test_w"] for o in D])
        summ.append(dict(deg=deg, n_params=D[0]["n_params"], R2=np.mean([o["R2"] for o in D]),
                         out_mean=np.mean([o["held_out"] for o in D]), out_sd=np.std([o["held_out"] for o in D]),
                         nn_out_mean=np.mean([o["nn_held_out"] for o in D]),
                         pooled=100 * math.sqrt(np.sum(w * e) / np.sum(w) / 2), D=np.mean([o["D_TW"] for o in D]),
                         shallow=np.mean([o["shallow"] for o in D]), coef_fold0=D[0]["coef"], terms=D[0]["terms"]))
    save("summary_distill.json", dict(rows=summ, folds=[{k: v for k, v in o.items() if k not in ("e_model", "test_w")}
                                                        for o in out], pd=pd))


def exp_slr(a):
    ctx = get_ctx(a)
    m0 = ctx.m
    m1 = T.Mesh(ctx.lat, ctx.lon, np.where(m0.maskb, ctx.Hraw + a.slr, ctx.Hraw))
    assert np.array_equal(m1.maskb, m0.maskb)
    cfg = base_cfg(a)
    gj, gi = np.array([r["j"] for r in ctx.g]), np.array([r["i"] for r in ctx.g])
    wA = np.broadcast_to(m0.area_np[:, None], m0.maskb.shape)
    summ = {}
    for spec in ["phys", "nn:8"]:
        C = T.make_closure(spec)
        dR, R0 = [], []
        for rec in run_cv(ctx, [spec], cfg, folds(a))[spec]:
            th = torch.tensor(rec["theta"])
            with torch.no_grad():
                R = [2 * (T.solve(mm, C, th, "M2").abs() + T.solve(mm, C, th, "S2").abs()).numpy() for mm in (m0, m1)]
            dR.append((R[1] - R[0]) / a.slr)
            R0.append(R[0])
        dR = np.array(dR)
        mean, sd = dR.mean(0), dR.std(0)
        agree = (np.sign(dR) == np.sign(mean)[None]).mean(0)
        np.savez_compressed(FIELDS / f"slr_{spec.replace(':', '_')}.npz", lat=ctx.lat, lon=ctx.lon, mask=m0.maskb,
                            dR_mean=mean, dR_sd=sd, agree=agree, R0=np.mean(R0, 0))
        mk = m0.maskb
        summ[spec] = dict(area_mean_cm_per_m=100 * np.sum(mean[mk] * wA[mk]) / wA[mk].sum(),
                          frac_cells_all_folds_agree=float(np.mean(agree[mk] == 1.0)),
                          p95_abs_cm_per_m=float(100 * np.percentile(np.abs(mean[mk]), 95)),
                          gauge_mean_cm_per_m=float(100 * mean[gj, gi].mean()),
                          gauge_frac_increase=float(np.mean(mean[gj, gi] > 0)), n_folds=len(dR))
        print(f"[slr] {spec}: {summ[spec]}")
    save("summary_slr.json", dict(delta_m=a.slr, rows=summ))


def exp_res(a):
    c_coarse, c_fine = get_ctx(a, a.res), get_ctx(a, a.res_fine)
    common = frozenset({r["id"] for r in c_coarse.g} & {r["id"] for r in c_fine.g})
    rows = []
    for res in (a.res, a.res_fine):
        ctx = get_ctx(a, res, common)
        for r in summarise(run_cv(ctx, ["phys", "nn:8"], base_cfg(a, res=res), folds(a)))["rows"]:
            r["res"] = res
            rows.append(r)
            print(f"[res] {res:g}° {r['label']:28s} out {r['out_mean']:.1f}±{r['out_sd']:.1f}  D {r['D']:.2f} TW  "
                  f"shallow {100 * r['shallow']:.0f}%")
    save("summary_res.json", dict(rows=rows, n_common=len(common)))


def _print_summary(S, title):
    print(f"\n==== {title} ====  Kendall tau(in, out) over NN widths = {S['kendall_in_vs_out']}")
    for r in S["rows"]:
        vr = r.get("vs_ref")
        print(f"{r['label']:36s}{r['n_params']:7d}  in {r['in_mean']:5.1f}±{r['in_sd']:4.1f}  out {r['out_mean']:5.1f}±"
              f"{r['out_sd']:4.1f}  pooled {r['pooled']:5.1f}  GP {r['pooled_gp']:5.1f}  P(>=GP) {r['vs_gp']['p']:.3f}  "
              + (f"vs PHYS {vr['diff']:+.1f} [{vr['lo']:+.1f},{vr['hi']:+.1f}] wins {r['wins_vs_ref']}  " if vr else "")
              + f"D {r['D']:.2f} TW shallow {100 * r['shallow']:.0f}%")


# ============================================================================ figures & tables
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
    return plt


def _save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=250)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[fig] {name}")


def _dat(name, header, rows):
    with open(FIGS / f"{name}.dat", "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.6g}" if isinstance(v, (float, int, np.floating, np.integer)) else str(v) for v in r) + "\n")


def _color(spec):
    if spec.startswith("physfc"): return "#56B4E9"
    if spec.startswith("phys"): return "#0072B2"
    if spec.startswith("poly"): return "#CC79A7"
    if "fw" in spec: return "#009E73"
    if spec.count(":") >= 3 and "lat" not in spec.split(":")[3]: return "#D55E00"
    return "#E69F00"


def _short(spec):
    p = spec.split(":")
    if p[0] == "phys": return "PHYS"
    if p[0] == "physfc": return "PHYS-fc"
    if p[0] == "poly": return f"POLY{p[1] if len(p) > 1 else 2}"
    f = p[3] if len(p) > 3 else "H+slope+lat"
    return f"NN-{p[1] if len(p) > 1 else 16}" + ("" if f == "H+slope+lat" else f"[{f.replace('H+slope', 'Hs')}]")


def _tex_label(spec):
    s = _short(spec)
    return "\\textsc{" + s.replace("[", "}[").replace("]", "]") + ("}" if "[" not in s else "")


def _tex(name, caption, label, header, rows, align=None, wide=True):
    env = "table*" if wide else "table"
    align = align or "l" + "c" * (len(header) - 1)
    L = [f"\\begin{{{env}}}[t]\\centering\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    L += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    L += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    (TABS / f"{name}.tex").write_text("\n".join(L) + "\n")
    print(f"[tab] {name}")


pm = lambda m, s: f"${m:.1f}\\pm{s:.1f}$"


def _map_ax(ax, lat, lon, Z, mask, cmap, vmin, vmax):
    ax.set_facecolor("#d0d0d0")
    im = ax.pcolormesh(lon, lat, np.ma.masked_where(~mask, Z), cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest",
                       rasterized=True)
    ax.set_xlim(0, 360); ax.set_ylim(-80, 80)
    ax.set_xticks(range(0, 361, 90)); ax.set_yticks(range(-60, 61, 30))
    ax.tick_params(length=2)
    for sp_ in ax.spines.values():
        sp_.set_visible(True)
    return im


def _fields(key):
    f = FIELDS / f"{key}.npz"
    return np.load(f) if f.exists() else None


def fig_check():
    C = load("check.json")
    if not C: return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    lab, val, col = [], [], []
    for r in C:
        tag = _short(r["spec_str"]) + ("+E" if r.get("energy_penalty") else "")
        for d in r["grad"]:
            lab.append(f"{tag}: {d['label']}"); val.append(max(d["rel"], 1e-16)); col.append(_color(r["spec_str"]))
    x = np.arange(len(val))
    ax.bar(x, val, color=col, width=0.75)
    ax.axhline(1e-6, ls="--", c="k", lw=0.6)
    ax.set_yscale("log"); ax.set_ylabel("rel. error, adjoint vs. FD")
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=75, ha="right", fontsize=5)
    _save(fig, "fig_gradcheck")
    _dat("gradcheck", ["idx", "rel", "label"], [(i, v, l.replace(" ", "_")) for i, (v, l) in enumerate(zip(val, lab))])
    rows = [[_tex_label(r["spec_str"]) + (" + energy" if r.get("energy_penalty") else ""), r["n_params"],
             f"{r['energy']['M2']['D_TW']:.3f}", f"{abs(r['energy']['M2']['rel_mismatch']):.1e}",
             f"{abs(r['energy']['K1']['rel_mismatch']):.1e}", f"{r['energy']['M2']['solver_resid']:.1e}",
             f"{max(d['rel'] for d in r['grad']):.1e}", f"{r['seconds_loss_and_grad']:.1f}"] for r in C]
    _tex("tab_check", "Verification at initial parameters: discrete energy identity, solver residual and adjoint "
         "gradient check (central FD, $h=10^{-3}$).", "tab:check",
         ["closure", "\\#p", "$D_{M_2}$ [TW]", "$|W-D|/W$ $M_2$", "$K_1$", "$\\|A x-b\\|/\\|b\\|$", "max grad err",
          "loss+grad [s]"], rows, wide=False)


def fig_cv_bars(S, name):
    plt = _plt()
    rows = S["rows"]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for i, r in enumerate(rows):
        ax.bar(i, r["out_mean"], yerr=r["out_sd"], color=_color(r["spec"]), width=0.7, capsize=2, error_kw=dict(lw=0.7))
        ax.plot(i, r["pooled"], "D", color="k", ms=2.5)
    ax.axhline(rows[0]["pooled_gp"], c="#7F7F7F", ls="--", lw=0.8)
    ax.axhline(rows[0]["pooled_nearest"], c="#7F7F7F", ls=":", lw=0.8)
    ax.text(len(rows) - 0.4, rows[0]["pooled_gp"], " GP", va="center", fontsize=6.5, color="#555")
    ax.text(len(rows) - 0.4, rows[0]["pooled_nearest"], " nearest", va="center", fontsize=6.5, color="#555")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([_short(r["spec"]) for r in rows], rotation=35, ha="right")
    ax.set_ylabel("held-out RMS [cm]"); ax.set_xlim(-0.6, len(rows) + 0.6)
    _save(fig, name)
    _dat(name, ["i", "out_mean", "out_sd", "pooled", "label"],
         [(i, r["out_mean"], r["out_sd"], r["pooled"], _short(r["spec"]).replace(" ", "")) for i, r in enumerate(rows)])


def fig_capacity(Sh, Sb):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    for S, ls, tag in ((Sh, "-", "hash"), (Sb, "--", "block")):
        if not S: continue
        R = {r["spec"]: r for r in S["rows"]}
        fam = [R[s] for s in FAMILY if s in R]
        n = [r["n_params"] for r in fam]
        a1.errorbar(n, [r["in_mean"] for r in fam], [r["in_sd"] for r in fam], color="#E69F00", ls=ls, marker="s",
                    mfc="white", ms=3.5, lw=1, capsize=2, label=f"NN in-sample ({tag})")
        a1.errorbar(n, [r["out_mean"] for r in fam], [r["out_sd"] for r in fam], color="#A06A00", ls=ls, marker="o",
                    ms=3.5, lw=1, capsize=2, label=f"NN held-out ({tag})")
        for s in ("phys", "poly:2"):
            if s in R:
                a1.errorbar(R[s]["n_params"], R[s]["out_mean"], R[s]["out_sd"], color=_color(s), marker="o", ms=3.5,
                            capsize=2, ls="none", mfc=_color(s) if tag == "hash" else "white")
                a1.plot(R[s]["n_params"], R[s]["in_mean"], marker="s", color=_color(s), mfc="white", ms=3.5, ls="none")
        a1.axhline(S["rows"][0]["pooled_gp"], color="#7F7F7F", ls=ls, lw=0.7)
        lad = [R[s] for s in ["phys", "poly:2"] + FAMILY if s in R]
        a2.plot([r["n_params"] for r in lad], [r["gap_mean"] for r in lad], ls=ls, marker="o", ms=3, color="#D55E00",
                label=tag)
        _dat(f"capacity_{tag}", ["n", "in", "in_sd", "out", "out_sd", "gap"],
             [(r["n_params"], r["in_mean"], r["in_sd"], r["out_mean"], r["out_sd"], r["gap_mean"]) for r in lad])
    a1.set_xscale("log"); a1.set_ylabel("RMS [cm]"); a1.legend(ncol=2, loc="upper right")
    a2.set_ylabel("gap [cm]"); a2.set_xlabel("trainable parameters"); a2.axhline(0, c="k", lw=0.5); a2.legend()
    _save(fig, "fig_capacity")


def fig_skill_energy(S):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    ax.axvspan(2.3, 2.5, color="#009E73", alpha=0.12, lw=0)
    ax.axvline(2.4, color="#009E73", ls="--", lw=0.8)
    sc = None
    for r in S["rows"]:
        sc = ax.scatter(r["D"], r["out_mean"], c=[100 * r["shallow"]], cmap="viridis", vmin=10, vmax=40,
                        s=28, edgecolor=_color(r["spec"]), linewidth=1.2, zorder=3)
        ax.errorbar(r["D"], r["out_mean"], yerr=r["out_sd"], xerr=r["D_sd"], color="#999", lw=0.6, zorder=2)
        ax.annotate(_short(r["spec"]), (r["D"], r["out_mean"]), xytext=(4, 3), textcoords="offset points", fontsize=6)
    fig.colorbar(sc, ax=ax, label="% of $D_{M_2}$ in $H<500$ m", pad=0.02)
    ax.set_xlabel("$D_{M_2}$ [TW]"); ax.set_ylabel("held-out RMS [cm]")
    _save(fig, "fig_skill_energy")
    _dat("skill_energy", ["D", "D_sd", "out", "out_sd", "shallow", "label"],
         [(r["D"], r["D_sd"], r["out_mean"], r["out_sd"], r["shallow"], _short(r["spec"])) for r in S["rows"]])


def fig_zonal(S):
    plt = _plt()
    lat = np.array(S["lat"])
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for r in S["rows"]:
        if r["spec"] in ("phys", "nn:8", "nn:128", "nn:32:2:H+slope"):
            ax.plot(lat, 1e3 * np.array(r["D_row"]), color=_color(r["spec"]), lw=1,
                    ls="--" if r["spec"] == "nn:128" else "-", label=_short(r["spec"]))
    for c in (74.5, -74.5):
        ax.axvline(c, color="k", ls=":", lw=0.6)
    ax.text(74.5, ax.get_ylim()[1] * 0.9, " $M_2$ crit.", fontsize=6)
    ax.set_xlim(-80, 80); ax.set_xlabel("latitude [°]"); ax.set_ylabel("$D_{M_2}$ per 2° row [GW]"); ax.legend()
    _save(fig, "fig_zonal_dissipation")


def fig_maps(S):
    R = {r["spec"]: r for r in S["rows"]}
    if "phys" not in R or "nn:8" not in R: return
    Fp, Fn = _fields(R["phys"]["keys"][0]), _fields(R["nn:8"]["keys"][0])
    if Fp is None or Fn is None: return
    plt = _plt()
    lat, lon, mask = Fn["lat"], Fn["lon"], Fn["mask"]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 3.6), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.08, hspace=0.18))
    lp, ln = np.log10(Fp["r_eff"] + 1e-12), np.log10(Fn["r_eff"] + 1e-12)
    im = _map_ax(axs[0, 0], lat, lon, lp, mask, "magma", -5, -0.5); axs[0, 0].set_title("(a) $\\log_{10} r_{\\rm eff}$, PHYS", loc="left")
    _map_ax(axs[0, 1], lat, lon, ln, mask, "magma", -5, -0.5); axs[0, 1].set_title("(b) $\\log_{10} r_{\\rm eff}$, NN-8", loc="left")
    fig.colorbar(im, ax=axs[0, :], label="m s$^{-1}$ (log$_{10}$)", pad=0.01, shrink=0.9)
    im2 = _map_ax(axs[1, 0], lat, lon, ln - lp, mask, "RdBu_r", -2, 2)
    axs[1, 0].set_title("(c) NN-8 / PHYS drag ratio (log$_{10}$)", loc="left")
    fig.colorbar(im2, ax=axs[1, 0], pad=0.01, shrink=0.9)
    eta = Fn["eta_M2"]
    amp, G = 100 * np.abs(eta), (-np.degrees(np.angle(eta))) % 360
    im3 = _map_ax(axs[1, 1], lat, lon, amp, mask, "viridis", 0, 120)
    axs[1, 1].contour(lon, lat, np.ma.masked_where(~mask, G), levels=np.arange(30, 360, 30), colors="w", linewidths=0.35)
    axs[1, 1].set_title("(d) NN-8 $M_2$ amplitude [cm], phase every 30°", loc="left")
    fig.colorbar(im3, ax=axs[1, 1], pad=0.01, shrink=0.9)
    _save(fig, "fig_maps")


def fig_gain_map(S):
    R = {r["spec"]: r for r in S["rows"]}
    if "phys" not in R or "nn:8" not in R: return
    F = _fields(R["nn:8"]["keys"][0])
    pa, pb = R["nn:8"]["pooled_arrays"], R["phys"]["pooled_arrays"]
    if F is None or pa["test_ids"] != pb["test_ids"]: return
    plt = _plt()
    d = 100 * (np.sqrt(np.array(pa["e_model"]) / 2) - np.sqrt(np.array(pb["e_model"]) / 2))
    fig, ax = plt.subplots(figsize=(W2 * 0.6, 2.2))
    ax.set_facecolor("white")
    ax.contourf(F["lon"], F["lat"], (~F["mask"]).astype(float), levels=[0.5, 1.5], colors=["#d0d0d0"])
    v = np.percentile(np.abs(d), 95)
    sc = ax.scatter(pa["test_lon"], pa["test_lat"], c=d, cmap="RdBu_r", vmin=-v, vmax=v, s=10, edgecolor="k", linewidth=0.2)
    fig.colorbar(sc, ax=ax, label="held-out error NN-8 − PHYS [cm]", pad=0.01)
    ax.set_xlim(0, 360); ax.set_ylim(-80, 80); ax.set_xlabel("longitude [°E]"); ax.set_ylabel("latitude [°]")
    _save(fig, "fig_gain_map")


def fig_convergence(S):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for r in S["rows"]:
        h = np.minimum.accumulate(np.array(r["history0"]))
        ax.plot(np.arange(1, len(h) + 1), h / h[0], color=_color(r["spec"]), lw=0.9, label=_short(r["spec"]))
    ax.set_yscale("log"); ax.set_xlabel("objective evaluations"); ax.set_ylabel("best loss / initial"); ax.legend(ncol=2)
    _save(fig, "fig_convergence")


def fig_block(Sh, Sb):
    if not (Sh and Sb): return
    plt = _plt()
    H, B = {r["spec"]: r for r in Sh["rows"]}, {r["spec"]: r for r in Sb["rows"]}
    specs = [s for s in CORE if s in H and s in B]
    x = np.arange(len(specs))
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    ax.bar(x - 0.18, [H[s]["out_mean"] for s in specs], 0.34, yerr=[H[s]["out_sd"] for s in specs],
           color=[_color(s) for s in specs], capsize=2, error_kw=dict(lw=0.6), label="hash folds")
    ax.bar(x + 0.18, [B[s]["out_mean"] for s in specs], 0.34, yerr=[B[s]["out_sd"] for s in specs],
           color=[_color(s) for s in specs], hatch="////", edgecolor="w", capsize=2, error_kw=dict(lw=0.6),
           label="spatial blocks")
    ax.axhline(Sh["rows"][0]["pooled_gp"], color="#7F7F7F", ls="--", lw=0.7)
    ax.axhline(Sb["rows"][0]["pooled_gp"], color="#7F7F7F", ls=":", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([_short(s) for s in specs], rotation=30, ha="right")
    ax.set_ylabel("held-out RMS [cm]"); ax.legend(loc="upper left")
    _save(fig, "fig_hash_vs_block")


def fig_block_map(Sb):
    if not Sb or "block_of" not in Sb: return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(W1, 1.8))
    ids_block = list(Sb["block_of"].values())
    ax.scatter(Sb["gauge_lon"], Sb["gauge_lat"], c=ids_block, cmap="tab10", vmin=0, vmax=9, s=6)
    ax.set_xlim(0, 360); ax.set_ylim(-80, 80); ax.set_xlabel("longitude [°E]"); ax.set_ylabel("latitude [°]")
    ax.set_title("spatial CV blocks (k-means on the sphere)", loc="left")
    _save(fig, "fig_blocks")


def fig_reg(Rg):
    if not Rg: return
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 1.8))
    for s in FAMILY:
        rows = sorted([r for r in Rg["rows"] if r["spec"] == s], key=lambda r: r["lam"])
        if not rows: continue
        lam = [r["lam"] for r in rows]
        c = {"nn:8": "#E69F00", "nn:32": "#A06A00", "nn:128": "#5A3C00"}[s]
        a1.errorbar(lam, [r["out_mean"] for r in rows], [r["out_sd"] for r in rows], color=c, marker="o", ms=3,
                    capsize=2, label=_short(s))
        a2.plot(lam, [r["gap_mean"] for r in rows], color=c, marker="o", ms=3)
    for a_ in (a1, a2):
        a_.set_xscale("log"); a_.set_xlabel("$\\lambda$")
    a1.set_ylabel("held-out RMS [cm]"); a2.set_ylabel("gap [cm]"); a1.legend()
    fig.tight_layout()
    _save(fig, "fig_reg")
    _tex("tab_reg", "Weight-decay sweep (hash CV, mean $\\pm$ s.d. over folds, cm).", "tab:reg",
         ["closure", "$\\lambda$", "in-sample", "held-out", "gap", "$D_{M_2}$ [TW]"],
         [[_tex_label(r["spec"]), f"{r['lam']:.0e}", pm(r["in_mean"], r["in_sd"]), pm(r["out_mean"], r["out_sd"]),
           f"{r['gap_mean']:+.1f}", f"{r['D']:.2f}"] for r in Rg["rows"]], wide=False)


def fig_transfer(Tr):
    if not Tr: return
    plt = _plt()
    rows = Tr["rows"]
    names = list(rows[0]["out_row"])
    x, wbar = np.arange(len(names)), 0.8 / (len(rows) + 1)
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 2.1))
    for k, r in enumerate(rows):
        ax.bar(x + (k - len(rows) / 2) * wbar, [r["out_row"][n] for n in names], wbar,
               yerr=[r["out_row_sd"][n] for n in names], color=_color(r["spec"]), capsize=1.5,
               error_kw=dict(lw=0.5), label=_short(r["spec"]))
    ax.bar(x + (len(rows) / 2) * wbar, [rows[0]["gp_row"][n] for n in names], wbar, color="#BBBBBB", label="GP (data)")
    ax.set_xticks(x); ax.set_xticklabels([n + (" (fitted)" if n == "M2" else " (no refit)") for n in names])
    ax.set_ylabel("held-out RMS [cm]"); ax.legend(ncol=3, loc="upper right")
    _save(fig, "fig_transfer")
    _tex("tab_transfer", "Frequency transfer: closures fitted to $M_2$ only, scored on held-out gauges for each "
         "constituent without refitting (cm, mean $\\pm$ s.d. over folds).", "tab:transfer",
         ["closure", "\\#p"] + [f"${n[0]}_{n[1]}$" for n in names] + ["$D_{M_2}$ [TW]"],
         [[_tex_label(r["spec"]), r["n_params"]] + [pm(r["out_row"][n], r["out_row_sd"][n]) for n in names]
          + [f"{r['D']:.2f}"] for r in rows] + [["GP (data-only)", "--"] + [f"{rows[0]['gp_row'][n]:.1f}" for n in names] + ["--"]])


def fig_energy(E):
    if not E: return
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 1.9))
    for s in ("phys", "nn:8"):
        rows = sorted([r for r in E["rows"] if r["spec"] == s], key=lambda r: r["mu"])
        if not rows: continue
        a1.plot([abs(r["D"] / 2.4 - 1) for r in rows], [r["out_mean"] for r in rows], marker="o", ms=3,
                color=_color(s), label=_short(s))
        for r in rows:
            a1.annotate(f"{r['mu']:g}", (abs(r["D"] / 2.4 - 1), r["out_mean"]), xytext=(2, 2), textcoords="offset points", fontsize=5)
        a2.plot([max(r["mu"], 0.01) for r in rows], [100 * r["shallow"] for r in rows], marker="o", ms=3, color=_color(s))
    a2.axhline(66.7, color="#009E73", ls="--", lw=0.7)
    a1.set_xlabel("$|D_{M_2}/2.4\\,\\mathrm{TW}-1|$"); a1.set_ylabel("held-out RMS [cm]"); a1.legend()
    a2.set_xscale("log"); a2.set_xlabel("$\\mu$ (0 plotted at 0.01)"); a2.set_ylabel("% in $H<500$ m")
    fig.tight_layout()
    _save(fig, "fig_energy_pareto")
    _tex("tab_energy", "Energy-constrained fits: penalty $\\mu[(D_{M_2}/2.4-1)^2+(f_{\\rm sh}-2/3)^2]$.", "tab:energy",
         ["closure", "$\\mu$", "held-out", "$D_{M_2}$ [TW]", "shelf [\\%]"],
         [[_tex_label(r["spec"]), f"{r['mu']:g}", pm(r["out_mean"], r["out_sd"]), f"{r['D']:.2f}",
           f"{100 * r['shallow']:.0f}"] for r in E["rows"]], wide=False)


def fig_distill(Dd):
    if not Dd: return
    plt = _plt()
    pd = Dd["pd"]
    s = np.array(pd["s"])
    latd = np.degrees(np.arcsin(s))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.9), gridspec_kw=dict(wspace=0.35))
    for k, (curves, ttl) in enumerate(((pd["lat_smooth"], "smooth deep floor"), (pd["lat_rough"], "rough deep floor"))):
        Cv = np.array(curves)
        for c in Cv:
            axs[k].plot(latd, 1e3 * c, color="#E69F00", lw=0.6, alpha=0.6)
        axs[k].plot(latd, 1e3 * Cv.mean(0), color="#A06A00", lw=1.4)
        for v in (74.5, -74.5):
            axs[k].axvline(v, color="k", ls=":", lw=0.6)
        for v in (30, -30):
            axs[k].axvline(v, color="#009E73", ls=":", lw=0.6)
        axs[k].set_xlabel("latitude [°]"); axs[k].set_ylabel("$r_{\\rm eff}$ [mm/s]")
        axs[k].set_title(f"({'ab'[k]}) NN-8, H=4 km, {ttl}", loc="left", fontsize=7)
    Hc = np.array(pd["H_curve"])
    for c in Hc:
        axs[2].plot(pd["H"], 1e3 * c, color="#E69F00", lw=0.6, alpha=0.6)
    axs[2].plot(pd["H"], 1e3 * Hc.mean(0), color="#A06A00", lw=1.4)
    axs[2].set_xscale("log"); axs[2].set_yscale("log"); axs[2].set_xlabel("H [m]"); axs[2].set_ylabel("$r_{\\rm eff}$ [mm/s]")
    axs[2].set_title("(c) depth dependence, 30°", loc="left", fontsize=7)
    _save(fig, "fig_partial_dependence")
    _tex("tab_distill", "Distillation of NN-8 into polynomial drag laws $h(\\phi)$ (no refit): area-weighted $R^2$ "
         "of $h$ and held-out RMS (cm).", "tab:distill",
         ["law", "\\#p", "$R^2$", "held-out", "NN-8 held-out", "$D_{M_2}$ [TW]", "shelf [\\%]"],
         [[f"degree {r['deg']}", r["n_params"], f"{r['R2']:.3f}", pm(r["out_mean"], r["out_sd"]),
           f"{r['nn_out_mean']:.1f}", f"{r['D']:.2f}", f"{100 * r['shallow']:.0f}"] for r in Dd["rows"]], wide=False)


def fig_slr():
    plt = _plt()
    fs = [(s, FIELDS / f"slr_{s.replace(':', '_')}.npz") for s in ("phys", "nn:8")]
    fs = [(s, f) for s, f in fs if f.exists()]
    if not fs: return
    fig, axs = plt.subplots(1, len(fs), figsize=(W2, 1.9), sharey=True, gridspec_kw=dict(wspace=0.05))
    axs = np.atleast_1d(axs)
    for ax, (s, f) in zip(axs, fs):
        d = np.load(f)
        Z = 100 * d["dR_mean"]
        v = np.percentile(np.abs(Z[d["mask"]]), 98)
        im = _map_ax(ax, d["lat"], d["lon"], Z, d["mask"], "RdBu_r", -v, v)
        low = np.ma.masked_where(~(d["mask"] & (d["agree"] < 1.0)), np.ones_like(Z))
        ax.contourf(d["lon"], d["lat"], low, levels=[0.5, 1.5], colors="none", hatches=["...."])
        ax.set_title(f"{_short(s)}: Δ spring range per m SLR [cm]; dots: folds disagree in sign", loc="left", fontsize=6.5)
        fig.colorbar(im, ax=ax, pad=0.01, shrink=0.85)
    _save(fig, "fig_slr")


def fig_res(Rs):
    if not Rs: return
    plt = _plt()
    rows = Rs["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 1.8))
    for k, s in enumerate(("phys", "nn:8")):
        rr = sorted([r for r in rows if r["spec"] == s], key=lambda r: -r["res"])
        lab = [f"{r['res']:g}°" for r in rr]
        x = np.arange(len(rr)) + 0.35 * (k - 0.5)
        a1.bar(x, [r["out_mean"] for r in rr], 0.33, yerr=[r["out_sd"] for r in rr], color=_color(s), capsize=2, label=_short(s))
        a2.bar(x, [100 * r["shallow"] for r in rr], 0.33, color=_color(s))
        a1.set_xticks(np.arange(len(rr))); a1.set_xticklabels(lab); a2.set_xticks(np.arange(len(rr))); a2.set_xticklabels(lab)
    a2.axhline(66.7, color="#009E73", ls="--", lw=0.7)
    a1.set_ylabel("held-out RMS [cm]"); a2.set_ylabel("% $D_{M_2}$ in $H<500$ m"); a1.legend()
    fig.tight_layout()
    _save(fig, "fig_resolution")


def tab_cv(S, name, caption, label):
    rows = []
    for r in S["rows"]:
        vr = r.get("vs_ref")
        rows.append([_tex_label(r["spec"]), f"{r['n_params']:,}".replace(",", "\\,"), pm(r["in_mean"], r["in_sd"]),
                     pm(r["out_mean"], r["out_sd"]), f"{r['gap_mean']:+.1f}", f"{r['pooled']:.1f}",
                     f"${vr['diff']:+.1f}\\ [{vr['lo']:+.1f},{vr['hi']:+.1f}]$" if vr else "--",
                     f"{r['wins_vs_ref']}/{r['n_folds']}" if vr else "--", f"{r['vs_gp']['p']:.3f}",
                     f"{r['D']:.2f}", f"{100 * r['shallow']:.0f}"])
    r0 = S["rows"][0]
    cap = (caption + f" GP {pm(r0['gp_mean'], r0['gp_sd'])}, nearest {pm(r0['nearest_mean'], r0['nearest_sd'])}, "
           f"equilibrium {r0['pooled_eq']:.1f} (pooled). Kendall $\\tau$ (in-sample vs held-out, NN widths) = "
           f"{S['kendall_in_vs_out'] if S['kendall_in_vs_out'] is not None else 'n/a'}.")
    _tex(name, cap, label, ["closure", "\\#p", "in-sample", "held-out", "gap", "pooled", "$\\Delta$ vs PHYS [95\\% CI]",
                            "wins", "$P(\\ge$GP)", "$D_{M_2}$", "shelf\\%"], rows)
    names = list(S["rows"][0]["out_row"])
    _tex(name + "_perconst", caption.split(".")[0] + ": held-out RMS per constituent (cm).", label + "pc",
         ["closure"] + [f"${n[0]}_{n[1]}$" for n in names],
         [[_tex_label(r["spec"])] + [pm(r["out_row"][n], r["out_row_sd"][n]) for n in names] for r in S["rows"]]
         + [["GP"] + [f"{S['rows'][0]['gp_row'][n]:.1f}" for n in names]], wide=False)


def exp_figures(a):
    Sh, Sb = load("summary_cv_hash.json"), load("summary_cv_block.json")
    jobs = [fig_check]
    if Sh:
        jobs += [lambda: fig_cv_bars(Sh, "fig_heldout"), lambda: fig_skill_energy(Sh), lambda: fig_zonal(Sh),
                 lambda: fig_maps(Sh), lambda: fig_gain_map(Sh), lambda: fig_convergence(Sh),
                 lambda: tab_cv(Sh, "tab_cv", "Five-fold hash CV (mean $\\pm$ s.d. over folds, cm; RMS is the "
                                "root-sum-square over the fitted constituents).", "tab:cv")]
    if Sh or Sb:
        jobs += [lambda: fig_capacity(Sh, Sb)]
    if Sb:
        jobs += [lambda: fig_block(Sh, Sb), lambda: fig_block_map(Sb),
                 lambda: tab_cv(Sb, "tab_block", "Spatial-block CV (k-means blocks).", "tab:block")]
    jobs += [lambda: fig_reg(load("summary_reg.json")), lambda: fig_transfer(load("summary_transfer.json")),
             lambda: fig_energy(load("summary_energy.json")), lambda: fig_distill(load("summary_distill.json")),
             fig_slr, lambda: fig_res(load("summary_res.json"))]
    for j in jobs:
        try:
            j()
        except Exception as e:                           # one broken figure must not stop the rest
            print(f"[fig] skipped: {type(e).__name__}: {e}")


EXPS = dict(check=exp_check, cv=exp_cv, block=exp_block, reg=exp_reg, transfer=exp_transfer, energy=exp_energy,
            distill=exp_distill, slr=exp_slr, res=exp_res, figures=exp_figures)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=["all"] + list(EXPS))
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--res-fine", type=float, default=1.0)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--max-gauges", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--constituents", default="M2,S2,K1,O1")
    ap.add_argument("--l2-nn", type=float, default=1e-6, help="weight decay for NN/POLY in the main table")
    ap.add_argument("--l2-phys", type=float, default=1e-3, help="prior weight for PHYS log-parameters")
    ap.add_argument("--buffer-km", type=float, default=0.0, help="spatial-block CV buffer")
    ap.add_argument("--slr", type=float, default=1.0, help="uniform sea-level rise for E7 [m]")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 folds, <=8 L-BFGS iterations")
    ap.add_argument("--skip-res", action="store_true", help="skip the expensive 1° run in 'all'")
    a = ap.parse_args()
    if a.quick:
        a.iters = min(a.iters, 8)
    for d in (T.CACHE, T.RESULTS, RUNS, FIELDS, FIGS, TABS):
        d.mkdir(parents=True, exist_ok=True)
    todo = list(EXPS) if a.exp == "all" else [a.exp]
    for e in todo:
        if e == "res" and a.exp == "all" and a.skip_res:
            continue
        t0 = time.time()
        EXPS[e](a)
        print(f"[{e}] finished in {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    main()