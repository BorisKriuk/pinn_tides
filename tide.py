#!/usr/bin/env python3
"""
tide.py — differentiable global barotropic tide model with hard-constrained, learnable dissipation
          closures. Library for experiments.py plus a small CLI.

Physics: frequency-domain linearised Laplace tidal equations on a lat/lon Arakawa C-grid
    i om u - f v + kappa u = -g dx Phi,   i om v + f u + kappa v = -g dy Phi,   i om eta + div(H u) = 0,
    Phi = (1 - beta) eta - eta_eq,   eta_eq = gamma_k A_k P_m(lat) e^{i m lon},   kappa = r_eff / H [1/s].
  The coupled (u, v, eta) system A(theta) x = b is solved by sparse LU. Coriolis uses shared corner weights
  and grad/div are summation-by-parts, so  work input == dissipation  holds to round-off for every theta,
  and r_eff > 0 makes A(theta) non-singular for every theta.
Learning: only the drag field r_eff(x; theta) > 0 and the scalar SAL factor beta are learned. Gradients come
  from the implicit-function adjoint, which reuses the forward LU (one back-substitution per constituent,
  independent of dim theta).

Closures (spec strings)
  phys                PHYS(4p):  kappa = r0/H + r_sh/H e^{-H/200} + c_it |grad H|^2 sig((H-500)/100) + kappa_min
  physfc              PHYS-fc(4p): IW term x sqrt(max(0, 1-(f/om)^2))   (critical-latitude cut-off)
  nn[:W[:D[:feats]]]  MLP:  r = 1e-3 exp(6 tanh(h/6)) m/s,  h = MLP(features),  tanh hidden layers
  poly[:deg[:feats]]  polynomial h(features); also used to distil a trained NN into a closed-form law
  features: H = 0.5 ln(H/1000 m), slope = ln(1+|grad H|/0.005), lat = sin(lat), abslat = |sin(lat)|,
            fw = min(|f|/om, 3)   (frequency dependent)
Loss: density-weighted complex misfit summed over fitted constituents + l2 * penalty (+ optional energy
  penalty  mu_D (D_M2/2.4TW - 1)^2 + mu_sh (shallow fraction - 2/3)^2 ).
Validation: hash folds (random w.r.t. space) or spatial blocks (k-means on the sphere, optional buffer),
  data-only GP / nearest-gauge / equilibrium baselines, paired bootstrap.

    python tide.py check [--synthetic] [--closure nn:8] [--energy-mu 0.1]
    python tide.py fit   --closure nn:8 [--fold 0] [--scheme hash|block]
    python tide.py eval
    python tide.py predict --lat 21.3 --lon 202.1 --start 2024-06-01 --hours 240
    python tide.py split [--scheme block]
  Every experiment in the paper:  python experiments.py all

Deps: numpy scipy torch requests  (+ matplotlib for experiments.py)
"""
import argparse, csv, hashlib, io, itertools, json, math, sys, time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import requests

torch.set_default_dtype(torch.float64)
CPLX = torch.complex128
ROOT = Path(__file__).resolve().parent
CACHE, RESULTS = ROOT / "cache", ROOT / "results"

# ----------------------------------------------------------------------------- constants
R_E, G0, OMEGA_E, RHO = 6.371e6, 9.81, 7.292115e-5, 1035.0
ALPHA_LOVE = 0.693                                              # 1 + k2 - h2 (semidiurnal)
GAMMA_DIURNAL = {"K1": 0.736, "O1": 0.695, "P1": 0.706, "Q1": 0.695}   # FCN-affected diurnal 1 + k - h
KAPPA_MIN = 1e-6              # s^-1 background damping
LAT_MAX = 80.0                # rows poleward of this are closed
H_MIN = 10.0                  # m, shallower cells are land
D_OBS_M2, SHALLOW_OBS = 2.4e12, 2.0 / 3.0                     # Egbert & Ray (2000, 2001)
R_REF, R_SPAN = 1e-3, 6.0     # learned drag velocity r = R_REF exp(R_SPAN tanh(h/R_SPAN)) in [2.5e-6, 0.40] m/s
FEATS = ("H", "slope", "lat", "abslat", "fw")
UHSLC = "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_hourly_fast.csv"
ETOPO = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.csv"

CONST = {
    "M2": ((2, 0, 0, 0, 0, 0), 0, 0.242334, 2),
    "S2": ((2, 2, -2, 0, 0, 0), 0, 0.112841, 2),
    "N2": ((2, -1, 0, 1, 0, 0), 0, 0.046398, 2),
    "K2": ((2, 2, 0, 0, 0, 0), 0, 0.030704, 2),
    "K1": ((1, 1, 0, 0, 0, 0), -90, 0.141565, 1),
    "O1": ((1, -1, 0, 0, 0, 0), 90, 0.100514, 1),
    "P1": ((1, 1, -2, 0, 0, 0), 90, 0.046843, 1),
    "Q1": ((1, -2, 0, 1, 0, 0), 90, 0.019256, 1),
    "2N2": ((2, -2, 0, 2, 0, 0), 0, 0, 2), "MU2": ((2, -2, 2, 0, 0, 0), 0, 0, 2),
    "NU2": ((2, -1, 2, -1, 0, 0), 0, 0, 2), "L2": ((2, 1, 0, -1, 0, 0), 180, 0, 2),
    "T2": ((2, 2, -3, 0, 0, 1), 0, 0, 2), "J1": ((1, 2, 0, -1, 0, 0), -90, 0, 1),
    "OO1": ((1, 3, 0, 0, 0, 0), -90, 0, 1), "M4": ((4, 0, 0, 0, 0, 0), 0, 0, 4),
    "MS4": ((4, 2, -2, 0, 0, 0), 0, 0, 4), "Mf": ((0, 2, 0, 0, 0, 0), 0, 0, 0),
    "Mm": ((0, 1, 0, -1, 0, 0), 0, 0, 0), "Ssa": ((0, 0, 2, 0, 0, 0), 0, 0, 0),
}
MODEL_SET = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
RATES = np.array([14.4920521, 0.5490165, 0.0410686, 0.0046418, -0.0022064, 0.0000020])  # deg/hr


def omega(name):
    return math.radians(float(np.dot(CONST[name][0], RATES))) / 3600.0


def love_factor(name):
    return GAMMA_DIURNAL.get(name, ALPHA_LOVE)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if torch.is_tensor(x):
        return _jsonable(x.detach().numpy())
    return x


# ----------------------------------------------------------------------------- astronomy
def astro(t):
    """Mean longitudes (deg) at times t (datetime64). Returns (..., 6): tau, s, h, p, N, p'."""
    jd = (t - np.datetime64("2000-01-01T12:00:00")) / np.timedelta64(1, "s") / 86400.0
    T = jd / 36525.0
    s = 218.3164477 + 481267.88123421 * T
    h = 280.46646 + 36000.76983 * T
    p = 83.3532465 + 4069.0137287 * T
    N = 125.04452 - 1934.136261 * T
    pp = 282.93735 + 1.71946 * T
    tau = (jd % 1.0) * 360.0 - s + h
    return np.stack([tau, s, h, p, N, pp], -1) % 360.0


def V_deg(name, A):
    d, ph = CONST[name][0], CONST[name][1]
    return (A @ np.array(d, float) + ph) % 360.0


def nodal(name, N_deg):
    """Nodal factor f and angle u (deg) — Schureman/Pugh formulas."""
    N = np.radians(N_deg)
    c1, c2, c3 = np.cos(N), np.cos(2 * N), np.cos(3 * N)
    s1, s2, s3 = np.sin(N), np.sin(2 * N), np.sin(3 * N)
    if name in ("M2", "N2", "2N2", "MU2", "NU2", "L2"):
        f, u = 1.0004 - 0.0373 * c1 + 0.0002 * c2, -2.14 * s1
    elif name in ("M4", "MS4"):
        k = 2 if name == "M4" else 1
        f, u = (1.0004 - 0.0373 * c1 + 0.0002 * c2) ** k, -2.14 * s1 * k
    elif name == "K1":
        f, u = 1.0060 + 0.1150 * c1 - 0.0088 * c2 + 0.0006 * c3, -8.86 * s1 + 0.68 * s2 - 0.07 * s3
    elif name in ("O1", "Q1"):
        f, u = 1.0089 + 0.1871 * c1 - 0.0147 * c2 + 0.0014 * c3, 10.80 * s1 - 1.34 * s2 + 0.19 * s3
    elif name == "K2":
        f, u = 1.0241 + 0.2863 * c1 + 0.0083 * c2 - 0.0015 * c3, -17.74 * s1 + 0.68 * s2 - 0.04 * s3
    elif name == "J1":
        f, u = 1.0129 + 0.1676 * c1 - 0.0170 * c2 + 0.0016 * c3, -12.94 * s1 + 1.34 * s2 - 0.19 * s3
    elif name == "OO1":
        f, u = 1.1027 + 0.6504 * c1 + 0.0317 * c2 - 0.0014 * c3, -36.68 * s1 + 4.02 * s2 - 0.57 * s3
    elif name == "Mf":
        f, u = 1.0429 + 0.4135 * c1 - 0.0040 * c2, -23.74 * s1 + 2.68 * s2 - 0.38 * s3
    elif name == "Mm":
        f, u = 1.0000 - 0.1300 * c1 + 0.0013 * c2, 0.0 * s1
    else:
        f, u = np.ones_like(N), np.zeros_like(N)
    return f, u


def harmonic_analysis(t, eta, names):
    """Least-squares tidal analysis: eta ≈ Σ f|Z|cos(V+u-G), Z=|Z|e^{-iG}. Returns ({name: Z}, residual std)."""
    ok = np.isfinite(eta)
    t, eta = t[ok], eta[ok]
    A = astro(t)
    cols = [np.ones(len(t)), (t - t[0]) / np.timedelta64(365, "D")]
    for n in names:
        f, u = nodal(n, A[:, 4])
        arg = np.radians(V_deg(n, A) + u)
        cols += [f * np.cos(arg), f * np.sin(arg)]
    X = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(X, eta, rcond=None)
    resid = eta - X @ coef
    Z = {n: coef[2 + 2 * k] - 1j * coef[3 + 2 * k] for k, n in enumerate(names)}
    return Z, float(np.std(resid))


def predict_series(Z, t):
    A = astro(t)
    out = np.zeros(len(t))
    for n, z in Z.items():
        f, u = nodal(n, A[:, 4])
        out += f * np.real(z * np.exp(1j * np.radians(V_deg(n, A) + u)))
    return out


# ----------------------------------------------------------------------------- data fetch
def http_text(url, tries=3):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(3 * (k + 1))
    return None


def load_bathymetry(res):
    """ETOPO1 block-averaged to res-degree grid. Returns (lat centres, lon centres 0..360, depth H>0 [m])."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"etopo_H_{res:g}deg.npz"
    if f.exists():
        d = np.load(f)
        return d["lat"], d["lon"], d["H"]
    raw = CACHE / "etopo_0.5deg.csv"
    if not raw.exists():
        print("[etopo] downloading 0.5° ETOPO1 (~3 MB) ...")
        txt = http_text(f"{ETOPO}?altitude[(-90):30:(90)][(-180):30:(180)]")
        if txt is None:
            raise RuntimeError("ETOPO download failed; use --synthetic or place cache/etopo_0.5deg.csv")
        raw.write_text(txt)
    arr = np.loadtxt(io.StringIO(raw.read_text()), delimiter=",", skiprows=2)
    lat_p, lon_p, z = arr[:, 0], arr[:, 1] % 360.0, arr[:, 2]
    nlat, nlon = int(round(180 / res)), int(round(360 / res))
    j = np.clip(((lat_p + 90) / res).astype(int), 0, nlat - 1)
    i = np.clip((lon_p / res).astype(int), 0, nlon - 1)
    ssum, cnt = np.zeros((nlat, nlon)), np.zeros((nlat, nlon))
    np.add.at(ssum, (j, i), z)
    np.add.at(cnt, (j, i), 1)
    H = np.maximum(-ssum / np.maximum(cnt, 1), 0.0)
    lat = -90 + res * (np.arange(nlat) + 0.5)
    lon = res * (np.arange(nlon) + 0.5)
    np.savez(f, lat=lat, lon=lon, H=H)
    return lat, lon, H


def synthetic_bathymetry(res):
    nlat, nlon = int(round(180 / res)), int(round(360 / res))
    lat = -90 + res * (np.arange(nlat) + 0.5)
    lon = res * (np.arange(nlon) + 0.5)
    H = np.full((nlat, nlon), 4000.0)
    LON, LAT = np.meshgrid(lon, lat)
    land = (np.abs(LON - 280) < 12) & (LAT > -55) & (LAT < 70)
    shelf = (np.abs(LON - 280) < 20) & ~land & (LAT > -55) & (LAT < 70)
    H[shelf] = 100.0
    H[land] = 0.0
    return lat, lon, H


def uhslc_stations():
    f = CACHE / "uhslc_stations.json"
    if f.exists():
        return json.loads(f.read_text())
    txt = http_text(f"{UHSLC}?uhslc_id,station_name,latitude,longitude&distinct()")
    if txt is None:
        raise RuntimeError("UHSLC station list download failed")
    rows = list(csv.reader(io.StringIO(txt)))[2:]
    st = [dict(id=int(r[0]), name=r[1], lat=float(r[2]), lon=float(r[3]) % 360) for r in rows if r and r[0]]
    f.write_text(json.dumps(st))
    return st


def uhslc_hourly(sid, year):
    f = CACHE / f"uhslc_{sid}_{year}.npz"
    if f.exists():
        d = np.load(f)
        return d["t"], d["eta"]
    q = f"time,sea_level&uhslc_id={sid}&time>={year}-01-01T00:00:00Z&time<{year + 1}-01-01T00:00:00Z"
    txt = http_text(f"{UHSLC}?{quote(q, safe='=&,:')}")
    if txt is None:
        np.savez(f, t=np.array([], dtype="datetime64[s]"), eta=np.array([]))
        return np.array([], dtype="datetime64[s]"), np.array([])
    ts, vs = [], []
    for line in txt.strip().splitlines()[2:]:
        a, b = line.split(",")[:2]
        ts.append(a.rstrip("Z"))
        vs.append(float(b) if b not in ("", "NaN") else np.nan)
    t = np.array(ts, dtype="datetime64[s]")
    eta = np.array(vs) / 1000.0
    t, idx = np.unique(t, return_index=True)
    eta = eta[idx]
    np.savez(f, t=t, eta=eta)
    return t, eta


# ----------------------------------------------------------------------------- mesh
def _north(x):
    return torch.cat([x[1:], torch.zeros_like(x[:1])], 0)


def _south(x):
    return torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)


def _T(a):
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64))


class Mesh:
    """Arakawa C-grid. eta at cell centres (j,i); u on the east face, v on the north face of (j,i).
    W[j,i] = f H A at the NE corner of (j,i): shared Coriolis weight -> Coriolis does no discrete work."""

    def __init__(self, lat, lon, H):
        self.res = float(lat[1] - lat[0])
        self.nlat, self.nlon = nlat, nlon = len(lat), len(lon)
        self.lat_np, self.lon_np = lat, lon
        phi, lam = np.radians(lat), np.radians(lon)
        dphi = dlam = np.radians(self.res)
        north = lambda a: np.vstack([a[1:], np.zeros((1, nlon))])
        south = lambda a: np.vstack([np.zeros((1, nlon)), a[:-1]])
        maskb = (H > H_MIN) & (np.abs(lat)[:, None] <= LAT_MAX)
        Hm = np.where(maskb, np.maximum(H, H_MIN), 0.0)
        mk = maskb.astype(float)
        self.maskb, self.mask_np, self.H_np = maskb, mk, Hm
        self.open_u_np = mk * np.roll(mk, -1, 1)
        self.open_v_np = mk * north(mk)
        self.H_u_np = 0.5 * (Hm + np.roll(Hm, -1, 1)) * self.open_u_np
        self.H_v_np = 0.5 * (Hm + north(Hm)) * self.open_v_np
        self.dx_np = R_E * np.cos(phi) * dlam
        self.dy = R_E * dphi
        self.len_v_np = R_E * np.cos(phi + dphi / 2) * dlam
        self.area_np = R_E ** 2 * np.cos(phi) * dlam * dphi
        self.area_v_np = R_E ** 2 * np.cos(phi + dphi / 2) * dlam * dphi
        f_z = 2 * OMEGA_E * np.sin(phi + dphi / 2)
        wet = np.stack([mk, np.roll(mk, -1, 1), north(mk), north(np.roll(mk, -1, 1))])
        dep = np.stack([Hm, np.roll(Hm, -1, 1), north(Hm), north(np.roll(Hm, -1, 1))])
        H_z = dep.sum(0) / np.maximum(wet.sum(0), 1)
        self.W_np = f_z[:, None] * H_z * self.area_v_np[:, None]
        Hx = (np.roll(Hm, -1, 1) - np.roll(Hm, 1, 1)) / (2 * self.dx_np[:, None])
        Hy = (north(Hm) - south(Hm)) / (2 * self.dy)
        self.slope_np = np.sqrt(Hx ** 2 + Hy ** 2) * mk
        for n in ("mask", "H", "open_u", "open_v", "H_u", "H_v", "dx", "len_v", "area", "area_v", "W", "slope"):
            setattr(self, n, _T(getattr(self, n + "_np")))
        self.phi, self.lam = _T(phi), _T(lam)
        self.fcor = _T(2 * OMEGA_E * np.sin(phi))
        self.cache = {}

        def number(open_):
            j, i = np.nonzero(open_ > 0)
            k = -np.ones((nlat, nlon), int)
            k[j, i] = np.arange(len(j))
            return k, (torch.as_tensor(j), torch.as_tensor(i)), len(j)

        self.ku, self.iu, self.Nu = number(self.open_u_np)
        self.kv, self.iv, self.Nv = number(self.open_v_np)
        self.k, self.ie, self.N = number(mk)
        self.last_resid = float("nan")

    def pack(self, u, v, e):
        return torch.cat([u[self.iu], v[self.iv], e[self.ie]])

    def unpack(self, x):
        z = lambda: torch.zeros((self.nlat, self.nlon), dtype=CPLX)
        return (z().index_put(self.iu, x[:self.Nu]),
                z().index_put(self.iv, x[self.Nu:self.Nu + self.Nv]),
                z().index_put(self.ie, x[self.Nu + self.Nv:]))

    def equilibrium(self, name):
        key = ("eq", name)
        if key not in self.cache:
            d, ph, A, spc = CONST[name]
            P = torch.cos(self.phi) ** 2 if spc == 2 else torch.sin(2 * self.phi)
            self.cache[key] = (love_factor(name) * A * P[:, None] * torch.exp(1j * spc * self.lam)[None, :]
                               * self.mask).to(CPLX)
        return self.cache[key]

    def snap(self, lat, lon):
        """Nearest ocean cell within one cell of (lat, lon), else None (this is the gauge sampling operator P)."""
        j0 = int(np.clip((lat + 90) / self.res, 0, self.nlat - 1))
        i0 = int(lon / self.res) % self.nlon
        best = None
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                j, i = j0 + dj, (i0 + di) % self.nlon
                if 0 <= j < self.nlat and self.maskb[j, i]:
                    d = gc_km(lat, lon, self.lat_np[j], self.lon_np[i])
                    if best is None or d < best[0]:
                        best = (d, j, i)
        return None if best is None else (best[1], best[2])


def gc_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_E / 1000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ----------------------------------------------------------------------------- closures (the learned part)
THETA0 = np.array([math.log(1e-3), math.log(0.1), 0.0, math.log(1e-3)])
PARAM_NAMES = ["log r0", "log c_it", "logit(beta/0.2)", "log r_shelf"]


def r_from_h(h):
    return R_REF * torch.exp(R_SPAN * torch.tanh(h / R_SPAN))


def features(m, feats, om):
    """Gauge-free per-cell features with fixed (not data-dependent) scaling. Cached per mesh/frequency."""
    key = ("feat", tuple(feats), round(om, 12) if "fw" in feats else None)
    if key not in m.cache:
        Hs = torch.clamp(m.H, min=H_MIN)
        s = torch.sin(m.phi)[:, None].expand(m.nlat, m.nlon)
        cols = dict(H=lambda: 0.5 * torch.log(Hs / 1000.0), slope=lambda: torch.log1p(m.slope / 0.005),
                    lat=lambda: s, abslat=lambda: s.abs(),
                    fw=lambda: torch.clamp((2 * OMEGA_E * s).abs() / om, max=3.0))
        m.cache[key] = torch.stack([cols[f]() for f in feats], -1).contiguous()
    return m.cache[key]


class PhysClosure:
    """kappa = r0/H + r_sh/H e^{-H/200} + c_it |grad H|^2 sig((H-500)/100) [x sqrt(1-(f/om)^2)_+] + kappa_min
    theta = (log r0 [m/s], log c_it [1/s], logit(beta/0.2), log r_sh [m/s]). Penalty: prior around THETA0."""
    n, names, l2_default = 4, PARAM_NAMES, 1e-3

    def __init__(self, critical=False):
        self.critical = bool(critical)
        self.label = "PHYS-fc(4p)" if self.critical else "PHYS(4p)"
        self.spec = dict(kind="phys", critical=self.critical)

    def theta0(self, seed=0):
        return THETA0.copy()

    def kappa_beta(self, theta, m, om):
        r0, c_it, r_sh = torch.exp(theta[0]), torch.exp(theta[1]), torch.exp(theta[3])
        beta = 0.2 * torch.sigmoid(theta[2])
        Hs = torch.clamp(m.H, min=H_MIN)
        it = m.slope ** 2 * torch.sigmoid((Hs - 500.0) / 100.0)
        if self.critical:
            key = ("crit", round(om, 12))
            if key not in m.cache:
                fw2 = (m.fcor[:, None] / om) ** 2
                m.cache[key] = torch.sqrt(torch.clamp(1.0 - fw2, min=0.0)).expand(m.nlat, m.nlon)
            it = it * m.cache[key]
        kappa = r0 / Hs + r_sh / Hs * torch.exp(-Hs / 200.0) + c_it * it
        return (kappa + KAPPA_MIN) * m.mask, beta

    def penalty(self, theta, theta0):
        return torch.sum((theta - theta0) ** 2)

    def describe(self, theta):
        t = np.asarray(theta, float)
        return dict(r0=float(np.exp(t[0])), c_it=float(np.exp(t[1])), beta=float(0.2 / (1 + np.exp(-t[2]))),
                    r_shelf=float(np.exp(t[3])))


class NNClosure:
    """kappa = r(x)/H + kappa_min, r = 1e-3 exp(6 tanh(MLP(x)/6)) m/s, tanh hidden layers.
    theta = [W1, b1, ..., W_out, b_out, logit(beta/0.2)]. Output layer initialised ~0 -> r = 1e-3 m/s.
    Penalty: weight decay on weight matrices only (biases and beta unpenalised)."""
    l2_default = 1e-4

    def __init__(self, width=16, depth=2, feats=("H", "slope", "lat")):
        self.width, self.depth, self.feats = int(width), int(depth), tuple(feats)
        bad = [f for f in self.feats if f not in FEATS]
        if bad:
            raise ValueError(f"unknown NN features {bad}; choose from {FEATS}")
        sizes = [len(self.feats)] + [self.width] * self.depth + [1]
        self.shapes = list(zip(sizes[:-1], sizes[1:]))
        self.n = sum(a * b + b for a, b in self.shapes) + 1
        wm = []
        for a, b in self.shapes:
            wm += [True] * (a * b) + [False] * b
        self.wmask = torch.as_tensor(np.array(wm + [False]))
        self.label = f"NN[{'+'.join(self.feats)}]w{self.width}d{self.depth}({self.n}p)"
        self.spec = dict(kind="nn", width=self.width, depth=self.depth, feats=list(self.feats))
        self.names = [f"w[{k}]" for k in range(self.n - 1)] + ["logit(beta/0.2)"]

    def theta0(self, seed=0):
        rng = np.random.RandomState(seed)
        parts = []
        for k, (a, b) in enumerate(self.shapes):
            sc = 1.0 / math.sqrt(a) if k < len(self.shapes) - 1 else 1e-3
            parts += [rng.randn(a * b) * sc, np.zeros(b)]
        parts.append(np.zeros(1))
        return np.concatenate(parts)

    def pre(self, theta, X):
        h, off = X, 0
        for k, (a, b) in enumerate(self.shapes):
            Wk = theta[off:off + a * b].reshape(a, b); off += a * b
            bk = theta[off:off + b]; off += b
            h = h @ Wk + bk
            if k < len(self.shapes) - 1:
                h = torch.tanh(h)
        return h[..., 0]

    def kappa_beta(self, theta, m, om):
        r = r_from_h(self.pre(theta, features(m, self.feats, om)))
        kappa = r / torch.clamp(m.H, min=H_MIN)
        return (kappa + KAPPA_MIN) * m.mask, 0.2 * torch.sigmoid(theta[-1])

    def penalty(self, theta, theta0):
        return torch.sum(theta[self.wmask] ** 2)

    def describe(self, theta):
        t = np.asarray(theta, float)
        return dict(beta=float(0.2 / (1 + np.exp(-t[-1]))), n_params=int(self.n),
                    weight_rms=float(np.sqrt(np.mean(t[:-1] ** 2))))


class PolyClosure:
    """h(x) = polynomial of total degree <= deg in the features; r = 1e-3 exp(6 tanh(h/6)) as for NN.
    Used as a low-capacity learnable closure and as the target of NN distillation."""
    l2_default = 1e-4

    def __init__(self, degree=2, feats=("H", "slope", "lat")):
        self.degree, self.feats = int(degree), tuple(feats)
        nf = len(self.feats)
        self.terms = [t for d in range(self.degree + 1) for t in itertools.combinations_with_replacement(range(nf), d)]
        self.n = len(self.terms) + 1
        self.label = f"POLY{self.degree}[{'+'.join(self.feats)}]({self.n}p)"
        self.spec = dict(kind="poly", degree=self.degree, feats=list(self.feats))
        self.names = ["*".join(self.feats[i] for i in t) or "1" for t in self.terms] + ["logit(beta/0.2)"]

    def theta0(self, seed=0):
        return np.zeros(self.n)

    def design(self, X):
        cols = [torch.ones(X.shape[:-1], dtype=X.dtype) if not t else torch.prod(X[..., list(t)], -1)
                for t in self.terms]
        return torch.stack(cols, -1)

    def pre(self, theta, X):
        return self.design(X) @ theta[:-1]

    def kappa_beta(self, theta, m, om):
        key = ("design", self.feats, self.degree, round(om, 12) if "fw" in self.feats else None)
        if key not in m.cache:
            m.cache[key] = self.design(features(m, self.feats, om))
        r = r_from_h(m.cache[key] @ theta[:-1])
        kappa = r / torch.clamp(m.H, min=H_MIN)
        return (kappa + KAPPA_MIN) * m.mask, 0.2 * torch.sigmoid(theta[-1])

    def penalty(self, theta, theta0):
        return torch.sum(theta[1:-1] ** 2)

    def describe(self, theta):
        t = np.asarray(theta, float)
        return dict(beta=float(0.2 / (1 + np.exp(-t[-1]))), n_params=int(self.n),
                    coef={k: float(v) for k, v in zip(self.names[:-1], t[:-1])})


def make_closure(spec):
    """'phys' | 'physfc' | 'nn:W:D:feat+feat' | 'poly:deg:feat+feat'"""
    p = spec.strip().split(":")
    get = lambda k, d: p[k] if len(p) > k and p[k] else d
    if p[0] in ("phys", "physfc"):
        return PhysClosure(critical=p[0] == "physfc")
    if p[0] == "nn":
        return NNClosure(int(get(1, 16)), int(get(2, 2)), tuple(get(3, "H+slope+lat").split("+")))
    if p[0] == "poly":
        return PolyClosure(int(get(1, 2)), tuple(get(2, "H+slope+lat").split("+")))
    raise ValueError(f"unknown closure spec '{spec}'")


def closure_from_spec(d):
    k = d.get("kind", "phys")
    if k == "phys":
        return PhysClosure(d.get("critical", False))
    if k == "nn":
        return NNClosure(d["width"], d["depth"], tuple(d["feats"]))
    if k == "poly":
        return PolyClosure(d["degree"], tuple(d["feats"]))
    raise ValueError(d)


# ----------------------------------------------------------------------------- physics
def residual(m, om, kappa, beta, u, v, eta, eta_eq):
    """r(x; theta) = A(theta) x - b for x = (u, v, eta); differentiable in theta."""
    Phi = ((1 - beta) * eta - eta_eq) * m.mask
    gx = (torch.roll(Phi, -1, 1) - Phi) / m.dx[:, None] * m.open_u
    gy = (_north(Phi) - Phi) / m.dy * m.open_v
    kap_u = 0.5 * (kappa + torch.roll(kappa, -1, 1))
    kap_v = 0.5 * (kappa + _north(kappa))
    Hu = torch.where(m.open_u > 0, m.H_u, torch.ones_like(m.H_u))
    Hv = torch.where(m.open_v > 0, m.H_v, torch.ones_like(m.H_v))
    Sv = m.W * (v + torch.roll(v, -1, 1))
    Cv = (Sv + _south(Sv)) / (4 * Hu * m.area[:, None])
    Su = m.W * (u + _north(u))
    Cu = (Su + torch.roll(Su, 1, 1)) / (4 * Hv * m.area_v[:, None])
    r_u = ((1j * om + kap_u) * u - Cv + G0 * gx) * m.open_u
    r_v = ((1j * om + kap_v) * v + Cu + G0 * gy) * m.open_v
    Fu, Fv = m.H_u * u * m.dy, m.H_v * v * m.len_v[:, None]
    div = (Fu - torch.roll(Fu, 1, 1) + Fv - _south(Fv)) / m.area[:, None]
    r_e = (1j * om * eta + div) * m.mask
    return r_u, r_v, r_e


def build_system(m, om, kappa, beta):
    """Sparse A matching residual() exactly. Unknowns ordered [u faces | v faces | eta cells]."""
    nlat, nlon = m.nlat, m.nlon
    kap = kappa.detach().numpy()
    J, I = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing="ij")
    Ip, Im = (I + 1) % nlon, (I - 1) % nlon
    Jn, Js = np.minimum(J + 1, nlat - 1), np.maximum(J - 1, 0)
    okN, okS = J + 1 < nlat, J - 1 >= 0
    ku, kv, ke, Nu, Nv = m.ku, m.kv, m.k, m.Nu, m.Nv
    ou, ov, oe = ku >= 0, kv >= 0, ke >= 0
    trip = []

    def add(rloc, roff, cloc, coff, val, ok):
        ok = ok & (rloc >= 0) & (cloc >= 0)
        trip.append((rloc[ok] + roff, cloc[ok] + coff, np.broadcast_to(val, rloc.shape)[ok]))

    A, Av = m.area_np[:, None], m.area_v_np[:, None]
    dx, lenv, dy, W = m.dx_np[:, None], m.len_v_np[:, None], m.dy, m.W_np
    Hu, Hv = m.H_u_np, m.H_v_np
    kap_u = 0.5 * (kap + np.roll(kap, -1, 1))
    kap_v = 0.5 * (kap + np.vstack([kap[1:], np.zeros((1, nlon))]))
    gb = G0 * (1 - beta)
    add(ku, 0, ku, 0, 1j * om + kap_u, ou)
    cu = 1.0 / (4 * np.where(ou, Hu, 1.0) * A)
    W_s = np.vstack([np.zeros((1, nlon)), W[:-1]])
    for cj, okj, w in ((J, ou, W * cu), (Js, ou & okS, W_s * cu)):
        add(ku, 0, kv[cj, I], Nu, -w, okj)
        add(ku, 0, kv[cj, Ip], Nu, -w, okj)
    add(ku, 0, ke[J, Ip], Nu + Nv, gb / dx, ou)
    add(ku, 0, ke, Nu + Nv, -gb / dx, ou)
    add(kv, Nu, kv, Nu, 1j * om + kap_v, ov)
    cv = 1.0 / (4 * np.where(ov, Hv, 1.0) * Av)
    for ci, w in ((I, W * cv), (Im, np.roll(W, 1, 1) * cv)):
        add(kv, Nu, ku[J, ci], 0, w, ov)
        add(kv, Nu, ku[Jn, ci], 0, w, ov & okN)
    add(kv, Nu, ke[Jn, I], Nu + Nv, gb / dy, ov & okN)
    add(kv, Nu, ke, Nu + Nv, -gb / dy, ov)
    add(ke, Nu + Nv, ke, Nu + Nv, 1j * om, oe)
    add(ke, Nu + Nv, ku, 0, Hu * dy / A, oe)
    add(ke, Nu + Nv, ku[J, Im], 0, -Hu[J, Im] * dy / A, oe)
    add(ke, Nu + Nv, kv, Nu, Hv * lenv / A, oe)
    add(ke, Nu + Nv, kv[Js, I], Nu, -(Hv * lenv)[Js, I] / A, oe & okS)
    r, c, v = (np.concatenate(z) for z in zip(*trip))
    n = Nu + Nv + m.N
    return sp.coo_matrix((v.astype(complex), (r, c)), shape=(n, n)).tocsc()


class _ImplicitSolve(torch.autograd.Function):
    """y = -A^{-1} r with A fixed (LU). Backward: grad_r = -A^{-H} g (exact adjoint, same factorisation)."""

    @staticmethod
    def forward(ctx, r, lu):
        ctx.lu = lu
        return torch.from_numpy(-lu.solve(np.ascontiguousarray(r.detach().numpy())))

    @staticmethod
    def backward(ctx, g):
        lam = ctx.lu.solve(np.ascontiguousarray(g.detach().numpy()), trans="H")
        return torch.from_numpy(np.ascontiguousarray(-lam)), None


def solve_full(m, C, theta, name):
    """Forward solve for one constituent; fields are differentiable in theta through the implicit adjoint."""
    om = omega(name)
    kappa, beta = C.kappa_beta(theta, m, om)
    Zeq = m.equilibrium(name)
    lu = spla.splu(build_system(m, om, kappa, float(beta.detach())))
    z = torch.zeros((m.nlat, m.nlon), dtype=CPLX)
    b = -m.pack(*residual(m, om, kappa, beta, z, z, z, Zeq))
    x0 = torch.from_numpy(lu.solve(np.ascontiguousarray(b.detach().numpy())))
    r0 = m.pack(*residual(m, om, kappa, beta, *m.unpack(x0), Zeq))
    m.last_resid = (r0.detach().norm() / b.detach().norm()).item()
    x = x0 + _ImplicitSolve.apply(r0, lu)
    u, v, eta = m.unpack(x)
    return dict(name=name, om=om, eta=eta, u=u, v=v, kappa=kappa, beta=beta)


def solve(m, C, theta, name, full=False):
    F = solve_full(m, C, theta, name)
    return (F["eta"], F["u"], F["v"]) if full else F["eta"]


def energy_terms(m, F):
    """Differentiable work input P [W] and per-face dissipation maps Du, Dv [W]. P == sum(D) to round-off."""
    om, eta, u, v, kappa, beta = (F[k] for k in ("om", "eta", "u", "v", "kappa", "beta"))
    Phi = ((1 - beta) * eta - m.equilibrium(F["name"])) * m.mask
    P = 0.5 * RHO * G0 * torch.sum(torch.real(Phi * torch.conj(-1j * om * eta)) * m.area[:, None])
    kap_u = 0.5 * (kappa + torch.roll(kappa, -1, 1))
    kap_v = 0.5 * (kappa + _north(kappa))
    Du = 0.5 * RHO * kap_u * m.H_u * (u.real ** 2 + u.imag ** 2) * m.area[:, None]
    Dv = 0.5 * RHO * kap_v * m.H_v * (v.real ** 2 + v.imag ** 2) * m.area_v[:, None]
    return P, Du, Dv


def shallow_part(m, Du, Dv, Hcut=500.0):
    return torch.sum(Du * (m.H_u < Hcut)) + torch.sum(Dv * (m.H_v < Hcut))


def energy_budget(m, C, theta, name, F=None):
    with torch.no_grad():
        if F is None:
            F = solve_full(m, C, torch.as_tensor(np.asarray(theta, float)), name)
        P, Du, Dv = energy_terms(m, F)
        D_row = torch.sum(Du + Dv, 1)
        D_sh = shallow_part(m, Du, Dv)
    return dict(P=float(P), D=float(D_row.sum()), D_row=D_row.numpy(), D_shallow=float(D_sh))


# ----------------------------------------------------------------------------- gauges and folds
def build_gauges(m, year, max_gauges):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"gauges_{year}_{m.res:g}deg_{max_gauges}.json"
    if f.exists():
        g = json.loads(f.read_text())
        for r in g:
            r["Z"] = np.array(r["Zr"]) + 1j * np.array(r["Zi"])
        return g
    names = list(CONST)
    out = []
    stations = sorted(uhslc_stations(), key=lambda s: s["id"])
    print(f"[gauges] {len(stations)} UHSLC stations; analysing year {year} ...")
    for s in stations:
        cell = m.snap(s["lat"], s["lon"])
        if cell is None:
            continue
        t, eta = uhslc_hourly(s["id"], year)
        if len(t) < 0.6 * 8760 or np.isfinite(eta).sum() < 0.6 * 8760:
            continue
        if (t[-1] - t[0]) < np.timedelta64(300, "D"):
            continue
        Z, rstd = harmonic_analysis(t, eta, names)
        zm = np.array([Z[n] for n in MODEL_SET])
        if rstd > 0.5 or abs(zm[0]) > 6 or not np.all(np.isfinite(zm)):
            continue
        out.append(dict(id=s["id"], name=s["name"], lat=s["lat"], lon=s["lon"], j=int(cell[0]), i=int(cell[1]),
                        resid=rstd, Zr=zm.real.tolist(), Zi=zm.imag.tolist()))
        print(f"  {len(out):3d} {s['id']:4d} {s['name'][:22]:22s} M2={abs(zm[0]):.2f} m  resid={rstd:.3f} m", flush=True)
        if len(out) >= max_gauges:
            break
    f.write_text(json.dumps(out))
    for r in out:
        r["Z"] = np.array(r["Zr"]) + 1j * np.array(r["Zi"])
    return out


def density_weights(g, radius_km=500.0):
    """w_g ∝ 1 / (number of gauges within radius) — de-weights dense coastlines; normalised to mean 1."""
    lat, lon = np.array([r["lat"] for r in g]), np.array([r["lon"] for r in g])
    D = gc_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    w = 1.0 / (D < radius_km).sum(1)
    return w / w.mean()


def exclude_ids():
    f = RESULTS / "exclude_ids.json"
    return set(json.loads(f.read_text())) if f.exists() else set()


def fold_of(sid, seed, k):
    h = hashlib.sha256(f"{seed}:{int(sid)}".encode()).digest()
    return int.from_bytes(h[:8], "big") % k


def digest(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in sorted(ids)).encode()).hexdigest()[:12]


_BLOCKS = {}


def spatial_blocks(g, k, seed=0, restarts=25):
    """Spherical k-means on gauge positions -> k contiguous blocks, ordered by centroid longitude."""
    key = (digest([r["id"] for r in g]), k, seed)
    if key in _BLOCKS:
        return _BLOCKS[key]
    la, lo = np.radians([r["lat"] for r in g]), np.radians([r["lon"] for r in g])
    X = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
    rng, best = np.random.RandomState(seed), None
    for _ in range(restarts):
        Cc = X[rng.choice(len(X), k, replace=False)]
        for _ in range(200):
            lab = np.argmax(X @ Cc.T, 1)
            new = np.array([X[lab == c].mean(0) if np.any(lab == c) else Cc[c] for c in range(k)])
            new /= np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-12)
            if np.allclose(new, Cc):
                break
            Cc = new
        inertia = float(np.sum(1 - np.sum(X * Cc[lab], 1)))
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy(), Cc.copy())
    lab, Cc = best[1], best[2]
    order = np.argsort(np.degrees(np.arctan2(Cc[:, 1], Cc[:, 0])) % 360)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    _BLOCKS[key] = remap[lab]
    return _BLOCKS[key]


def split_gauges(g, seed=0, fold=0, k=5, scheme="hash", buffer_km=0.0, verbose=True):
    """scheme='hash': fold = sha256(seed:id) mod k (random in space).
       scheme='block': spatial k-means blocks; training gauges within buffer_km of any test gauge are dropped."""
    ex = exclude_ids()
    ids = np.array([r["id"] for r in g])
    keep = np.array([i not in ex for i in ids])
    if scheme == "hash":
        lab = np.array([fold_of(i, seed, k) for i in ids])
    elif scheme == "block":
        lab = spatial_blocks(g, k, seed)
    else:
        raise ValueError(scheme)
    te = keep & (lab == fold)
    tr = keep & ~te
    if buffer_km > 0 and te.any():
        lat, lon = np.array([r["lat"] for r in g]), np.array([r["lon"] for r in g])
        D = gc_km(lat[tr][:, None], lon[tr][:, None], lat[te][None, :], lon[te][None, :])
        tri = np.nonzero(tr)[0]
        tr[tri[D.min(1) < buffer_km]] = False
    if verbose:
        print(f"[split] {scheme} seed={seed} fold={fold}/{k}: {tr.sum()} train / {te.sum()} test / "
              f"{(~keep).sum()} excluded; test={digest(ids[te])}")
    return np.nonzero(tr)[0], np.nonzero(te)[0]


def gauge_Z(g):
    return np.stack([r["Z"] for r in g])


def model_at_gauges(m, C, theta, g, names):
    gj = torch.as_tensor([r["j"] for r in g]); gi = torch.as_tensor([r["i"] for r in g])
    return {n: solve(m, C, theta, n)[gj, gi] for n in names}


# ----------------------------------------------------------------------------- objective
def objective(m, C, theta, g, idx, names, w, theta0=None, l2=0.0, energy=None):
    """L = Σ_k Σ_g w_g |eta_k(x_g) - etahat_gk|^2 / Σ w  +  l2 * penalty(theta)
           [+ mu_D (D_M2/D_obs - 1)^2 + mu_sh (D_shallow/D - 2/3)^2]  — no PDE residual: the PDE is solved."""
    Zo = gauge_Z(g)
    sub = [g[k] for k in idx]
    gj = torch.as_tensor([r["j"] for r in sub]); gi = torch.as_tensor([r["i"] for r in sub])
    wt = torch.as_tensor(np.asarray(w)[idx])
    loss = torch.zeros(())
    if energy and "M2" not in names:
        raise ValueError("energy penalty needs M2 among the fitted constituents")
    for n in names:
        F = solve_full(m, C, theta, n)
        d = F["eta"][gj, gi] - torch.as_tensor(Zo[idx, MODEL_SET.index(n)])
        loss = loss + torch.sum(wt * (d.real ** 2 + d.imag ** 2)) / wt.sum()
        if energy and n == "M2":
            _, Du, Dv = energy_terms(m, F)
            D = Du.sum() + Dv.sum()
            loss = loss + energy.get("mu_D", 0.0) * (D / energy.get("D_obs", D_OBS_M2) - 1.0) ** 2 \
                        + energy.get("mu_sh", 0.0) * (shallow_part(m, Du, Dv) / D - SHALLOW_OBS) ** 2
    if l2 > 0:
        th0 = torch.as_tensor(np.asarray(theta0 if theta0 is not None else C.theta0(), float))
        loss = loss + l2 * C.penalty(theta, th0)
    return loss


# ----------------------------------------------------------------------------- baselines & scoring
def rms_cm(Zp, Zo, w):
    return 100.0 * math.sqrt(np.sum(w * np.abs(Zp - Zo) ** 2) / np.sum(w) / 2.0)


def baseline_nearest(g, tr, te, Zo):
    lat, lon = np.array([r["lat"] for r in g]), np.array([r["lon"] for r in g])
    D = gc_km(lat[te][:, None], lon[te][:, None], lat[tr][None, :], lon[tr][None, :])
    return Zo[tr][D.argmin(1)]


def baseline_gp(g, tr, te, Zo, ells=(300, 600, 1200, 2500, 5000), nugs=(0.01, 0.05, 0.2)):
    """Complex GP (exponential kernel on great-circle distance), length scale and nugget by LOO, per constituent."""
    lat, lon = np.array([r["lat"] for r in g]), np.array([r["lon"] for r in g])
    Dtt = gc_km(lat[tr][:, None], lon[tr][:, None], lat[tr][None, :], lon[tr][None, :])
    Dst = gc_km(lat[te][:, None], lon[te][:, None], lat[tr][None, :], lon[tr][None, :])
    out = np.zeros((len(te), Zo.shape[1]), complex)
    Kis = {(l, n): np.linalg.inv(np.exp(-Dtt / l) + n * np.eye(len(tr))) for l in ells for n in nugs}
    for c in range(Zo.shape[1]):
        y, best = Zo[tr, c], None
        for (l, n), Ki in Kis.items():
            loo = np.mean(np.abs((Ki @ y) / np.diag(Ki)) ** 2)
            if best is None or loo < best[0]:
                best = (loo, l, Ki)
        out[:, c] = np.exp(-Dst / best[1]) @ (best[2] @ y)
    return out


def paired_bootstrap(e1, e2, w, n=4000, seed=0):
    """e1, e2: per-gauge squared complex errors (summed over constituents). Returns RMS1-RMS2 [cm], its 95% CI,
    and p = P(RMS1 - RMS2 >= 0), i.e. probability that model 1 is NOT better than model 2."""
    e1, e2, w = (np.asarray(a, float) for a in (e1, e2, w))
    rms = lambda e, ww: 100 * np.sqrt((ww * e).sum(-1) / ww.sum(-1) / 2)
    idx = np.random.RandomState(seed).randint(0, len(e1), (n, len(e1)))
    d = rms(e1[idx], w[idx]) - rms(e2[idx], w[idx])
    return dict(diff=float(rms(e1, w) - rms(e2, w)), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p=float(np.mean(d >= 0)))


def score_split(m, C, theta, g, tr, te, names, w, names_eval=None, keep_fields=False):
    """In-sample RMS on (tr, names), held-out RMS on (te, names_eval), baselines, energy diagnostics."""
    names_eval = list(names_eval or names)
    alln = list(dict.fromkeys(list(names) + names_eval + ["M2"]))
    th = torch.as_tensor(np.asarray(theta, float))
    with torch.no_grad():
        F = {n: solve_full(m, C, th, n) for n in alln}
    resid = m.last_resid
    gj, gi = np.array([r["j"] for r in g]), np.array([r["i"] for r in g])
    Zm = {n: F[n]["eta"].numpy()[gj, gi] for n in alln}
    Zeq = {n: m.equilibrium(n).numpy()[gj, gi] for n in alln}
    Zall = gauge_Z(g)

    def rss(idx, nms, Z):
        row = {n: rms_cm(Z[n][idx], Zall[idx, MODEL_SET.index(n)], w[idx]) for n in nms}
        return math.sqrt(sum(v * v for v in row.values())), row

    ins, ins_row = rss(tr, names, Zm)
    out, out_row = rss(te, names_eval, Zm)
    eq_out, _ = rss(te, names_eval, Zeq)
    Zo = Zall[:, [MODEL_SET.index(n) for n in names_eval]]
    Zmt = np.stack([Zm[n][te] for n in names_eval], 1)
    Zgp, Znn = baseline_gp(g, tr, te, Zo), baseline_nearest(g, tr, te, Zo)
    err = lambda Zp: np.sum(np.abs(Zp - Zo[te]) ** 2, 1)
    e_m, e_gp, e_nn = err(Zmt), err(Zgp), err(Znn)
    e_eq = err(np.stack([Zeq[n][te] for n in names_eval], 1))
    gp_row = {n: rms_cm(Zgp[:, c], Zo[te, c], w[te]) for c, n in enumerate(names_eval)}
    en = {n: energy_budget(m, C, th, n, F[n]) for n in alln}
    eM = en["M2"]
    r_eff = (F["M2"]["kappa"] * m.H).numpy()
    hi = np.abs(m.lat_np) > 66
    res = dict(
        in_sample=ins, held_out=out, in_row=ins_row, held_out_row=out_row,
        gp=math.sqrt(sum(v * v for v in gp_row.values())), gp_row=gp_row,
        nearest=100 * math.sqrt(np.sum(w[te] * e_nn) / np.sum(w[te]) / 2), equilibrium=eq_out,
        p_vs_gp=paired_bootstrap(e_m, e_gp, w[te])["p"], n_train=len(tr), n_test=len(te),
        test_ids=[int(g[k]["id"]) for k in te], test_lat=[g[k]["lat"] for k in te],
        test_lon=[g[k]["lon"] for k in te], test_w=w[te], e_model=e_m, e_gp=e_gp, e_nearest=e_nn, e_eq=e_eq,
        D_TW=eM["D"] / 1e12, D_TW_all={n: en[n]["D"] / 1e12 for n in alln},
        energy_closure={n: (en[n]["P"] - en[n]["D"]) / max(abs(en[n]["P"]), 1e-30) for n in alln},
        shallow_frac=eM["D_shallow"] / max(eM["D"], 1e-30), polar_frac=float(eM["D_row"][hi].sum() / max(eM["D"], 1e-30)),
        D_row_M2=eM["D_row"] / 1e12, r_eff_q=np.percentile(r_eff[m.maskb], [5, 50, 95]),
        beta=float(F["M2"]["beta"]), solver_resid=resid, describe=C.describe(np.asarray(theta, float)))
    fields = None
    if keep_fields:
        fields = dict(r_eff=r_eff.astype(np.float32), mask=m.maskb, lat=m.lat_np, lon=m.lon_np,
                      **{f"eta_{n}": F[n]["eta"].numpy().astype(np.complex64) for n in alln})
    return res, fields


# ----------------------------------------------------------------------------- fitting / checking
def fit_theta(m, C, g, tr, names, w, theta0, l2, iters, fixed=None, energy=None, verbose=True):
    """L-BFGS (strong Wolfe) on the free entries of theta. Returns dict(theta, loss, n_evals, seconds, history)."""
    fixed = fixed or {}
    free = [p for p in range(len(theta0)) if p not in fixed]
    pos = {p: k for k, p in enumerate(free)}
    fixed_t = {p: torch.tensor(float(v)) for p, v in fixed.items()}
    phi = torch.tensor(theta0[free], requires_grad=True)

    def full(ph):
        if not fixed:
            return ph
        return torch.stack([fixed_t[p] if p in fixed else ph[pos[p]] for p in range(len(theta0))])

    opt = torch.optim.LBFGS([phi], lr=1.0, max_iter=iters, history_size=10, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12)
    hist, best = [], [float("inf"), np.asarray(theta0, float).copy()]
    t0 = time.time()

    def step():
        opt.zero_grad()
        th = full(phi)
        L = objective(m, C, th, g, tr, names, w, theta0, l2, energy)
        L.backward()
        hist.append(L.item())
        if L.item() < best[0]:
            best[0], best[1] = L.item(), th.detach().numpy().copy()
        if verbose:
            print(f"  eval {len(hist):3d}  loss={L.item():.5e}", flush=True)
        return L

    opt.step(step)
    sec = time.time() - t0
    if verbose:
        print(f"[fit] {C.label}: {len(hist)} evaluations, {sec:.0f} s, best loss {best[0]:.4e}")
    return dict(theta=best[1], loss=best[0], n_evals=len(hist), seconds=sec, history=hist)


def run_check(m, C, g, names=("M2", "K1"), energy=None, seed=0, h=1e-3):
    """Energy identity per constituent + adjoint vs central finite differences (directional if many params)."""
    names = list(names)
    th0 = C.theta0(seed)
    if C.n > 6:                                  # move off the near-zero output layer so all paths are active
        th0 = th0 + 0.1 * np.random.RandomState(seed + 7).randn(C.n)
    eb = {}
    for n in names:
        e = energy_budget(m, C, th0, n)
        eb[n] = dict(P_TW=e["P"] / 1e12, D_TW=e["D"] / 1e12, rel_mismatch=(e["P"] - e["D"]) / max(abs(e["P"]), 1e-30),
                     shallow_frac=e["D_shallow"] / max(e["D"], 1e-30), solver_resid=m.last_resid)
    idx, w = np.arange(len(g)), density_weights(g)
    theta = torch.tensor(th0, requires_grad=True)
    t0 = time.time()
    L = objective(m, C, theta, g, idx, names, w, energy=energy)
    L.backward()
    t_grad = time.time() - t0
    grad = theta.grad.numpy().copy()
    if C.n <= 6:
        dirs, labels = np.eye(C.n), list(C.names)
    else:
        dirs = np.random.RandomState(0).randn(4, C.n)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        labels = [f"dir {k}" for k in range(4)]
    rows = []
    for d, lab in zip(dirs, labels):
        with torch.no_grad():
            fp = objective(m, C, torch.tensor(th0 + h * d), g, idx, names, w, energy=energy).item()
            fm = objective(m, C, torch.tensor(th0 - h * d), g, idx, names, w, energy=energy).item()
        fd, ad = (fp - fm) / (2 * h), float(grad @ d)
        rows.append(dict(label=lab, adjoint=ad, fd=fd, rel=abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)))
    return dict(label=C.label, spec=C.spec, n_params=C.n, energy=eb, loss=L.item(), grad=rows,
                seconds_loss_and_grad=t_grad, energy_penalty=energy, n_gauges=len(g))


def synthetic_gauges(m, C, names, seed=0):
    rng, g = np.random.RandomState(1), []
    for k, (la, lo) in enumerate(zip(rng.uniform(-60, 60, 40), rng.uniform(0, 360, 40))):
        c = m.snap(float(la), float(lo))
        if c:
            g.append(dict(id=k, lat=float(la), lon=float(lo), j=c[0], i=c[1]))
    with torch.no_grad():
        Zt = model_at_gauges(m, C, torch.tensor(C.theta0(seed) + 0.3), g, names)
    for k, r in enumerate(g):
        r["Z"] = np.array([Zt[n][k].item() if n in Zt else 0 for n in MODEL_SET], complex)
    return g


# ----------------------------------------------------------------------------- CLI
def save_params(theta, C, a, best, names, path=None):
    RESULTS.mkdir(exist_ok=True)
    d = dict(theta=np.asarray(theta).tolist(), closure=C.spec, label=C.label, describe=C.describe(theta),
             res=a.res, year=a.year, seed=a.seed, fold=a.fold, kfolds=a.kfolds, scheme=a.scheme,
             train_loss=best, constituents=names)
    (path or RESULTS / "params.json").write_text(json.dumps(_jsonable(d), indent=1))


def load_params():
    f = RESULTS / "params.json"
    if not f.exists():
        return PhysClosure(), THETA0.copy(), {}
    d = json.loads(f.read_text())
    return closure_from_spec(d.get("closure", dict(kind="phys"))), np.array(d["theta"]), d


def make_mesh(a):
    lat, lon, H = synthetic_bathymetry(a.res) if a.synthetic else load_bathymetry(a.res)
    m = Mesh(lat, lon, H)
    print(f"[mesh] {m.nlat}x{m.nlon} @ {a.res:g}°, {m.N} ocean cells, {m.Nu + m.Nv + m.N} unknowns")
    return m


def cmd_check(a):
    m = make_mesh(a)
    C = make_closure(a.closure)
    names = ["M2", "K1"]
    g = synthetic_gauges(m, C, names, a.seed) if a.synthetic else build_gauges(m, a.year, a.max_gauges)
    energy = dict(mu_D=a.energy_mu, mu_sh=a.energy_mu) if a.energy_mu else None
    r = run_check(m, C, g, names, energy, a.seed)
    ok = True
    for n, e in r["energy"].items():
        ok &= abs(e["rel_mismatch"]) < 1e-6
        print(f"[energy] {n}: work {e['P_TW']:.3f} TW, dissipation {e['D_TW']:.3f} TW, mismatch {e['rel_mismatch']:.1e}, "
              f"|Ax-b|/|b|={e['solver_resid']:.1e}, {100 * e['shallow_frac']:.0f}% in H<500 m")
    for d in r["grad"]:
        ok &= d["rel"] < 1e-3
        print(f"   {d['label']:18s} adjoint={d['adjoint']: .6e}  FD={d['fd']: .6e}  rel.err={d['rel']:.2e}")
    print(f"[check] {C.label}: loss+grad in {r['seconds_loss_and_grad']:.1f} s  ->", "PASS" if ok else "FAIL")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"check_{a.closure.replace(':', '_')}.json").write_text(json.dumps(_jsonable(r), indent=1))


def cmd_fit(a):
    if a.synthetic:
        sys.exit("fit needs real gauges; drop --synthetic")
    m = make_mesh(a)
    g = build_gauges(m, a.year, a.max_gauges)
    w = density_weights(g)
    tr, te = split_gauges(g, a.seed, a.fold, a.kfolds, a.scheme, a.buffer_km)
    C = make_closure(a.closure)
    names = a.constituents.split(",")
    theta0, fixed = C.theta0(a.seed), {}
    if a.fix_r0 is not None:
        if not isinstance(C, PhysClosure):
            sys.exit("--fix-r0 only applies to phys closures")
        fixed[0] = theta0[0] = math.log(a.fix_r0)
    l2 = a.l2 if a.l2 is not None else C.l2_default
    energy = dict(mu_D=a.energy_mu, mu_sh=a.energy_mu) if a.energy_mu else None
    print(f"[fit] {C.label}: {len(tr)} train / {len(te)} test, constituents {names}, l2={l2:g}, energy={energy}")
    fit = fit_theta(m, C, g, tr, names, w, theta0, l2, a.iters, fixed, energy, verbose=not a.quiet)
    save_params(fit["theta"], C, a, fit["loss"], names)
    print("[fit] saved results/params.json:", C.describe(fit["theta"]))


def cmd_eval(a):
    if a.synthetic:
        sys.exit("eval needs real gauges; drop --synthetic")
    m = make_mesh(a)
    g = build_gauges(m, a.year, a.max_gauges)
    w = density_weights(g)
    C, theta, pf = load_params()
    if pf and (pf.get("fold") != a.fold or float(pf.get("res", a.res)) != a.res or pf.get("scheme", "hash") != a.scheme):
        print(f"[eval] WARNING: params.json is from res={pf.get('res')} fold={pf.get('fold')} scheme={pf.get('scheme')}")
    tr, te = split_gauges(g, a.seed, a.fold, a.kfolds, a.scheme, a.buffer_km)
    names = pf.get("constituents", a.constituents.split(","))
    sc, _ = score_split(m, C, theta, g, tr, te, names, w, MODEL_SET)
    print(f"\nheld-out RMS [cm], {len(te)} gauges, closure {C.label}")
    print(f"{'':12s}" + "".join(f"{n:>7s}" for n in MODEL_SET) + f"{'RSS':>8s}")
    print(f"{'MODEL':12s}" + "".join(f"{sc['held_out_row'][n]:7.1f}" for n in MODEL_SET) + f"{sc['held_out']:8.1f}")
    print(f"{'GP':12s}" + "".join(f"{sc['gp_row'][n]:7.1f}" for n in MODEL_SET) + f"{sc['gp']:8.1f}")
    print(f"nearest {sc['nearest']:.1f}  equilibrium {sc['equilibrium']:.1f}  P(model>=GP) {sc['p_vs_gp']:.3f}")
    print(f"D_M2 {sc['D_TW']:.2f} TW, {100 * sc['shallow_frac']:.0f}% in H<500 m, closure {sc['energy_closure']['M2']:+.1e}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(_jsonable(sc), indent=1))


def cmd_predict(a):
    m = make_mesh(a)
    cell = m.snap(a.lat, a.lon % 360)
    if cell is None:
        sys.exit("point is not within one cell of ocean on this grid")
    C, theta, _ = load_params()
    theta = torch.tensor(theta)
    with torch.no_grad():
        Z = {n: solve(m, C, theta, n)[cell].item() for n in MODEL_SET}
    t = np.datetime64(a.start) + np.arange(a.hours) * np.timedelta64(1, "h")
    eta = predict_series(Z, t)
    for n in MODEL_SET:
        print(f"  {n:3s} amp={abs(Z[n]) * 100:6.1f} cm  phase={(-np.degrees(np.angle(Z[n]))) % 360:6.1f}°")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"predict_{a.lat:g}_{a.lon:g}.csv"
    np.savetxt(out, np.column_stack([t.astype("datetime64[s]").astype(str), np.round(eta, 3)]), fmt="%s",
               delimiter=",", header="time_utc,eta_m", comments="")
    print(f"[predict] wrote {out}")


def cmd_split(a):
    m = make_mesh(a)
    g = build_gauges(m, a.year, a.max_gauges)
    for f in range(a.kfolds):
        split_gauges(g, a.seed, f, a.kfolds, a.scheme, a.buffer_km)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "fit", "eval", "predict", "split"])
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--max-gauges", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--scheme", default="hash", choices=["hash", "block"])
    ap.add_argument("--buffer-km", type=float, default=0.0)
    ap.add_argument("--closure", default="phys")
    ap.add_argument("--l2", type=float, default=None)
    ap.add_argument("--energy-mu", type=float, default=0.0)
    ap.add_argument("--fix-r0", type=float, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--constituents", default="M2,S2,K1,O1")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--lat", type=float, default=21.3); ap.add_argument("--lon", type=float, default=202.1)
    ap.add_argument("--start", default="2024-06-01T00:00"); ap.add_argument("--hours", type=int, default=240)
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    {"check": cmd_check, "fit": cmd_fit, "eval": cmd_eval, "predict": cmd_predict, "split": cmd_split}[a.cmd](a)


if __name__ == "__main__":
    main()