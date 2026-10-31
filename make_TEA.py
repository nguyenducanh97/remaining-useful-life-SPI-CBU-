import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import rcParams, font_manager
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, Polygon
from matplotlib.ticker import FixedLocator, NullFormatter

OUT = os.path.join(os.getcwd(), "outputs")
F_MAIN = os.path.join(OUT, "fig_main"); SD = os.path.join(OUT, "source_data")
for d in (F_MAIN, SD): os.makedirs(d, exist_ok=True)

FONT = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "Liberation Sans"
rcParams.update({
    "font.family": FONT, "font.weight": "bold", "font.size": 6.5, "axes.labelweight": "bold", "axes.linewidth": 0.5,
    "axes.labelsize": 6.8, "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.4, "ytick.major.size": 2.4, "xtick.minor.size": 1.3,
    "ytick.minor.size": 1.3, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2, "mathtext.fontset": "custom", "mathtext.rm": FONT + ":bold",
    "mathtext.it": FONT + ":bold:italic", "mathtext.bf": FONT + ":bold", "mathtext.default": "rm",
    "pdf.fonttype": 42, "svg.fonttype": "none"})
C = dict(salt="#a8c3de", saltrib="#dde7f1", h2="#7ea5cc", k="#3a78b3", po4="#1f4e7c", navy="#1f4e7c", pink="#d7337c",
         pinkrib="#fae6ef", cost="#543771", acid="#dcd0e6", elec="#543771", bead="#7e599f", pt="#9b80b8", kco="#bca6cd",
         teal="#1b9e8c", tealfill="#99d2c9", tealband="#eef8f6", dupink="#d8327e", dufill="#ea9dbf", pinkband="#fdf2f6",
         ink="#1a1a1a", grey="#7a7a7a", negdark="#624285", posdark="#215388")


def shade(c, f): r, g, b = to_rgb(c); return (min(1, r * f), min(1, g * f), min(1, b * f))
def tint(c, t): r, g, b = to_rgb(c); return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


# ===== inputs =====
SALT_Q = 132.14 / (2 * 14.007)          # kg (NH4)2SO4 per kg N
ACID_Q = 98.08 / (2 * 14.007)           # kg H2SO4 per kg N
M_KCO, PT_AREA, PT_LIFE, BEAD_DOSE, EFF_FRAC = 0.682, 0.017, 8760, 200.0, 0.80
TECH = dict(cR=(9.077, 0.15), sec=(1.973, 0.042), yh2=(0.0726, 0.0007), nc=(0.01815, 0.0004),
            cK=(0.9445, 0.0049), cP=(1.923, 0.0266))
TRI = dict(salt=(0.157, 0.5235, 0.890), h2=(2.50, 3.75, 5.00), k=(0.565, 0.708, 0.851), po4=(0.890, 1.080, 1.270),
           kco=(0.00210, 0.006225, 0.01035), bead=(0.0007, 0.0008, 0.0009), pt=(0.008, 0.0265, 0.045),
           Lkco=(168426, 189033, 200118), Lbead=(1312, 1702, 2256))
CAMPUS = {"SKKU": dict(V=2840116.625, elec=(0.011, 0.0115, 0.012), acid=(0.060, 0.0675, 0.075), col=C["teal"],
                       fill=C["tealfill"], band=C["tealband"], title="SKKU"),
          "DU": dict(V=32795335.0, elec=(0.067, 0.085, 0.103), acid=(0.12, 0.135, 0.15), col=C["dupink"],
                     fill=C["dufill"], band=C["pinkband"], title="Univ. of Delhi")}
UNIT = dict(salt="USD/kg", h2="USD/kg", k="USD/kg K", po4="USD/kg PO4", kco="USD/g", bead="USD/g", pt="USD/cm2",
            Lkco="cycles", Lbead="cycles", elec="USD/kWh", acid="USD/kg", cR="g N/L urine", sec="kWh/kg N",
            yh2="kg H2/kg N", nc="g N/cycle", cK="g/L", cP="g/L")
PRICE_KEYS = ["salt", "h2", "k", "po4", "kco", "bead", "pt", "Lkco", "Lbead", "elec", "acid"]
TECH_KEYS = list(TECH)
ALL_KEYS = PRICE_KEYS + TECH_KEYS
REV = ("salt", "h2", "po4", "k")
COST = ("acid", "elec", "bead", "pt", "kco")


def items(p):
    return dict(salt=SALT_Q * p["salt"], h2=p["yh2"] * p["h2"], k=p["cK"] / p["cR"] * p["k"],
                po4=p["cP"] / p["cR"] * p["po4"], acid=ACID_Q * p["acid"], elec=p["sec"] * p["elec"],
                bead=p["bead"] * BEAD_DOSE * 1000.0 / (p["Lbead"] * p["cR"]),
                pt=p["pt"] * PT_AREA * 1000.0 / (PT_LIFE * p["nc"]),
                kco=p["kco"] * M_KCO * 1000.0 / (p["Lkco"] * p["nc"]))


def margin(p):
    it = items(p)
    return sum(it[k] for k in REV) - sum(it[k] for k in COST)


def annual_N(p, V): return V * p["cR"] / 1000.0


def central(cn="SKKU"):
    p = {k: TRI[k][1] for k in TRI}
    p["elec"] = CAMPUS[cn]["elec"][1]; p["acid"] = CAMPUS[cn]["acid"][1]
    p.update({k: TECH[k][0] for k in TECH})
    return p


def tri_ppf(u, lo, mo, hi):
    f = (mo - lo) / (hi - lo)
    return np.where(u < f, lo + np.sqrt(u * (hi - lo) * (mo - lo)), hi - np.sqrt((1 - u) * (hi - lo) * (hi - mo)))


# ===== Monte Carlo (same random numbers for both campuses) =====
N_ITER, SEED = 20000, 2026
rng = np.random.default_rng(SEED)
U = {k: rng.random(N_ITER) for k in PRICE_KEYS}
Zn = {k: rng.standard_normal(N_ITER) for k in TECH_KEYS}
MC = {}
for cn, cp in CAMPUS.items():
    s = {k: tri_ppf(U[k], *(cp[k] if k in ("elec", "acid") else TRI[k])) for k in PRICE_KEYS}
    s.update({k: TECH[k][0] + TECH[k][1] * Zn[k] for k in TECH_KEYS})
    m = margin(s); n = annual_N(s, cp["V"])
    MC[cn] = dict(samples=s, perkg=m, N=n, annual=m * n)

BASE = central("SKKU"); B_IT = items(BASE)
REV_T = sum(B_IT[k] for k in REV); COST_T = sum(B_IT[k] for k in COST); MAR = REV_T - COST_T
OUTPUT = {}
for cn, cp in CAMPUS.items():
    pc = central(cn); Nn = annual_N(pc, cp["V"])
    OUTPUT[cn] = dict(N=Nn, salt_t=Nn * SALT_Q / 1000, h2_t=Nn * pc["yh2"] / 1000, eff_m3=cp["V"] * EFF_FRAC / 1000,
                      margin_central=margin(pc), annual_central=margin(pc) * Nn)


def rank(a): r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r


def sensitivity(cn):
    s = MC[cn]["samples"]; rm = rank(MC[cn]["annual"])
    keys = [k for k in ALL_KEYS if np.std(s[k]) > 0]
    rho = {k: np.corrcoef(rank(s[k]), rm)[0, 1] for k in keys}
    tot = sum(v ** 2 for v in rho.values())
    return rho, {k: 100 * rho[k] ** 2 / tot for k in keys}


def worst_case(cn="both"):
    w = {}
    for k in PRICE_KEYS:
        if k in ("elec", "acid"):
            if cn == "both": lo = min(CAMPUS["SKKU"][k][0], CAMPUS["DU"][k][0]); hi = max(CAMPUS["SKKU"][k][2], CAMPUS["DU"][k][2])
            else: lo, hi = CAMPUS[cn][k][0], CAMPUS[cn][k][2]
        else:
            lo, hi = TRI[k][0], TRI[k][2]
        w[k] = lo if k in ("salt", "h2", "k", "po4", "Lkco", "Lbead") else hi
    for k in TECH_KEYS:
        m_, s_ = TECH[k]
        w[k] = m_ + 2 * s_ if k == "sec" else m_ - 2 * s_
    return w


# ===== Fig. 7E : value of 1 kg recovered N at SKKU central inputs =====
def ribbon(ax, x0, a0, b0, x1, a1, b1, color, alpha=1.0, z=1):
    xm = 0.5 * (x0 + x1)
    v = [(x0, b0), (xm, b0), (xm, b1), (x1, b1), (x1, a1), (xm, a1), (xm, a0), (x0, a0), (x0, b0)]
    c = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(v, c), fc=color, ec="none", alpha=alpha, zorder=z))


def node(ax, x, y0, y1, color, w=0.13, z=4):
    ax.add_patch(Rectangle((x - w / 2, y0), w, y1 - y0, fc=color, ec="none", lw=0, zorder=z))


def panel_E(ax, width):
    W = 0.075; XL, XC, XR, XZ = 0.74, 1.30, 1.86, 2.07; g = 0.10
    names = {"salt": "(NH$_4$)$_2$SO$_4$", "h2": "H$_2$", "po4": "PO$_4$ fertilizer", "k": "K fertilizer"}
    ribc = {"salt": C["saltrib"], "h2": tint(C["h2"], 0.45), "po4": tint(C["po4"], 0.62), "k": tint(C["k"], 0.55)}
    top = REV_T + 3 * g; left = {}; y = top
    for k in REV:
        left[k] = (y - B_IT[k], y); y -= B_IT[k] + g
    bottom = y + g
    c0 = 0.5 * (top + bottom) - REV_T / 2 + 0.05; cen = (c0, c0 + REV_T)
    cost = (bottom, bottom + COST_T); mar = (cost[1] + 0.16, cost[1] + 0.16 + MAR)
    yc = cen[1]
    for k in REV:
        a, b = left[k]; v = B_IT[k]
        ribbon(ax, XL + W / 2, a, b, XC - W / 2, yc - v, yc, ribc[k], alpha=0.9); yc -= v
    ribbon(ax, XC + W / 2, cen[1] - MAR, cen[1], XR - W / 2, mar[0], mar[1], C["pinkrib"])
    ribbon(ax, XC + W / 2, cen[0], cen[0] + COST_T, XR - W / 2, cost[0], cost[1], tint(C["cost"], 0.72))
    for k in REV: node(ax, XL, *left[k], C[k], W)
    node(ax, XC, *cen, C["navy"], W); node(ax, XR, *mar, C["pink"], W); node(ax, XR, *cost, C["cost"], W)
    stroke = [pe.withStroke(linewidth=1.0, foreground="white")]
    for k in REV:
        a, b = left[k]; yy = 0.5 * (a + b)
        ax.text(XL - W / 2 - 0.035, yy, names[k], ha="right", va="center", fontsize=5.8)
        ax.text(XL + W / 2 + 0.03, yy, "%.2f" % B_IT[k], ha="left", va="center", fontsize=5.6, color=C["ink"], path_effects=stroke, zorder=6)
    ax.text(XC, cen[1] + 0.06, "Revenue\n%.2f" % REV_T, ha="center", va="bottom", fontsize=6.3, color=C["navy"], linespacing=1.0)
    xm = 0.5 * (XC + XR); ym = 0.5 * (mar[0] + mar[1])
    ax.text(xm, ym + 0.30, "Operating\nmargin", ha="center", va="bottom", fontsize=6.0, color=C["pink"], linespacing=1.0)
    ax.text(xm, ym + 0.27, "%.2f" % MAR, ha="center", va="top", fontsize=6.6, color=C["pink"])
    ax.text(xm, ym + 0.02, "%.1f%% of\nrevenue" % (100 * MAR / REV_T), ha="center", va="top", fontsize=6.0, color=C["pink"], linespacing=1.0)
    Z = 10.0; zb = cost[0]; zt = zb + COST_T * Z; zw = 0.09
    ax.add_patch(Polygon([(XR + W / 2, cost[0]), (XR + W / 2, cost[1]), (XZ, zt), (XZ, zb)], closed=True,
                         fc="#f0f0f0", ec="#b8b8b8", lw=0.3, ls=(0, (2, 1)), zorder=0.5))
    labs = dict(acid="H$_2$SO$_4$", elec="Electricity", bead="Beads", pt="Pt–Ti", kco="KCoHCF")
    yz = zt; centres = []
    for k in COST:
        h = B_IT[k] * Z
        ax.add_patch(Rectangle((XZ, yz - h), zw, h, fc=C[k], ec="none", lw=0, zorder=4)); centres.append(yz - h / 2); yz -= h
    lab_y = [max(centres[0], zb + 1.30), zb + 0.94, zb + 0.64, zb + 0.34, zb + 0.04]
    xr = XZ + zw
    for k, yc_, yl in zip(COST, centres, lab_y):
        v = B_IT[k] * 100
        ax.plot([xr, xr + 0.04, xr + 0.07], [yc_, yl, yl], lw=0.3, color=C["grey"], zorder=3)
        ax.text(xr + 0.09, yl, "%s  %s US¢" % (labs[k], ("%.1f" % v) if v >= 1 else ("%.2f" % v)), va="center", ha="left", fontsize=5.2)
    ax.text(XZ + zw / 2, zt + 0.05, "Costs %.2f\n(×10)" % COST_T, ha="center", va="bottom", fontsize=5.8, color=C["cost"], linespacing=1.0)
    ax.set_xlim(0, width); ax.set_ylim(bottom - 0.08, top + 0.30); ax.axis("off")


# ===== Fig. 7F : annual operating margin, Monte Carlo =====
def kde(x, grid):
    xs = x[::10]; bw = 1.06 * xs.std() * len(xs) ** (-1 / 5)
    z = (grid[:, None] - xs[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(1) / (len(xs) * bw * np.sqrt(2 * np.pi))


def fmt_usd(v): return "$%.0fk" % (v / 1e3) if v < 1e6 else "$%.2fM" % (v / 1e6)


def tags(cn):
    o = OUTPUT[cn]
    n = ("%.1f" % (o["N"] / 1000)) if o["N"] < 1e5 else ("%.0f" % (o["N"] / 1000))
    st = ("%.1f" % o["salt_t"]) if o["salt_t"] < 1000 else "{:,.0f}".format(o["salt_t"])
    h2 = ("H$_2$ %.2f t" % o["h2_t"]) if o["h2_t"] < 10 else ("H$_2$ %.1f t" % o["h2_t"])
    return ["%s t N per year" % n, "(NH$_4$)$_2$SO$_4$ %s t" % st, h2, "Fert. effluent {:,.0f} m$^3$".format(float("%.3g" % o["eff_m3"]))]


def panel_F(ax):
    rr = np.random.default_rng(11)
    for y, cn in ((1.0, "SKKU"), (0.0, "DU")):
        cp = CAMPUS[cn]; a = MC[cn]["annual"]; la = np.log10(a); col = cp["col"]
        ax.axhspan(y - 0.5, y + 0.5, color=cp["band"], zorder=0, lw=0)
        g = np.linspace(la.min() - 0.04, la.max() + 0.04, 400)
        d = kde(la, g); d = d / d.max() * 0.34
        ax.fill_between(10 ** g, y, y + d, color=cp["fill"], lw=0, zorder=2)
        ax.plot(10 ** g, y + d, color=shade(col, .7), lw=0.55, zorder=3)
        sub = rr.choice(a, 450, replace=False)
        ax.scatter(sub, y - 0.29 + rr.uniform(-0.085, 0.085, len(sub)), s=1.2, color=col, alpha=0.6, lw=0, zorder=2)
        q = np.percentile(a, [5, 10, 25, 50, 75, 90, 95])
        ax.plot([q[1], q[5]], [y - 0.11] * 2, color=C["ink"], lw=0.45, zorder=4, solid_capstyle="butt")
        for e in (q[1], q[5]): ax.plot([e, e], [y - 0.135, y - 0.085], color=C["ink"], lw=0.45, zorder=4)
        ax.add_patch(Rectangle((q[2], y - 0.145), q[4] - q[2], 0.07, fc=col, ec=C["ink"], lw=0.4, zorder=5))
        ax.scatter([q[3]], [y - 0.11], s=10, color="white", ec=C["ink"], lw=0.45, zorder=6)
        ax.text(10 ** g[np.argmax(d)], y + 0.355, fmt_usd(q[3]), ha="center", va="bottom", fontsize=7.4, color=col)
        xa, ha = (0.97, "right") if cn == "SKKU" else (0.03, "left")
        T = ax.get_yaxis_transform()
        ax.text(xa, y + 0.44, cp["title"], transform=T, ha=ha, va="top", fontsize=7.4, color=col)
        for j, t in enumerate(tags(cn)):
            ax.text(xa, y + 0.27 - j * 0.115, t, transform=T, ha=ha, va="top", fontsize=5.6, color=C["ink"], fontweight="normal")
    ax.axhline(0.5, color="white", lw=0.8, zorder=1)
    ax.set_xscale("log"); ax.set_xlim(1.6e4, 3.6e6); ax.set_ylim(-0.5, 1.5); ax.set_yticks([])
    ax.xaxis.set_major_locator(FixedLocator([2e4, 5e4, 1e5, 3e5, 1e6, 3e6]))
    ax.set_xticklabels(["20k", "50k", "100k", "300k", "1M", "3M"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Annual operating margin (USD yr$^{-1}$)")


# ===== Fig. 7G : margin vs (NH4)2SO4 and electricity price =====
def margin_grid(xs, ys, p):
    P = dict(p); X, Y = np.meshgrid(xs, ys); P["salt"] = X; P["elec"] = Y
    return X, Y, margin(P)


def breakeven_elec(p, salt):
    q = dict(p); q["salt"] = salt; q["elec"] = 0.0
    return margin(q) / q["sec"]


VMIN, VMAX = -1.3, 5.2
CMAP_N = LinearSegmentedColormap.from_list("n", [(0.0, C["negdark"]), (1.0, "#ece6f2")])
CMAP_P = LinearSegmentedColormap.from_list("p", [(0.0, "#f5f9fc"), (0.35, "#c9dcec"), (0.7, "#6f9cc8"), (1.0, C["posdark"])])


def cm_lookup(v):
    v = np.asarray(v, float)
    return np.where((v < 0)[..., None], CMAP_N(np.clip((v - VMIN) / (0 - VMIN), 0, 1)), CMAP_P(np.clip(v / VMAX, 0, 1)))


def panel_G(ax, cax):
    xs = np.linspace(0, 1.0, 201); ys = np.linspace(0, 0.6, 201)
    X, Y, Z = margin_grid(xs, ys, BASE)
    lv = np.round(np.arange(VMIN, VMAX + 1e-9, 0.02), 4)
    Xc, Yc, Zc = margin_grid(np.linspace(0, 1.0, 5), np.linspace(0, 0.6, 5), BASE)
    cf = ax.contourf(Xc, Yc, Zc, levels=lv, colors=[cm_lookup(0.5 * (a + b)) for a, b in zip(lv[:-1], lv[1:])], zorder=0, antialiased=False)
    cf.set_edgecolor("face"); cf.set_linewidth(0.15)
    cs = ax.contour(X, Y, Z, levels=[1, 2, 3, 4], colors="white", linewidths=0.35, zorder=2)
    yl = 0.22; c0 = margin(dict(BASE, salt=0.0, elec=0.0))
    ax.clabel(cs, fmt=lambda v: "%d" % v, fontsize=5.2, inline_spacing=1.5,
              manual=[((L - c0 + BASE["sec"] * yl) / SALT_Q, yl) for L in (1, 2, 3, 4)])
    ax.contour(X, Y, Z, levels=[0], colors=C["ink"], linewidths=0.7, zorder=3)
    _, _, Zw = margin_grid(xs, ys, worst_case())
    ax.contour(X, Y, Zw, levels=[0], colors=C["ink"], linewidths=0.6, linestyles=[(0, (3, 1.6))], zorder=3)
    for xv in (TRI["salt"][0], TRI["salt"][2]): ax.axvline(xv, color=C["ink"], lw=0.35, ls=(0, (1, 1.3)), zorder=2)
    ax.annotate("", xy=(TRI["salt"][2], 0.565), xytext=(TRI["salt"][0], 0.565),
                arrowprops=dict(arrowstyle="<->", lw=0.4, color=C["ink"], shrinkA=0, shrinkB=0), zorder=4)
    ax.text(0.5 * (TRI["salt"][0] + TRI["salt"][2]), 0.572, "Market price range", ha="center", va="bottom", fontsize=5.6, fontweight="normal", zorder=5)
    rr = np.random.default_rng(5)
    for cn, mk in (("SKKU", "o"), ("DU", "D")):
        s = MC[cn]["samples"]; idx = rr.choice(N_ITER, 260, replace=False); col = CAMPUS[cn]["col"]
        ax.scatter(s["salt"][idx], s["elec"][idx], s=1.4, color=col, alpha=0.8, lw=0, zorder=4)
        cx, cy = TRI["salt"][1], CAMPUS[cn]["elec"][1]
        ax.scatter([cx], [cy], s=22, marker=mk, color=col, ec="white", lw=0.5, zorder=6)
        ax.text(cx, cy + (0.016 if cn == "SKKU" else 0.02), cn, fontsize=6.0, va="bottom", ha="center", color=col, zorder=7)
    ax.text(0.03, 0.52, "Loss", fontsize=6.0, color=C["ink"], ha="left", va="center", fontweight="normal", zorder=5)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.6)
    ax.set_xlabel("(NH$_4$)$_2$SO$_4$ selling price (USD kg$^{-1}$)"); ax.set_ylabel("Electricity price (USD kWh$^{-1}$)", labelpad=2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.tick_params(which="both", color="white", top=True, right=True)
    ax.plot([0.24, 0.32], [0.43, 0.43], color=C["ink"], lw=0.7, zorder=6)
    ax.plot([0.24, 0.32], [0.385, 0.385], color=C["ink"], lw=0.6, ls=(0, (3, 1.6)), zorder=6)
    for yv, t in ((0.43, "Break-even (central)"), (0.385, "Break-even (worst case)")):
        ax.text(0.335, yv, t, fontsize=5.4, va="center", ha="left", color=C["ink"], fontweight="normal", zorder=6)
    for a, b in zip(lv[:-1], lv[1:]):
        cax.add_patch(Rectangle((a, 0), b - a, 1, fc=cm_lookup(0.5 * (a + b)), ec=cm_lookup(0.5 * (a + b)), lw=0.15))
    cax.set_ylim(0, 1); cax.set_yticks([]); cax.set_xlim(VMIN, VMAX)
    cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cax.set_xticks([-1, 0, 1, 2, 3, 4, 5]); cax.tick_params(labelsize=5.6, length=2, pad=1)
    cax.set_xlabel("Operating margin (USD kg$^{-1}$ N)", fontsize=6.0, labelpad=2); cax.axvline(0, color=C["ink"], lw=0.5)
    for sp in cax.spines.values(): sp.set_linewidth(0.4)


def save(fig, name):
    for ext in ("png", "svg"):
        fn = os.path.join(F_MAIN, "%s.%s" % (name, ext))
        fig.savefig(fn, dpi=600, facecolor="white")
        if ext == "svg" and FONT != "Arial":
            t = open(fn, encoding="utf8").read().replace("'%s'" % FONT, "Arial").replace(FONT, "Arial")
            open(fn, "w", encoding="utf8").write(t)
    plt.close(fig)


fig = plt.figure(figsize=(3.0, 2.55)); ax = fig.add_axes([0, 0, 1, 0.90]); panel_E(ax, 3.0)
fig.text(0.004, 0.998, "(E)", fontsize=9, va="top", ha="left"); save(fig, "Fig7E")
fig = plt.figure(figsize=(2.10, 2.55)); ax = fig.add_axes([0.03, 0.155, 0.94, 0.745]); panel_F(ax)
fig.text(0.004, 0.998, "(F)", fontsize=9, va="top", ha="left"); save(fig, "Fig7F")
fig = plt.figure(figsize=(2.25, 2.55)); ax = fig.add_axes([0.205, 0.155, 0.755, 0.645]); cax = fig.add_axes([0.205, 0.865, 0.755, 0.032])
panel_G(ax, cax); fig.text(0.004, 0.998, "(G)", fontsize=9, va="top", ha="left"); save(fig, "Fig7G")

# ===== source data and summary tables =====
NAMES = dict(salt="(NH4)2SO4", h2="H2", po4="PO4 fertilizer", k="K fertilizer", acid="H2SO4", elec="Electricity",
             bead="PePurease beads", pt="Pt-Ti mesh", kco="KCoHCF")
rows = [(NAMES[k], "Revenue", B_IT[k]) for k in REV] + [("Revenue", "Operating margin", MAR), ("Revenue", "Operating costs", COST_T)]
rows += [("Operating costs", NAMES[k], B_IT[k]) for k in COST]
pd.DataFrame([dict(source=a, target=b, usd_per_kg_N=round(v, 5), share_of_revenue_pct=round(100 * v / REV_T, 3)) for a, b, v in rows]).to_csv(
    os.path.join(SD, "SourceData_Fig7E.csv"), index=False)

S, D = MC["SKKU"]["samples"], MC["DU"]["samples"]
draws = pd.DataFrame({"draw": np.arange(1, N_ITER + 1)})
for k in ["salt", "h2", "k", "po4", "kco", "bead", "pt", "Lkco", "Lbead"] + TECH_KEYS:
    draws["%s (%s)" % (k, UNIT[k])] = S[k]
for cn, s in (("SKKU", S), ("DU", D)):
    draws["%s elec (USD/kWh)" % cn] = s["elec"]; draws["%s acid (USD/kg)" % cn] = s["acid"]
    draws["%s margin (USD/kg N)" % cn] = MC[cn]["perkg"]; draws["%s N (kg/yr)" % cn] = MC[cn]["N"]
    draws["%s annual margin (USD/yr)" % cn] = MC[cn]["annual"]
draws.round(6).to_csv(os.path.join(SD, "SourceData_Fig7F.csv"), index=False)

gx = np.round(np.linspace(0, 1.0, 101), 4); gy = np.round(np.linspace(0, 0.6, 61), 4)
_, _, Zg = margin_grid(gx, gy, BASE)
grid = pd.DataFrame(np.round(Zg, 5), index=pd.Index(gy, name="elec_USD_per_kWh"), columns=gx)
grid.to_csv(os.path.join(SD, "SourceData_Fig7G.csv"))

CRF = 0.08 * 1.08 ** 20 / (1.08 ** 20 - 1)
rb = np.random.default_rng(7)
summ = []
for cn in CAMPUS:
    a = MC[cn]["annual"]; m = MC[cn]["perkg"]
    B = np.array([np.percentile(a[rb.integers(0, N_ITER, N_ITER)], [5, 50, 95]) for _ in range(1000)])
    q = np.percentile(a, [5, 10, 25, 50, 75, 90, 95])
    r = dict(campus=cn, margin_central_usd_per_kgN=OUTPUT[cn]["margin_central"], N_kg_per_yr=OUTPUT[cn]["N"],
             salt_t_per_yr=OUTPUT[cn]["salt_t"], h2_t_per_yr=OUTPUT[cn]["h2_t"], margin_mean_usd_per_kgN=m.mean(),
             margin_sd_usd_per_kgN=m.std(ddof=1), annual_mean=a.mean(), annual_sd=a.std(ddof=1), positive_fraction=(a > 0).mean())
    r.update({"annual_P%d" % p: v for p, v in zip([5, 10, 25, 50, 75, 90, 95], q)})
    r.update({"bootCI95_P%d" % p: "%.0f-%.0f" % tuple(np.percentile(B[:, i], [2.5, 97.5])) for i, p in enumerate((5, 50, 95))})
    r.update({"capex_P%d" % p: np.percentile(a, p) / CRF for p in (5, 50, 95)})
    summ.append(r)
pd.DataFrame(summ).round(4).to_csv(os.path.join(OUT, "tea_monte_carlo_summary.csv"), index=False)

var = []
for cn in CAMPUS:
    rho, share = sensitivity(cn)
    for k in sorted(share, key=lambda k: -share[k]):
        var.append(dict(campus=cn, input=k, unit=UNIT[k], spearman_rho=round(rho[k], 5), variance_share_pct=round(share[k], 4)))
pd.DataFrame(var).to_csv(os.path.join(OUT, "tea_variance_shares.csv"), index=False)

print("revenue %.3f  costs %.4f  margin %.3f USD/kg N" % (REV_T, COST_T, MAR))
for r in summ:
    print("%-4s median %s (P5 %s, P95 %s), positive %.1f%%" % (r["campus"], fmt_usd(r["annual_P50"]), fmt_usd(r["annual_P5"]),
                                                             fmt_usd(r["annual_P95"]), 100 * r["positive_fraction"]))
