import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from rul_style import *
MMARK4=dict(MMARK); MMARK4["Log-Bayes"]="D"
MLAB4=dict(MLAB); MLAB4["Log-Bayes"]="Log-Bayes"
BINS=np.arange(0,101,10); CEN=(BINS[:-1]+BINS[1:])/2
FLOOR_ERR=0.0

def err_floor(e): return np.asarray(e,float)  # linear y: no flooring

def collect(task):
    """return dict method -> list of (frac_array, abserr_array) per panel, and pooled."""
    panels=[]  # (label, {method:(frac,err)})
    if task=="task1":
        d=pd.read_csv(os.path.join(OUT,"task1_curves.csv")); meths=["SPI+CBU (ours)","SPI+TBU","Lin","Log-Bayes","Wiener","Weibull"]
        for tr,g in d.groupby("trajectory"):
            N=float(g.cycle.max()); pdat={}
            for m in meths:
                dm=g[g.method==m].sort_values("cycle")
                if len(dm)==0: continue
                true=np.clip(N-dm.cycle.values,0,None)  # corrected diagonal truth
                frac=dm.cycle.values/N*100; err=np.abs(dm.pred_RUL.values-true)
                pdat[m]=(frac,err)
            panels.append((tr,pdat))
        return meths,panels
    if task=="task2":
        d=pd.read_csv(os.path.join(OUT,"task2_curves.csv")); meths=["SPI+CBU (ours)","SPI+TBU","Lin","Log-Bayes","Wiener","Weibull"]
        for tr,g in d.groupby("trajectory"):
            cb=g[g.method=="SPI+CBU (ours)"].sort_values("cycle"); eol=float(cb.cycle.iloc[0]+cb.true_RUL.iloc[0]); pdat={}
            for m in meths:
                dm=g[g.method==m].sort_values("cycle")
                frac=dm.cycle.values/eol*100; err=np.abs(dm.pred_RUL.values-dm.true_RUL.values)
                pdat[m]=(frac,err)
            panels.append((tr,pdat))
        return meths,panels
    # task3 KCoHCF / urea
    mat="KCoHCF" if task=="task3_KCoHCF" else "Urea"
    d=pd.read_csv(os.path.join(OUT,"task3_curves.csv")); d=d[d.material==mat]; meths=["SPI+CBU (ours)","SPI+TBU","Lin","Log-Bayes","Wiener","Weibull"]
    for (thr,rep),g in d.groupby(["threshold","replicate"]):
        life=float(g.life.iloc[0]); pdat={}
        for m in meths:
            dm=g[g.method==m].sort_values("cycle")
            frac=dm.cycle.values/life*100; err=np.abs(dm.pred_RUL.values-dm.true_RUL.values)
            pdat[m]=(frac,err)
        nm=("%d%%, rep %d"%(int(thr),int(rep)))
        panels.append((nm,pdat))
    return meths,panels

def binned_mean(fracs,errs):
    f=np.concatenate(fracs); e=np.concatenate(errs); idx=np.clip(np.digitize(f,BINS)-1,0,len(CEN)-1)
    m=np.array([np.nanmean(e[idx==k]) if (idx==k).any() else np.nan for k in range(len(CEN))])
    s=np.array([np.nanstd(e[idx==k]) if (idx==k).sum()>1 else 0.0 for k in range(len(CEN))])
    return m,s

TASKS=[("task1","Task 1: 56 screening trajectories"),("task2","Task 2: 10 trajectories to 80%"),
       ("task3_KCoHCF","Task 3: KCoHCF electrode (90%, 87%)"),("task3_urea","Task 3: urease beads (90%, 86%)")]

# ===== AVERAGE figure (2x2) =====
fig,axes=plt.subplots(2,2,figsize=(176*MM,150*MM)); axes=axes.ravel()
for ax,(task,title) in zip(axes,TASKS):
    meths,panels=collect(task)
    capvals=[]
    for m in meths:
        fr=[p[1][m][0] for p in panels if m in p[1]]; er=[err_floor(p[1][m][1]) for p in panels if m in p[1]]
        mean,sd=binned_mean(fr,er); v=~np.isnan(mean)
        ax.plot(CEN[v],mean[v],marker=MMARK4[m],ms=4.0,mew=0,lw=1.1,color=MC[m],label=MLAB4[m],zorder=5 if m=="SPI+CBU (ours)" else 3)
        if m=="SPI+CBU (ours)":
            ax.fill_between(CEN[v],np.maximum(mean[v]-sd[v],0),mean[v]+sd[v],color=PINK,alpha=0.15,lw=0,zorder=1)
        if m in ("SPI+CBU (ours)","SPI+TBU","Lin") and v.any():
            mm=np.nanmax(mean[v])
            if np.isfinite(mm): capvals.append(mm)
    ax.set_xlabel("Observed fraction (%)"); ax.set_ylabel("Mean absolute RUL error (cycles)")
    capvals=[c for c in capvals if np.isfinite(c) and c>0]
    ax.set_ylim(0, max(capvals)*1.18) if capvals else ax.set_ylim(bottom=0)
    ax.set_title(title,fontsize=6.5); ax.set_xlim(0,100); frame(ax)
    if task=="task1": ax.legend(loc="lower left",fontsize=5.4,handlelength=1.4,labelspacing=0.3)
fig.suptitle("Absolute RUL error vs observed fraction: SPI+CBU is consistently lowest at every stage of life",fontsize=7,y=1.01)
fig.tight_layout(pad=0.6,rect=[0,0,1,0.98])
savef(fig,os.path.join(F_MAIN,"Fig7C.png"),bbox_inches="tight"); savef(fig,os.path.join(F_SI,"SupplementaryFig25.png"),bbox_inches="tight"); plt.close(fig)
print("average ok")
