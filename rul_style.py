#!/usr/bin/env python3
"""Shared plotting style and projection engine for the RUL figures."""
import os, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

ROOT=os.getcwd(); OUT=os.path.join(ROOT,"outputs"); DD=os.path.join(ROOT,"Electrode_RUL_Data")
F_MAIN=os.path.join(OUT,"fig_main"); F_ED=os.path.join(OUT,"fig_ED"); F_SI=os.path.join(OUT,"fig_SI")
for d in (F_MAIN,F_ED,F_SI): os.makedirs(d,exist_ok=True)
PINK="#D6337C"; BLUE="#2E6FAE"; GREY="#8A8A8A"; DARK="#1A1A1A"; FLOOR="#C0792E"; OLIVE="#B8A11E"; TEAL="#1F9E89"; PURPLE="#7B4FA3"
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
  "font.size":7,"axes.titlesize":7,"axes.labelsize":7,"xtick.labelsize":6,"ytick.labelsize":6,
  "legend.fontsize":6,"axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
  "xtick.major.size":2.5,"ytick.major.size":2.5,"xtick.direction":"in","ytick.direction":"in",
  "lines.linewidth":1.0,"savefig.dpi":400,"figure.dpi":400,"legend.frameon":False,
  "axes.spines.top":True,"axes.spines.right":True})
MM=1/25.4
MC={"SPI+CBU (ours)":PINK,"SPI+TBU":BLUE,"Lin":GREY,"Log-Bayes":OLIVE,"Wiener":TEAL,"Weibull":PURPLE}
MLAB={"SPI+CBU (ours)":"SPI+CBU (ours)","SPI+TBU":"SPI+TBU","Lin":"Linear","Log-Bayes":"Log-Bayes","Wiener":"Wiener","Weibull":"Weibull"}

def frame(ax):
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.6)

def movmean(y,k): return pd.Series(y).rolling(k,center=False,min_periods=1).mean().to_numpy()
def _rmse(a,b): a,b=map(np.asarray,(a,b)); return float(np.sqrt(np.mean((a-b)**2)))
def _logfit(c0,c1,ys): x=np.log(np.arange(c0+1,c1+1)+1); b1,b0=np.polyfit(x,ys[c0:c1],1); return b0,b1
def _fit(c0,c1,ys,yraw):
    t=np.arange(c0+1,c1+1); X=np.column_stack([np.ones(len(t)),np.log(t+1)])
    beta,_,_,_=np.linalg.lstsq(X,ys[c0:c1],rcond=None); s2=np.sum((yraw[c0:c1]-X@beta)**2)/max(len(t)-2,1)
    return beta,s2,np.linalg.inv(X.T@X)
def _pred(b,s2,Xi,tg): Xg=np.column_stack([np.ones(len(tg)),np.log(tg+1)]); return Xg@b,np.sqrt(s2*(1+np.sum((Xg@Xi)*Xg,1)))
THR=80.0
def project(yraw,w,step,DROP,xmax,LOGX):
    ys=movmean(yraw,w); N=len(ys); t=np.arange(1,N+1); n0=15
    TT=[];PR=[];pc=0;pf=None
    for c in range(step,N+1,step):
        if pc>=n0 and pf is not None: TT.append(c); PR.append(pf[0]+pf[1]*np.log(c+1))
        sden=min(int(DROP*c),max(c-6,0))
        if c-sden>=4: pf=_logfit(sden,c,ys)
        pc=c
    TT=np.array(TT);PR=np.array(PR); idx=TT.astype(int)-1; trk=_rmse(PR,ys[idx]) if len(TT) else np.nan
    cut=int(0.7*N); sH=int(DROP*cut); bH,s2H,XiH=_fit(sH,cut,ys,yraw); th=np.arange(cut+1,N+1); mH,seH=_pred(bH,s2H,XiH,th)
    lam=np.quantile(np.abs(yraw[cut:N]-mH)/np.maximum(seH,1e-9),0.90)
    sF=int(DROP*N); bF,s2F,XiF=_fit(sF,N,ys,yraw)
    tg=(np.geomspace(N,xmax,3000) if LOGX else np.linspace(N,xmax,3000)); mm,se=_pred(bF,s2F,XiF,tg)
    cb=tg[np.where(mm<=THR)[0][0]] if np.any(mm<=THR) else np.inf
    lo=mm-lam*se; hi=mm+lam*se
    csnap=np.linspace(int(0.25*N),N,10).astype(int); fan=[]
    for c in csnap:
        bl1,bl0=np.polyfit(np.arange(1,c+1).astype(float),ys[:c],1)
        if bl1>=0: continue
        clin=(THR-bl0)/bl1
        xend=min(max(clin*1.03,c*1.5),xmax)
        tgl=(np.geomspace(c,xend,200) if LOGX else np.linspace(c,xend,200))
        fan.append(dict(x=tgl,y=bl0+bl1*tgl,clin=clin))
    fcross=min(f["clin"] for f in fan) if fan else np.inf
    return dict(t=t,yraw=yraw,ys=ys,N=N,now=round(ys[-1],2),trk=round(trk,3),TT=TT,PR=PR,
                tg=tg,mm=mm,lo=lo,hi=hi,cb=cb,fan=fan,fcross=fcross,xmax=xmax)

MATS={"KCoHCF":("KCoHCF_main_material_5000_cycles_new_triplicate.csv",
        ["capacity_retention_1","capacity_retention_2","capacity_retention_3"],21,10,0.85,400000,"KCoHCF electrode","Capacity retention (%)"),
      "Urea":("urea_hydrolysis_urease_beads_300_cycles_triplicate.csv",
        ["Relative_activity_1","Relative_activity_2","Relative_activity_3"],21,3,0.5,4000,"Urease beads","Relative activity (%)")}

def draw_proj(ax,R,LOGX,ylab,showleg=True,track_ms=2.0):
    import matplotlib.cm as cm
    ax.plot(R["t"],R["yraw"],".",ms=0.8,color="0.82",zorder=2)
    fan=R.get("fan",[]); gcols=cm.Greys(np.linspace(0.30,0.80,max(len(fan),1)))
    for k,f in enumerate(fan):
        ax.plot(f["x"],f["y"],"-",color=gcols[k],lw=0.6,alpha=0.85,zorder=5,
                label="Per-step linear floor" if k==len(fan)-1 else None)
        if np.isfinite(f["clin"]) and f["clin"]<R["xmax"]:
            ax.plot(f["clin"],THR,"o",color=gcols[k],ms=2.0,mew=0,zorder=5)
    ax.fill_between(R["tg"],R["lo"],R["hi"],color=PINK,alpha=0.16,label="90% interval",zorder=4)
    ax.plot(R["tg"],R["mm"],"-",color=PINK,lw=1.5,label="Projection",zorder=7)
    ax.plot(R["t"],R["ys"],"-",color=DARK,lw=1.3,label="Observed",zorder=8)
    ax.plot(R["TT"],R["PR"],"o",color=PINK,ms=track_ms,alpha=0.45,mew=0,label="Tracking",zorder=9)
    if np.isfinite(R["cb"]): ax.plot(R["cb"],THR,"o",color=PINK,ms=6,mec="white",mew=0.5,zorder=10,label="80% life")
    ax.axhline(THR,color=GREY,ls=(0,(3,2)),lw=0.8); ax.axvline(R["N"],color="0.6",ls=":",lw=0.7)
    if LOGX: ax.set_xscale("log")
    ax.set_ylim(78,101); ax.set_ylabel(ylab); ax.set_xlabel("Cycle (log scale)" if LOGX else "Cycle")
    if showleg: ax.legend(loc="upper right",handlelength=1.3,labelspacing=0.22,fontsize=5.0,borderpad=0.3)
    frame(ax)

def rul_panel_lims(ax,cyc,true_rul,cbu_pred):
    """Axis limits that keep the CBU prediction and flat baselines visible."""
    top=max(float(np.nanmax(cyc)),float(np.nanmax(true_rul)),float(np.nanpercentile(cbu_pred,98)))*1.06
    ax.set_xlim(0,top); ax.set_ylim(-0.045*top,top)
    return top

MMARK={"SPI+CBU (ours)":"o","SPI+TBU":"s","Lin":"^","Log-Bayes":"D","Wiener":"v","Weibull":"P"}
def plot_methods(ax,d,valcol,top=None,ms=2.0,lwn=0.6):
    for m in ["SPI+TBU","Lin","Log-Bayes","Wiener","Weibull","SPI+CBU (ours)"]:
        dm=d[d.method==m].sort_values("cycle")
        if len(dm)==0: continue
        y=dm[valcol].to_numpy(float)
        if top is not None: y=np.clip(y,0,top)
        z=5 if m=="SPI+CBU (ours)" else 3
        ax.plot(dm.cycle,y,marker=MMARK[m],ms=ms,mew=0,lw=lwn,color=MC[m],zorder=z)
def legend_handles_scatter(truth_label="Ground-truth RUL",interval=True):
    from matplotlib.lines import Line2D; from matplotlib.patches import Patch
    h=[Line2D([],[],color=MC["SPI+CBU (ours)"],marker="o",ms=3.5,mew=0,lw=0.7,label="SPI+CBU (ours)"),
       Line2D([],[],color=MC["SPI+TBU"],marker="s",ms=3.0,mew=0,lw=0.7,label="SPI+TBU"),
       Line2D([],[],color=MC["Lin"],marker="^",ms=3.0,mew=0,lw=0.7,label="Linear"),
       Line2D([],[],color=MC["Log-Bayes"],marker="D",ms=3.0,mew=0,lw=0.7,label="Log-Bayes"),
       Line2D([],[],color=MC["Wiener"],marker="v",ms=3.0,mew=0,lw=0.7,label="Wiener"),
       Line2D([],[],color=MC["Weibull"],marker="P",ms=3.0,mew=0,lw=0.7,label="Weibull"),
       Line2D([],[],color=DARK,lw=1.0,ls=(0,(3,2)),label=truth_label)]
    if interval: h.append(Patch(facecolor=MC["SPI+CBU (ours)"],alpha=0.16,label="Calibrated 90% interval"))
    return h

def savef(fig,path,**kw):
    fig.savefig(path,**kw)
    if path.lower().endswith(".png"): fig.savefig(path[:-4]+".svg",**kw)
