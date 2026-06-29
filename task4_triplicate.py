"""Task 4: forward extrapolation to the 80% point for the triplicate materials."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
HERE=os.path.dirname(os.path.abspath(__file__)); DD=os.path.join(HERE,"Electrode_RUL_Data")
FIG=os.path.join(HERE,"outputs","figs"); MET=os.path.join(HERE,"outputs"); os.makedirs(FIG,exist_ok=True)
THR=80.0
# paper symbols: THR=D end-of-life threshold, q=split-conformal interval factor (S52)
def movmean(y,k): return pd.Series(y).rolling(k,center=False,min_periods=1).mean().to_numpy()
def _rmse(a,b): a,b=map(np.asarray,(a,b)); return float(np.sqrt(np.mean((a-b)**2)))
def _logfit(c0,c1,ys): x=np.log(np.arange(c0+1,c1+1)+1); b1,b0=np.polyfit(x,ys[c0:c1],1); return b0,b1
def _fit(c0,c1,ys,yraw):
    t=np.arange(c0+1,c1+1); X=np.column_stack([np.ones(len(t)),np.log(t+1)])
    beta,_,_,_=np.linalg.lstsq(X,ys[c0:c1],rcond=None); s2=np.sum((yraw[c0:c1]-X@beta)**2)/max(len(t)-2,1)
    return beta,s2,np.linalg.inv(X.T@X)
def _pred(b,s2,Xi,tg): Xg=np.column_stack([np.ones(len(tg)),np.log(tg+1)]); return Xg@b,np.sqrt(s2*(1+np.sum((Xg@Xi)*Xg,1)))
def project(yraw,w,step,DROP,xmax,LOGX,ax):
    ys=movmean(yraw,w); N=len(ys); t=np.arange(1,N+1); n0=15
    TT=[];PR=[];pc=0;pf=None
    for c in range(step,N+1,step):
        if pc>=n0 and pf is not None: TT.append(c); PR.append(pf[0]+pf[1]*np.log(c+1))
        sden=min(int(DROP*c),max(c-6,0))
        if c-sden>=4: pf=_logfit(sden,c,ys)
        pc=c
    TT=np.array(TT);PR=np.array(PR); idx=TT.astype(int)-1; trk=_rmse(PR,ys[idx]) if len(TT) else np.nan
    cut=int(0.7*N); sH=int(DROP*cut); bH,s2H,XiH=_fit(sH,cut,ys,yraw); th=np.arange(cut+1,N+1); mH,seH=_pred(bH,s2H,XiH,th)
    q=np.quantile(np.abs(yraw[cut:N]-mH)/np.maximum(seH,1e-9),0.90)  # split-conformal interval factor (paper: q)
    sF=int(DROP*N); bF,s2F,XiF=_fit(sF,N,ys,yraw)
    tg=(np.geomspace(N,xmax,3000) if LOGX else np.linspace(N,xmax,3000)); mm,se=_pred(bF,s2F,XiF,tg)
    cb=tg[np.where(mm<=THR)[0][0]] if np.any(mm<=THR) else np.inf
    lo=mm-q*se; hi=mm+q*se
    cb_lo=tg[np.where(hi<=THR)[0][0]] if np.any(hi<=THR) else np.inf   # optimistic edge
    cb_hi=tg[np.where(lo<=THR)[0][0]] if np.any(lo<=THR) else np.inf   # pessimistic edge
    csnap=np.linspace(int(0.25*N),N,12).astype(int); cols=cm.viridis(np.linspace(0,1,len(csnap))); f0=f1=None
    for c,col in zip(csnap,cols):
        bl1,bl0=np.polyfit(np.arange(1,c+1).astype(float),ys[:c],1); clin=(THR-bl0)/bl1
        tgl=(np.geomspace(c,min(max(clin*1.05,c*2),xmax),300) if LOGX else np.linspace(c,min(max(clin*1.05,c*2),xmax),300))
        ax.plot(tgl,bl0+bl1*tgl,"-",color=col,lw=0.8,alpha=0.6,zorder=3)
        if np.isfinite(clin)&(clin<xmax): ax.plot(clin,THR,"o",color=col,ms=3,zorder=4)
        f0=clin if f0 is None else f0; f1=clin
    ax.plot(t,yraw,".",ms=1.0,color="0.85"); ax.plot(t,ys,"-",color="k",lw=1.8,label="observed",zorder=8)
    ax.plot(TT,PR,"o",color="tab:red",ms=2.0,alpha=0.7,label=f"SPI+CBU track (RMSE={trk:.2f}%)",zorder=9)
    ax.fill_between(tg,lo,hi,color="tab:red",alpha=0.18,label="90% CI",zorder=6)
    ax.plot(tg,mm,"-",color="tab:red",lw=1.6,label=f"->80% @ {cb:.0f}",zorder=7)
    if np.isfinite(cb)&(cb<xmax): ax.plot(cb,THR,"o",color="tab:red",ms=7,zorder=9)
    ax.plot([],[],color=cm.viridis(0.5),lw=1.2,label=f"linear floor ({f0:.0f}->{f1:.0f})")
    ax.axhline(THR,color="0.4",ls=":",lw=1); ax.axvline(N,color="0.6",ls=":",lw=1)
    if LOGX: ax.set_xscale("log"); ax.set_xlim(1,xmax)
    else: ax.set_xlim(0,xmax)
    ax.set_ylim(78,101)
    return dict(N=N,now=round(ys[-1],2),trk=round(trk,3),cb=cb,cb_lo=cb_lo,cb_hi=cb_hi,floor0=f0,floor1=f1,q=round(q,2))

MATS=[("KCoHCF","KCoHCF_main_material_5000_cycles_new_triplicate.csv",["capacity_retention_1","capacity_retention_2","capacity_retention_3"],21,10,0.85,400000),
      ("Urea","urea_hydrolysis_urease_beads_300_cycles_triplicate.csv",["Relative_activity_1","Relative_activity_2","Relative_activity_3"],21,3,0.5,4000)]
rows=[]
for LOGX in [True,False]:
    for name,fn,cols,w,step,DROP,xmax in MATS:
        df=pd.read_csv(os.path.join(DD,fn))
        fig,axes=plt.subplots(1,3,figsize=(16,4.6)); 
        for i,ax in enumerate(axes):
            r=project(df[cols[i]].to_numpy(float),w,step,DROP,xmax,LOGX,ax)
            ax.set_title(f"{name} rep {i+1}: now {r['now']:.1f}% @{r['N']}, 80%@{r['cb']:.0f} (RUL {r['cb']-r['N']:.0f})",fontsize=9)
            ax.set_xlabel("cycle (log)" if LOGX else "cycle"); ax.set_ylabel("retention/activity (%)"); ax.legend(fontsize=6.5,loc="lower left")
            if LOGX: rows.append(dict(material=name,replicate=i+1,now_cycle=r["N"],now_ret=r["now"],tracking_RMSE_pct=r["trk"],
                                      cycle_at_80=round(r["cb"],0),RUL_to_80=round(r["cb"]-r["N"],0),
                                      cb_lo=round(r["cb_lo"],0),cb_hi=round(r["cb_hi"],0),floor_first=round(r["floor0"],0),floor_last=round(r["floor1"],0),CI90_q=r["q"]))
        fig.suptitle(f"Task 4 - {name}: project to 80% per replicate (90% CI + per-step linear floor), {'log' if LOGX else 'linear'} x",fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(FIG,f"task4_{name}_{'log' if LOGX else 'lin'}.png"),dpi=140); plt.close(fig)
        print(f"saved task4_{name}_{'log' if LOGX else 'lin'}.png")
M=pd.DataFrame(rows); M.to_csv(os.path.join(MET,"task4_projection_triplicate.csv"),index=False)
print("\nTask 4 projection (cycle@80% per replicate):")
print(M[["material","replicate","now_ret","tracking_RMSE_pct","cycle_at_80","RUL_to_80","cb_lo","cb_hi","floor_last"]].to_string(index=False))
