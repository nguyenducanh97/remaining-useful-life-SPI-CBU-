import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from rul_style import *

# ---- Fig 7A : short-term RUL validation (scatter + thin line; dashed-black truth) ----
t2=pd.read_csv(os.path.join(OUT,"task2_curves.csv")); TRAJ="NH4Cl | NaCoHCF"
d=t2[t2.trajectory==TRAJ]; cbu=d[d.method=="SPI+CBU (ours)"].sort_values("cycle")
life=float(cbu.true_RUL.iloc[0]+cbu.cycle.iloc[0]); top=life*1.06
fig,ax=plt.subplots(figsize=(88*MM,80*MM))
ax.fill_between(cbu.cycle,np.clip(cbu.RUL_lo_cal90,0,top),np.clip(cbu.RUL_hi_cal90,0,top),color=PINK,alpha=0.16,lw=0,zorder=1)
ax.plot(cbu.cycle,cbu.true_RUL,color=DARK,lw=1.3,ls=(0,(4,2)),zorder=6)
plot_methods(ax,d,"pred_RUL",top=top,ms=6.0,lwn=0.8)
ax.set_xlim(0,top); ax.set_ylim(-0.045*top,top)
ax.set_xlabel("Cycle"); ax.set_ylabel("Remaining useful life (cycles)")
ax.set_title("Walk-forward RUL to end-of-life\nNH4Cl | NaCoHCF (life = 70; MAE 5.3 vs 152/94)")
ax.legend(handles=legend_handles_scatter(),loc="lower left",fontsize=5.4,handlelength=1.3,labelspacing=0.3,borderpad=0.4); frame(ax)
fig.tight_layout(pad=0.5)
savef(fig,os.path.join(F_MAIN,"Fig7A.png")); savef(fig,os.path.join(F_MAIN,"Fig7A.pdf")); plt.close(fig)

# ---- Fig 7B : capacity/activity projection to 80% (all 3 replicates) ----
fig,axes=plt.subplots(1,2,figsize=(176*MM,70*MM))
for ax,(mat,info) in zip(axes,MATS.items()):
    fn,cols,w,step,DROP,xmax,title,ylab=info; df=pd.read_csv(os.path.join(DD,fn))
    Rs=[project(df[c].to_numpy(float),w,step,DROP,xmax,True) for c in cols]
    import matplotlib.cm as cm
    for j,R in enumerate(Rs):
        fan=R.get("fan",[]); gcols=cm.Greys(np.linspace(0.30,0.78,max(len(fan),1)))
        for k,f in enumerate(fan):
            ax.plot(f["x"],f["y"],"-",color=gcols[k],lw=0.45,alpha=0.6,zorder=5,
                    label="Per-step linear floor" if (j==0 and k==len(fan)-1) else None)
        ax.fill_between(R["tg"],R["lo"],R["hi"],color=PINK,alpha=0.10,lw=0,label="90% interval" if j==0 else None,zorder=4)
        ax.plot(R["tg"],R["mm"],"-",color=PINK,lw=1.1,alpha=0.9,label="Projection" if j==0 else None,zorder=7)
        ax.plot(R["t"],R["ys"],"-",color=DARK,lw=1.0,alpha=0.95,label="Observed" if j==0 else None,zorder=8)
        ax.plot(R["TT"],R["PR"],"o",color=PINK,ms=2.0,alpha=0.40,mew=0,label="Tracking" if j==0 else None,zorder=9)
        if np.isfinite(R["cb"]): ax.plot(R["cb"],THR,"o",color=PINK,ms=5,mec="white",mew=0.5,zorder=10,label="80% life" if j==0 else None)
    ax.axhline(THR,color=GREY,ls=(0,(3,2)),lw=0.8)
    ax.set_xscale("log"); ax.set_ylim(78,101); ax.set_ylabel(ylab); ax.set_xlabel("Cycle (log scale)")
    c80=[R["cb"] for R in Rs if np.isfinite(R["cb"])]
    ax.set_title("%s\nProjected 80%% life: %s-%s cycles"%(title,format(int(min(c80)),","),format(int(max(c80)),",")))
    if mat=="KCoHCF": ax.legend(loc="upper right",fontsize=5.4,handlelength=1.3,labelspacing=0.25,borderpad=0.4)
    frame(ax)
fig.tight_layout(pad=0.6)
savef(fig,os.path.join(F_MAIN,"Fig7B.png"),bbox_inches="tight"); savef(fig,os.path.join(F_MAIN,"Fig7B.pdf"),bbox_inches="tight"); plt.close(fig)
print("main ok")
