import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from rul_style import *

# ===== Walk-forward RUL across all 56 screening trajectories =====
t1=pd.read_csv(os.path.join(OUT,"task1_curves.csv")); trajs=sorted(t1.trajectory.unique())
ncol=8; nrow=int(np.ceil(len(trajs)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(180*MM,(180/ncol*nrow*0.95)*MM)); axes=axes.ravel()
for i,tr in enumerate(trajs):
    ax=axes[i]; d=t1[t1.trajectory==tr].copy(); cbu=d[d.method=="SPI+CBU (ours)"].sort_values("cycle")
    if len(cbu)==0: ax.axis("off"); continue
    N=float(cbu.cycle.max()); top=N*1.06
    true_diag=np.clip(N-cbu.cycle.values,0,None)
    ax.fill_between(cbu.cycle,np.clip(cbu.RUL_lo_cal90,0,top),np.clip(cbu.RUL_hi_cal90,0,top),color=PINK,alpha=0.13,lw=0,zorder=1)
    ax.plot(cbu.cycle,true_diag,color=DARK,lw=0.9,ls=(0,(3,2)),zorder=6)
    plot_methods(ax,d,"pred_RUL",top=top,ms=2.25,lwn=0.5)
    ax.set_xlim(0,top); ax.set_ylim(-0.045*top,top)
    ax.set_title(tr,fontsize=4.3,pad=1.5); ax.tick_params(labelsize=4); frame(ax)
for j in range(len(trajs),len(axes)): axes[j].axis("off")
fig.suptitle("Extended Data Fig. 8 | Walk-forward RUL to each curve's endpoint across all 56 screening trajectories. "
             "SPI+CBU (pink circles), SPI+TBU (blue squares), linear (grey triangles); black dashed, ground-truth RUL; pink shading, calibrated 90% interval. "
             "SPI+CBU follows the true diagonal; the unconstrained and linear baselines collapse early on the hard panels.",fontsize=5.5,y=1.004)
fig.supxlabel("Cycle",fontsize=7); fig.supylabel("Remaining useful life (cycles)",fontsize=7)
fig.tight_layout(pad=0.3,rect=[0.024,0.02,1,0.985])
savef(fig,os.path.join(F_ED,"ExtendedDataFig8.png"),bbox_inches="tight"); plt.close(fig)

# ===== Triplicate RUL for the deployed materials =====
t3=pd.read_csv(os.path.join(OUT,"task3_curves.csv")); cov3=pd.read_csv(os.path.join(OUT,"task3_coverage.csv"))
cases=[("KCoHCF",90.0),("KCoHCF",87.0),("Urea",90.0),("Urea",86.0)]
fig,axes=plt.subplots(4,3,figsize=(132*MM,132*MM))
for ri,(mat,thr) in enumerate(cases):
    for ci,rep in enumerate([1,2,3]):
        ax=axes[ri,ci]; d=t3[(t3.material==mat)&(t3.threshold==thr)&(t3.replicate==rep)]
        cbu=d[d.method=="SPI+CBU (ours)"].sort_values("cycle")
        if len(cbu)==0: ax.axis("off"); continue
        top=rul_panel_lims(ax,cbu.cycle.values,cbu.true_RUL.values,cbu.pred_RUL.values)
        ax.fill_between(cbu.cycle,np.clip(cbu.RUL_lo_cal90,0,top),np.clip(cbu.RUL_hi_cal90,0,top),color=PINK,alpha=0.16,lw=0,zorder=1)
        ax.plot(cbu.cycle,cbu.true_RUL,color=DARK,lw=0.9,ls=(0,(3,2)),zorder=6)
        plot_methods(ax,d,"pred_RUL",top=top,ms=3.6,lwn=0.6)
        cc=cov3[(cov3.material==mat)&(cov3.threshold==thr)&(cov3.replicate==rep)]; cov=cc.cov90.values[0] if len(cc) else np.nan
        nm="Urease beads" if mat=="Urea" else "KCoHCF electrode"
        ax.set_title("%s, %d%%, rep %d (coverage %.0f%%)"%(nm,int(thr),rep,cov),fontsize=5.3); ax.tick_params(labelsize=5); frame(ax)
axes[0,2].legend(handles=legend_handles_scatter(),loc="upper right",fontsize=4.6,handlelength=1.3,labelspacing=0.25,borderpad=0.3)
fig.suptitle("Extended Data Fig. 9 | Triplicate RUL for the deployed materials: KCoHCF electrode (90%, 87%) and urease beads (90%, 86%).",fontsize=6.4,y=1.002)
fig.supxlabel("Cycle",fontsize=7); fig.supylabel("Remaining useful life (cycles)",fontsize=7)
fig.tight_layout(pad=0.3,rect=[0.03,0.02,1,0.99])
savef(fig,os.path.join(F_ED,"ExtendedDataFig9.png"),bbox_inches="tight"); plt.close(fig)

# ===== Forward projection to 80% (per replicate, log axis) =====
fig,axes=plt.subplots(2,3,figsize=(180*MM,120*MM))
for ri,(mat,info) in enumerate(MATS.items()):
    fn,cols,w,step,DROP,xmax,title,ylab=info; df=pd.read_csv(os.path.join(DD,fn))
    for ci,c in enumerate(cols):
        ax=axes[ri,ci]; R=project(df[c].to_numpy(float),w,step,DROP,xmax,True)
        draw_proj(ax,R,True,ylab,showleg=(ri==0 and ci==0))
        nm="Urease beads" if mat=="Urea" else "KCoHCF electrode"
        ax.set_title("%s, rep %d: 80%% @ %s (RUL %s)"%(nm,ci+1,format(int(R["cb"]),","),format(int(R["cb"]-R["N"]),",")),fontsize=5.4)
fig.suptitle("Forward projection to 80% with conservative linear floor, near-term tracking, and future projection (log axis).",fontsize=6.2,y=1.003)
fig.tight_layout(pad=0.5,rect=[0,0.01,1,0.99])
savef(fig,os.path.join(F_ED,"forward_projection_log_perrep.png"),bbox_inches="tight"); plt.close(fig)
print("ED ok")
