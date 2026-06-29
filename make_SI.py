import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from rul_style import *

GMY=(-0.06,1.06)
def gm_panel(ax,x,no,yes):
    ax.plot(x,no,marker="^",ms=3.0,mew=0,lw=0.6,color=GREY,zorder=3)
    ax.plot(x,yes,marker="o",ms=3.0,mew=0,lw=0.7,color=PINK,zorder=4)
    ax.set_ylim(*GMY)

# ===== 4 named Task-4 projection figures =====
for LOGX,tag in [(True,"log"),(False,"lin")]:
    for mat,info in MATS.items():
        fn,cols,w,step,DROP,xmax,title,ylab=info; df=pd.read_csv(os.path.join(DD,fn))
        fig,axes=plt.subplots(1,3,figsize=(180*MM,64*MM))
        for ci,c in enumerate(cols):
            ax=axes[ci]; R=project(df[c].to_numpy(float),w,step,DROP,xmax,LOGX)
            draw_proj(ax,R,LOGX,ylab,showleg=(ci==0))
            ax.set_title("Replicate %d: 80%% @ %s (RUL %s, tracking RMSE %.2f%%)"%(ci+1,format(int(R["cb"]),","),format(int(R["cb"]-R["N"]),","),R["trk"]),fontsize=5.0)
        nm="Urease beads" if mat=="Urea" else "KCoHCF electrode"
        fig.suptitle("%s: forward projection to 80%% (%s axis) with conservative linear floor, near-term tracking, and 90%% interval."%(nm,tag),fontsize=6.2,y=1.02)
        fig.tight_layout(pad=0.5,rect=[0,0,1,0.95])
        fn={("KCoHCF","lin"):"SupplementaryFig20",("Urea","lin"):"SupplementaryFig21",("KCoHCF","log"):"forward_projection_KCoHCF_log",("Urea","log"):"forward_projection_Urea_log"}[(mat,tag)]
        savef(fig,os.path.join(F_SI,fn+".png"),bbox_inches="tight"); plt.close(fig)
print("task4 ok")

# ===== Capacity tracking across all 56 screening trajectories =====
t1=pd.read_csv(os.path.join(OUT,"task1_curves.csv")); trajs=sorted(t1.trajectory.unique())
ncol=8; nrow=int(np.ceil(len(trajs)/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(180*MM,(180/ncol*nrow*0.95)*MM)); axes=axes.ravel()
for i,tr in enumerate(trajs):
    ax=axes[i]; d=t1[t1.trajectory==tr]; cbu=d[d.method=="SPI+CBU (ours)"].sort_values("cycle")
    if len(cbu)==0: ax.axis("off"); continue
    ax.plot(cbu.cycle,cbu.actual_retention,color=DARK,lw=0.8,ls=(0,(3,2)),zorder=6)
    plot_methods(ax,d,"pred_retention",top=None,ms=2.1,lwn=0.5)
    ax.set_title(tr,fontsize=4.3,pad=1.5); ax.tick_params(labelsize=4); frame(ax)
for j in range(len(trajs),len(axes)): axes[j].axis("off")
fig.suptitle("Supplementary Fig. 22 | Capacity tracking across all 56 screening trajectories. SPI+CBU (pink circles), SPI+TBU (blue squares), linear (grey triangles); black dashed, measured.",fontsize=5.6,y=1.005)
fig.supxlabel("Cycle",fontsize=7); fig.supylabel("Capacity retention (%)",fontsize=7)
fig.tight_layout(pad=0.3,rect=[0.024,0.02,1,0.985])
savef(fig,os.path.join(F_SI,"SupplementaryFig22.png"),bbox_inches="tight"); plt.close(fig)

# ===== RUL + capacity tracking to 80% on the reaching trajectories =====
t2=pd.read_csv(os.path.join(OUT,"task2_curves.csv")); cov2=pd.read_csv(os.path.join(OUT,"task2_coverage_per_trajectory.csv")).set_index("traj")["cov90"].to_dict()
tr2=sorted(t2.trajectory.unique())
def t2grid(band,fname,ylab,title,showcov=False):
    fig,axes=plt.subplots(2,5,figsize=(180*MM,80*MM)); axes=axes.ravel()
    for i,tr in enumerate(tr2):
        ax=axes[i]; d=t2[t2.trajectory==tr]; cbu=d[d.method=="SPI+CBU (ours)"].sort_values("cycle")
        if band:
            top=rul_panel_lims(ax,cbu.cycle.values,cbu.true_RUL.values,cbu.pred_RUL.values)
            ax.fill_between(cbu.cycle,np.clip(cbu.RUL_lo_cal90,0,top),np.clip(cbu.RUL_hi_cal90,0,top),color=PINK,alpha=0.16,lw=0,zorder=1)
            ax.plot(cbu.cycle,cbu.true_RUL,color=DARK,lw=0.9,ls=(0,(3,2)),zorder=6)
            plot_methods(ax,d,"pred_RUL",top=top,ms=3.0,lwn=0.55)
        else:
            ax.plot(cbu.cycle,cbu.actual_retention,color=DARK,lw=0.9,ls=(0,(3,2)),zorder=6)
            plot_methods(ax,d,"pred_retention",top=None,ms=3.0,lwn=0.55); ax.set_ylim(78,101)
        ttl=tr+(" (cov %.0f%%)"%cov2.get(tr,0) if showcov else "")
        ax.set_title(ttl,fontsize=4.7,pad=1.5); ax.tick_params(labelsize=4.5); frame(ax)
    fig.suptitle(title,fontsize=5.7,y=1.01)
    fig.supxlabel("Cycle",fontsize=7); fig.supylabel(ylab,fontsize=7)
    fig.tight_layout(pad=0.3,rect=[0.024,0.02,1,0.98])
    savef(fig,os.path.join(F_SI,fname),bbox_inches="tight"); plt.close(fig)
t2grid(True,"SupplementaryFig18.png","Remaining useful life to 80% (cycles)",
       "Supplementary Fig. 18 | RUL prediction to 80% on the reaching trajectories. SPI+CBU (pink circles), SPI+TBU (blue squares), linear (grey triangles); black dashed, ground truth; pink, calibrated 90% interval; per-panel coverage shown.",showcov=True)
t2grid(False,"SupplementaryFig19.png","Capacity retention (%)",
       "Supplementary Fig. 19 | Capacity tracking to 80% on the reaching trajectories. SPI+CBU (pink circles), SPI+TBU (blue squares), linear (grey triangles); black dashed, measured.")
print("retention + to-80% panels ok")


# ===== Task 3 capacity/activity tracking (3 replicates, all methods) =====
t3r=pd.read_csv(os.path.join(OUT,"task3_curves.csv"))
def t3ret(mat,thrs,fname,title,ylab):
    fig,axes=plt.subplots(2,3,figsize=(150*MM,92*MM))
    for ri,thr in enumerate(thrs):
        for ci,rep in enumerate([1,2,3]):
            ax=axes[ri,ci]; d=t3r[(t3r.material==mat)&(t3r.threshold==thr)&(t3r.replicate==rep)]
            cbu=d[d.method=="SPI+CBU (ours)"].sort_values("cycle")
            if len(cbu)==0: ax.axis("off"); continue
            ax.plot(cbu.cycle,cbu.actual,color=DARK,lw=0.9,ls=(0,(3,2)),zorder=6)
            plot_methods(ax,d,"pred_retention",top=None,ms=2.4,lwn=0.6)
            ax.axhline(thr,color=GREY,lw=0.7,ls=(0,(2,2)))
            lo=min(float(thr)-1.0,float(np.nanmin(cbu.actual))-0.5); ax.set_ylim(lo,101)
            nm="Urease beads" if mat=="Urea" else "KCoHCF electrode"
            ax.set_title("%s, %d%%, rep %d"%(nm,int(thr),rep),fontsize=5.3); ax.tick_params(labelsize=5); frame(ax)
    axes[0,2].legend(handles=legend_handles_scatter(truth_label="Measured",interval=False),loc="lower left",fontsize=4.4,handlelength=1.3,labelspacing=0.25,borderpad=0.3)
    fig.suptitle(title,fontsize=6.2,y=1.0)
    fig.supxlabel("Cycle",fontsize=7); fig.supylabel(ylab,fontsize=7)
    fig.tight_layout(pad=0.3,rect=[0.03,0.02,1,0.98])
    savef(fig,os.path.join(F_SI,fname),bbox_inches="tight"); plt.close(fig)
t3ret("KCoHCF",[90.0,87.0],"SupplementaryFig23.png","Supplementary Fig. 23 | Triplicate capacity retention tracking for the KCoHCF electrode (90%, 87%). SPI+CBU pink circles, SPI+TBU blue squares, linear grey triangles, Log-Bayes orange diamonds, Wiener teal, Weibull purple; black dashed, measured.","Capacity retention (%)")
t3ret("Urea",[90.0,86.0],"SupplementaryFig24.png","Supplementary Fig. 24 | Triplicate activity tracking for the PePurease bead (90%, 86%). Same key as Supplementary Fig. 23.","Relative activity (%)")
print("task3 retention tracking ok")

# ===== GM ablation averaged across tasks =====
g1=pd.read_csv(os.path.join(OUT,"gm_effect_task1_average.csv")); g2=pd.read_csv(os.path.join(OUT,"gm_effect_task2.csv")); g3=pd.read_csv(os.path.join(OUT,"gm_effect_task3_average.csv"))
fig,axes=plt.subplots(1,3,figsize=(180*MM,56*MM))
ax=axes[0]
ax.plot(g1.cycle,g1.acc_noGM_mean,color=GREY,lw=1.1,label="Without GM"); ax.fill_between(g1.cycle,g1.acc_noGM_mean-g1.acc_noGM_sd,g1.acc_noGM_mean+g1.acc_noGM_sd,color=GREY,alpha=0.15,lw=0)
ax.plot(g1.cycle,g1.acc_GM_mean,color=PINK,lw=1.3,label="With GM"); ax.fill_between(g1.cycle,g1.acc_GM_mean-g1.acc_GM_sd,g1.acc_GM_mean+g1.acc_GM_sd,color=PINK,alpha=0.15,lw=0)
ax.set_title("Task 1 (56 trajectories)"); ax.set_xlabel("Cycle"); ax.set_ylabel("PPC acceptance"); ax.set_ylim(*GMY); ax.legend(loc="center right",fontsize=5.5); frame(ax)
ax=axes[1]
ax.plot(g2.frac_pct,g2.acc_noGM_mean,color=GREY,lw=1.1,label="Without GM"); ax.fill_between(g2.frac_pct,g2.acc_noGM_mean-g2.acc_noGM_sd,g2.acc_noGM_mean+g2.acc_noGM_sd,color=GREY,alpha=0.15,lw=0)
ax.plot(g2.frac_pct,g2.acc_GM_mean,color=PINK,lw=1.3,label="With GM"); ax.fill_between(g2.frac_pct,g2.acc_GM_mean-g2.acc_GM_sd,g2.acc_GM_mean+g2.acc_GM_sd,color=PINK,alpha=0.15,lw=0)
ax.set_title("Task 2 (10 trajectories)"); ax.set_xlabel("Observation fraction (%)"); ax.set_ylabel("PPC acceptance"); ax.set_ylim(*GMY); ax.legend(loc="center right",fontsize=5.5); frame(ax)
ax=axes[2]; cm=[PINK,BLUE,OLIVE,"#5BA053"]
for ci,case in enumerate(list(g3.case.unique())):
    dd=g3[g3.case==case]; ax.plot(dd.frac_pct,dd.acc_GM_mean,color=cm[ci%4],lw=1.0,label=case); ax.plot(dd.frac_pct,dd.acc_noGM_mean,color=cm[ci%4],lw=0.8,ls=(0,(2,2)),alpha=0.7)
ax.set_title("Task 3 (solid, with GM; dashed, without)",fontsize=6); ax.set_xlabel("Observation fraction (%)"); ax.set_ylabel("PPC acceptance"); ax.set_ylim(*GMY); ax.legend(loc="lower right",fontsize=4.3); frame(ax)
fig.suptitle("Extended Data Fig. 10 | Gaussian-mutation ablation: PPC acceptance without vs with GM (mean +/-1 s.d.).",fontsize=6.2,y=1.04)
fig.tight_layout(pad=0.5,rect=[0,0,1,0.96])
savef(fig,os.path.join(F_ED,"ExtendedDataFig10.png"),bbox_inches="tight"); plt.close(fig)

# ===== Per-trajectory GM effect, 56 screening trajectories =====
gp1=pd.read_csv(os.path.join(OUT,"figs","gm_effect_task1.csv")); names=sorted(gp1.name.unique())
nrow=int(np.ceil(len(names)/8))
fig,axes=plt.subplots(nrow,8,figsize=(180*MM,(180/8*nrow*0.9)*MM)); axes=axes.ravel()
for i,nm in enumerate(names):
    ax=axes[i]; d=gp1[gp1.name==nm].sort_values("cycle"); gm_panel(ax,d.cycle,d.acc_noGM,d.acc_GM)
    ax.set_title(nm,fontsize=4.0,pad=1.2); ax.tick_params(labelsize=3.6); frame(ax)
for j in range(len(names),len(axes)): axes[j].axis("off")
fig.suptitle("Supplementary Fig. 26 | Per-trajectory Gaussian-mutation effect on PPC acceptance for 56 trajectories. Pink circles, with GM; grey triangles, without GM.",fontsize=5.7,y=1.005)
fig.supxlabel("Cycle",fontsize=7); fig.supylabel("PPC acceptance",fontsize=7)
fig.tight_layout(pad=0.3,rect=[0.024,0.02,1,0.985])
savef(fig,os.path.join(F_SI,"SupplementaryFig26.png"),bbox_inches="tight"); plt.close(fig)

# ===== Per-trajectory / per-replicate GM effect (Task 2, Task 3) =====
gp2=pd.read_csv(os.path.join(OUT,"gm_effect_task2_panels.csv")); tr2b=sorted(gp2.traj.unique())
fig,axes=plt.subplots(2,5,figsize=(180*MM,80*MM)); axes=axes.ravel()
for i,tr in enumerate(tr2b):
    ax=axes[i]; d=gp2[gp2.traj==tr].sort_values("cycle"); gm_panel(ax,d.cycle,d.acc_noGM,d.acc_GM)
    ax.set_title(tr,fontsize=4.8,pad=1.5); ax.tick_params(labelsize=4.5); frame(ax)
fig.suptitle("Supplementary Fig. 27 | Per-trajectory GM effect on PPC acceptance for 10 trajectories to 80% threshold. Pink circles, with GM; grey triangles, without GM.",fontsize=5.8,y=1.01)
fig.supxlabel("Cycle",fontsize=7); fig.supylabel("PPC acceptance",fontsize=7)
fig.tight_layout(pad=0.3,rect=[0.024,0.02,1,0.98])
savef(fig,os.path.join(F_SI,"SupplementaryFig27.png"),bbox_inches="tight"); plt.close(fig)

gp3=pd.read_csv(os.path.join(OUT,"gm_effect_task3_panels.csv"))
def t3gm(cases,fname,suptitle):
    fig,axes=plt.subplots(2,3,figsize=(150*MM,90*MM))
    for ri,case in enumerate(cases):
        for ci,rep in enumerate([1,2,3]):
            ax=axes[ri,ci]; d=gp3[(gp3.case==case)&(gp3.replicate==rep)].sort_values("frac_pct"); gm_panel(ax,d.frac_pct,d.acc_noGM,d.acc_GM)
            ax.set_title("%s, rep %d"%(case,rep),fontsize=5.2); ax.tick_params(labelsize=4.6); frame(ax)
    fig.suptitle(suptitle,fontsize=5.9,y=1.0)
    fig.supxlabel("Observation fraction (%)",fontsize=7); fig.supylabel("PPC acceptance",fontsize=7)
    fig.tight_layout(pad=0.3,rect=[0.03,0.02,1,0.98])
    savef(fig,os.path.join(F_SI,fname),bbox_inches="tight"); plt.close(fig)
t3gm(["KCoHCF 90%","KCoHCF 87%"],"SupplementaryFig28.png","Supplementary Fig. 28 | Per-replicate Gaussian-mutation effect on the PPC acceptance for KCoHCF electrode (90% and 87% thresholds). Pink circles, with GM; grey triangles, without GM.")
t3gm(["Urea 90%","Urea 86%"],"SupplementaryFig29.png","Supplementary Fig. 29 | Per-replicate Gaussian-mutation effect on the PPC acceptance for PePurease beads (90% and 86% thresholds). Pink circles, with GM; grey triangles, without GM.")
print("GM panels ok")

# ===== Soft vs strict GM =====
pi=pd.read_csv(os.path.join(OUT,"figs","gm_pushto1_impact.csv"),index_col=0)
fig,axes=plt.subplots(1,3,figsize=(180*MM,58*MM)); x=[0,1]; lab=["Soft\n(triggered)","Strict\n(forced)"]
for ax,(row,ylab,ttl) in zip(axes,[("acc","PPC acceptance","Constraint satisfaction"),("RUL_MAE","RUL MAE (cycles)","Accuracy (unchanged)"),("track_RMSE","Tracking RMSE (%)","Tracking (collapses)")]):
    ax.bar(x,[pi.loc[row,"trig"],pi.loc[row,"strict"]],color=[PINK,GREY],width=0.6); ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=5.5); ax.set_ylabel(ylab); ax.set_title(ttl); frame(ax)
axes[0].set_ylim(0,1.08)
fig.suptitle("Supplementary Fig. 30 | Soft versus strict Gaussian mutation performance.",fontsize=5.8,y=1.04)
fig.tight_layout(pad=0.5,rect=[0,0,1,0.95]); savef(fig,os.path.join(F_SI,"SupplementaryFig30.png"),bbox_inches="tight"); plt.close(fig)

print("GM soft-vs-strict ok")
