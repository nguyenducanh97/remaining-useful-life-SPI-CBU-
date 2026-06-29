"""Conformal calibration reliability, averaged across tasks (raw vs conformal coverage)."""
import os, re, numpy as np, pandas as pd
OUT="outputs"; DD=os.path.join(os.getcwd(),"Electrode_RUL_Data")
# reuse the Task 3 engine
src=open("task3_triplicate.py").read().split("MATS={")[0].replace('HERE=os.path.dirname(os.path.abspath(__file__))','HERE=os.getcwd()')
g={}; exec(src,g)
basis=g["basis"]; sbl=g["sbl"]; online_update=g["online_update"]; life_of=g["life_of"]; ppc=g["ppc"]; crossing=g["crossing"]; movmean=g["movmean"]
PRIOR_SCALE=g["PRIOR_SCALE"]; NSTEPS=g["NSTEPS"]; N0FRAC=g.get("N0FRAC",0)
LV=[0.5,0.68,0.8,0.9,0.95]; ZR={0.5:0.4098,0.68:0.6044,0.8:0.7791,0.9:1.0,0.95:1.1915}  # z_p / z_0.90

def run_traj(ys,thr,life,prior,band,tmax,kind,ff):
    H=life; step=max(1,round(H/NSTEPS)); n0=max(step,int(N0FRAC*H)); cyc=list(range(n0,int(H)+1,step))
    if not cyc: cyc=[int(H)]
    mu_p,Sg,s2=prior; Sg=Sg*PRIOR_SCALE; prev_c=0; out=[]
    for c in cyc:
        if c>len(ys): break
        Sg=Sg/ff
        tnew=np.arange(prev_c+1,c+1,dtype=float)
        mu_pos,Sg_pos=online_update(mu_p,Sg,s2,ys[prev_c:c],basis(tnew,kind))
        if np.any(~np.isfinite(mu_pos)): mu_pos,Sg_pos=mu_p,Sg
        R=[max(band[0],c),max(band[1],c+1)]; v,pdf,mu_u,Sg_u=ppc(mu_pos,Sg_pos,R,kind,thr,c,tmax)
        if v>0:
            mu_p,Sg=mu_u,Sg_u
            if len(pdf)>=5: out.append((life-c, np.asarray(pdf,float)))
        else: mu_p,Sg=mu_pos,Sg_pos*10
        prev_c=c
    return out

def reliab_from_pdf(per_traj):
    raw={p:[] for p in LV}; scores=[]
    for steps in per_traj:
        sc=[]; rh={p:[] for p in LV}
        for true,pdf in steps:
            med=np.median(pdf); hw=max((np.percentile(pdf,84)-np.percentile(pdf,16))/2,1e-6); sc.append(abs(true-med)/hw)
            for p in LV:
                lo=np.percentile(pdf,50-100*p/2); hi=np.percentile(pdf,50+100*p/2); rh[p].append(lo<=true<=hi)
        scores.append(np.array(sc))
        for p in LV: raw[p].append(np.mean(rh[p]) if rh[p] else np.nan)
    return raw,scores

def conformal_cov(scores):
    cov={p:[] for p in LV}
    for i in range(len(scores)):
        cal=np.concatenate([scores[j] for j in range(len(scores)) if j!=i and len(scores[j])])
        if len(cal)==0 or len(scores[i])==0:
            for p in LV: cov[p].append(np.nan)
            continue
        for p in LV: cov[p].append(float(np.mean(scores[i]<=np.quantile(cal,p))))
    return cov

# ---------- Task 2 (10 reaching trajectories) ----------
t2=pd.read_csv(os.path.join(OUT,"task2_curves.csv")); trajs2=sorted(t2.trajectory.unique())
el=pd.read_csv(os.path.join(DD,"all_cycling_data_long.csv"))
S2={n:movmean(el[(el.electrolyte==n.split(" | ")[0])&(el.electrode==n.split(" | ")[1])].sort_values("cycle")["capacity_retention"].to_numpy(float),13) for n in trajs2}
life2={n:crossing(S2[n],80.0) for n in trajs2}; life2={n:(v if np.isfinite(v) else len(S2[n])) for n,v in life2.items()}
band2=[min(life2.values()),max(life2.values())]
def prior2(name):
    P=[sbl(S2[o][:int(life2[o])],basis(np.arange(1,int(life2[o])+1,dtype=float),"loglin")) for o in trajs2 if o!=name]
    return (np.mean([p[0] for p in P],0),np.mean([p[1] for p in P],0),float(np.mean([p[2] for p in P])))
pt2=[run_traj(S2[n],80.0,life2[n],prior2(n),band2,2000,"loglin",1.0) for n in trajs2]
raw2,sc2=reliab_from_pdf(pt2); con2=conformal_cov(sc2)
print("task2 done; n traj", len(pt2))

# ---------- Task 3 KCoHCF & Urea ----------
M3={"KCoHCF":("KCoHCF_main_material_5000_cycles_new_triplicate.csv",["capacity_retention_1","capacity_retention_2","capacity_retention_3"],11,[90.0,87.0],8000,0.75),
    "Urea":("urea_hydrolysis_urease_beads_300_cycles_triplicate.csv",["Relative_activity_1","Relative_activity_2","Relative_activity_3"],9,[90.0,86.0],1500,0.8)}
res3={}
for mat,(fn,cols,w,thrs,tmax,ff) in M3.items():
    df=pd.read_csv(os.path.join(DD,fn)); sm={i:movmean(df[cols[i]].to_numpy(float),w) for i in range(3)}
    per=[]
    for thr in thrs:
        lives={i:crossing(sm[i],thr) for i in range(3)}; band=[min(lives.values()),max(lives.values())]
        for i in range(3):
            others=[j for j in range(3) if j!=i]
            P=[sbl(sm[j][:int(lives[j])],basis(np.arange(1,int(lives[j])+1,dtype=float),"loglin")) for j in others]
            pri=(np.mean([p[0] for p in P],0),np.mean([p[1] for p in P],0),float(np.mean([p[2] for p in P])))
            per.append(run_traj(sm[i],thr,lives[i],pri,band,tmax,"loglin",ff))
    raw,sc=reliab_from_pdf(per); con=conformal_cov(sc); res3[mat]=(raw,con)
    print(mat,"done; n series",len(per))

# ---------- Task 1 (from stored raw 90% interval) ----------
t1=pd.read_csv(os.path.join(OUT,"task1_curves.csv")); cb=t1[t1.method=="SPI+CBU (ours)"].copy()
raw1={p:[] for p in LV}; sc1=[]
for tr,gg in cb.groupby("trajectory"):
    gg=gg.sort_values("cycle"); N=gg.cycle.max(); true=np.clip(N-gg.cycle.values,0,None); pred=gg.pred_RUL.values
    hw=np.maximum((gg.RUL_hi90.values-gg.RUL_lo90.values)/2,1e-6); res=np.abs(true-pred)
    sc1.append(res/hw); rh={p:[] for p in LV}
    for p in LV: rh[p]=(res<=hw*ZR[p])
    for p in LV: raw1[p].append(float(np.mean(rh[p])))
con1=conformal_cov(sc1)
print("task1 done; n traj",len(sc1))

# ---------- assemble + save ----------
def ms(d):  # mean,std across trajectories per level (ignoring nan)
    return {p:(float(np.nanmean(d[p])),float(np.nanstd(d[p]))) for p in LV}
TASKS={"Task 1 (56 screen)":(ms(raw1),ms(con1)),
       "Task 2 (10 to 80%)":(ms(raw2),ms(con2)),
       "Task 3 KCoHCF":(ms(res3["KCoHCF"][0]),ms(res3["KCoHCF"][1])),
       "Task 3 PePurease":(ms(res3["Urea"][0]),ms(res3["Urea"][1]))}
rows=[]
for t,(r,c) in TASKS.items():
    for p in LV: rows.append(dict(task=t,nominal_pct=int(p*100),raw_mean=round(100*r[p][0],1),raw_std=round(100*r[p][1],1),conformal_mean=round(100*c[p][0],1),conformal_std=round(100*c[p][1],1)))
pd.DataFrame(rows).to_csv(os.path.join(OUT,"calibration_per_task.csv"),index=False)
import pickle; pickle.dump(TASKS,open(os.path.join(OUT,"_calib_tasks.pkl"),"wb"))
print("saved calibration_per_task.csv")
print(pd.DataFrame(rows).to_string(index=False))

# ---------- calibration reliability figure (averaged across tasks) ----------
import matplotlib.pyplot as plt
from rul_style import PINK, GREY, MM, F_SI, frame, savef
x=np.array([p*100 for p in LV]); order=list(TASKS.keys())
raw_m=np.array([[TASKS[t][0][p][0] for p in LV] for t in order])
con_m=np.array([[TASKS[t][1][p][0] for p in LV] for t in order])
rm=raw_m.mean(0)*100; rs=raw_m.std(0)*100; cm=con_m.mean(0)*100; cs=con_m.std(0)*100
fig,ax=plt.subplots(figsize=(88*MM,82*MM))
ax.plot([40,100],[40,100],ls=(0,(3,2)),color="0.55",lw=0.8,label="Ideal")
ax.fill_between(x,np.clip(rm-rs,0,100),np.clip(rm+rs,0,100),color=GREY,alpha=0.18,lw=0)
ax.plot(x,rm,marker="^",ms=4.5,mew=0,lw=1.1,color=GREY,label="Raw model")
ax.fill_between(x,np.clip(cm-cs,0,100),np.clip(cm+cs,0,100),color=PINK,alpha=0.16,lw=0)
ax.plot(x,cm,marker="o",ms=4.5,mew=0,lw=1.4,color=PINK,label="After conformal")
ax.set_xlim(45,100); ax.set_ylim(0,100)
ax.set_xlabel("Nominal coverage (%)"); ax.set_ylabel("Empirical coverage (%)")
ax.legend(loc="upper left",fontsize=6); frame(ax)
fig.suptitle("Supplementary Fig. 31 | Conformal calibration reliability, averaged across tasks (mean +/-1 STD). Grey, raw model interval; pink, after split-conformal; dashed, ideal.",fontsize=5.8,y=1.02)
fig.tight_layout(pad=0.5)
savef(fig,os.path.join(F_SI,"SupplementaryFig31.png"),bbox_inches="tight"); plt.close(fig)
print("saved SupplementaryFig31")
