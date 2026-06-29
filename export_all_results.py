"""Export plot-ready CSVs for Tasks 1-4 + coverage + GM (average & per panel)."""
import re,numpy as np,pandas as pd,os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DD="Electrode_RUL_Data"
OUT="outputs"; FIG="outputs/figs"; os.makedirs(OUT,exist_ok=True)
src=open("Electrode_RUL_Colab.py").read(); cells=re.split(r'\n# %%(?: \[markdown\])?\n',src)
g=dict(np=np,pd=pd,os=os,plt=plt)
def run(c):
    try: exec(c,g); return True
    except Exception as e: return False
for cell in cells:
    if "drive.mount" in cell or cell.strip().startswith("from google.colab"):
        cell=re.sub(r'^from google\.colab.*$','',cell,flags=re.M); cell=re.sub(r'^drive\.mount.*$','',cell,flags=re.M)
        cell=re.sub(r'^DATA_DIR\s*=.*$',f'DATA_DIR={DD!r}',cell,flags=re.M); cell=re.sub(r'^OUT_DIR\s*=.*$',f'OUT_DIR={OUT!r}',cell,flags=re.M); run(cell); g["FIG"]=FIG; continue
    if any(k in cell for k in ["## 6)","8b)","prognostic","Uncertainty calibration","GaussianProcess","Robustness","## 9)","run_task3","run_task4","run_task2_gm","# ## 8c","# ## 8d","# ## 8e"]): continue
    if any(k in cell for k in ["def basis","def sbl","def walk(","def eol_crossing"]): run(cell)

# ---- TASK 1 ----
walk=g["walk"]; basis=g["basis"]; sbl=g["sbl"]; movmean=g["movmean"]; np=g["np"]
SMOOTH_A=5; KIND="loglin"; TMAX_A=2000; H=100; NSTEPS_A=8
el=pd.read_csv(os.path.join(DD,"all_cycling_data_long.csv"))
ALL=[(f"{a} | {b}",gg.sort_values("cycle")["capacity_retention"].to_numpy(float)) for (a,b),gg in el.groupby(["electrolyte","electrode"])]
sm={n:movmean(y,SMOOTH_A) for n,y in ALL}
fits={n:sbl(sm[n],basis(np.arange(1,H+1,dtype=float),KIND)) for n,_ in ALL}
fits_lin={n:sbl(sm[n],basis(np.arange(1,H+1,dtype=float),"linear")) for n,_ in ALL}
def loo(fd,name,d):
    mus=np.array([fd[n][0] for n,_ in ALL if n!=name]); Sgs=np.array([fd[n][1] for n,_ in ALL if n!=name]); s2s=np.array([fd[n][2] for n,_ in ALL if n!=name])
    return mus.mean(0),np.cov(mus,rowvar=False)+Sgs.mean(0)+1e-8*np.eye(d),float(s2s.max())
def eolc(ys,thr):
    for i in range(1,len(ys)):
        if ys[i]<=thr: return i+(ys[i-1]-thr)/(ys[i-1]-ys[i]+1e-12)
    return len(ys)
rows=[]
for name,yraw in ALL:
    ys=sm[name]; thr=ys[-1]; life=eolc(ys,thr); el_e,el_d=name.split(" | ")
    cfgs={"SPI+CBU (ours)":dict(kind=KIND,prior=loo(fits,name,3),R_pop=(0.8*H,1.3*H),mode="cbu"),
          "SPI+TBU":dict(kind=KIND,prior=loo(fits,name,3),R_pop=None,mode="tbu"),
          "Lin":dict(kind="linear",prior=loo(fits_lin,name,2),R_pop=None,mode="tbu"),
          "Log-Bayes":dict(kind=None,prior=None,R_pop=None,mode=None,bayes_refit="log")}
    for mn,cf in cfgs.items():
        o=walk(y=ys,thr=thr,tmax=TMAX_A,H=H,nsteps=NSTEPS_A,**cf)
        for k in range(len(o["C"])):
            cyc=o["C"][k]; rl=o["Rlo"][k]; rh=o["Rhi"][k]
            rows.append(dict(trajectory=name,electrolyte=el_e,electrode=el_d,method=mn,cycle=float(cyc),
                actual_retention=float(ys[min(int(cyc)-1,len(ys)-1)]),true_RUL=float(max(life-cyc,0)),
                pred_RUL=float(o["RUL"][k]),RUL_lo90=(float(rl) if np.isfinite(rl) else np.nan),RUL_hi90=(float(rh) if np.isfinite(rh) else np.nan),
                pred_retention=float(o["P"][k]),retention_lo=float(o["Plo"][k]),retention_hi=float(o["Phi"][k])))
t1=pd.DataFrame(rows)
# conformal-calibrate the CBU RUL interval (pooled over the gradual subset, life>=10) -> ~90% coverage
cb=t1[(t1.method=="SPI+CBU (ours)")&t1.RUL_lo90.notna()].copy()
life_by_traj=cb.groupby("trajectory").true_RUL.max(); grad_traj=set(life_by_traj[life_by_traj>=10].index)
cbg=cb[cb.trajectory.isin(grad_traj)]
hw=np.maximum((cbg.RUL_hi90-cbg.RUL_lo90)/2,1e-6); Q1=float(np.quantile(np.abs(cbg.true_RUL-cbg.pred_RUL)/hw,0.90))
hw_all=np.maximum((t1.RUL_hi90-t1.RUL_lo90)/2,1e-6)
t1["RUL_lo_cal90"]=np.where(t1.method=="SPI+CBU (ours)",np.maximum(t1.pred_RUL-Q1*hw_all,0),np.nan)
t1["RUL_hi_cal90"]=np.where(t1.method=="SPI+CBU (ours)",t1.pred_RUL+Q1*hw_all,np.nan)
t1.to_csv(os.path.join(OUT,"task1_curves.csv"),index=False); print("task1_curves.csv",len(t1),"| conformal Q1=",round(Q1,2))
cb=t1[(t1.method=="SPI+CBU (ours)")&t1.RUL_lo_cal90.notna()].copy(); cb["cov"]=((cb.true_RUL>=cb.RUL_lo_cal90)&(cb.true_RUL<=cb.RUL_hi_cal90))
t1cov=cb.groupby(["trajectory","electrode"]).agg(life_max=("true_RUL","max"),cov90_cal_pct=("cov",lambda s:round(100*s.mean())),n=("cov","size")).reset_index()
t1cov.to_csv(os.path.join(OUT,"task1_coverage.csv"),index=False); print("task1_coverage.csv ; calibrated gradual(life>=10) mean cov=%.0f%%"%t1cov[t1cov.life_max>=10].cov90_cal_pct.mean())

# ---- TASK 2 ----
t2=[c for c in cells if "Task 2 figures saved" in c][0]; run(t2.split("for view in")[0])
SEL=g["SEL"]; W2=g["W2"]; Q90=g["Q90"]
rows=[]
for nm,ys,eol in SEL:
    for mn,o in W2[nm].items():
        C,RUL,RT,RET,RLO,RHI=o; hw=np.maximum((RHI-RLO)/2,1e-6)
        for k in range(len(C)):
            cyc=C[k]; row=dict(trajectory=nm,method=mn,cycle=float(cyc),actual_retention=float(ys[min(int(cyc)-1,len(ys)-1)]),
                true_RUL=float(max(eol-cyc,0)),pred_RUL=float(RUL[k]),pred_retention=float(RET[k]) if k<len(RET) else np.nan)
            if mn.startswith("SPI+CBU"): row["RUL_lo_cal90"]=float(max(RUL[k]-Q90*hw[k],0)); row["RUL_hi_cal90"]=float(RUL[k]+Q90*hw[k])
            rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT,"task2_curves.csv"),index=False); print("task2_curves.csv",len(rows))

# ---- TASK 3 ----
t3src=open("task3_triplicate.py").read().split("MATS={")[0].replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=os.getcwd()")
n3={}; exec(t3src,n3)
basis=n3["basis"];sbl=n3["sbl"];walk3=n3["walk"];ens_walk=n3["ens_walk"];movmean=n3["movmean"];crossing=n3["crossing"]
MATS3={"KCoHCF":dict(fn="KCoHCF_main_material_5000_cycles_new_triplicate.csv",cols=["capacity_retention_1","capacity_retention_2","capacity_retention_3"],w=11,thr=[90.0,87.0],tmax=8000,kind="loglin",ff=0.75,ens=False,ci_ens=True),
       "Urea":dict(fn="urea_hydrolysis_urease_beads_300_cycles_triplicate.csv",cols=["Relative_activity_1","Relative_activity_2","Relative_activity_3"],w=9,thr=[90.0,86.0],tmax=1500,kind="loglin",ff=0.8,ens=True,pkind=None)}
METH3={"SPI+CBU (ours)":"cbu","SPI+TBU":"tbu","Lin":"lin"}
rows=[]
for mat,cf in MATS3.items():
    df=pd.read_csv(os.path.join(DD,cf["fn"])); cols=cf["cols"]; raw={i:df[cols[i]].to_numpy(float) for i in range(3)}; sm2={i:movmean(raw[i],cf["w"]) for i in raw}
    for thr in cf["thr"]:
        lives={i:crossing(sm2[i],thr) for i in raw}; band=[min(lives.values()),max(lives.values())]
        def agg(i,kind):
            others=[j for j in range(3) if j!=i]; P=[sbl(sm2[j][:int(lives[j])],basis(np.arange(1,int(lives[j])+1,dtype=float),kind)) for j in others]
            return (np.mean([p[0] for p in P],0),np.mean([p[1] for p in P],0),float(np.mean([p[2] for p in P])))
        res={}; cbu_true={}; cbu_hw={}
        for i in range(3):
            ys=sm2[i]; life=lives[i]; tg=np.arange(1,len(ys)+1); trueR=np.maximum(life-tg,0)
            for mn,mode in METH3.items():
                if mode=="cbu" and cf.get("ens"): o=ens_walk(ys,thr,{k:agg(i,k) for k in ("loglin","log")},band,cf["tmax"],life,ff=cf["ff"],point_kind=cf.get("pkind"))
                elif mode=="cbu" and cf.get("ci_ens"):
                    o=walk3(ys,thr,cf["kind"],agg(i,cf["kind"]),band,cf["tmax"],life,mode="cbu",ff=cf["ff"]); ol=walk3(ys,thr,"log",agg(i,"log"),band,cf["tmax"],life,mode="cbu",ff=cf["ff"]); o["RLO"]=np.minimum(o["RLO"],ol["RLO"]); o["RHI"]=np.maximum(o["RHI"],ol["RHI"])
                else:
                    kk="linear" if mode=="lin" else cf["kind"]; o=walk3(ys,thr,kk,agg(i,kk),band,cf["tmax"],life,mode="tbu",ff=cf["ff"])
                res[(i,mn)]=o
            oc=res[(i,"SPI+CBU (ours)")]; cbu_true[i]=np.interp(oc["C"],tg,trueR); cbu_hw[i]=np.maximum((oc["RHI"]-oc["RLO"])/2,1e-6)
        sc=np.concatenate([np.abs(cbu_true[j]-res[(j,"SPI+CBU (ours)")]["RUL"])/cbu_hw[j] for j in range(3)]); Q=float(np.quantile(sc,0.90))
        for i in range(3):
            ys=sm2[i]; life=lives[i]; tg=np.arange(1,len(ys)+1)
            for mn in METH3:
                o=res[(i,mn)]
                for k in range(len(o["C"])):
                    cyc=o["C"][k]; row=dict(material=mat,threshold=thr,replicate=i+1,method=mn,life=round(life,1),band_lo=round(band[0],1),band_hi=round(band[1],1),
                        cycle=float(cyc),actual=float(ys[min(int(cyc)-1,len(ys)-1)]),true_RUL=float(max(life-cyc,0)),pred_RUL=float(o["RUL"][k]),pred_retention=float(o["RET"][k]))
                    if mn.startswith("SPI+CBU"):
                        hw=max((o["RHI"][k]-o["RLO"][k])/2,1e-6); row["RUL_lo_cal90"]=float(max(o["RUL"][k]-Q*hw,0)); row["RUL_hi_cal90"]=float(o["RUL"][k]+Q*hw)
                    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT,"task3_curves.csv"),index=False); print("task3_curves.csv",len(rows))

# ---- TASK 4 ----
n4={}; exec(open("task4_triplicate.py").read().split("MATS=[")[0].replace("HERE=os.path.dirname(os.path.abspath(__file__))","HERE=os.getcwd()"),n4)
movmean=n4["movmean"];_fit=n4["_fit"];_pred=n4["_pred"]
MATS4=[("KCoHCF","KCoHCF_main_material_5000_cycles_new_triplicate.csv",["capacity_retention_1","capacity_retention_2","capacity_retention_3"],21,0.85,400000),
       ("Urea","urea_hydrolysis_urease_beads_300_cycles_triplicate.csv",["Relative_activity_1","Relative_activity_2","Relative_activity_3"],21,0.5,4000)]
obs=[]; proj=[]
for name,fn,cols,w,DROP,xmax in MATS4:
    df=pd.read_csv(os.path.join(DD,fn))
    for i in range(3):
        yraw=df[cols[i]].to_numpy(float); ys=movmean(yraw,w); N=len(ys)
        for c in range(N): obs.append(dict(material=name,replicate=i+1,cycle=c+1,raw=float(yraw[c]),smoothed=float(ys[c])))
        cut=int(0.7*N); sH=int(DROP*cut); bH,s2H,XiH=_fit(sH,cut,ys,yraw); th=np.arange(cut+1,N+1); mH,seH=_pred(bH,s2H,XiH,th)
        q=np.quantile(np.abs(yraw[cut:N]-mH)/np.maximum(seH,1e-9),0.90)  # split-conformal interval factor (paper: q)
        sF=int(DROP*N); bF,s2F,XiF=_fit(sF,N,ys,yraw); tg=np.geomspace(N,xmax,800); mm,se=_pred(bF,s2F,XiF,tg)
        for k in range(len(tg)): proj.append(dict(material=name,replicate=i+1,cycle=float(tg[k]),pred_retention=float(mm[k]),ci_lo=float(mm[k]-q*se[k]),ci_hi=float(mm[k]+q*se[k])))
pd.DataFrame(obs).to_csv(os.path.join(OUT,"task4_observed.csv"),index=False)
pd.DataFrame(proj).to_csv(os.path.join(OUT,"task4_projection_curve.csv"),index=False)
print("task4_observed.csv",len(obs),"| task4_projection_curve.csv",len(proj))

# ---- COVERAGE + GM AVERAGES ----
m3=pd.read_csv(os.path.join(OUT,"task3_triplicate_metrics.csv"))
m3[m3.method=="SPI+CBU (ours)"][["material","threshold","replicate","RUL_MAE","track_RMSE","cov90"]].to_csv(os.path.join(OUT,"task3_coverage.csv"),index=False)
g1=pd.read_csv(os.path.join(FIG,"gm_effect_task1.csv"))
g1.groupby("cycle").agg(acc_noGM_mean=("acc_noGM","mean"),acc_noGM_sd=("acc_noGM","std"),acc_GM_mean=("acc_GM","mean"),acc_GM_sd=("acc_GM","std")).reset_index().to_csv(os.path.join(OUT,"gm_effect_task1_average.csv"),index=False)
g3=pd.read_csv(os.path.join(OUT,"gm_effect_task3_panels.csv"))
g3.groupby(["case","frac_pct"]).agg(acc_noGM_mean=("acc_noGM","mean"),acc_noGM_sd=("acc_noGM","std"),acc_GM_mean=("acc_GM","mean"),acc_GM_sd=("acc_GM","std")).reset_index().to_csv(os.path.join(OUT,"gm_effect_task3_average.csv"),index=False)
print("task3_coverage.csv, gm_effect_task1_average.csv, gm_effect_task3_average.csv written")
print("DONE")
