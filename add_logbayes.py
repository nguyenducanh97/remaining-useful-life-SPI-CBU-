import os, numpy as np, pandas as pd
OUT="outputs"; DD=os.path.join(os.getcwd(),"Electrode_RUL_Data")
def movmean(y,k): return pd.Series(y).rolling(k,center=False,min_periods=1).mean().to_numpy()

def naive_log(ys,thr,c,tmax):
    """Naive log refit up to cycle c, extrapolated to the threshold."""
    c=int(c); c=min(c,len(ys))
    if c<4: return np.nan,np.nan
    t=np.arange(1,c+1,dtype=float); x=np.log(1+t); b,a=np.polyfit(x,ys[:c],1)  # y=a+b*x ; polyfit returns [slope,intercept]
    pr=a+b*np.log(1+c)
    if b>=-1e-9: return float(min(tmax,tmax)),float(pr)   # not decaying -> cannot reach -> cap
    tcross=np.exp((thr-a)/b)-1
    return float(np.clip(tcross-c,0,tmax)),float(pr)

# ---------- TASK 2 ----------
t2=pd.read_csv(os.path.join(OUT,"task2_curves.csv"))
t2=t2[t2.method!="Log-Bayes"]  # idempotent
el=pd.read_csv(os.path.join(DD,"all_cycling_data_long.csv"))
series={f"{a} | {b}":gg.sort_values("cycle")["capacity_retention"].to_numpy(float) for (a,b),gg in el.groupby(["electrolyte","electrode"])}
W2=13; THR2=80.0; rows=[]
for n in t2.trajectory.unique():
    cb=t2[(t2.trajectory==n)&(t2.method=="SPI+CBU (ours)")].sort_values("cycle")
    ys=movmean(series[n],W2); eol=float(cb.cycle.iloc[0]+cb.true_RUL.iloc[0]); tmax=8*eol
    for _,r in cb.iterrows():
        c=r.cycle; rul,pr=naive_log(ys,THR2,c,tmax)
        rows.append(dict(trajectory=n,method="Log-Bayes",cycle=float(c),
            actual_retention=float(ys[min(int(c)-1,len(ys)-1)]),true_RUL=float(r.true_RUL),
            pred_RUL=rul,pred_retention=pr))
t2=pd.concat([t2,pd.DataFrame(rows)],ignore_index=True)
t2.to_csv(os.path.join(OUT,"task2_curves.csv"),index=False)

# ---------- TASK 3 ----------
t3=pd.read_csv(os.path.join(OUT,"task3_curves.csv")); t3=t3[t3.method!="Log-Bayes"]
MATS3={"KCoHCF":("KCoHCF_main_material_5000_cycles_new_triplicate.csv",["capacity_retention_1","capacity_retention_2","capacity_retention_3"],11),
       "Urea":("urea_hydrolysis_urease_beads_300_cycles_triplicate.csv",["Relative_activity_1","Relative_activity_2","Relative_activity_3"],9)}
rows=[]
for mat,(fn,cols,w) in MATS3.items():
    df=pd.read_csv(os.path.join(DD,fn)); sm={i:movmean(df[cols[i]].to_numpy(float),w) for i in range(3)}
    sub=t3[t3.material==mat]
    for (thr,rep),g in sub.groupby(["threshold","replicate"]):
        cb=g[g.method=="SPI+CBU (ours)"].sort_values("cycle"); ys=sm[int(rep)-1]
        life=float(cb.life.iloc[0]); bl=cb.band_lo.iloc[0]; bh=cb.band_hi.iloc[0]; tmax=20*life
        for _,r in cb.iterrows():
            c=r.cycle; rul,pr=naive_log(ys,float(thr),c,tmax)
            rows.append(dict(material=mat,threshold=thr,replicate=rep,method="Log-Bayes",life=life,band_lo=bl,band_hi=bh,
                cycle=float(c),actual=float(ys[min(int(c)-1,len(ys)-1)]),true_RUL=float(r.true_RUL),pred_RUL=rul,pred_retention=pr))
t3=pd.concat([t3,pd.DataFrame(rows)],ignore_index=True)
t3.to_csv(os.path.join(OUT,"task3_curves.csv"),index=False)

def mae(d,truecol,mcol="method"):
    out={}
    for m in d[mcol].unique():
        dm=d[d[mcol]==m]; out[m]=float(np.mean(np.abs(dm.pred_RUL-dm[truecol])))
    return out
print("TASK2 MAE:",{k:round(v,1) for k,v in mae(t2,"true_RUL").items()})
for mat in ["KCoHCF","Urea"]:
    for thr in sorted(t3[t3.material==mat].threshold.unique()):
        d=t3[(t3.material==mat)&(t3.threshold==thr)]
        print(f"TASK3 {mat} {thr}: ",{k:round(v,1) for k,v in mae(d,"true_RUL").items()})
