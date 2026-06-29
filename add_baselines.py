"""Add Wiener and Weibull baselines to the per-cycle RUL CSVs (Task 1/2/3)."""
import os, numpy as np, pandas as pd
OUT="outputs"; DD=os.path.join(os.getcwd(),"Electrode_RUL_Data")
def movmean(y,k): return pd.Series(y).rolling(k,center=False,min_periods=1).mean().to_numpy()

def wiener_rul(ys,thr,c,tmax):
    """Linear-drift Wiener model; RUL = first-passage time to the threshold."""
    c=int(min(c,len(ys)))
    if c<4: return np.nan,np.nan
    t=np.arange(1,c+1,dtype=float); slope,inter=np.polyfit(t,ys[:c],1)
    pr=float(inter+slope*c)
    if slope>=-1e-9 or pr<=thr: return (float(tmax) if pr>thr else 0.0),pr
    return float(np.clip((pr-thr)/(-slope),0,tmax)),pr

def weibull_rul(ys,thr,c,tmax,y0):
    """Weibull degradation path u(t)=(y0-y)/(y0-thr); RUL from a log-log fit."""
    c=int(min(c,len(ys)))
    if c<4: return np.nan,np.nan
    y0=float(np.nanmax(ys[:c]))                 # running peak (handles initial rise/activation)
    if y0<=thr: return 0.0,float(ys[c-1])
    t=np.arange(1,c+1,dtype=float); u=(y0-ys[:c])/(y0-thr)
    m=(u>1e-3)&(u<0.999)
    if m.sum()<3: return float(tmax),y0
    lt=np.log(t[m]); lu=np.log(u[m]); beta,inter=np.polyfit(lt,lu,1)
    if beta<=1e-6: return float(tmax),y0
    eta=np.exp(-inter/beta)
    pr=float(y0-(y0-thr)*(c/eta)**beta)
    return float(np.clip(eta-c,0,tmax)),pr

def add(df,key_cols,series_of,thr_of,y0_of,tmax_of,cyc_col,true_col,actual_col):
    df=df[~df.method.isin(["Wiener","Weibull"])].copy()
    rows=[]
    for keys,g in df[df.method=="SPI+CBU (ours)"].groupby(key_cols):
        if not isinstance(keys,tuple): keys=(keys,)
        ys=series_of(keys); thr=thr_of(keys); y0=y0_of(keys,ys); tmax=tmax_of(keys,ys)
        for _,r in g.sort_values(cyc_col).iterrows():
            c=r[cyc_col]
            for mname,(rul,pr) in [("Wiener",wiener_rul(ys,thr,c,tmax)),("Weibull",weibull_rul(ys,thr,c,tmax,y0))]:
                row={kc:kv for kc,kv in zip(key_cols,keys)}
                row.update(dict(method=mname,**{cyc_col:float(c),true_col:float(r[true_col]),"pred_RUL":rul,"pred_retention":pr,
                                actual_col:float(ys[min(int(c)-1,len(ys)-1)])}))
                rows.append(row)
    return pd.concat([df,pd.DataFrame(rows)],ignore_index=True)

# ---------- TASK 1 ----------
t1=pd.read_csv(os.path.join(OUT,"task1_curves.csv"))
el=pd.read_csv(os.path.join(DD,"all_cycling_data_long.csv"))
S1={f"{a} | {b}":movmean(gg.sort_values("cycle")["capacity_retention"].to_numpy(float),5) for (a,b),gg in el.groupby(["electrolyte","electrode"])}
N1={n:len(S1[n]) for n in S1}
t1["true_RUL"]=t1.apply(lambda r: max(N1[r.trajectory]-r.cycle,0), axis=1)
t1=add(t1,["trajectory"],lambda k:S1[k[0]],lambda k:S1[k[0]][-1],lambda k,ys:ys[0],lambda k,ys:8*len(ys),"cycle","true_RUL","actual_retention")
t1.to_csv(os.path.join(OUT,"task1_curves.csv"),index=False)

# ---------- TASK 2 ----------
t2=pd.read_csv(os.path.join(OUT,"task2_curves.csv"))
S2={n:movmean(el[(el.electrolyte==n.split(" | ")[0])&(el.electrode==n.split(" | ")[1])].sort_values("cycle")["capacity_retention"].to_numpy(float),13) for n in t2.trajectory.unique()}
EOL2={}
for n in t2.trajectory.unique():
    cb=t2[(t2.trajectory==n)&(t2.method=="SPI+CBU (ours)")]; EOL2[n]=float(cb.cycle.iloc[0]+cb.true_RUL.iloc[0])
t2=add(t2,["trajectory"],lambda k:S2[k[0]],lambda k:80.0,lambda k,ys:ys[0],lambda k,ys:8*EOL2[k[0]],"cycle","true_RUL","actual_retention")
t2.to_csv(os.path.join(OUT,"task2_curves.csv"),index=False)

# ---------- TASK 3 ----------
t3=pd.read_csv(os.path.join(OUT,"task3_curves.csv"))
M3={"KCoHCF":("KCoHCF_main_material_5000_cycles_new_triplicate.csv",["capacity_retention_1","capacity_retention_2","capacity_retention_3"],11),
    "Urea":("urea_hydrolysis_urease_beads_300_cycles_triplicate.csv",["Relative_activity_1","Relative_activity_2","Relative_activity_3"],9)}
SER3={}
for mat,(fn,cols,w) in M3.items():
    df=pd.read_csv(os.path.join(DD,fn))
    for i in range(3): SER3[(mat,i+1)]=movmean(df[cols[i]].to_numpy(float),w)
def life3(k): 
    g=t3[(t3.material==k[0])&(t3.threshold==k[1])&(t3.replicate==k[2])&(t3.method=="SPI+CBU (ours)")]; return float(g.life.iloc[0])
t3=add(t3,["material","threshold","replicate"],lambda k:SER3[(k[0],int(k[2]))],lambda k:float(k[1]),lambda k,ys:ys[0],lambda k,ys:20*life3(k),"cycle","true_RUL","actual")
# fill life/band columns for new rows
for col in ["life","band_lo","band_hi"]:
    fill=t3[t3.method=="SPI+CBU (ours)"].set_index(["material","threshold","replicate"])[col]
    t3[col]=t3.apply(lambda r: r[col] if pd.notna(r.get(col)) else fill.get((r.material,r.threshold,r.replicate),np.nan),axis=1)
t3.to_csv(os.path.join(OUT,"task3_curves.csv"),index=False)

def mae(d,tc="true_RUL"):
    return {m:round(float(np.mean(np.abs(d[d.method==m].pred_RUL-d[d.method==m][tc]))),1) for m in d.method.unique()}
print("TASK1:",mae(t1))
print("TASK2:",mae(t2))
for mat in ["KCoHCF","Urea"]:
    for thr in sorted(t3[t3.material==mat].threshold.unique()):
        print(f"TASK3 {mat} {thr}:",mae(t3[(t3.material==mat)&(t3.threshold==thr)]))
