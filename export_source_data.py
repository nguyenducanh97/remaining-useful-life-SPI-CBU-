"""Source data: one CSV per figure."""
import os, numpy as np, pandas as pd, shutil
from rul_style import OUT, DD, MATS, project, movmean
SD=os.path.join(OUT,"source_data"); os.makedirs(SD,exist_ok=True)
def w(df,name):
    try: df.to_csv(os.path.join(SD,name),index=False); print(name,len(df))
    except PermissionError: print("LOCKED (skipped, close the file):",name)
RULCOLS=["method","cycle","true_RUL","pred_RUL","RUL_lo_cal90","RUL_hi_cal90"]

t1=pd.read_csv(os.path.join(OUT,"task1_curves.csv"))
t2=pd.read_csv(os.path.join(OUT,"task2_curves.csv"))
t3=pd.read_csv(os.path.join(OUT,"task3_curves.csv"))

# ---- Fig. 7B : short-term RUL validation (NH4Cl|NaCoHCF) ----
d=t2[t2.trajectory=="NH4Cl | NaCoHCF"][RULCOLS]
w(d,"SourceData_Fig7B.csv")

# ---- forward-projection source (shared by Fig. 7C and the linear/log projection figures) ----
def projection_dataset():
    rows=[]
    for mat,info in MATS.items():
        fn,cols,wn,step,DROP,xmax,title,ylab=info; df=pd.read_csv(os.path.join(DD,fn))
        for ci,c in enumerate(cols):
            R=project(df[c].to_numpy(float),wn,step,DROP,xmax,True); rep=ci+1
            for x,y in zip(R["t"],R["ys"]): rows.append(dict(material=mat,replicate=rep,series="observed",cycle=float(x),value=float(y),ci_lo=np.nan,ci_hi=np.nan))
            for x,y in zip(R["TT"],R["PR"]): rows.append(dict(material=mat,replicate=rep,series="tracking",cycle=float(x),value=float(y),ci_lo=np.nan,ci_hi=np.nan))
            for x,y,lo,hi in zip(R["tg"],R["mm"],R["lo"],R["hi"]): rows.append(dict(material=mat,replicate=rep,series="projection",cycle=float(x),value=float(y),ci_lo=float(lo),ci_hi=float(hi)))
            for k,f in enumerate(R["fan"]):
                for x,y in zip(f["x"],f["y"]): rows.append(dict(material=mat,replicate=rep,series=f"linear_floor_step{k+1}",cycle=float(x),value=float(y),ci_lo=np.nan,ci_hi=np.nan))
            if np.isfinite(R["cb"]): rows.append(dict(material=mat,replicate=rep,series="projected_80pct_life",cycle=float(R["cb"]),value=80.0,ci_lo=np.nan,ci_hi=np.nan))
    return pd.DataFrame(rows)
proj=projection_dataset()
w(proj,"SourceData_Fig7C.csv")
w(proj,"SourceData_forward_projection_log_perrep.csv")
w(proj,"SourceData_SupplementaryFig58_59.csv")

# ---- Supplementary Fig. 54 : 56-trajectory RUL (diagonal truth already in CSV) ----
w(t1[["trajectory"]+RULCOLS],"SourceData_SupplementaryFig54.csv")
# ---- Supplementary Fig. 70 : triplicate deployed-material RUL ----
w(t3[["material","threshold","replicate","method","cycle","true_RUL","pred_RUL","RUL_lo_cal90","RUL_hi_cal90"]],"SourceData_SupplementaryFig70.csv")

# ---- Supplementary Fig. 60 : 56-trajectory capacity tracking ----
w(t1[["trajectory","method","cycle","actual_retention","pred_retention"]],"SourceData_SupplementaryFig60.csv")
# ---- Supplementary Fig. 56 RUL / Fig. 57 tracking, reaching-to-80% set ----
cov2=pd.read_csv(os.path.join(OUT,"task2_coverage_per_trajectory.csv")).rename(columns={"traj":"trajectory"})
s7a=t2[["trajectory"]+RULCOLS].merge(cov2,on="trajectory",how="left")
w(s7a,"SourceData_SupplementaryFig56.csv")
w(t2[["trajectory","method","cycle","actual_retention","pred_retention"]],"SourceData_SupplementaryFig57.csv")
# ---- Supplementary Figs 61/62 : Task 3 triplicate tracking ----
w(t3[t3.material=="KCoHCF"][["material","threshold","replicate","method","cycle","actual","pred_retention"]],"SourceData_SupplementaryFig61.csv")
w(t3[t3.material=="Urea"][["material","threshold","replicate","method","cycle","actual","pred_retention"]],"SourceData_SupplementaryFig62.csv")


# ---- GM figures ----
try: shutil.copy(os.path.join(OUT,"figs","gm_effect_task1.csv"),os.path.join(SD,"SourceData_SupplementaryFig64.csv")); print("Fig64 copied")
except PermissionError: print("LOCKED:",("Fig64"))
try: shutil.copy(os.path.join(OUT,"gm_effect_task2_panels.csv"),os.path.join(SD,"SourceData_SupplementaryFig65.csv")); print("Fig65 copied")
except PermissionError: print("LOCKED: Fig65")
g3p=pd.read_csv(os.path.join(OUT,"gm_effect_task3_panels.csv"))
w(g3p[g3p.case.str.startswith("KCoHCF")],"SourceData_SupplementaryFig66.csv")
w(g3p[g3p.case.str.startswith("Urea")],"SourceData_SupplementaryFig67.csv")
# Supplementary Fig. 55 : averaged GM ablation (combine 3 panels)
g1=pd.read_csv(os.path.join(OUT,"gm_effect_task1_average.csv")); g1.insert(0,"panel","Task 1"); g1=g1.rename(columns={"cycle":"x"})
g2=pd.read_csv(os.path.join(OUT,"gm_effect_task2.csv")); g2.insert(0,"panel","Task 2"); g2=g2.rename(columns={"frac_pct":"x"})
g3=pd.read_csv(os.path.join(OUT,"gm_effect_task3_average.csv")); g3.insert(0,"panel","Task 3 ("+g3["case"]+")"); g3=g3.rename(columns={"frac_pct":"x"}).drop(columns=["case"])
w(pd.concat([g1,g2,g3],ignore_index=True),"SourceData_SupplementaryFig55.csv")

# ---- Supplementary Fig. 68 strict-vs-soft / Fig. 69 calibration ----
w(pd.read_csv(os.path.join(OUT,"figs","gm_pushto1_impact.csv")).rename(columns={"Unnamed: 0":"metric"}),"SourceData_SupplementaryFig68.csv")
try: shutil.copy(os.path.join(OUT,"calibration_per_task.csv"),os.path.join(SD,"SourceData_SupplementaryFig69.csv")); print("Fig69 copied")
except PermissionError: print("LOCKED: Fig69")

# ---- Fig. 7D / Supplementary Fig. 63 error average ----
try:
    shutil.copy(os.path.join(OUT,"error_vs_fraction_average.csv"),os.path.join(SD,"SourceData_Fig7D.csv"))
    shutil.copy(os.path.join(OUT,"error_vs_fraction_average.csv"),os.path.join(SD,"SourceData_SupplementaryFig63.csv")); print("Fig7D/Fig63 copied")
except PermissionError: print("LOCKED: error-average")

print("\nSOURCE DATA FILES:")
for f in sorted(os.listdir(SD)): print(" ",f)
