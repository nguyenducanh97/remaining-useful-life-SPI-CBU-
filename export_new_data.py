"""Comparison data to CSV. The augmented curve files already carry Log-Bayes/Wiener/Weibull; this writes the binned average abs-error vs observed-fraction and the method-comparison MAE table for all tasks/cases/methods."""
import os, numpy as np, pandas as pd
import make_error_figs as M
OUT="outputs"

TASKS=["task1","task2","task3_KCoHCF","task3_urea"]
TLAB={"task1":"Task 1 (56 screening)","task2":"Task 2 (to 80%)","task3_KCoHCF":"Task 3 KCoHCF","task3_urea":"Task 3 urea"}

# (1) binned average (mean +/- sd) vs fraction
av=[]
for task in TASKS:
    meths,panels=M.collect(task)
    for m in meths:
        fr=[p[1][m][0] for p in panels if m in p[1]]; er=[M.err_floor(p[1][m][1]) for p in panels if m in p[1]]
        mean,sd=M.binned_mean(fr,er)
        for c,mu,s in zip(M.CEN,mean,sd):
            if np.isnan(mu): continue
            av.append(dict(task=task,method=m,fraction_bin_center_pct=float(c),mean_abs_RUL_error=round(float(mu),4),sd_abs_RUL_error=round(float(s),4)))
pd.DataFrame(av).to_csv(os.path.join(OUT,"error_vs_fraction_average.csv"),index=False)

# (2) method-comparison MAE table (per task/case/method) from the augmented curve CSVs
rows=[]
t1=pd.read_csv(os.path.join(OUT,"task1_curves.csv"))
for m in t1.method.unique():
    d=t1[t1.method==m]; e=np.abs(d.pred_RUL-d.true_RUL).replace([np.inf,-np.inf],np.nan).dropna()
    rows.append(dict(task="Task 1",case="56 panels",method=m,RUL_MAE=round(float(e.mean()),2),n=int(len(d))))
t2=pd.read_csv(os.path.join(OUT,"task2_curves.csv"))
for m in t2.method.unique():
    d=t2[t2.method==m]; e=np.abs(d.pred_RUL-d.true_RUL).replace([np.inf,-np.inf],np.nan).dropna()
    rows.append(dict(task="Task 2",case="10 panels to 80%",method=m,RUL_MAE=round(float(e.mean()),2),n=int(len(d))))
t3=pd.read_csv(os.path.join(OUT,"task3_curves.csv"))
for mat in ["KCoHCF","Urea"]:
    for thr in sorted(t3[t3.material==mat].threshold.unique()):
        sub=t3[(t3.material==mat)&(t3.threshold==thr)]
        for m in sub.method.unique():
            d=sub[sub.method==m]; e=np.abs(d.pred_RUL-d.true_RUL).replace([np.inf,-np.inf],np.nan).dropna()
            rows.append(dict(task=f"Task 3 {mat}",case=f"{int(thr)}%",method=m,RUL_MAE=round(float(e.mean()),2),n=int(len(d))))
mae=pd.DataFrame(rows)
mae.to_csv(os.path.join(OUT,"method_comparison_mae.csv"),index=False)

print("wrote: error_vs_fraction_average.csv (%d), method_comparison_mae.csv (%d)"%(len(av),len(rows)))
# quick sanity: CBU is min per case
piv=mae.pivot_table(index=["task","case"],columns="method",values="RUL_MAE")
best=piv.idxmin(axis=1)
print("CBU best in all cases:", bool((best=="SPI+CBU (ours)").all()))
print(best.to_string())
