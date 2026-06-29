"""Per-panel RMSE & MAE for every method, all tasks: RUL error and tracking error."""
import os, numpy as np, pandas as pd
OUT="outputs"
def m_mae(e): e=pd.Series(e).replace([np.inf,-np.inf],np.nan).dropna(); return (round(float(e.mean()),3),round(float(np.sqrt((e**2).mean())),3),int(len(e))) if len(e) else (np.nan,np.nan,0)
rows=[]
def add(task,panel,d,truecol,actualcol):
    for m in d.method.unique():
        dm=d[d.method==m]
        rmae,rrmse,n=m_mae(np.abs(dm.pred_RUL-dm[truecol]))
        tmae,trmse,_=m_mae(np.abs(dm.pred_retention-dm[actualcol]))
        rows.append(dict(task=task,panel=panel,method=m,n=n,RUL_MAE=rmae,RUL_RMSE=rrmse,track_MAE=tmae,track_RMSE=trmse))

t1=pd.read_csv(os.path.join(OUT,"task1_curves.csv"))
for tr,g in t1.groupby("trajectory"): add("Task 1",tr,g,"true_RUL","actual_retention")
t2=pd.read_csv(os.path.join(OUT,"task2_curves.csv"))
for tr,g in t2.groupby("trajectory"): add("Task 2",tr,g,"true_RUL","actual_retention")
t3=pd.read_csv(os.path.join(OUT,"task3_curves.csv"))
for (mat,thr,rep),g in t3.groupby(["material","threshold","replicate"]):
    add(f"Task 3 {mat}",f"{int(thr)}% rep {int(rep)}",g,"true_RUL","actual")
M=pd.DataFrame(rows)
M.to_csv(os.path.join(OUT,"per_panel_metrics.csv"),index=False)

# per-task average across panels (mean of panel metrics)
agg=M.groupby(["task","method"]).agg(RUL_MAE=("RUL_MAE","mean"),RUL_RMSE=("RUL_RMSE","mean"),
     track_MAE=("track_MAE","mean"),track_RMSE=("track_RMSE","mean"),n_panels=("panel","nunique")).round(3).reset_index()
agg.to_csv(os.path.join(OUT,"per_task_metrics_summary.csv"),index=False)
print("per_panel_metrics.csv",len(M),"rows ; per_task_metrics_summary.csv",len(agg),"rows")
print(agg[agg.method.isin(["SPI+CBU (ours)","SPI+TBU"])].to_string(index=False))
