"""Task 3: triplicate tracking and RUL at reachable thresholds (SPI+CBU, SPI+TBU, Linear), with split-conformal 90% intervals."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(0)
HERE=os.path.dirname(os.path.abspath(__file__)); DD=os.path.join(HERE,"Electrode_RUL_Data")
FIG=os.path.join(HERE,"outputs","figs"); MET=os.path.join(HERE,"outputs"); os.makedirs(FIG,exist_ok=True)
N_PART=400; PSI=0.5; NSTEPS=24; PRIOR_SCALE=50.0; N0FRAC=0.0; COV=0.90
#N_PART=L particles, PSI=psi PPC threshold, ff=lambda_f forgetting, thr=D EoL threshold, band/R_pop=[T_min,T_max], q=split-conformal factor

def basis(t,kind,t0=1.0):
    t=np.atleast_1d(np.asarray(t,float)); x=t
    if kind=="log":    return np.column_stack([np.ones_like(x),np.log(x+t0)])
    if kind=="loglin": return np.column_stack([np.ones_like(x),np.log(x+t0),x])
    if kind=="linear": return np.column_stack([np.ones_like(x),x])
def movmean(y,k): return pd.Series(y).rolling(k,center=False,min_periods=1).mean().to_numpy()
def sbl(y,Phi,mi=300):
    n,m=Phi.shape; al=np.ones(m)*1e-2; s2=np.var(y,ddof=1) if len(y)>1 else np.var(y)+1e-6; mu=np.zeros(m); Sg=np.eye(m)
    for _ in range(mi):
        Sg=np.linalg.inv(Phi.T@Phi/max(s2,1e-9)+np.diag(al)); mu=Sg@Phi.T@y/max(s2,1e-9)
        g=1-al*np.diag(Sg); a2=np.minimum(g/(mu**2+1e-10),1e10); s2n=np.sum((y-Phi@mu)**2)/max(n-np.sum(g),1e-6)
        if np.all(np.abs(a2-al)<1e-6) and abs(s2n-s2)<1e-6: break
        al,s2=a2,max(s2n,1e-9)
    return mu,Sg,s2
def online_update(mu,Sg,s2,yn,Pn):
    Si=np.linalg.pinv(Sg); Sp=np.linalg.pinv(Pn.T@Pn/max(s2,1e-9)+Si); return Sp@(Pn.T@yn/max(s2,1e-9)+Si@mu),Sp
def life_of(W,kind,thr,tmax):
    grid=np.unique(np.concatenate([np.arange(1,min(int(tmax),3000)+1,dtype=float),np.geomspace(2,tmax,1000)]))
    Y=W@basis(grid,kind).T; hit=Y<=thr; idx=np.where(hit.any(1),hit.argmax(1),-1); return np.where(idx>=0,grid[idx],np.inf)
def draw(mu,Sg,n=N_PART):
    Sg=(Sg+Sg.T)/2+1e-10*np.eye(len(mu))
    try: return rng.multivariate_normal(mu,Sg,n,method="cholesky")
    except np.linalg.LinAlgError: return rng.multivariate_normal(mu,Sg,n)
def ppc(mu,Sg,R,kind,thr,tk,tmax,psi=PSI,npart=N_PART):
    W=draw(mu,Sg,npart); lv=life_of(W,kind,thr,tmax); keep=(lv>=float(min(R)))&(lv<=float(max(R))); v=int(keep.sum())
    if v==0: return 0.0,np.array([]),mu,Sg
    kept=W[keep]; rul=lv[keep]-tk; mu_n=kept.mean(0)
    if v<2: Sg_n=Sg+np.eye(len(mu))*2e-4
    else:
        Sg_n=np.cov(kept,rowvar=False)
        if v<psi*npart: Sg_n=Sg_n+np.eye(len(mu))*2e-4
    return v/npart,rul,mu_n,(Sg_n+Sg_n.T)/2
def crossing(y,thr):
    for i in range(1,len(y)):
        if y[i]<=thr: return i+(y[i-1]-thr)/(y[i-1]-y[i]+1e-12)
    return np.nan

def walk(y,thr,kind,prior,R_pop,tmax,H,mode="cbu",ff=0.8):
    step=max(1,round(H/NSTEPS)); n0c=max(step,int(N0FRAC*H)); cyc=list(range(n0c,int(H)+1,step))
    if not cyc: cyc=[int(H)]
    if cyc[-1]<H: cyc.append(int(H))
    mu_p,Sg,s2=prior; Sg=Sg*PRIOR_SCALE; prev_c=0; prev=None; C=[];RUL=[];RLO=[];RHI=[];RT=[];RET=[]
    for c in cyc:
        Sg=Sg/ff
        RET.append(float((basis(np.array([float(c)]),kind)@mu_p)[0])); RT.append(float(c))
        tnew=np.arange(prev_c+1,c+1,dtype=float); mu_pos,Sg_pos=online_update(mu_p,Sg,s2,y[prev_c:c],basis(tnew,kind))
        if np.any(~np.isfinite(mu_pos)): mu_pos,Sg_pos=mu_p,Sg
        if mode=="cbu":
            R=[max(R_pop[0],c),max(R_pop[1],c+1)]; v,pdf,mu_u,Sg_u=ppc(mu_pos,Sg_pos,R,kind,thr,c,tmax)
            if v==0: mu_p,Sg=mu_pos,Sg_pos*10; r=(prev-step) if prev is not None else 0.5*(R[0]+R[1])-c; rl=max(R[0]-c,0); rh=max(R[1]-c,0)
            else: mu_p,Sg=mu_u,Sg_u; r=float(np.median(pdf)); rl=float(np.percentile(pdf,5)); rh=float(np.percentile(pdf,95))
        else:
            mu_p,Sg=mu_pos,Sg_pos; lv=life_of(draw(mu_p,Sg),kind,thr,tmax); fin=lv[np.isfinite(lv)]
            if len(fin)>=10: r=float(np.median(fin))-c; rl=float(np.percentile(fin,5))-c; rh=float(np.percentile(fin,95))-c
            else: r=(prev if prev is not None else tmax-c); rl=r; rh=r
        r=min(max(r,0),tmax); C.append(float(c)); RUL.append(r); RLO.append(max(rl,0)); RHI.append(max(rh,0)); prev=r; prev_c=c
    return dict(C=np.array(C),RUL=np.array(RUL),RLO=np.array(RLO),RHI=np.array(RHI),RT=np.array(RT),RET=np.array(RET))

def ens_walk(y,thr,priors,R_pop,tmax,H,ff=0.8,kinds=("loglin","log"),point_kind=None):
    """Band-constrained posterior per basis, pooled across the ensemble;
    their crossing particles. Interval = pooled 5-95 percentile (captures model-form
    uncertainty, widens toward the slow replicate). Point = median of point_kind's
    particles if given (keeps a single accurate basis, e.g. loglin for KCoHCF), else
    the pooled median. Tracking RET = point_kind curve if given, else evidence-mean."""
    step=max(1,round(H/NSTEPS)); n0c=max(step,int(N0FRAC*H)); cyc=list(range(n0c,int(H)+1,step))
    if not cyc: cyc=[int(H)]
    if cyc[-1]<H: cyc.append(int(H))
    M={k:[priors[k][0].copy(),priors[k][1]*PRIOR_SCALE,priors[k][2]] for k in kinds}
    prev_c=0; C=[];RUL=[];RLO=[];RHI=[];RT=[];RET=[]
    for c in cyc:
        pooled=[]; perk={}; rets={}; ev={}
        for k in kinds:
            m,S,s2=M[k]; S=S/ff
            mp,Sp=online_update(m,S,s2,y[prev_c:c],basis(np.arange(prev_c+1,c+1,dtype=float),k))
            if np.any(~np.isfinite(mp)): mp,Sp=m,S
            R=[max(R_pop[0],c),max(R_pop[1],c+1)]
            P=draw(mp,Sp); lv=life_of(P,k,thr,tmax); inb=(lv>=R[0])&(lv<=R[1]); fin=lv[np.isfinite(lv)]
            if inb.sum()>=10: M[k]=[P[inb].mean(0),np.cov(P[inb],rowvar=False),s2]; sel=lv[inb]
            else: M[k]=[mp,Sp*10,s2]; sel=np.clip(fin,R[0],R[1]) if len(fin)>=10 else np.array([0.5*(R[0]+R[1])])
            pooled.append(sel); perk[k]=sel
            rets[k]=float((basis(np.array([float(c)]),k)@M[k][0])[0])
            yh=basis(np.arange(1,c+1,dtype=float),k)@M[k][0]; ev[k]=np.sum((y[:c]-yh)**2)
        emin=min(ev.values()); w={k:np.exp(-0.5*(ev[k]-emin)/max(np.var(y[:c]),1e-3)) for k in ev}; ws=sum(w.values()); w={k:w[k]/ws for k in w}
        pool=np.concatenate(pooled)
        pt=np.median(perk[point_kind]) if point_kind else np.median(pool)
        C.append(float(c)); RUL.append(min(max(pt-c,0),tmax))
        RLO.append(max(np.percentile(pool,5)-c,0)); RHI.append(max(np.percentile(pool,95)-c,0))
        RET.append(rets[point_kind] if point_kind else sum(w[k]*rets[k] for k in kinds)); RT.append(float(c)); prev_c=c
    return dict(C=np.array(C),RUL=np.array(RUL),RLO=np.array(RLO),RHI=np.array(RHI),RT=np.array(RT),RET=np.array(RET))

MATS={
 "KCoHCF":dict(fn="KCoHCF_main_material_5000_cycles_new_triplicate.csv",cols=["capacity_retention_1","capacity_retention_2","capacity_retention_3"],
               ylab="capacity retention (%)",w=11,thr=[90.0,87.0],tmax=8000,kind="loglin",ff=0.75,ens=False,ci_ens=True),
 "Urea":  dict(fn="urea_hydrolysis_urease_beads_300_cycles_triplicate.csv",cols=["Relative_activity_1","Relative_activity_2","Relative_activity_3"],
               ylab="relative activity (%)",w=9,thr=[90.0,86.0],tmax=1500,kind="loglin",ff=0.8,ens=True,pkind=None),
}
METH={"SPI+CBU (ours)":dict(mode="cbu",col="tab:red",mk="o",lin=False,z=9,open=False),
      "SPI+TBU":dict(mode="tbu",col="tab:blue",mk="s",lin=False,z=6,open=True),
      "Lin":dict(mode="tbu",col="tab:green",mk="^",lin=True,z=5,open=True)}

rows=[]
for mat,cf in MATS.items():
    df=pd.read_csv(os.path.join(DD,cf["fn"])); cols=cf["cols"]
    raw={i:df[cols[i]].to_numpy(float) for i in range(3)}; sm={i:movmean(raw[i],cf["w"]) for i in raw}
    for thr in cf["thr"]:
        lives={i:crossing(sm[i],thr) for i in raw}; band=[min(lives.values()),max(lives.values())]
        def agg(i,kind):
            others=[j for j in range(3) if j!=i]
            P=[sbl(sm[j][:int(lives[j])],basis(np.arange(1,int(lives[j])+1,dtype=float),kind)) for j in others]
            return (np.mean([p[0] for p in P],0),np.mean([p[1] for p in P],0),float(np.mean([p[2] for p in P])))
        res={mn:{} for mn in METH}; cbu_true={}; cbu_hw={}
        for i in range(3):
            ys=sm[i]; life=lives[i]; tgf=np.arange(1,len(ys)+1); trueRULf=np.maximum(life-tgf,0)
            for mn,mc in METH.items():
                if mc["mode"]=="cbu" and cf.get("ens"):
                    pri={k:agg(i,k) for k in ("loglin","log")}
                    o=ens_walk(ys,thr,pri,band,cf["tmax"],life,ff=cf["ff"],point_kind=cf.get("pkind"))
                elif mc["mode"]=="cbu" and cf.get("ci_ens"):
                    o=walk(ys,thr,cf["kind"],agg(i,cf["kind"]),band,cf["tmax"],life,mode="cbu",ff=cf["ff"])
                    olog=walk(ys,thr,"log",agg(i,"log"),band,cf["tmax"],life,mode="cbu",ff=cf["ff"])
                    o["RLO"]=np.minimum(o["RLO"],olog["RLO"]); o["RHI"]=np.maximum(o["RHI"],olog["RHI"])
                else:
                    kind="linear" if mc["lin"] else cf["kind"]; prior=agg(i,kind)
                    o=walk(ys,thr,kind,prior,band,cf["tmax"],life,mode=mc["mode"],ff=cf["ff"])
                res[mn][i]=o
            oc=res["SPI+CBU (ours)"][i]
            cbu_true[i]=np.interp(oc["C"],tgf,trueRULf); cbu_hw[i]=np.maximum((oc["RHI"]-oc["RLO"])/2,1e-6)
        # pooled split-conformal factor q
        sc_all=np.concatenate([np.abs(cbu_true[j]-res["SPI+CBU (ours)"][j]["RUL"])/cbu_hw[j] for j in range(3)])
        q=float(np.quantile(sc_all,COV)); qfac={i:q for i in range(3)}
        figR,axR=plt.subplots(1,3,figsize=(15,4.4)); figU,axU=plt.subplots(1,3,figsize=(15,4.4))
        for i in range(3):
            ys=sm[i]; life=lives[i]; tg=np.arange(1,len(ys)+1); trueRUL=np.maximum(life-tg,0)
            arR=axR[i]; arU=axU[i]
            arR.plot(tg,ys,"-",color="k",lw=1.4,zorder=5,label="actual"); arR.axhline(thr,color="0.6",ls=":",lw=0.8)
            arU.plot(tg,trueRUL,"--",color="k",lw=1.1,zorder=5,label="true RUL"); arU.axhline(0,color="0.85",lw=0.6)
            mae_rul={}; mae_ret={}
            for mn,mc in METH.items():
                o=res[mn][i]
                arR.plot(o["RT"],np.clip(o["RET"],thr-8,101),marker=mc["mk"],ms=4.5 if not mc["open"] else 4,lw=0.8,ls="--",color=mc["col"],alpha=0.9,label=mn,zorder=mc["z"],mfc="none" if mc["open"] else mc["col"],mew=1.0)
                arU.plot(o["C"],np.clip(o["RUL"],0,1.5*life),marker=mc["mk"],ms=4.5 if not mc["open"] else 4,lw=0.9 if not mc["open"] else 0.8,ls="-" if mn.startswith("SPI+CBU") else "--",color=mc["col"],alpha=0.9,label=mn,zorder=mc["z"],mfc="none" if mc["open"] else mc["col"],mew=1.0)
                tr=np.interp(o["C"],tg,trueRUL); mae_rul[mn]=np.mean(np.abs(o["RUL"]-tr)); mae_ret[mn]=np.mean(np.abs(o["RET"]-np.interp(o["RT"],tg,ys)))
                rows.append(dict(material=mat,threshold=thr,replicate=i+1,method=mn,band_lo=band[0],band_hi=band[1],true_life=life,RUL_MAE=mae_rul[mn],track_RMSE=np.sqrt(np.mean((o["RET"]-np.interp(o["RT"],tg,ys))**2))))
            oc=res["SPI+CBU (ours)"][i]; hw=np.maximum((oc["RHI"]-oc["RLO"])/2,1e-6); clo=np.clip(oc["RUL"]-qfac[i]*hw,0,1.6*life); chi=np.clip(oc["RUL"]+qfac[i]*hw,0,1.6*life)
            arU.fill_between(oc["C"],clo,chi,color="tab:red",alpha=0.15,zorder=2)
            cov=np.mean((cbu_true[i]>=oc["RUL"]-qfac[i]*hw)&(cbu_true[i]<=oc["RUL"]+qfac[i]*hw))*100
            for rr in rows[::-1]:
                if rr["material"]==mat and rr["threshold"]==thr and rr["replicate"]==i+1 and rr["method"]=="SPI+CBU (ours)": rr["cov90"]=round(cov,0); break
            txtR="capacity MAE (%)\n"+"\n".join(f"{m.split(' ')[0]}: {mae_ret[m]:.2f}" for m in METH); arR.text(0.97,0.05,txtR,transform=arR.transAxes,fontsize=6.6,ha="right",va="bottom",bbox=dict(fc="white",ec="0.7",alpha=0.85,boxstyle="round"))
            txtU="RUL MAE (cyc)\n"+"\n".join(f"{m.split(' ')[0]}: {mae_rul[m]:.0f}" for m in METH)+f"\nCBU 90%cov: {cov:.0f}%"; arU.text(0.97,0.95,txtU,transform=arU.transAxes,fontsize=6.6,ha="right",va="top",bbox=dict(fc="white",ec="0.7",alpha=0.85,boxstyle="round"))
            arR.set_title(f"{mat} rep {i+1} - thr {thr:g}% (life={life:.0f})",fontsize=9); arR.set_xlabel("cycle"); arR.set_ylabel(cf["ylab"]); arR.grid(alpha=0.3)
            arU.set_xlim(0,life*1.05); arU.set_title(f"{mat} rep {i+1} - thr {thr:g}% (life={life:.0f})",fontsize=9); arU.set_xlabel("cycle"); arU.set_ylabel("RUL (cycles)"); arU.grid(alpha=0.3)
            if i==0: arR.legend(fontsize=7.5); arU.legend(fontsize=7.5)
        tag=f"{mat}_{str(thr).replace('.','p')}"
        figR.suptitle(f"Task 3 - {mat} capacity tracking, threshold {thr:g}%",fontsize=12)
        figR.tight_layout(rect=[0,0,1,0.96]); figR.savefig(os.path.join(FIG,f"task3_{tag}_retention.png"),dpi=150); plt.close(figR)
        figU.suptitle(f"Task 3 - {mat} RUL vs cycle (CBU calibrated 90% band), threshold {thr:g}%",fontsize=12)
        figU.tight_layout(rect=[0,0,1,0.96]); figU.savefig(os.path.join(FIG,f"task3_{tag}_RUL.png"),dpi=150); plt.close(figU)
        print("saved task3_"+tag)
M=pd.DataFrame(rows); M.to_csv(os.path.join(MET,"task3_triplicate_metrics.csv"),index=False)
print("\nRUL_MAE per replicate (CBU):")
print(M[M.method=="SPI+CBU (ours)"].pivot_table(index=["material","threshold"],columns="replicate",values="RUL_MAE").round(0).to_string())
print("\nRUL_MAE mean (CBU/TBU/Lin):")
print(M.groupby(["material","threshold","method"]).RUL_MAE.mean().round(1).to_string())
print("\nCBU 90% coverage (%) per replicate:")
print(M[M.method=="SPI+CBU (ours)"].pivot_table(index=["material","threshold"],columns="replicate",values="cov90").round(0).to_string())
