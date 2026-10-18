"""Figures from raw files only; no curve replacement, axis manipulation or seed selection."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from common import read_json,read_csv,write_csv,save_json

LABELS={"TMAPO":"TMAPO","IPPO":"IPPO","MARATD3-Adapted":"MARATD3 (适配)","DPFL-Adapted":"DPFL (适配)",
        "TMAPO-NoAmp":"TMAPO-NoAmp","Greedy":"Greedy","Local-Only":"Local-Only",
        "TMAPO-NoPred":"TMAPO-NoPred","TMAPO-NoWait":"TMAPO-NoWait"}
COLORS={"TMAPO":"#b02a37","IPPO":"#355fa0","MARATD3-Adapted":"#247f72","DPFL-Adapted":"#ab6c29","TMAPO-NoAmp":"#ae7020",
        "Greedy":"#6b6575","Local-Only":"#424242","TMAPO-NoPred":"#9767a6","TMAPO-NoWait":"#477fbd"}
MARKERS={"TMAPO":"o","IPPO":"s","MARATD3-Adapted":"D","DPFL-Adapted":"^","TMAPO-NoAmp":"^","Greedy":"v","Local-Only":"h","TMAPO-NoPred":"P","TMAPO-NoWait":"X"}


def style():
    fonts={f.name for f in font_manager.fontManager.ttflist}
    for font in ("SimSun","Microsoft YaHei","Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Micro Hei"):
        if font in fonts:
            plt.rcParams["font.family"]=font; break
    plt.rcParams.update({"font.size":10,"axes.unicode_minus":False,"svg.fonttype":"none",
                         "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":220})


def save(fig,path):
    fig.savefig(path.with_suffix(".png"),bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"),bbox_inches="tight")
    plt.close(fig)


def plot_run(run):
    run=Path(run); style()
    plan=read_json(run/"plan.json"); kind=plan["kind"]
    figs=run/"figs"; figs.mkdir(exist_ok=True)
    allrows=[]; status=[]; selection={}
    for job in plan["jobs"]:
        p=run/job["id"]
        status.append({"job":job["id"],"complete":(p/"complete.json").exists(),"raw_train_exists":(p/"train_raw.csv").exists()})
        if (p/"test_raw.csv").exists():
            allrows.extend(dict(**job,**row) for row in read_csv(p/"test_raw.csv"))
        if (p/"validation_raw.csv").exists():
            raw=read_csv(p/"validation_raw.csv"); by={}
            for row in raw: by.setdefault(int(row["update"]),[]).append(float(row["mean_task_cost"]))
            update,cost=min(((k,float(np.mean(v))) for k,v in by.items()),key=lambda p:p[1])
            if job["method"] not in selection or cost<selection[job["method"]]["validation_cost"]:
                selection[job["method"]]={"lr":job["lr"],"update":update,"validation_cost":cost,"job":job["id"],"single_seed_pilot":True}
    write_csv(run/"status.csv",status)
    if allrows: write_csv(run/"summary_raw.csv",allrows)
    if kind in ("convergence", "baseline-tuning"):
        if all(s["complete"] for s in status): save_json(run/"selection.json",selection)
        fig,ax=plt.subplots(figsize=(8,4.6))
        for job in plan["jobs"]:
            p=run/job["id"]
            if not (p/"train_raw.csv").exists(): continue
            rows=read_csv(p/"train_raw.csv")
            x=np.array([float(r["environment_steps"]) for r in rows]); y=np.array([float(r["mean_step_reward"]) for r in rows])
            w=min(25,len(y)); smooth=np.convolve(y,np.ones(w)/w,mode="valid")
            color=COLORS[job["method"]]
            ax.plot(x,y,color=color,alpha=.18,lw=.7)
            ax.plot(x[w-1:],smooth,color=color,lw=1.8,ls="-" if job["lr"]==0 else "--",label=job["method"]+f'-LR{job["lr"]+1}')
        ax.set(xlabel="累计环境交互步数（全局时隙）",ylabel="平均每步奖励",title="算法与学习率收敛对比（单训练种子试验）")
        ax.grid(alpha=.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(1.02,1))
        fig.tight_layout(); save(fig,figs/"convergence")
        save_json(figs/"convergence_caption.json",{"raw":"thin lines","smoothing":"25 logged-update trailing mean, one point per rollout; plotted only where a full window exists",
                  "confidence_interval":"not estimable with one training seed; no fake CI", "x_includes_drain_slots":True})
        fig,ax=plt.subplots(figsize=(8,4.6))
        for job in plan["jobs"]:
            p=run/job["id"]/"validation_raw.csv"
            if not p.exists(): continue
            grouped={}
            for r in read_csv(p):
                grouped.setdefault(int(r["environment_steps"]),[]).append(float(r["mean_task_cost"]))
            x=sorted(grouped)
            ax.plot(x,[np.mean(grouped[k]) for k in x],color=COLORS[job["method"]],
                    ls="-" if job["lr"]==0 else "--",marker="o",ms=3,
                    label=job["method"]+f'-LR{job["lr"]+1}')
        reference=run/"greedy_validation.csv"
        if reference.exists():
            cost=np.mean([float(r["mean_task_cost"]) for r in read_csv(reference)])
            ax.axhline(cost,color=COLORS["Greedy"],ls=":",lw=1.8,label="Greedy（相同验证轨迹）")
        ax.set(xlabel="累计环境交互步数（全局时隙）",ylabel="独立验证平均任务成本（越低越好）",title="独立验证表现与 Greedy 参照")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18); ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(1.02,1))
        fig.tight_layout(); save(fig,figs/"validation_cost")
        completed=[]
        for method, choice in selection.items():
            p=run/choice["job"]/"test_raw.csv"
            if p.exists():
                completed.append((method,read_csv(p)))
        greedy=next((run/j["id"]/"test_raw.csv" for j in plan["jobs"] if j["method"]=="Greedy"),None)
        if greedy is not None and greedy.exists(): completed.append(("Greedy",read_csv(greedy)))
        if completed:
            fig,axes=plt.subplots(1,3,figsize=(12,3.8))
            specs=[("delay_s","完成任务平均时延（s）",1),("energy_j_per_arrival","每到达任务系统能耗（J）",1),("qos_rate","QoS 违约率（%）",100)]
            for ax,(metric,label,scale) in zip(axes,specs):
                means=[np.mean([float(r[metric]) for r in rows])*scale for _,rows in completed]
                ax.bar([m for m,_ in completed],means,color=[COLORS[m] for m,_ in completed],width=.55)
                ax.set(ylabel=label); ax.grid(axis="y",alpha=.18)
            fig.tight_layout(); save(fig,figs/"selected_test_performance")
        return
    if not allrows: return
    if kind=="privacy":
        specs=[("ql_m","平均实际位置质量损失 QL（m）"),("qos_rate","QoS 违约率（%）")]
        xlabel="批次隐私参数 ε_req"; xkey="epsilon_req"
    else:
        specs=[("delay_s","完成任务平均时延（s）"),("energy_j_per_arrival","每到达任务系统能耗（J）"),("qos_rate","QoS 违约率（%）")]
        xlabel="每区域用户数"; xkey="users_per_region"
    fig,axes=plt.subplots(1,len(specs),figsize=(5*len(specs),3.9),squeeze=False)
    summary=[]
    for method in dict.fromkeys(j["method"] for j in plan["jobs"]):
        for x in sorted({float(r[xkey]) for r in allrows if r["method"]==method}):
            subset=[r for r in allrows if r["method"]==method and float(r[xkey])==x]
            s=dict(method=method,**{xkey:x},test_traces=len(subset))
            for metric in ("delay_s","energy_j_per_arrival","qos_rate","ql_m","publication_ratio","report_weighted_batch_n","mean_task_cost","completion_rate"):
                vals=[float(r[metric]) for r in subset if r[metric] not in (None,"")]
                s[metric]=float(np.mean(vals)) if vals else None
                s[metric+"_trace_sd"]=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            summary.append(s)
        subset=[r for r in summary if r["method"]==method]
        for ax,(metric,ylabel) in zip(axes[0],specs):
            y=[np.nan if r[metric] is None else r[metric]*(100 if metric=="qos_rate" else 1) for r in subset]
            ax.plot([r[xkey] for r in subset],y,marker=MARKERS[method],ms=5 if method=="TMAPO" else 4,
                    markerfacecolor=COLORS[method] if method=="TMAPO" else "white",
                    lw=1.6,color=COLORS[method],label=LABELS[method])
            ax.set(xlabel=xlabel,ylabel=ylabel); ax.grid(alpha=.18)
            ax.set_xticks(sorted({float(j[xkey]) for j in plan["jobs"]}))
    for i,ax in enumerate(axes[0]):
        ax.set_title(f"({chr(97+i)}) "+("位置报告质量" if kind=="privacy" and i==0 else "QoS 违约率" if specs[i][0]=="qos_rate" else "任务时延" if specs[i][0]=="delay_s" else "系统能耗"))
    write_csv(run/"summary.csv",summary)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=min(3,len(labels)),frameon=False)
    fig.tight_layout(rect=(0,0,1,.84)); save(fig,figs/kind)
    save_json(figs/"caption.json",{"evidence":"single training seed; do not interpret test-trace SD as training uncertainty",
              "missing_points":"not completed, or undefined when no published reports/no completions; never filled",
              "privacy_scope":"fixed-public-membership conditional batch only", "all_jobs_complete":all(s["complete"] for s in status)})
