"""C1: fixed300 synthetic users, one report/member, two mechanisms, epsilon sweep.

This controlled mechanism benchmark does not simulate service delay or claim
complete-transcript DP. Public region batches contain100 distinct members;
100 independent Monte Carlo repetitions are not one composed private trace.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/"model")]
import hybrid_campaign as h
from privacy import calibrate, delta_bound, planar_noise

EPS = (.1, .2, .3, .5, 1.)


def run(out, repetitions=100):
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("At least two independent repetitions are required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper_four_studies import full_config
    cfg, _ = full_config(users=300)
    identity = dict(schema="privacy-quality-C1-v1", users=300, regions=3,
        distinct_members_each_batch=100, epsilon=list(EPS), delta=cfg["delta_req"],
        repetitions=repetitions, seed=602021, projection=False,
        release="one report for each distinct region member per independent repetition",
        workload_service_simulated=False, source_hashes=h.code_hashes(),
        privacy_claim="Exploratory conditional one-batch accountant, not full trajectory DP")
    if out.exists():
        if (out/"complete.json").exists() and h.read_json(out/"manifest.json") == identity:
            return h.read_json(out/"complete.json")
        raise FileExistsError("Do not overwrite previous raw C1 results")
    out.mkdir(parents=True)
    h.write_json(out/"manifest.json", identity)
    columns = ("epsilon", "repetition", "region", "user", "n_distinct", "true_x", "true_y",
               "geoi_x", "geoi_y", "shuffle_x", "shuffle_y", "ql_geoi_m", "ql_shuffle_m",
               "epsilon0_shuffle", "batch_id", "shuffled_output_index")
    summaries = []
    with (out/"reports_raw.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for eps in EPS:
            eps0 = calibrate(100, eps, cfg["delta_req"], cfg["accountant_grid"], cfg["accountant_max_eps0"])
            bound = delta_bound(eps0, 100, eps)
            if bound > cfg["delta_req"]+1e-12 or eps0 < eps:
                raise AssertionError("Accountant feasibility failed")
            geoi, shuffled = [], []
            for rep in range(repetitions):
                a, b = [], []
                for region in range(3):
                    rng = np.random.default_rng(np.random.SeedSequence([602021, rep, region]))
                    draws = rng.random((100, 2))
                    angle, radius = 2*np.pi*draws[:,0], cfg["region_radius_m"]*np.sqrt(draws[:,1])
                    locations = np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
                    locations += np.asarray(cfg["region_centers_m"][region])
                    standard = planar_noise(rng, 1., size=100)
                    permutation = rng.permutation(100)
                    output_index = np.argsort(permutation)
                    one = locations+standard/(eps/(2*cfg["region_radius_m"]))
                    two = locations+standard/(eps0/(2*cfg["region_radius_m"]))
                    q1, q2 = np.linalg.norm(one-locations, axis=1), np.linalg.norm(two-locations, axis=1)
                    members = [region*100+i for i in range(100)]
                    if len(set(members)) != 100:
                        raise AssertionError("Not an actual distinct-member batch")
                    for i, user in enumerate(members):
                        writer.writerow(dict(epsilon=eps, repetition=rep, region=region, user=user,
                            n_distinct=len(set(members)), true_x=locations[i,0], true_y=locations[i,1],
                            geoi_x=one[i,0], geoi_y=one[i,1], shuffle_x=two[i,0], shuffle_y=two[i,1],
                            ql_geoi_m=q1[i], ql_shuffle_m=q2[i], epsilon0_shuffle=eps0,
                            batch_id=f"eps{eps}_rep{rep}_region{region}",
                            shuffled_output_index=int(output_index[i])))
                    a.extend(q1)
                    b.extend(q2)
                geoi.append(float(np.mean(a)))
                shuffled.append(float(np.mean(b)))
            summaries.append(dict(epsilon=eps, users=300, effective_batch_n=100, epsilon0_shuffle=eps0,
                ql_geoi_m=float(np.mean(geoi)), ql_shuffle_m=float(np.mean(shuffled)),
                geoi_ci95=1.96*float(np.std(geoi, ddof=1))/np.sqrt(repetitions),
                shuffle_ci95=1.96*float(np.std(shuffled, ddof=1))/np.sqrt(repetitions),
                improvement_fraction=1-float(np.mean(shuffled))/float(np.mean(geoi)),
                accountant_delta=bound, reports_per_method=300*repetitions,
                repetitions=repetitions, theoretical_geoi_ql_m=4*cfg["region_radius_m"]/eps,
                theoretical_shuffle_ql_m=4*cfg["region_radius_m"]/eps0))
    with (out/"summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    x = [r["epsilon"] for r in summaries]
    for label, key, ci, color in (("Geo-I", "ql_geoi_m", "geoi_ci95", "#24659e"),
                                  ("Geo-I + Shuffle", "ql_shuffle_m", "shuffle_ci95", "#bd543f")):
        y = np.asarray([r[key] for r in summaries])
        error = np.asarray([r[ci] for r in summaries])
        ax.plot(x, y, marker="o", color=color, label=label, linewidth=2)
        ax.fill_between(x, y-error, y+error, color=color, alpha=.12)
    ax.set(xlabel="Privacy budget epsilon", ylabel="Raw location quality loss QL (m)",
           title="300 users | 100 distinct reports per public-region batch")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.text(.5, .01, "Controlled one-release quality comparison; exploratory conditional accountant; no delay/energy claim.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out/"privacy_C1.png", dpi=220)
    plt.close(fig)
    audit = dict(passed=True, samples=5*300*repetitions, actual_distinct_members=100,
                 paired_standard_noise=True, no_projection=True, dp_proven=False,
                 service_delay_measured=False, raw_sha256=h.file_hash(out/"reports_raw.csv"),
                 summary_sha256=h.file_hash(out/"summary.csv"))
    h.write_json(out/"complete.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2))
