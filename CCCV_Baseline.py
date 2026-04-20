"""
CC-CV Baseline Comparison
=========================
Simulates standard Constant-Current / Constant-Voltage charging using the
identical ECM physics as BMS_Pipeline.py, then compares results against the
pipeline's Pareto-optimal policies loaded from the saved dataset.

Usage:
    python cccv_baseline.py                         # default state
    python cccv_baseline.py --soc 0.3 --soh 0.88 --temp 35
    python cccv_baseline.py --soc 0.3 --soh 0.90 --temp 50

Output:
    - Console comparison table
    - cccv_baseline_results.csv  (all trajectory data)
    - cccv_comparison.png        (4-panel trajectory plot)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths — edit to match your environment ────────────────────────────────────
DATA_DIR     = "/Users/arujatiwary/Desktop/ECM_processed_cycles"
DATASET_BASE = "/Users/arujatiwary/Desktop/BMS Codes/nsga2_synthetic_dataset.csv"


def get_dataset_path(soc, soh, temp_C):
    """Mirror the state-keyed naming used by BMS_Pipeline.py.
    Uses raw sensor values (same as --soc/--soh/--temp CLI args)
    so the filename matches what the pipeline saved.
    temp_C is in Celsius; converted to Kelvin then floored at T_amb.
    """
    temp_K = max(temp_C + 273.15, BATTERY_PARAMS["T_amb"])
    tag    = f"soc{soc:.2f}_soh{soh:.2f}_temp{temp_K:.0f}"
    base   = os.path.splitext(DATASET_BASE)[0]
    return f"{base}_{tag}.csv"

# ── shared physics parameters (identical to BMS_Pipeline.py) ─────────────────
BATTERY_PARAMS = {
    "capacity_Ah": 2.3,
    "R0_nominal":  0.02,
    "R1_nominal":  0.01,
    "C1_nominal":  2000.0,
    "R_heat":      0.35,
    "V_max":       4.2,
    "V_min":       3.0,
    "V_cutoff":    2.7,
    "T_amb":       298.0,
    "T_max":       333.0,
    "R_th":        5.0,
    "C_th":        50.0,
    "dt":          1.0,
}

# CC-CV protocol parameters
CC_CV_CONFIGS = {
    "CC-CV 1C":  {"I_cc": 2.3,  "I_cutoff": 0.1},   # 1C  = 2.3 A
    "CC-CV 2C":  {"I_cc": 4.6,  "I_cutoff": 0.1},   # 2C  = 4.6 A (capped at I_MAX=4A in sim)
    "CC-CV 0.5C":{"I_cc": 1.15, "I_cutoff": 0.1},   # 0.5C = 1.15 A
}

SEP  = "=" * 64
SEP2 = "-" * 64


# ── ECM physics (copied verbatim from BMS_Pipeline.py) ───────────────────────

def _build_ocv_function(data_dir=None):
    if data_dir and os.path.isdir(data_dir):
        csv_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".csv")
        ])
        soc_all, volt_all = [], []
        for fp in csv_files[:20]:
            try:
                df_c = pd.read_csv(fp)
                df_c["mode"] = df_c["mode"].astype(str).str.lower().str.strip()
                df_dis = df_c[df_c["mode"] == "discharge"].reset_index(drop=True)
                if len(df_dis) == 0:
                    continue
                df_dis = df_dis[df_dis["Voltage_measured"] > BATTERY_PARAMS["V_cutoff"]]
                soc  = df_dis["SOC"].values
                volt = df_dis["Voltage_measured"].values
                mask = (soc > 0.15) & (soc < 0.95)
                soc_all.append(soc[mask])
                volt_all.append(volt[mask])
            except Exception:
                continue
        if soc_all:
            soc_arr  = np.concatenate(soc_all)
            volt_arr = np.concatenate(volt_all)
            bins = np.linspace(soc_arr.min(), soc_arr.max(), 50)
            soc_centers, volt_means = [], []
            for i in range(len(bins) - 1):
                idx = (soc_arr >= bins[i]) & (soc_arr < bins[i + 1])
                if np.sum(idx) > 5:
                    soc_centers.append(np.mean(soc_arr[idx]))
                    volt_means.append(np.mean(volt_arr[idx]))
            if len(soc_centers) >= 6:
                coeffs   = np.polyfit(soc_centers, volt_means, deg=5)
                ocv_poly = np.poly1d(coeffs)
                print(f"  [OCV] Fitted polynomial from {len(soc_centers)} bins")
                return lambda soc: float(ocv_poly(np.clip(soc, 0.0, 1.0)))

    print("  [OCV] Analytical fallback")
    def _analytical(soc):
        soc = np.clip(soc, 0.0, 1.0)
        return 3.0 + 1.2*soc - 0.3*np.exp(-5*soc) + 0.1*np.exp(-5*(1-soc))
    return _analytical


_OCV_FUNC = None
def _get_ocv_func():
    global _OCV_FUNC
    if _OCV_FUNC is None:
        _OCV_FUNC = _build_ocv_function(data_dir=DATA_DIR)
    return _OCV_FUNC


class BatteryECM:
    def __init__(self, Q, R0, R1, C1, dt, ocv_func, soc_init=1.0):
        self.Q = Q; self.R0 = R0; self.R1 = R1; self.C1 = C1
        self.dt = dt; self.ocv_func = ocv_func
        self.SOC = soc_init; self.V_RC = 0.0

    def step(self, I):
        self.SOC += (I * self.dt) / (3600.0 * self.Q)
        self.SOC  = np.clip(self.SOC, 0.0, 1.0)
        alpha     = np.exp(-self.dt / (self.R1 * self.C1))
        self.V_RC = alpha * self.V_RC + self.R1 * (1.0 - alpha) * I
        OCV       = self.ocv_func(self.SOC)
        V_ecm     = OCV - I * self.R0 - self.V_RC
        return self.SOC, self.V_RC, OCV, V_ecm


def thermal_step(temp, current, params, dt):
    heat_gen  = (current ** 2) * params["R_heat"]
    heat_loss = (temp - params["T_amb"]) / params["R_th"]
    return temp + (dt / params["C_th"]) * (heat_gen - heat_loss)


def degradation_step(soh, current, temp, dt):
    stress = abs(current) * max(0.0, temp - 298.0)
    return max(0.0, soh - 5e-9 * stress * dt)


# ── CC-CV simulation ──────────────────────────────────────────────────────────

def simulate_cccv(state, params, I_cc, I_cutoff, max_time_s=1200):
    """
    Simulate CC-CV charging from the given battery state.

    Protocol
    --------
    Phase 1 — Constant Current (CC):
        Apply I_cc until terminal voltage reaches V_max.
    Phase 2 — Constant Voltage (CV):
        Hold voltage at V_max; current tapers as battery fills.
        Current approximated each step via: I = (V_max - OCV) / (R0 + R1)
        Stop when I < I_cutoff or SoC >= 0.99.

    Returns
    -------
    dict with trajectory arrays and summary metrics, or None if aborted.
    """
    soc  = state["soc"]
    soh  = state["soh"]
    temp = state["temp"]
    dt   = params["dt"]
    ocv_func = _get_ocv_func()

    Q  = params["capacity_Ah"] * max(soh, 0.01)
    R0 = params["R0_nominal"]  / max(soh, 0.01)
    R1 = params["R1_nominal"]
    C1 = params["C1_nominal"]
    ecm = BatteryECM(Q=Q, R0=R0, R1=R1, C1=C1, dt=dt,
                     ocv_func=ocv_func, soc_init=soc)

    # Cap CC current at I_MAX to stay within safe hardware limits
    I_cc = min(I_cc, 4.0)

    soc_t, temp_t, soh_t, volt_t, cur_t, phase_t = [], [], [], [], [], []
    peak_temp  = temp
    phase      = "CC"
    aborted    = False

    for t in range(int(max_time_s / dt)):
        if phase == "CC":
            current = I_cc
            # Check if this step would exceed V_max; if so, switch to CV
            _, _, _, v_test = ecm.step(0)   # peek at OCV
            ecm.SOC -= (0 * dt) / (3600.0 * Q)  # undo peek (no-op)
            v_projected = ocv_func(ecm.SOC) + I_cc * R0
            if v_projected >= params["V_max"]:
                phase = "CV"
                current = I_cc  # will be recomputed below

        if phase == "CV":
            # Taper current: I = (V_max - OCV) / total_resistance
            # This approximates the CV behaviour without a closed-form solver
            OCV_now = ocv_func(ecm.SOC)
            current = max(0.0, (params["V_max"] - OCV_now) / (R0 + R1))
            current = min(current, I_cc)   # can't exceed CC phase current
            if current < I_cutoff or ecm.SOC >= 0.99:
                break   # charging complete
            current = max(0.0, current)   # clamp — never pull current

        soc, V_RC, OCV, voltage = ecm.step(current)
        soh  = degradation_step(soh, current, temp, dt)
        temp = thermal_step(temp, current, params, dt)
        if temp > peak_temp:
            peak_temp = temp

        # Safety abort
        if temp > params["T_max"]:
            aborted = True
            break

        soc_t.append(soc);    temp_t.append(temp)
        soh_t.append(soh);    volt_t.append(voltage)
        cur_t.append(current); phase_t.append(phase)

    if aborted:
        return None

    return {
        "soc_traj":     np.array(soc_t),
        "temp_traj":    np.array(temp_t),
        "soh_traj":     np.array(soh_t),
        "volt_traj":    np.array(volt_t),
        "current_traj": np.array(cur_t),
        "phase_traj":   phase_t,
        # summary metrics
        "soc_gain":        soc_t[-1] - state["soc"] if soc_t else 0,
        "peak_temp":       peak_temp,
        "soh_loss":        state["soh"] - soh_t[-1] if soh_t else 0,
        "charging_time_s": len(soc_t) * dt,
        "final_soc":       soc_t[-1] if soc_t else state["soc"],
        "start_soc":       state["soc"],
    }


# ── Pipeline policy loader ────────────────────────────────────────────────────

def load_pipeline_policies(dataset_path, state):
    """Load the three representative policies from the saved NSGA-II dataset."""
    if not os.path.exists(dataset_path):
        print(f"  [WARN] Dataset not found at {dataset_path}")
        print("  Run BMS_Pipeline.py with the same --soc/--soh/--temp flags first.")
        return {}

    df = pd.read_csv(dataset_path)
    policies = {}
    metrics_rows = []

    for sid, grp in df.groupby("solution_id"):
        soc_t  = grp["SoC"].values
        temp_t = grp["temperature_K"].values
        soh_t  = grp["SoH"].values
        metrics_rows.append({
            "solution_id": sid,
            "soc_gain":    soc_t[-1] - soc_t[0],  # relative to trajectory start
            "peak_temp":   np.max(temp_t),
            "soh_loss":    soh_t[0] - soh_t[-1],
            "soc_traj":    soc_t,
            "temp_traj":   temp_t,
            "soh_traj":    soh_t,
            "cur_traj":    grp["current_A"].values,
            "charging_time_s": len(soc_t),
        })

    mdf = pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ("soc_traj","temp_traj","soh_traj","cur_traj")}
                         for r in metrics_rows])

    fast_idx     = mdf["soc_gain"].idxmax()
    gentle_idx   = mdf["soh_loss"].idxmin()
    balanced_idx = (mdf["soc_gain"] - mdf["soc_gain"].median()).abs().idxmin()

    for name, idx in [("Pipeline — Fast", fast_idx),
                       ("Pipeline — Balanced", balanced_idx),
                       ("Pipeline — Gentle", gentle_idx)]:
        r = metrics_rows[idx]
        policies[name] = {
            "soc_traj":        r["soc_traj"],
            "temp_traj":       r["temp_traj"],
            "soh_traj":        r["soh_traj"],
            "current_traj":    r["cur_traj"],
            "soc_gain":        r["soc_gain"],
            "peak_temp":       r["peak_temp"],
            "soh_loss":        r["soh_loss"],
            "charging_time_s": r["charging_time_s"],
            "final_soc":       r["soc_traj"][-1],
            "start_soc":       r["soc_traj"][0],
        }
    return policies


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(all_results):
    print(f"\n{SEP}")
    print("  COMPARISON TABLE")
    print(SEP)
    # Note: Pipeline SoC Gain is relative to the predictor's starting SoC
    # (which may differ from the raw --soc input). CC-CV gain is from raw input.
    header = f"  {'Method':<22} {'Start SoC':>10} {'SoC Gain':>9} {'Final SoC':>10} {'Peak Temp (K)':>14} {'SoH Loss':>10} {'Time (s)':>9}"
    print(header)
    print(f"  {'-'*22} {'-'*10} {'-'*9} {'-'*10} {'-'*14} {'-'*10} {'-'*9}")

    for name, res in all_results.items():
        if res is None:
            print(f"  {name:<22} {'ABORTED':>9}")
            continue
        start_soc = res.get("start_soc", res["final_soc"] - res["soc_gain"])
        print(
            f"  {name:<22}"
            f" {start_soc:>10.4f}"
            f" {res['soc_gain']:>9.4f}"
            f" {res['final_soc']:>10.4f}"
            f" {res['peak_temp']:>14.2f}"
            f" {res['soh_loss']:>10.6f}"
            f" {res['charging_time_s']:>9.0f}"
        )
    print()

    # Improvement summary: each pipeline policy vs CC-CV 1C
    if "CC-CV 1C" in all_results and all_results["CC-CV 1C"] is not None:
        baseline = all_results["CC-CV 1C"]
        print(f"  {'─'*60}")
        print("  Pipeline policies vs CC-CV 1C (same 1200s window):")
        for name, res in all_results.items():
            if "Pipeline" not in name or res is None:
                continue
            temp_delta = res["peak_temp"] - baseline["peak_temp"]
            soh_delta  = res["soh_loss"]  - baseline["soh_loss"]
            print(f"    {name:<22}  "
                  f"Peak temp {temp_delta:+.2f} K | "
                  f"SoH loss {soh_delta:+.6f}")
    print()


# ── Trajectory plot ───────────────────────────────────────────────────────────

COLORS = {
    "CC-CV 0.5C":          "#6b7194",
    "CC-CV 1C":            "#fbbf24",
    "CC-CV 2C":            "#f97066",
    "Pipeline — Gentle":   "#34d4a0",
    "Pipeline — Balanced": "#7c6af7",
    "Pipeline — Fast":     "#f97066",
}
STYLES = {
    "CC-CV 0.5C":          "--",
    "CC-CV 1C":            "--",
    "CC-CV 2C":            "--",
    "Pipeline — Gentle":   "-",
    "Pipeline — Balanced": "-",
    "Pipeline — Fast":     "-",
}


def plot_comparison(all_results, state, out_path="cccv_comparison.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes.flat:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#6b7194")
        ax.xaxis.label.set_color("#c8cde8")
        ax.yaxis.label.set_color("#c8cde8")
        ax.title.set_color("#c8cde8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3250")
        ax.grid(True, color="#2e3250", linewidth=0.6)

    ax_soc, ax_temp, ax_soh, ax_cur = axes.flat

    for name, res in all_results.items():
        if res is None:
            continue
        color  = COLORS.get(name, "#ffffff")
        ls     = STYLES.get(name, "-")
        lw     = 1.5 if "Pipeline" in name else 1.2
        alpha  = 0.95 if "Pipeline" in name else 0.65
        t      = np.arange(len(res["soc_traj"])) / 60   # minutes

        ax_soc.plot(t,  res["soc_traj"],     color=color, ls=ls, lw=lw, alpha=alpha, label=name)
        ax_temp.plot(t, res["temp_traj"],     color=color, ls=ls, lw=lw, alpha=alpha)
        ax_soh.plot(t,  res["soh_traj"],      color=color, ls=ls, lw=lw, alpha=alpha)
        ax_cur.plot(t,  res["current_traj"],  color=color, ls=ls, lw=lw, alpha=alpha)

    # Safety reference lines
    ax_temp.axhline(320.0, color="#f97066", lw=0.9, ls=":", label="Abort limit (320 K)")
    ax_temp.axhline(315.0, color="#fbbf24", lw=0.9, ls=":", label="Override limit (315 K)")
    ax_temp.legend(fontsize=7, labelcolor="#c8cde8",
                   facecolor="#1a1d27", edgecolor="#2e3250")

    ax_soc.set_title("State of Charge");    ax_soc.set_ylabel("SoC")
    ax_temp.set_title("Temperature (K)");   ax_temp.set_ylabel("K")
    ax_soh.set_title("State of Health");    ax_soh.set_ylabel("SoH")
    ax_cur.set_title("Charging Current");   ax_cur.set_ylabel("A")

    for ax in axes[1]:
        ax.set_xlabel("Time (min)")

    ax_soc.legend(fontsize=7, labelcolor="#c8cde8",
                  facecolor="#1a1d27", edgecolor="#2e3250")

    fig.suptitle(
        f"CC-CV Baseline vs Pipeline  |  "
        f"SoC₀={state['soc']:.2f}  SoH₀={state['soh']:.2f}  "
        f"T₀={state['temp']-273.15:.1f}°C",
        color="#c8cde8", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    print(f"  Plot saved → {out_path}")


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save_results_csv(all_results, out_path="cccv_baseline_results.csv"):
    rows = []
    for name, res in all_results.items():
        if res is None:
            rows.append({"method": name, "aborted": True})
            continue
        for t in range(len(res["soc_traj"])):
            rows.append({
                "method":       name,
                "time_s":       t,
                "SoC":          res["soc_traj"][t],
                "temperature_K": res["temp_traj"][t],
                "SoH":          res["soh_traj"][t],
                "current_A":    res["current_traj"][t],
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  CSV saved  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CC-CV Baseline Comparison")
    p.add_argument("--soc",  type=float, default=0.30, help="Initial SoC [0-1]")
    p.add_argument("--soh",  type=float, default=0.88, help="Initial SoH [0-1]")
    p.add_argument("--temp", type=float, default=35.0, help="Initial temperature (°C)")
    p.add_argument("--plot-out", type=str, default="cccv_comparison.png")
    p.add_argument("--csv-out",  type=str, default="cccv_baseline_results.csv")
    return p.parse_args()


def main():
    args = parse_args()

    state = {
        "soc":  args.soc,
        "soh":  args.soh,
        "temp": max(args.temp + 273.15, BATTERY_PARAMS["T_amb"]),
    }

    print(f"\n{SEP}")
    print("  CC-CV BASELINE COMPARISON")
    print(SEP)
    print(f"  Initial state:  SoC={state['soc']:.2f}  "
          f"SoH={state['soh']:.2f}  "
          f"T={state['temp']-273.15:.1f}°C ({state['temp']:.2f} K)")

    _ = _get_ocv_func()   # build OCV once, log the source

    all_results = {}

    # ── CC-CV baselines ───────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  Simulating CC-CV baselines")
    print(SEP2)
    for label, cfg in CC_CV_CONFIGS.items():
        res = simulate_cccv(state, BATTERY_PARAMS,
                            I_cc=cfg["I_cc"], I_cutoff=cfg["I_cutoff"])
        all_results[label] = res
        if res is None:
            print(f"  {label:<14} → ABORTED (thermal limit exceeded)")
        else:
            print(f"  {label:<14} → "
                  f"SoC gain={res['soc_gain']:.4f}  "
                  f"peak_temp={res['peak_temp']:.2f} K  "
                  f"SoH loss={res['soh_loss']:.6f}  "
                  f"time={res['charging_time_s']:.0f}s  "
                  f"({res['charging_time_s']/60:.1f} min)")

    # ── Pipeline policies ─────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  Loading pipeline Pareto policies")
    print(SEP2)
    state_dataset_path = get_dataset_path(args.soc, args.soh, args.temp)
    print(f"  Looking for: {state_dataset_path}")
    pipeline_policies = load_pipeline_policies(state_dataset_path, state)

    if pipeline_policies:
        for name, res in pipeline_policies.items():
            all_results[name] = res
            print(f"  {name:<24} → "
                  f"SoC gain={res['soc_gain']:.4f}  "
                  f"peak_temp={res['peak_temp']:.2f} K  "
                  f"SoH loss={res['soh_loss']:.6f}  "
                  f"time={res['charging_time_s']:.0f}s")
    else:
        print("  No pipeline dataset found — showing CC-CV only")

    # ── Output ────────────────────────────────────────────────────────────────
    print_comparison_table(all_results)
    plot_comparison(all_results, state, out_path=args.plot_out)
    save_results_csv(all_results, out_path=args.csv_out)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()