"""
BMS Master Pipeline
===================
Links all five agents end-to-end.
Run from the directory containing:
  - best_model.pt
  - predictor_globals.pkl
  - nsga2_synthetic_dataset.csv

Usage:
  python bms_pipeline.py                        # uses default input
  python bms_pipeline.py --soc 0.45             # override any input field
  python bms_pipeline.py --soc 0.3 --mode fast  # manual charging mode
"""

import os
import re
import sys
import math
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── optional: suppress pymoo verbose output ──────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — edit these to match your Kaggle/local environment
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/Users/arujatiwary/Desktop/BMS Codes/best_model.pt"
GLOBALS_PATH = "/Users/arujatiwary/Desktop/BMS Codes/predictor_globals.pkl"
DATASET_PATH = "/Users/arujatiwary/Desktop/BMS Codes/nsga2_synthetic_dataset.csv"
DATA_DIR     = "/Users/arujatiwary/Desktop/ECM_processed_cycles"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SEP  = "=" * 60
SEP2 = "-" * 60

def banner(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def section(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BatteryTransformer(nn.Module):
    def __init__(self, input_dim=11, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.query_embed      = nn.Parameter(torch.randn(1, 3, d_model))
        self.soc_mu_head      = nn.Linear(d_model, 1)
        self.soc_logvar_head  = nn.Linear(d_model, 1)
        self.soh_mu_head      = nn.Linear(d_model, 1)
        self.soh_logvar_head  = nn.Linear(d_model, 1)
        self.temp_mu_head     = nn.Linear(d_model, 1)
        self.temp_logvar_head = nn.Linear(d_model, 1)

    def forward(self, x):
        bs      = x.size(0)
        x       = self.pos_encoder(self.input_proj(x))
        memory  = self.encoder(x)
        queries = self.query_embed.expand(bs, -1, -1)
        decoded = self.decoder(queries, memory)

        soc_f, soh_f, temp_f = decoded[:, 0], decoded[:, 1], decoded[:, 2]

        soc  = torch.cat([self.soc_mu_head(soc_f),
                          self.soc_logvar_head(soc_f.detach())],  dim=1)
        soh  = torch.cat([self.soh_mu_head(soh_f),
                          self.soh_logvar_head(soh_f.detach())],  dim=1)
        temp = torch.cat([self.temp_mu_head(temp_f),
                          self.temp_logvar_head(temp_f.detach())], dim=1)
        return soc, soh, temp


def build_input_sequence(battery_input, global_mean, global_std, seq_len=64):
    """
    Builds a (seq_len, 11) feature tensor from a single battery state dict.
    In production this would come from real sensor readings.
    Here we construct a plausible sequence from the given scalar inputs.
    """
    num_cols = [
        'Voltage_measured', 'Current_measured', 'dV_dt', 'dT_dt',
        'V_RC_masked', 'V_ECM_masked', 'power', 'Temperature_measured'
    ]

    # Build synthetic sequence — constant values with tiny noise
    soc   = battery_input["soc"]
    temp  = battery_input["temp_C"]      # Celsius
    curr  = battery_input["current_A"]   # negative = discharge
    # Use the ECM OCV function for a physics-consistent voltage estimate
    volt  = _get_ocv_func()(soc)

    rows = []
    for t in range(seq_len):
        noise = np.random.normal(0, 0.001, 8)
        row = np.array([
            volt  + noise[0],          # Voltage_measured
            curr  + noise[1],          # Current_measured
            noise[2],                  # dV_dt
            noise[3],                  # dT_dt
            0.0   + noise[4],          # V_RC_masked (discharge only)
            0.0   + noise[5],          # V_ECM_masked
            volt * curr + noise[6],    # power
            temp  + noise[7],          # Temperature_measured
        ])
        rows.append(row)

    data = np.stack(rows)  # (seq_len, 8)

    # normalise the 8 numeric cols
    mean_vals = global_mean[num_cols].values
    std_vals  = global_std[num_cols].values
    data_norm = (data - mean_vals) / std_vals

    # extra features (not normalised)
    mode_flag   = 1.0 if curr < 0 else 0.0
    time_uniform = np.linspace(0, 1, seq_len).reshape(-1, 1)
    cycle_index  = np.full((seq_len, 1), battery_input.get("cycle_norm", 0.5))
    mode_col     = np.full((seq_len, 1), mode_flag)

    # final feature order matches BatteryDataset:
    # Voltage, Current, Time_uniform, dV_dt, dT_dt,
    # mode_flag, V_RC_masked, V_ECM_masked, power, cycle_index, Temperature
    features = np.concatenate([
        data_norm[:, 0:1],   # Voltage_measured
        data_norm[:, 1:2],   # Current_measured
        time_uniform,        # Time_uniform
        data_norm[:, 2:3],   # dV_dt
        data_norm[:, 3:4],   # dT_dt
        mode_col,            # mode_flag
        data_norm[:, 4:5],   # V_RC_masked
        data_norm[:, 5:6],   # V_ECM_masked
        data_norm[:, 6:7],   # power
        cycle_index,         # cycle_index
        data_norm[:, 7:8],   # Temperature_measured
    ], axis=1)               # (seq_len, 11)

    return features.astype(np.float32)


def run_predictor(battery_input, model, global_mean, global_std, device):
    banner("AGENT 1 — PREDICTOR")
    print(f"  Input battery state:")
    print(f"    SoC (initial)  : {battery_input['soc']:.3f}")
    print(f"    SoH (initial)  : {battery_input['soh']:.3f}")
    print(f"    Temperature    : {battery_input['temp_C']:.1f} °C")
    print(f"    Current        : {battery_input['current_A']:.2f} A")

    seq = build_input_sequence(battery_input, global_mean, global_std)
    x   = torch.tensor(seq).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        soc_out, soh_out, temp_out = model(x)

    soc_mu      = float(soc_out[0, 0])
    soc_logvar  = float(torch.clamp(soc_out[0, 1], -2, 2))
    soh_mu      = float(soh_out[0, 0])
    soh_logvar  = float(torch.clamp(soh_out[0, 1], -2, 2))
    temp_mu     = float(temp_out[0, 0])
    temp_logvar = float(torch.clamp(temp_out[0, 1], -2, 2))

    # confidence: higher = more certain
    soc_conf  = float(1 / (1 + np.exp(soc_logvar)))
    soh_conf  = float(1 / (1 + np.exp(soh_logvar)))
    temp_conf = float(1 / (1 + np.exp(temp_logvar)))
    avg_conf  = (soc_conf + soh_conf + temp_conf) / 3.0

    # temperature is normalised — denormalise back to Celsius
    temp_mean = float(global_mean["Temperature_measured"])
    temp_std  = float(global_std["Temperature_measured"])
    temp_real = temp_mu * temp_std + temp_mean

    predictor_output = {
        "soc":        np.clip(soc_mu, 0.0, 1.0),
        "soh":        np.clip(soh_mu, 0.0, 1.0),
        "temperature": temp_real,
        "confidence": avg_conf,
        # per-target confidences
        "soc_conf":   soc_conf,
        "soh_conf":   soh_conf,
        "temp_conf":  temp_conf,
    }

    section("Predictor Output")
    print(f"  SoC prediction   : {predictor_output['soc']:.4f}  (confidence {soc_conf:.3f})")
    print(f"  SoH prediction   : {predictor_output['soh']:.4f}  (confidence {soh_conf:.3f})")
    print(f"  Temp prediction  : {predictor_output['temperature']:.2f} °C  (confidence {temp_conf:.3f})")
    print(f"  Overall confidence: {avg_conf:.3f}  {'[HIGH UNCERTAINTY — pipeline will be cautious]' if avg_conf < 0.5 else '[OK]'}")

    return predictor_output


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — SIMULATOR + OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_PARAMS = {
    "capacity_Ah": 2.3,    # NASA Li-ion rated capacity (corrected from 2.0)
    "R0_nominal":  0.02,   # Ohm — ohmic resistance, electrical model (ECM.ipynb)
    "R1_nominal":  0.01,   # Ohm — RC branch resistance (ECM.ipynb)
    "C1_nominal":  2000.0, # F   — RC branch capacitance (ECM.ipynb)
    "R_heat":      0.35,   # Ohm — effective resistance for heat generation;
                           #       decoupled from R0: captures reaction heat,
                           #       contact resistance, electrolyte losses.
                           #       Calibrated: 2.5A->+1K rise, 4A->+18K rise.
    "V_max":       4.2,
    "V_min":       3.0,
    "V_cutoff":    2.7,    # hard discharge cutoff from ECM.ipynb
    "T_amb":       298.0,
    "T_max":       333.0,
    "R_th":        5.0,    # K/W — thermal resistance to ambient
    "C_th":        50.0,   # J/K — thermal mass; tau=R_th*C_th=250s
    "dt":          1.0,
}

POP_SIZE      = 40
N_GENERATIONS = 25
HORIZON_SEC   = 1200
N_GENES       = int(HORIZON_SEC / 1.0)
I_MIN, I_MAX  = 0.5, 4.0
ELITE_FRAC    = 0.2
MUT_PROB      = 0.1
MUT_STD       = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# ECM PHYSICS  (ported from ECM.ipynb — BatteryECM class + OCV fitting)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ocv_function(data_dir=None):
    """
    Build the OCV–SOC polynomial from real NASA discharge data if available
    (mirrors ECM.ipynb Cells 7, 11, 12).  Falls back to the analytical
    approximation when the processed data directory is not present.
    """
    if data_dir and os.path.isdir(data_dir):
        csv_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv")
        ])
        soc_all, volt_all = [], []
        for fp in csv_files[:20]:                     # use up to 20 cycles
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
            # bin-average to reduce noise (ECM.ipynb Cell 11)
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
                print("  [ECM] OCV–SOC curve: 5th-degree polynomial fitted "
                      f"from {len(soc_centers)} bins across {len(csv_files[:20])} cycles")
                return lambda soc: float(ocv_poly(np.clip(soc, 0.0, 1.0)))

    # Fallback: analytical approximation (ECM.ipynb Cell 6 extended)
    print("  [ECM] OCV–SOC curve: analytical fallback (data dir not found or empty)")
    def _analytical(soc):
        soc = np.clip(soc, 0.0, 1.0)
        return 3.0 + 1.2 * soc - 0.3 * np.exp(-5 * soc) + 0.1 * np.exp(-5 * (1 - soc))
    return _analytical


class BatteryECM:
    """
    First-order RC Equivalent Circuit Model.
    Ported directly from ECM.ipynb Cell 5.

    State variables
    ---------------
    SOC   : state of charge [0, 1]
    V_RC  : voltage across the RC branch (transient polarisation)

    Parameters
    ----------
    Q        : capacity in Ah (derated by SoH before instantiation)
    R0       : ohmic resistance (Ω) — scales with 1/SoH to model ageing
    R1, C1   : RC branch parameters
    dt       : timestep (s)
    ocv_func : callable SOC → OCV voltage (V)
    soc_init : starting SOC
    """

    def __init__(self, Q, R0, R1, C1, dt, ocv_func, soc_init=1.0):
        self.Q        = Q
        self.R0       = R0
        self.R1       = R1
        self.C1       = C1
        self.dt       = dt
        self.ocv_func = ocv_func
        self.SOC      = soc_init
        self.V_RC     = 0.0          # initialise RC branch at rest

    def step(self, I):
        """
        Advance one timestep.

        Parameters
        ----------
        I : current in Amperes.
            Convention (matching ECM.ipynb): positive I = charging current
            (opposite to the discharge-positive convention in ECM.ipynb itself,
            which is corrected when calling simulate_charging).

        Returns
        -------
        SOC, V_RC, OCV, V_ecm
        """
        # SOC update — capacity derated by SoH already baked into self.Q
        self.SOC += (I * self.dt) / (3600.0 * self.Q)
        self.SOC  = np.clip(self.SOC, 0.0, 1.0)

        # RC branch dynamics (ECM.ipynb Cell 5)
        alpha     = np.exp(-self.dt / (self.R1 * self.C1))
        self.V_RC = alpha * self.V_RC + self.R1 * (1.0 - alpha) * I

        # Terminal voltage
        OCV   = self.ocv_func(self.SOC)
        V_ecm = OCV - I * self.R0 - self.V_RC   # drops for charging current

        return self.SOC, self.V_RC, OCV, V_ecm


# Module-level OCV function — built once, reused by every simulation call
_OCV_FUNC = None

def _get_ocv_func():
    global _OCV_FUNC
    if _OCV_FUNC is None:
        _OCV_FUNC = _build_ocv_function(data_dir=DATA_DIR)
    return _OCV_FUNC


def thermal_step(temp, current, params, dt):
    """Newton cooling + I^2*R_heat heating.
    R_heat is decoupled from electrical R0 — calibrated to capture all
    internal heat sources (reaction heat, contact resistance, electrolyte).
    """
    heat_gen  = (current ** 2) * params["R_heat"]
    heat_loss = (temp - params["T_amb"]) / params["R_th"]
    return temp + (dt / params["C_th"]) * (heat_gen - heat_loss)


def degradation_step(soh, current, temp, dt):
    stress = abs(current) * max(0.0, temp - 298.0)
    return max(0.0, soh - 5e-9 * stress * dt)


def simulate_charging(profile, state, params, log_trajectory=False):
    """
    Simulate a charging profile using the full first-order RC ECM.

    Key differences from the old stub
    -----------------------------------
    - Uses BatteryECM with a real RC branch (R1, C1, V_RC state)
    - OCV is the fitted polynomial from real NASA data (or analytical fallback)
    - R0 and capacity are derated by SoH at each step as the cell ages
    - Voltage limit checked against ECM terminal voltage (not just OCV + R0)
    """
    soc  = state["soc"]
    soh  = state["soh"]
    temp = state["temp"]   # Kelvin
    dt   = params["dt"]
    ocv_func = _get_ocv_func()

    # Build ECM instance for this simulation; derate capacity by SoH
    Q  = params["capacity_Ah"] * max(soh, 0.01)
    R0 = params["R0_nominal"] / max(soh, 0.01)   # resistance rises as SoH falls
    R1 = params["R1_nominal"]
    C1 = params["C1_nominal"]
    ecm = BatteryECM(Q=Q, R0=R0, R1=R1, C1=C1, dt=dt,
                     ocv_func=ocv_func, soc_init=soc)

    soc_t, temp_t, soh_t = [], [], []
    peak_temp = temp   # track running maximum for consistent NSGA-II objective

    for current in profile:
        soc, V_RC, OCV, voltage = ecm.step(current)

        # Sync SoH-dependent parameters every step (ageing during charging)
        soh  = degradation_step(soh, current, temp, dt)
        # Recompute R0 for thermal calc after SoH update
        temp = thermal_step(temp, current, params, dt)
        if temp > peak_temp:
            peak_temp = temp

        # Safety limits — terminal voltage and temperature
        if voltage > params["V_max"] or temp > params["T_max"]:
            return None

        if log_trajectory:
            soc_t.append(soc)
            temp_t.append(temp)
            soh_t.append(soh)

    charging_time = len(profile) * dt
    soh_loss      = state["soh"] - soh

    if log_trajectory:
        return charging_time, soh_loss, soc_t, temp_t, soh_t
    return charging_time, peak_temp, soh_loss, soc   # peak_temp, not final temp


def fitness_function(sim_result, state):
    if sim_result is None:
        return -1e6
    charging_time, peak_temp, soh_loss, final_soc = sim_result
    soc_gain     = final_soc - state["soc"]
    temp_penalty = max(0.0, peak_temp - 318.0)
    return 100.0 * soc_gain - 1e4 * soh_loss - 10.0 * temp_penalty


def run_ga(state):
    population = [np.random.uniform(I_MIN, I_MAX, N_GENES) for _ in range(POP_SIZE)]

    for gen in range(N_GENERATIONS):
        fitnesses = np.array([
            fitness_function(simulate_charging(ind, state, BATTERY_PARAMS), state)
            for ind in population
        ])
        elite_n  = int(ELITE_FRAC * POP_SIZE)
        elites   = [population[i] for i in np.argsort(fitnesses)[-elite_n:]]
        next_pop = elites.copy()
        while len(next_pop) < POP_SIZE:
            i1, i2 = np.random.choice(len(elites), 2, replace=False)
            pt     = np.random.randint(1, N_GENES - 1)
            child  = np.concatenate([elites[i1][:pt], elites[i2][pt:]])
            mask   = np.random.rand(N_GENES) < MUT_PROB
            child[mask] += np.random.normal(0, MUT_STD, mask.sum())
            child  = np.clip(child, I_MIN, I_MAX)
            next_pop.append(child)
        population = next_pop
        if (gen + 1) % 5 == 0 or gen == 0:
            print(f"    GA gen {gen+1:2d}/{N_GENERATIONS} | best fitness {max(fitnesses):.2f}")

    fitnesses    = np.array([
        fitness_function(simulate_charging(ind, state, BATTERY_PARAMS), state)
        for ind in population
    ])
    return population[np.argmax(fitnesses)]


def run_nsga2(state):
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    class ChargingProblem(Problem):
        def __init__(self):
            super().__init__(n_var=N_GENES, n_obj=3, n_constr=0,
                             xl=I_MIN, xu=I_MAX)
        def _evaluate(self, X, out, *args, **kwargs):
            F = []
            for ind in X:
                sim = simulate_charging(ind, state, BATTERY_PARAMS)
                if sim is None:
                    F.append([1e3, 1e3, 1e3])
                else:
                    _, peak_temp, soh_loss, final_soc = sim
                    F.append([-(final_soc - state["soc"]), peak_temp, soh_loss])
            out["F"] = np.array(F)

    result = minimize(
        ChargingProblem(),
        NSGA2(pop_size=60, sampling=FloatRandomSampling(),
              crossover=SBX(prob=0.9, eta=15),
              mutation=PM(eta=20), eliminate_duplicates=True),
        termination=("n_gen", 40), seed=1, verbose=False
    )
    return result.X, result.F


def build_synthetic_dataset(pareto_profiles, state):
    rows = []
    for sol_id, profile in enumerate(pareto_profiles):
        res = simulate_charging(profile, state, BATTERY_PARAMS, log_trajectory=True)
        if res is None:
            continue
        _, _, soc_t, temp_t, soh_t = res
        for t in range(len(profile)):
            rows.append({
                "solution_id":   sol_id,
                "time_s":        t,
                "current_A":     profile[t],
                "SoC":           soc_t[t],
                "temperature_K": temp_t[t],
                "SoH":           soh_t[t],
            })
    return pd.DataFrame(rows)


def run_simulator_optimiser(predictor_output, battery_input):
    banner("AGENT 2 — SIMULATOR + OPTIMISER")

    # Temperature: use predictor output only if its confidence is acceptable.
    # The predictor consistently undershoots (~15-16 °C regardless of input),
    # so fall back to the raw sensor value when temp_conf < 0.85.
    pred_temp_K   = predictor_output["temperature"] + 273.15
    raw_temp_K    = battery_input["temp_C"] + 273.15
    temp_conf     = predictor_output.get("temp_conf", 0.0)
    if temp_conf < 0.85:
        sim_temp_K = max(raw_temp_K, BATTERY_PARAMS["T_amb"])
        print(f"  [SIM] Temp predictor confidence {temp_conf:.3f} < 0.85 — "
              f"using raw sensor temp {battery_input['temp_C']:.1f} °C ({sim_temp_K:.2f} K)")
    else:
        sim_temp_K = max(pred_temp_K, BATTERY_PARAMS["T_amb"])
        if sim_temp_K != pred_temp_K:
            print(f"  [SIM] Predictor temp {pred_temp_K:.2f} K below ambient — "
                  f"clamped to {BATTERY_PARAMS['T_amb']} K")
    transformer_state = {
        "soc":        predictor_output["soc"],
        "soh":        predictor_output["soh"],
        "temp":       sim_temp_K,
        "confidence": predictor_output["confidence"],
    }

    section("Simulator — ECM physics (first-order RC model)")
    soh_now = transformer_state["soh"]
    Q_eff   = BATTERY_PARAMS["capacity_Ah"] * soh_now
    R0_eff  = BATTERY_PARAMS["R0_nominal"] / max(soh_now, 0.01)
    print(f"  ECM parameters for this session:")
    print(f"    Effective capacity : {Q_eff:.3f} Ah  (Q_rated x SoH)")
    print(f"    R0 (ohmic)         : {R0_eff:.4f} Ohm  (scales with 1/SoH)")
    print(f"    R1 (RC branch)     : {BATTERY_PARAMS['R1_nominal']:.4f} Ohm")
    print(f"    C1 (RC branch)     : {BATTERY_PARAMS['C1_nominal']:.0f} F")
    # trigger OCV curve build now so the log line appears before test sim
    _ = _get_ocv_func()
    test_profile = np.full(60, 1.5)   # 1-minute test at 1.5A
    test_result  = simulate_charging(test_profile, transformer_state, BATTERY_PARAMS)
    if test_result:
        _, peak_t, soh_loss, final_soc = test_result
        print(f"  1-min test sim at 1.5A:")
        print(f"    Final SoC  : {final_soc:.4f}")
        print(f"    Peak temp  : {peak_t:.2f} K")
        print(f"    SoH loss   : {soh_loss:.6f}")
    else:
        print("  Test simulation hit safety limit — battery state extreme")

    # ── state-keyed dataset path — keyed on raw sensor inputs so the
    # filename matches what cccv_baseline.py looks for.
    raw_temp_K_for_tag = max(battery_input["temp_C"] + 273.15, BATTERY_PARAMS["T_amb"])
    dataset_tag = (f"soc{battery_input['soc']:.2f}_"
                   f"soh{battery_input['soh']:.2f}_"
                   f"temp{raw_temp_K_for_tag:.0f}")
    state_dataset_path = (os.path.splitext(DATASET_PATH)[0]
                          + f"_{dataset_tag}.csv")

    # ── check if dataset already exists for this battery state ───────────────
    if os.path.exists(state_dataset_path):
        section(f"Loading cached dataset")
        print(f"  Path: {state_dataset_path}")
        df = pd.read_csv(state_dataset_path)
        print(f"  Loaded {len(df):,} rows | {df['solution_id'].nunique()} solutions")
        pareto_profiles = None   # not needed — dataset already built
    else:
        section("GA optimisation")
        best_ga = run_ga(transformer_state)
        res_ga  = simulate_charging(best_ga, transformer_state, BATTERY_PARAMS)
        if res_ga:
            _, _, _, best_soc = res_ga
            print(f"\n  GA best profile → final SoC: {best_soc:.4f}")

        section("NSGA-II multi-objective optimisation")
        print("  Running NSGA-II (40 generations, 60 individuals)...")
        pareto_profiles, pareto_F = run_nsga2(transformer_state)
        soc_gains  = -pareto_F[:, 0]
        peak_temps =  pareto_F[:, 1]
        soh_losses =  pareto_F[:, 2]
        print(f"  Pareto front: {len(pareto_profiles)} solutions")
        print(f"    SoC gain range  : {soc_gains.min():.4f} — {soc_gains.max():.4f}")
        print(f"    Peak temp range : {peak_temps.min():.2f} — {peak_temps.max():.2f} K")
        print(f"    SoH loss range  : {soh_losses.min():.6f} — {soh_losses.max():.6f}")

        section("Building synthetic dataset")
        df = build_synthetic_dataset(pareto_profiles, transformer_state)
        df.to_csv(state_dataset_path, index=False)
        print(f"  Saved {len(df):,} rows → {state_dataset_path}")

    return df, transformer_state


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — META-AGENT
# ─────────────────────────────────────────────────────────────────────────────

def extract_policies(df):
    policies = {}
    for sid, grp in df.groupby("solution_id"):
        policies[sid] = {
            "current_profile": grp["current_A"].values,
            "soc_traj":        grp["SoC"].values,
            "temp_traj":       grp["temperature_K"].values,
            "soh_traj":        grp["SoH"].values,
        }
    return policies


def compute_policy_metrics(policies):
    rows = []
    for pid, data in policies.items():
        rows.append({
            "solution_id": pid,
            "soc_gain":    data["soc_traj"][-1]  - data["soc_traj"][0],
            "peak_temp":   np.max(data["temp_traj"]),
            "soh_loss":    data["soh_traj"][0]   - data["soh_traj"][-1],
        })
    return pd.DataFrame(rows)


def identify_representative_policies(metrics_df):
    fast_idx     = metrics_df["soc_gain"].idxmax()
    gentle_idx   = metrics_df["soh_loss"].idxmin()
    balanced_idx = (metrics_df["soc_gain"] - metrics_df["soc_gain"].median()).abs().idxmin()
    return {
        "fast":     metrics_df.loc[fast_idx,     "solution_id"],
        "balanced": metrics_df.loc[balanced_idx, "solution_id"],
        "gentle":   metrics_df.loc[gentle_idx,   "solution_id"],
    }


def meta_agent_select(policy_choices, transformer_state, mode="auto"):
    soc        = transformer_state["soc"]
    soh        = transformer_state["soh"]
    confidence = transformer_state.get("confidence", 1.0)

    if confidence < 0.5:
        return policy_choices["gentle"], "predictor confidence low — defaulting to gentle"

    if mode == "fast":
        return policy_choices["fast"],     "manual mode: fast"
    if mode == "balanced":
        return policy_choices["balanced"], "manual mode: balanced"
    if mode == "battery_care":
        return policy_choices["gentle"],   "manual mode: battery_care"

    if soh < 0.9:
        return policy_choices["gentle"],   f"SoH={soh:.2f} below 0.90 — gentle"
    if soc < 0.4 and soh >= 0.9:
        return policy_choices["fast"],     f"low SoC={soc:.2f} and healthy — fast"
    if soc < 0.4 and soh < 0.9:
        return policy_choices["balanced"], f"low SoC={soc:.2f} but degraded — balanced"

    return policy_choices["balanced"], "default: balanced"


def run_meta_agent(df, transformer_state, mode="auto"):
    banner("AGENT 3 — META-AGENT")

    policies       = extract_policies(df)
    metrics_df     = compute_policy_metrics(policies)
    policy_choices = identify_representative_policies(metrics_df)

    section("Policy landscape")
    print(f"  Total Pareto policies : {len(policies)}")
    for name, pid in policy_choices.items():
        row = metrics_df[metrics_df["solution_id"] == pid].iloc[0]
        print(f"  {name:>8}  id={int(pid):3d} | "
              f"soc_gain={row.soc_gain:.4f} | "
              f"peak_temp={row.peak_temp:.2f} K | "
              f"soh_loss={row.soh_loss:.6f}")

    selected_policy, reason = meta_agent_select(
        policy_choices, transformer_state, mode=mode
    )

    section("Meta-Agent Decision")
    print(f"  Selected policy : {int(selected_policy)}")
    print(f"  Reason          : {reason}")

    sel_row = metrics_df[metrics_df["solution_id"] == selected_policy].iloc[0]
    print(f"\n  Selected policy metrics:")
    print(f"    SoC gain   : {sel_row.soc_gain:.4f}")
    print(f"    Peak temp  : {sel_row.peak_temp:.2f} K")
    print(f"    SoH loss   : {sel_row.soh_loss:.6f}")

    return selected_policy, policies, metrics_df, policy_choices


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — KILL AGENT
# ─────────────────────────────────────────────────────────────────────────────

def extract_policy(df, policy_id):
    data = df[df["solution_id"] == policy_id]
    return {
        "current": data["current_A"].values,
        "soc":     data["SoC"].values,
        "temp":    data["temperature_K"].values,
        "soh":     data["SoH"].values,
    }


def compute_metrics(policy):
    return {
        "soc_gain":           policy["soc"][-1]  - policy["soc"][0],
        "peak_temp":          np.max(policy["temp"]),
        "temp_rise":          np.max(np.diff(policy["temp"])),
        "soh_loss":           policy["soh"][0]   - policy["soh"][-1],
        "high_temp_duration": int(np.sum(policy["temp"] > 315)),
    }


def kill_agent(policy_metrics, battery_state,
               peak_temp_limit=320, temp_rise_limit=5,
               soh_loss_limit=0.001, health_limit=0.80,
               high_temp_limit=5, confidence_limit=0.5):

    checks = []

    def chk(name, value, limit, decision, reason):
        breached = value > limit
        checks.append({"rule": name, "value": value, "limit": limit,
                        "breached": breached})
        return breached

    if chk("Peak temperature",    policy_metrics["peak_temp"],          peak_temp_limit,   "abort",    "temperature limit exceeded"):
        return {"decision": "abort",    "reason": "temperature limit exceeded"},    checks
    if chk("Rapid temp rise",     policy_metrics["temp_rise"],          temp_rise_limit,   "abort",    "rapid thermal rise"):
        return {"decision": "abort",    "reason": "rapid thermal rise"},            checks
    if chk("Sustained overheat",  policy_metrics["high_temp_duration"], high_temp_limit,   "override", "sustained high temperature"):
        return {"decision": "override", "reason": "sustained high temperature"},    checks
    if chk("SoH loss",            policy_metrics["soh_loss"],           soh_loss_limit,    "override", "excessive degradation"):
        return {"decision": "override", "reason": "excessive degradation"},         checks
    if chk("Battery health",      1 - battery_state["soh"],             1 - health_limit,  "override", "battery health low"):
        return {"decision": "override", "reason": "battery health low"},            checks
    if chk("Predictor confidence",battery_state.get("confidence", 1.0) < confidence_limit, 0.5, "override", "predictor uncertainty"):
        return {"decision": "override", "reason": "predictor uncertainty"},         checks

    return {"decision": "allow", "reason": "policy safe"}, checks


def run_kill_agent(df, selected_policy, transformer_state, policies, metrics_df):
    banner("AGENT 4 — KILL AGENT")

    battery_state = {
        "soc":        transformer_state["soc"],
        "soh":        transformer_state["soh"],
        "temp":       transformer_state["temp"],
        "confidence": transformer_state.get("confidence", 1.0),
    }

    policy  = extract_policy(df, selected_policy)
    metrics = compute_metrics(policy)

    section("Policy safety metrics")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.6f}")

    decision, checks = kill_agent(metrics, battery_state)

    section("Rule trace")
    for c in checks:
        status = "BREACHED" if c["breached"] else "ok"
        bar    = c["value"] / (c["limit"] + 1e-9)
        marker = "!!" if c["breached"] else "  "
        print(f"  {marker} {c['rule']:<24} {c['value']:.4f} / {c['limit']:.4f}  [{bar:.2f}x]  {status}")

    section("Kill Agent Decision")
    dec_str = decision["decision"].upper()
    print(f"  Decision : {dec_str}")
    print(f"  Reason   : {decision['reason']}")

    # resolve final policy
    if decision["decision"] == "allow":
        final_policy = selected_policy
        print(f"\n  Final policy : {int(final_policy)} — APPROVED, charging may proceed")

    elif decision["decision"] == "override":
        # find safest approved policy from Pareto set
        safe_candidates = []
        for pid in df["solution_id"].unique():
            pol  = extract_policy(df, pid)
            mets = compute_metrics(pol)
            dec, _ = kill_agent(mets, battery_state)
            if dec["decision"] == "allow":
                safe_candidates.append((pid, mets["soh_loss"]))

        if safe_candidates:
            final_policy = min(safe_candidates, key=lambda x: x[1])[0]
            print(f"\n  Override → safest approved policy: {int(final_policy)}")
        else:
            final_policy = None
            print("\n  Override → no safe policy found, charging aborted")

    else:  # abort
        final_policy = None
        print("\n  ABORT — charging stopped, no policy executed")

    return final_policy, decision


# ─────────────────────────────────────────────────────────────────────────────
# FINAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_final_output(predictor_output, transformer_state,
                       selected_policy, decision, final_policy,
                       df, policies):
    banner("PIPELINE COMPLETE — FINAL OUTPUT")

    print(f"\n  {'─'*50}")
    print(f"  PREDICTOR STATE ESTIMATE")
    print(f"  {'─'*50}")
    print(f"  SoC          : {predictor_output['soc']:.4f}")
    print(f"  SoH          : {predictor_output['soh']:.4f}")
    print(f"  Temperature  : {predictor_output['temperature']:.2f} °C")
    print(f"  Confidence   : {predictor_output['confidence']:.3f}")

    print(f"\n  {'─'*50}")
    print(f"  KILL AGENT DECISION")
    print(f"  {'─'*50}")
    print(f"  Decision     : {decision['decision'].upper()}")
    print(f"  Reason       : {decision['reason']}")

    print(f"\n  {'─'*50}")
    print(f"  FINAL CHARGING COMMAND")
    print(f"  {'─'*50}")
    if final_policy is not None:
        pol  = extract_policy(df, final_policy)
        mets = compute_metrics(pol)
        print(f"  Policy ID    : {int(final_policy)}")
        print(f"  SoC gain     : {mets['soc_gain']:.4f}")
        print(f"  Peak temp    : {mets['peak_temp']:.2f} K")
        print(f"  SoH loss     : {mets['soh_loss']:.6f}")
        print(f"  Duration     : {len(pol['current'])} seconds")
        print(f"  Avg current  : {np.mean(pol['current']):.2f} A")
        print(f"\n  STATUS: CHARGING APPROVED")
    else:
        print(f"  STATUS: CHARGING ABORTED — no safe policy available")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BMS Multi-Agent Pipeline")
    p.add_argument("--soc",     type=float, default=0.45,  help="State of Charge [0-1]")
    p.add_argument("--soh",     type=float, default=0.95,  help="State of Health [0-1]")
    p.add_argument("--temp",    type=float, default=27.0,  help="Temperature in Celsius")
    p.add_argument("--current", type=float, default=-1.5,  help="Current in A (negative=discharge)")
    p.add_argument("--cycle",   type=float, default=0.5,   help="Normalised cycle index [0-1]")
    p.add_argument("--mode",    type=str,   default="auto",
                   choices=["auto", "fast", "balanced", "battery_care"],
                   help="Charging mode override")
    return p.parse_args()


def main():
    args = parse_args()

    battery_input = {
        "soc":       args.soc,
        "soh":       args.soh,
        "temp_C":    args.temp,
        "current_A": args.current,
        "cycle_norm": args.cycle,
    }

    print(f"\n{SEP}")
    print("  BMS MULTI-AGENT PIPELINE")
    print(f"  Mode: {args.mode.upper()}")
    print(SEP)

    # ── load model and globals ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print(f"  Loading model from {MODEL_PATH}...")
    model = BatteryTransformer(input_dim=11).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print(f"  Loading globals from {GLOBALS_PATH}...")
    with open(GLOBALS_PATH, "rb") as f:
        globs = pickle.load(f)
    global_mean = globs["global_mean"]
    global_std  = globs["global_std"]

    # ── run pipeline ──────────────────────────────────────────────────────────
    predictor_output = run_predictor(battery_input, model, global_mean, global_std, device)

    df, transformer_state = run_simulator_optimiser(predictor_output, battery_input)

    # attach confidence to transformer_state for downstream agents
    transformer_state["confidence"] = predictor_output["confidence"]

    selected_policy, policies, metrics_df, policy_choices = run_meta_agent(
        df, transformer_state, mode=args.mode
    )

    final_policy, decision = run_kill_agent(
        df, selected_policy, transformer_state, policies, metrics_df
    )

    print_final_output(
        predictor_output, transformer_state,
        selected_policy, decision, final_policy,
        df, policies
    )


if __name__ == "__main__":
    main()