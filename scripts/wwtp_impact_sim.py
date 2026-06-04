"""
Anomaly detection AS A COMPONENT of the wastewater-treatment control loop:
what does it change if the plant has the microfauna anomaly detector vs not?

This is an illustrative, literature-grounded *scenario* simulation (not a
calibrated plant model).  It encodes the well-established premise behind the
Sludge Biotic Index (Madoni, 1994): the microfauna community shifts from
healthy ciliates/rotifers to stressed nematodes/testate amoebae *before* the
macroscopic effluent quality fails — i.e. microfauna are a LEADING indicator.

We simulate one sludge-degradation episode under two operating policies:

  * WITHOUT anomaly detection (reactive): the operator only reacts once the
    effluent itself violates the discharge limit (a lagging indicator).
  * WITH anomaly detection  (proactive): the DINOv3/PatchCore detector samples
    sludge microscopy daily; it raises an alarm when the flagged-stressed
    fraction climbs.  The detector's reliability is its REAL operating point,
    read from results/csv/threshold_metrics.csv (TPR, FPR).

Outputs:
  results/csv/wwtp_simulation_timeseries.csv
  results/csv/wwtp_impact_summary.csv
  results/figures/wwtp_with_vs_without_ad.png
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# --------------------------------------------------------------------------- #
# Scenario parameters (documented, illustrative)
# --------------------------------------------------------------------------- #
DAYS = 90
D0 = 20                 # day the disturbance (toxic/organic shock) begins
SHOCK_DUR = 18          # days the degradation pressure is active
SHOCK = 0.060           # daily downward pressure on sludge health
H0 = 0.92               # baseline health (1 = healthy community)
H_TARGET = 0.92         # health the biology relaxes back toward
K_REC_NAT = 0.040       # natural recovery rate
K_REC_INT = 0.230       # recovery rate once a corrective action is taken
EFF_LAG = 3             # days effluent lags behind the microfauna state

# process couplings (health -> process variables)
SVI_BASE, SVI_GAIN = 90.0, 135.0      # mL/g  : 90 healthy -> ~225 fully stressed
TSS_BASE, TSS_GAIN = 6.0, 34.0        # mg/L
TSS_SVI_PEN = 0.06                    # extra effluent TSS from poor settling
TSS_LIMIT = 30.0                      # discharge limit (mg/L)
SVI_BULK = 150.0                      # bulking threshold

E_BASE = 1000.0         # baseline aeration energy (kWh/day) for healthy sludge
AER_HEALTH = 0.50       # aeration rises as health falls (operators raise DO)
AER_BULK = 0.40         # extra aeration per 100 SVI units above the bulking line
ENERGY_COST = 0.12      # $/kWh (illustrative)
PENALTY_VIOL = 500.0    # $/day of effluent-limit violation (compliance cost)

N_FIELDS = 30           # microscopy fields imaged per day
ALARM_LEVEL = 0.45      # alarm if smoothed flagged-stressed fraction exceeds
CONSEC = 2              # consecutive days required to act (both policies)
SEED = C.SEED


def detector_operating_point():
    """Read the detector's real TPR / FPR from the anomaly-pipeline results."""
    try:
        df = pd.read_csv(os.path.join(C.RES_CSV, "threshold_metrics.csv"))
        row = df[(df.backbone == "dinov3") &
                 (df.operating_point == "youden_J")].iloc[0]
        tpr = float(row["recall"]); fpr = 1.0 - float(row["specificity"])
        return tpr, fpr, "youden_J (from results)"
    except Exception:
        return 0.678, 0.244, "youden_J (fallback)"


def simulate(policy, tpr, fpr, rng):
    """policy in {'none','ad'}. Returns a per-day DataFrame."""
    H = np.zeros(DAYS); svi = np.zeros(DAYS); tss = np.zeros(DAYS)
    flagged = np.zeros(DAYS); energy = np.zeros(DAYS)
    h = H0
    intervened = False; intervene_day = -1
    eff_alarm_run = 0; det_alarm_run = 0
    flag_hist = []

    for t in range(DAYS):
        # ---- biology dynamics ------------------------------------------- #
        shock = SHOCK if (D0 <= t < D0 + SHOCK_DUR) else 0.0
        k_rec = K_REC_INT if intervened else K_REC_NAT
        h = h - shock + k_rec * (H_TARGET - h)
        h = float(np.clip(h, 0.05, 0.95))
        H[t] = h

        # health that the effluent "sees" (lagged) -> microfauna lead it
        h_eff = np.mean(H[max(0, t - EFF_LAG):t + 1])
        svi[t] = SVI_BASE + SVI_GAIN * (1 - h_eff)
        tss[t] = (TSS_BASE + TSS_GAIN * (1 - h_eff)
                  + TSS_SVI_PEN * max(0.0, svi[t] - SVI_BULK))

        # ---- detector reading (real TPR/FPR) ---------------------------- #
        true_stressed = 1 - h
        p_flag = true_stressed * tpr + (1 - true_stressed) * fpr
        obs = rng.binomial(N_FIELDS, p_flag) / N_FIELDS
        flagged[t] = obs
        flag_hist.append(obs)
        smooth = np.mean(flag_hist[-2:])

        # ---- decide on corrective action -------------------------------- #
        if not intervened:
            if policy == "ad":
                det_alarm_run = det_alarm_run + 1 if smooth >= ALARM_LEVEL else 0
                if det_alarm_run >= CONSEC:
                    intervened = True; intervene_day = t
            else:  # reactive: only the effluent limit triggers action
                eff_alarm_run = eff_alarm_run + 1 if tss[t] > TSS_LIMIT else 0
                if eff_alarm_run >= CONSEC:
                    intervened = True; intervene_day = t

        # ---- aeration energy (scales with how poor the sludge state is) -- #
        # Operators raise DO as health falls; filamentous/bulking sludge needs
        # markedly more air.  A prolonged, unmanaged episode therefore costs more
        # energy than one caught early -- regardless of when action is "declared".
        energy[t] = E_BASE * (1
                              + AER_HEALTH * (1 - h_eff)
                              + AER_BULK * max(0.0, svi[t] - SVI_BULK) / 100.0)

    return pd.DataFrame(dict(day=np.arange(DAYS), policy=policy, health=H,
                            svi=svi, eff_tss=tss, flagged_frac=flagged,
                            energy_kwh=energy)), intervene_day


def kpis(df, intervene_day, label):
    viol = df.eff_tss > TSS_LIMIT
    excess = np.clip(df.eff_tss - TSS_LIMIT, 0, None).sum()      # mg/L-days
    energy = float(df.energy_kwh.sum())
    penalty = int(viol.sum()) * PENALTY_VIOL
    return dict(scenario=label,
                action_day=int(intervene_day),
                violation_days=int(viol.sum()),
                cumulative_excess_TSS_mgL_days=round(float(excess), 1),
                days_in_bulking=int((df.svi > SVI_BULK).sum()),
                peak_SVI=round(float(df.svi.max()), 1),
                peak_eff_TSS=round(float(df.eff_tss.max()), 1),
                total_energy_kWh=round(energy, 0),
                energy_cost_usd=round(energy * ENERGY_COST, 0),
                compliance_penalty_usd=round(penalty, 0),
                total_operating_cost_usd=round(energy * ENERGY_COST + penalty, 0))


def main():
    C.ensure_dirs()
    rng = np.random.default_rng(SEED)
    tpr, fpr, src = detector_operating_point()
    print(f"detector operating point: TPR={tpr:.3f} FPR={fpr:.3f} [{src}]")

    df_none, day_none = simulate("none", tpr, fpr, np.random.default_rng(SEED))
    df_ad, day_ad = simulate("ad", tpr, fpr, np.random.default_rng(SEED))

    ts = pd.concat([df_none, df_ad], ignore_index=True)
    ts.to_csv(os.path.join(C.RES_CSV, "wwtp_simulation_timeseries.csv"),
              index=False)

    k_none = kpis(df_none, day_none, "WITHOUT anomaly detection (reactive)")
    k_ad = kpis(df_ad, day_ad, "WITH anomaly detection (proactive)")
    lead = day_none - day_ad
    e_save = k_none["total_energy_kWh"] - k_ad["total_energy_kWh"]
    cost_save = k_none["total_operating_cost_usd"] - k_ad["total_operating_cost_usd"]

    # generic "with - without" difference for every numeric KPI
    impr = {"scenario": "IMPROVEMENT (with - without)"}
    for key in k_none:
        if key == "scenario":
            continue
        impr[key] = round(k_ad[key] - k_none[key], 1)
    impr["action_day"] = -lead          # express as days earlier (positive=better)

    summ = pd.DataFrame([k_none, k_ad, impr])
    summ.to_csv(os.path.join(C.RES_CSV, "wwtp_impact_summary.csv"), index=False)
    print(summ.to_string(index=False))
    print(f"\nEarly-warning lead time: {lead} days earlier | "
          f"energy saved: {e_save:.0f} kWh | "
          f"operating cost saved: ${cost_save:.0f}")

    _figure(df_none, df_ad, day_none, day_ad, tpr, fpr)
    print("wrote wwtp_simulation_timeseries.csv, wwtp_impact_summary.csv, "
          "wwtp_with_vs_without_ad.png")


def _figure(df_none, df_ad, day_none, day_ad, tpr, fpr):
    d = df_none.day
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # health
    ax[0, 0].plot(d, df_none.health, label="without AD", color="#e76f51")
    ax[0, 0].plot(d, df_ad.health, label="with AD", color="#2a9d8f")
    ax[0, 0].axvspan(D0, D0 + SHOCK_DUR, color="grey", alpha=0.12,
                     label="disturbance")
    ax[0, 0].axvline(day_ad, color="#2a9d8f", ls="--", lw=1)
    ax[0, 0].axvline(day_none, color="#e76f51", ls="--", lw=1)
    ax[0, 0].set_title("Sludge health (microfauna community state)")
    ax[0, 0].set_ylabel("health  (1 = healthy)"); ax[0, 0].legend(fontsize=8)

    # detector signal
    ax[0, 1].plot(d, df_ad.flagged_frac, color="#264653",
                  label="detector flagged-stressed fraction")
    ax[0, 1].plot(d, 1 - df_ad.health, color="#888", ls=":",
                  label="true stressed fraction")
    ax[0, 1].axhline(ALARM_LEVEL, color="r", ls="--", lw=1, label="alarm level")
    ax[0, 1].axvline(day_ad, color="#2a9d8f", ls="--", lw=1,
                     label=f"AD acts (day {day_ad})")
    ax[0, 1].set_title(f"DINOv3 detector signal (TPR={tpr:.2f}, FPR={fpr:.2f})")
    ax[0, 1].set_ylabel("fraction"); ax[0, 1].legend(fontsize=8)

    # effluent
    ax[1, 0].plot(d, df_none.eff_tss, label="without AD", color="#e76f51")
    ax[1, 0].plot(d, df_ad.eff_tss, label="with AD", color="#2a9d8f")
    ax[1, 0].axhline(TSS_LIMIT, color="k", ls="--", lw=1, label="discharge limit")
    ax[1, 0].fill_between(d, TSS_LIMIT, df_none.eff_tss,
                          where=df_none.eff_tss > TSS_LIMIT,
                          color="#e76f51", alpha=0.25)
    ax[1, 0].fill_between(d, TSS_LIMIT, df_ad.eff_tss,
                          where=df_ad.eff_tss > TSS_LIMIT,
                          color="#2a9d8f", alpha=0.25)
    ax[1, 0].set_title("Effluent TSS (treatment quality)")
    ax[1, 0].set_xlabel("day"); ax[1, 0].set_ylabel("effluent TSS (mg/L)")
    ax[1, 0].legend(fontsize=8)

    # cumulative energy
    ax[1, 1].plot(d, np.cumsum(df_none.energy_kwh), label="without AD",
                  color="#e76f51")
    ax[1, 1].plot(d, np.cumsum(df_ad.energy_kwh), label="with AD",
                  color="#2a9d8f")
    saved = np.cumsum(df_none.energy_kwh).iloc[-1] - np.cumsum(df_ad.energy_kwh).iloc[-1]
    ax[1, 1].set_title(f"Cumulative aeration energy (saved {saved:.0f} kWh)")
    ax[1, 1].set_xlabel("day"); ax[1, 1].set_ylabel("kWh"); ax[1, 1].legend(fontsize=8)

    fig.suptitle("Microfauna anomaly detection as a WWTP component: "
                 "with vs without", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RES_FIG, "wwtp_with_vs_without_ad.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
