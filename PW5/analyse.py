import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram


def bandpass(x, fs, a, b):
    N = len(x)
    f = np.fft.fftfreq(N, 1 / fs)
    X = np.fft.fft(x)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real


def analyse(csv_file):
    d = pd.read_csv(csv_file)
    t = d["time_s"].to_numpy()
    z = d["Z"].to_numpy()
    n = d["N"].to_numpy()
    e = d["E"].to_numpy()

    fs = 1 / (t[1] - t[0])

    z_clean = bandpass(z, fs, 0.5, 10)
    n_clean = bandpass(n, fs, 0.5, 10)
    e_clean = bandpass(e, fs, 0.5, 10)

    w = int(0.5 * fs)
    window = np.ones(w) / w
    z_env = np.convolve(np.abs(z_clean), window, mode="same")
    horiz_motion = np.sqrt(n_clean**2 + e_clean**2)
    horiz_env = np.convolve(np.abs(horiz_motion), window, mode="same")

    tp_idx = np.argmax(z_env > 0.05 * np.max(z_env))
    tp = t[tp_idx]

    search_start_idx = tp_idx + int(2 * fs)
    ts_idx = search_start_idx + np.argmax(
        horiz_env[search_start_idx:] > 0.1 * np.max(horiz_env)
    )
    ts = t[ts_idx]

    v_p = 6.2
    v_s = 3.6
    R = (ts - tp) / ((1 / v_s) - (1 / v_p))
    origin_time = tp - (R / v_p)

    raw_horiz = np.sqrt(n**2 + e**2) * 100
    pgv = np.max(raw_horiz)
    magnitude = (np.log10(pgv) + np.log10(R) + 1.0) / 0.55

    return {
        "file": csv_file,
        "tp": round(tp, 2),
        "ts": round(ts, 2),
        "R_km": round(R, 2),
        "origin_time": round(origin_time, 2),
        "pgv_cm_s": round(pgv, 4),
        "magnitude": round(magnitude, 2),
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = []
    for f_path in ["data/seismogram_A.csv", "data/seismogram_B.csv"]:
        results.append(analyse(f_path))

    df_res = pd.DataFrame(results)
    df_res.to_csv("results/events.csv", index=False)
    print("Results saved to results/events.csv")