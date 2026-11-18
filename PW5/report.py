import pandas as pd

df = pd.read_csv("results/events.csv")

report_lines = ["# Seismic Events Summary Report\n"]
for idx, row in df.iterrows():
    report_lines.append(
        f"File: {row['file']}\n"
        f"  - P-wave arrival (tp): {row['tp']} s\n"
        f"  - S-wave arrival (ts): {row['ts']} s\n"
        f"  - Distance (R): {row['R_km']} km\n"
        f"  - Origin time: {row['origin_time']} s\n"
        f"  - Peak Ground Velocity (PGV): {row['pgv_cm_s']} cm/s\n"
        f"  - Magnitude (M): {row['magnitude']}\n"
    )

r_a, r_b = df.loc[0, "R_km"], df.loc[1, "R_km"]
m_a, m_b = df.loc[0, "magnitude"], df.loc[1, "magnitude"]

closer = "A" if r_a < r_b else "B"
stronger = "A" if m_a > m_b else "B"

report_lines.append(
    f"\nConclusion: Earthquake {closer} was closer to the station, and earthquake {stronger} was stronger."
)

with open("results/report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Report generated in results/report.txt")