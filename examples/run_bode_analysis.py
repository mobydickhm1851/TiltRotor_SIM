"""Print the local Bode-analysis metrics used by the dashboard and Q4 report."""
from __future__ import annotations

from rotorpy_tiltrotor.stability_analysis import summary_rows


def main() -> None:
    print("Local small-signal Bode summary")
    print("loop | stable | PM_deg | wgc_rad_s | BW_rad_s | peak_T_dB | grade")
    for row in summary_rows():
        print(
            f"{row['loop']} | {row['stable']} | "
            f"{row['phase_margin_deg']:.3f} | "
            f"{row['gain_crossover_rad_s']:.3f} | "
            f"{row['bandwidth_rad_s']:.3f} | "
            f"{row['closed_loop_peak_db']:.3f} | "
            f"{row['grade']}"
        )
        print(f"  poles: {row['poles']}")


if __name__ == "__main__":
    main()
