"""v0.4.2 operational dashboard corrections.

For this project's low-altitude urban transition demonstration, the default
continuous-turbulence benchmark is based on the EASA CS-AWO low-altitude model:

* Gaussian process;
* Von Karman-type spectrum;
* scale length L = 183 m (600 ft);
* RMS turbulence intensity sigma = 0.15 U, where U is reference mean wind.

The finite synthesis is restricted to 0.005--0.5 Hz because the current vehicle
is a reduced-order rigid-body model without structural vibration modes.  The
underlying Von Karman shape and CS-AWO scale/intensity assumptions are retained,
but this is an engineering controller benchmark, not certification evidence.
"""
from __future__ import annotations

import numpy as np
from bokeh.models import Div, NumericInput, Select

from . import dashboard as base_dashboard
from . import enhanced_dashboard as enhanced


CS_AWO_TURBULENCE_SCALE_M = 183.0
CS_AWO_SIGMA_RATIO = 0.15
CS_AWO_REFERENCE_AIRSPEED_MPS = 15.0
CS_AWO_COMPONENTS = 256
CS_AWO_MIN_FREQUENCY_HZ = 0.005
CS_AWO_MAX_FREQUENCY_HZ = 0.5
CS_AWO_FADE_IN_S = 3.0
CS_AWO_LABEL = "Continuous turbulence (CS-AWO low-altitude style)"


class LowAltitudeCSAWOWindModel(enhanced.FixedUrbanWindModel):
    """One-dimensional vertical CS-AWO-style low-altitude turbulence.

    The generic dashboard disturbance input is interpreted as reference mean
    wind U for this mode. The turbulence RMS is sigma=0.15 U. If a larger
    horizontal base-wind magnitude is selected, that magnitude is used as U so
    turbulence intensity is not understated relative to mean wind.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._awo_key = None
        self._awo_omega = np.zeros(0)
        self._awo_amplitude = np.zeros(0)
        self._awo_phase = np.zeros(0)

    def reset(self) -> None:
        super().reset()
        self._awo_key = None
        self._awo_omega = np.zeros(0)
        self._awo_amplitude = np.zeros(0)
        self._awo_phase = np.zeros(0)

    def _ensure_awo_basis(self, cfg) -> None:
        key = (
            int(cfg.random_seed),
            CS_AWO_REFERENCE_AIRSPEED_MPS,
            CS_AWO_MIN_FREQUENCY_HZ,
            CS_AWO_MAX_FREQUENCY_HZ,
        )
        if key == self._awo_key:
            return

        # At V=15 m/s and L=183 m, the characteristic rigid-body turbulence
        # frequency is O(V/L) ~ 0.08 Hz. The 0.5-Hz upper bound preserves the
        # relevant low-frequency tail while avoiding artificial structural-band
        # content that this 6-DOF model cannot represent faithfully.
        frequency_hz = np.logspace(
            np.log10(CS_AWO_MIN_FREQUENCY_HZ),
            np.log10(CS_AWO_MAX_FREQUENCY_HZ),
            CS_AWO_COMPONENTS,
        )
        omega = 2.0 * np.pi * frequency_hz
        reduced = omega / CS_AWO_REFERENCE_AIRSPEED_MPS
        x = 1.339 * CS_AWO_TURBULENCE_SCALE_M * reduced
        phi_spatial = (
            CS_AWO_TURBULENCE_SCALE_M
            / np.pi
            * (1.0 + (8.0 / 3.0) * x * x)
            / np.power(1.0 + x * x, 11.0 / 6.0)
        )
        phi_temporal = phi_spatial / CS_AWO_REFERENCE_AIRSPEED_MPS
        domega = np.gradient(omega)
        amplitude = np.sqrt(np.maximum(0.0, 2.0 * phi_temporal * domega))
        expected_rms = float(np.sqrt(0.5 * np.sum(amplitude * amplitude)))
        if expected_rms > 1e-12:
            amplitude /= expected_rms

        rng = np.random.default_rng(int(cfg.random_seed))
        self._awo_omega = omega
        self._awo_amplitude = amplitude
        self._awo_phase = rng.uniform(
            0.0, 2.0 * np.pi, len(self._awo_omega)
        )
        self._awo_key = key

    def _continuous_turbulence(self, t, elapsed, cfg):
        del t
        if elapsed < cfg.start_time_s:
            return np.zeros(3), False

        self._ensure_awo_basis(cfg)
        local_time = float(elapsed - cfg.start_time_s)
        unit_value = float(np.sum(
            self._awo_amplitude
            * np.cos(self._awo_omega * local_time + self._awo_phase)
        ))
        fade = self._smoothstep01(local_time / CS_AWO_FADE_IN_S)
        horizontal_base = float(np.linalg.norm(
            np.asarray(cfg.base_wind_mps, dtype=float)[:2]
        ))
        reference_mean_wind = max(
            float(cfg.disturbance_amplitude_mps),
            horizontal_base,
        )
        sigma = CS_AWO_SIGMA_RATIO * reference_mean_wind
        vertical_velocity = sigma * fade * unit_value
        return np.array([0.0, 0.0, vertical_velocity]), True


def _find(doc, model_type, title=None):
    for model in doc.select({"type": model_type}):
        if title is None or getattr(model, "title", None) == title:
            yield model


def build_dashboard(doc, simulation=None):
    """Build the v0.4.2 low-altitude operational dashboard."""
    enhanced.FixedUrbanWindModel = LowAltitudeCSAWOWindModel
    sim = enhanced.build_dashboard(doc, simulation=simulation)

    wind_mode = next(_find(doc, Select, title="Wind scenario"))
    old_label = "Continuous turbulence (FAR/CS 25.341-style)"
    wind_mode.options = [
        CS_AWO_LABEL if item == old_label else item
        for item in wind_mode.options
    ]
    base_dashboard.WIND_MODE_LABELS[CS_AWO_LABEL] = "continuous_turbulence"
    base_dashboard.WIND_MODE_LABELS.pop(old_label, None)

    amplitude = next(
        model for model in doc.select({"type": NumericInput})
        if model.title in {
            "Disturbance amplitude [m/s]",
            "Turbulence RMS sigma [m/s]",
        }
    )

    def set_low_alt_semantics(attr, old, new):
        del attr, old, new
        if wind_mode.value == CS_AWO_LABEL:
            amplitude.title = "Reference mean wind U [m/s] (sigma = 0.15 U)"
            if float(amplitude.value or 0.0) <= 1.1:
                amplitude.value = 5.0
        elif amplitude.title.startswith("Reference mean wind"):
            amplitude.title = "Disturbance amplitude [m/s]"

    wind_mode.on_change("value", set_low_alt_semantics)

    for div in doc.select({"type": Div}):
        if "v0.4.2 interpretation" in str(div.text):
            div.text = (
                "<h3>v0.4.2 interpretation</h3>"
                "<b>Comfort guard:</b> 1.50 m/s³ remains the measured passenger-"
                "motion target; internal command slew uses 60% headroom to "
                "account for plant dynamics. <b>Continuous turbulence:</b> the "
                "operational low-altitude benchmark is CS-AWO-style, with "
                "Gaussian/Von-Karman behaviour, L=183 m (600 ft), sigma=0.15U, "
                "and a 0.005–0.5 Hz rigid-body synthesis band. The input is "
                "reference mean wind U, not turbulence standard deviation. "
                "FAR/CS 25.341 is a separate transport-airplane structural-load "
                "benchmark and is not the routine 30-m urban-transition default."
            )
            break

    set_low_alt_semantics(None, None, None)
    return sim
