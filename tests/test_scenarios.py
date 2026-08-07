import numpy as np

from rotorpy_tiltrotor.scenarios import (
    ComfortLimits,
    PassengerComfortMonitor,
    UrbanWindModel,
    WindScenarioConfig,
)


def test_discrete_gust_is_smooth_bounded_and_relative_to_enable():
    model = UrbanWindModel(WindScenarioConfig(
        enabled=True,
        mode="discrete_gust",
        base_wind_mps=np.array([1.0, 0.0, 0.0]),
        disturbance_amplitude_mps=6.0,
        disturbance_direction_deg=90.0,
        start_time_s=10.0,
        duration_s=4.0,
    ))
    state = {"x": np.array([0.0, 0.0, 30.0])}

    enabled = model.sample(100.0, state)
    before = model.sample(109.0, state)
    middle = model.sample(112.0, state)
    after = model.sample(115.0, state)

    assert np.allclose(enabled.vector_mps, [1.0, 0.0, 0.0])
    assert np.allclose(before.vector_mps, [1.0, 0.0, 0.0])
    assert middle.active
    assert middle.disturbance_mps[1] > 5.9
    assert np.linalg.norm(middle.disturbance_mps) <= 6.1
    assert np.allclose(after.vector_mps, [1.0, 0.0, 0.0])


def test_wind_shear_depends_on_altitude_and_is_limited():
    model = UrbanWindModel(WindScenarioConfig(
        enabled=True,
        mode="wind_shear",
        disturbance_amplitude_mps=4.0,
        disturbance_direction_deg=0.0,
        shear_gradient_per_m=0.2,
        reference_altitude_m=20.0,
    ))
    low = model.sample(0.0, {"x": np.array([0.0, 0.0, 10.0])})
    high = model.sample(0.0, {"x": np.array([0.0, 0.0, 50.0])})

    assert np.isclose(low.disturbance_mps[0], -2.0)
    assert np.isclose(high.disturbance_mps[0], 4.0)


def test_urban_wake_is_repeatable():
    cfg = WindScenarioConfig(
        enabled=True,
        mode="urban_wake",
        disturbance_amplitude_mps=5.0,
        start_time_s=1.0,
        duration_s=20.0,
        wake_frequency_hz=0.4,
    )
    state = {"x": np.array([0.0, 0.0, 30.0])}
    model_a = UrbanWindModel(cfg)
    model_b = UrbanWindModel(cfg)
    model_a.sample(0.0, state)
    model_b.sample(0.0, state)
    a = model_a.sample(5.0, state)
    b = model_b.sample(5.0, state)

    assert a.active
    assert np.allclose(a.vector_mps, b.vector_mps)
    assert np.linalg.norm(a.disturbance_mps) > 0.0


def test_comfort_monitor_reports_rms_peaks_and_exceedance():
    monitor = PassengerComfortMonitor(
        limits=ComfortLimits(
            acceleration_mps2=1.0,
            jerk_mps3=2.0,
            angular_rate_deg_s=5.0,
            rms_window_s=2.0,
        ),
        enabled=True,
    )

    first = monitor.update(
        0.0,
        np.array([0.5, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.deg2rad(np.array([2.0, 0.0, 0.0])),
    )
    second = monitor.update(
        0.5,
        np.array([2.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
        np.deg2rad(np.array([8.0, 0.0, 0.0])),
    )

    assert first["comfort_alert"] == 0.0
    assert second["comfort_alert"] == 1.0
    assert second["comfort_index"] >= 2.0
    assert second["comfort_accel_rms_mps2"] > 1.0
    assert second["comfort_peak_accel_mps2"] == 2.0
    assert second["comfort_peak_jerk_mps3"] == 3.0
    assert second["comfort_peak_rate_deg_s"] == 8.0
    assert np.isclose(second["comfort_exceedance_time_s"], 0.5)


def test_disabled_comfort_monitor_does_not_accumulate_exceedance():
    monitor = PassengerComfortMonitor(enabled=False)
    result = monitor.update(
        1.0,
        np.array([9.0, 0.0, 0.0]),
        np.array([9.0, 0.0, 0.0]),
        np.deg2rad(np.array([90.0, 0.0, 0.0])),
    )

    assert result["comfort_enabled"] == 0.0
    assert result["comfort_alert"] == 0.0
    assert result["comfort_exceedance_time_s"] == 0.0
