from bokeh.document import Document
from bokeh.models import Toggle

from rotorpy_tiltrotor.dashboard import build_dashboard


def test_dashboard_builds_wind_and_comfort_toggles():
    doc = Document()
    sim = build_dashboard(doc)

    assert sim is not None
    assert len(doc.roots) == 1
    labels = {toggle.label for toggle in doc.select({"type": Toggle})}
    assert "Urban wind / gust: OFF" in labels
    assert "Comfort monitor: ON" in labels
    assert "Comfort accel guard: OFF" in labels
