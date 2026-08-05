from rotorpy_tiltrotor import CommandManager, FlightPhase


def test_command_sequence_api():
    c = CommandManager()
    c.vertical_takeoff(0.0, 30.0)
    assert c.setpoint.phase == FlightPhase.VERTICAL_TAKEOFF
    c.transition_to_cruise(10.0, 30.0, 15.0)
    assert c.setpoint.phase == FlightPhase.TRANSITION_TO_CRUISE
    c.cruise(25.0, 30.0, 15.0)
    assert c.setpoint.phase == FlightPhase.CRUISE
    c.transition_to_hover(40.0, 30.0, 200.0, 0.0)
    assert c.setpoint.phase == FlightPhase.TRANSITION_TO_HOVER
    c.vertical_land(60.0, 200.0, 0.0)
    assert c.setpoint.phase == FlightPhase.VERTICAL_LANDING
