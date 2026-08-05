"""RotorPy-compatible reduced-order tiltrotor simulation package."""
from .commands import AutomaticMission, CommandManager, FlightPhase, FlightSetpoint
from .controller import TiltrotorController
from .mission import MissionTrajectory
from .parameters import default_tiltrotor_params, initial_state
from .simulator import TiltrotorSimulation
from .vehicle import Tiltrotor

__all__ = [
    "AutomaticMission",
    "CommandManager",
    "FlightPhase",
    "FlightSetpoint",
    "TiltrotorController",
    "MissionTrajectory",
    "default_tiltrotor_params",
    "initial_state",
    "TiltrotorSimulation",
    "Tiltrotor",
]
