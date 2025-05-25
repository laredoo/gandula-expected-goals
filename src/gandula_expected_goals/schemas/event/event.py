from gandula_expected_goals.schemas.base import Time
from gandula_expected_goals.schemas.event.event_pass import Pass, Point, Team, Player
from typing import Dict, Any
from pydantic import BaseModel


class EventGeneric(BaseModel):
    event_id: str
    time: Time
    coordinates: Point
    team: Team
    player: Player
    ball_owning_team: Team
    ball_state: str
    raw_event: Dict
    attacking_direction: str


class EventHandler(EventGeneric):
    event_generic=None
    event_pass: Pass
    shot=None
    takeOn=None
    carry=None
    clearance=None
    interception=None
    duel=None
    substitution=None
    card=None
    playerOn=None
    playerOff=None
    recovery=None
    miscontrol=None
    ballOut=None
    foulCommitted=None
    pressure=None
    formationChange=None

