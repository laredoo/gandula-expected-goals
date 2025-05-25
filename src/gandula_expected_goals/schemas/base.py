from datetime import datetime, timedelta
from typing import Optional, Union
from pydantic import BaseModel


class Period(BaseModel):
    id: int
    start_timestamp: Union[datetime, timedelta]
    end_timestamp: Optional[Union[datetime, timedelta]] = None
    duration: timedelta


class Time(BaseModel):
    period: Period
    timestamp: timedelta


class Point(BaseModel):
    x: float
    y: float


class Team(BaseModel):
    team_id: str
    name: str


class Player(BaseModel):
    player_id: str
    team: Team
    jersey_no: int
    name: str
    starting: bool = False
