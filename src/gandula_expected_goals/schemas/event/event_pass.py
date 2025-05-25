from pydantic import BaseModel


class Pass(BaseModel):
    event_type: str
    event_name: str
    x: float
    y: float
    target_x: float
    target_y: float
