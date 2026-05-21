from pydantic import BaseModel

class FacialAnalyst(BaseModel):
    frames: list[str]
    duration: int