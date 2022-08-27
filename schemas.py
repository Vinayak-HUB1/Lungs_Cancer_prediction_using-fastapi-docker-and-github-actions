
from pydantic import BaseModel



class data_file(BaseModel):
    GENDER: int = 1 
    AGE: int = 26 
    SMOKING: int = 2 
    YELLOW_FINGERS: int = 1 
    ANXIETY: int= 1 
    PEER_PRESSURE: int = 1 
    CHRONIC_DISEASE : int = 1 
    FATIGUE: int = 1 
    ALLERGY: int = 1 
    WHEEZING: int = 1 
    ALCOHOL_CONSUMING: int = 2 
    COUGHING: int = 2 
    SHORTNESS_OF_BREATH: int = 2 
    SWALLOWING_DIFFICULTY: int = 2 
    CHEST_PAIN : int = 2
    