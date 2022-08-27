from fastapi import FastAPI
import joblib
import uvicorn
import logging
import os
from schemas import data_file

LOG_DIR = "LOGS"
os.makedirs(LOG_DIR, exist_ok=True)



logging.basicConfig(filename=os.path.join(LOG_DIR, "app_logs.log"),    level=logging.INFO,
                    filemode='a',
                    format='[%(asctime)s: %(levelname)s: %(module)s]: >>>>>>>>>  %(message)s')

MODEL = joblib.load("SAVED_MODELS/model.pkl")

app = FastAPI(debug=True,title="lungs cancer prediction",description="an application which can help to predict the lungs_cancer based on trained features")



@app.get("/")
def home():
    return "service is live"


@app.post('/lungs_cancer_prediction')
def prediction(input:data_file):
    """_summary_:

            Args:
                input (data_file): datafile containing all the features in numeric form as follows
                GENDER = 1 represents male, 0 represents female
                SMOKING = 1 represents yes, 2 represents NO
                YELLOW_FINGERS = 1 represents yes, 2 represents NO
                ANXIETY = 1 represents yes, 2 represents NO
                PEER_PRESSURE = 1 represents yes, 2 represents NO
                CHRONIC_DISEASE = 1 represents yes, 2 represents NO
                FATIGUE = 1 represents yes, 2 represents NO
                ALLERGY = 1 represents yes, 2 represents NO
                WHEEZING = 1 represents yes, 2 represents NO
                ALCOHOL_CONSUMING = 1 represents yes, 2 represents NO
                COUGHING = 1 represents yes, 2 represents NO
                SHORTNESS_OF_BREATH = 1 represents yes, 2 represents NO
                SWALLOWING_DIFFICULTY = 1 represents yes, 2 represents NO
                CHEST_PAIN = 1 represents yes, 2 represents NO


        Returns:

            str: it will return weather patient have a lungs cancer or not
    """
    try:
        data = input.dict()
        
    
        GENDER = data["GENDER"]
        AGE  =    data["AGE"]
        SMOKING  =     data["SMOKING"]
        YELLOW_FINGERS   =     data["YELLOW_FINGERS"]
        ANXIETY = data["ANXIETY"]
        PEER_PRESSURE     =   data["PEER_PRESSURE"]
        CHRONIC_DISEASE    =    data["CHRONIC_DISEASE"]
        FATIGUE    =   data["FATIGUE"]
        ALLERGY     =   data["ALLERGY"]
        WHEEZING     =   data["WHEEZING"]
        ALCOHOL_CONSUMING =       data["ALCOHOL_CONSUMING"]
        COUGHING    =    data["COUGHING"]
        SHORTNESS_OF_BREATH =       data["SHORTNESS_OF_BREATH"]
        SWALLOWING_DIFFICULTY =       data["SWALLOWING_DIFFICULTY"]
        CHEST_PAIN     =   data["CHEST_PAIN"]
            

        features = [[GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,
        ALLERGY,WHEEZING,ALCOHOL_CONSUMING,COUGHING,
        SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]]

        
        prediction = (MODEL.predict(features)).tolist()[0]
        logging.info(f"Prediction is {prediction}")

        if prediction==0:
            result =  "Patient is more likely to not have lungs cancer"
        else:
            result =  "Patient is more likely to have lungs cancer"
        logging.info(f"The result for the features {features} is {result}")
        return result
    except Exception as e:
        logging.error(e)



if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port="8000")