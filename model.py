import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_absolute_error,confusion_matrix,f1_score
import pickle
import os
import logging

MODEL_DIR = "SAVED_MODELS"
LOG_DIR = "LOGS"
os.makedirs(LOG_DIR, exist_ok=True)

data_path = "Processed_data\\final_data.csv"

logging.basicConfig(filename=os.path.join(LOG_DIR, "running_logs.log"),    level=logging.INFO,
                    filemode='a',
                    format='[%(asctime)s: %(levelname)s: %(module)s]: >>>>>>>>>  %(message)s')



def main(data_path):
        data = pd.read_csv(data_path)
        data = data.drop_duplicates()
        Encoder = LabelEncoder()
        data["GENDER"] = Encoder.fit_transform(data["GENDER"])
        data["LUNG_CANCER"] = Encoder.fit_transform(data["LUNG_CANCER"])
        logging.info("categorical features converted into numerical format")
    
        X = data.drop(["LUNG_CANCER"],axis=1)
        Y = data["LUNG_CANCER"]
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
        print(x_test.columns)
        logging.info("data is splitted in training and test data")

        CLASSIFIER = LogisticRegression(random_state=0)
        CLASSIFIER.fit(x_train,y_train)
        logging.info("Model Training is completed")

        PREDICTIONS = CLASSIFIER.predict(x_test)
        print("x_test",x_test)
        logging.info(f"predictions are {PREDICTIONS}")

        accuracy_score_ = accuracy_score(y_test,PREDICTIONS)
        mean_absolute_error_ = mean_absolute_error(y_test,PREDICTIONS)
        confusion_matrix_ = confusion_matrix(y_test,PREDICTIONS)
        f1_score_ = f1_score(y_test,PREDICTIONS)
        logging.info(f"model accuracy score is = {accuracy_score_} \n mean_absolute_error_ is = {mean_absolute_error_} \n confusion_matrix_ is \n {confusion_matrix_} \n f1_score_ is  = {f1_score_}")
        

        os.makedirs(MODEL_DIR,exist_ok=True)
        with open(f"{MODEL_DIR}/model.pkl","wb") as f:
            pickle.dump(CLASSIFIER,f)
    

if __name__=="__main__":
    main(data_path=data_path)
   