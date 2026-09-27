from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/raw/dataset.csv"; MODEL_DIR=ROOT/"models"; MODEL_DIR.mkdir(exist_ok=True)
if not DATA.exists():
    DATA.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve("https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",DATA)
df=pd.read_csv(DATA); X=df.drop(columns=["target"]); y=df["target"]
num=X.select_dtypes(include=np.number).columns.tolist()
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num)])
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={"Logistic Regression":LogisticRegression(max_iter=2000),"SVM":SVC(probability=True,random_state=42),"Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1)}
for name,est in models.items():
    pipe=Pipeline([("preprocess",pre),("model",est)]); pipe.fit(Xt,yt); pred=pipe.predict(Xv)
    print(name,{"accuracy":accuracy_score(yv,pred),"precision":precision_score(yv,pred,zero_division=0),"recall":recall_score(yv,pred,zero_division=0),"f1":f1_score(yv,pred,zero_division=0),"roc_auc":roc_auc_score(yv,pipe.predict_proba(Xv)[:,1])})
final=Pipeline([("preprocess",pre),("model",models["Random Forest"])]); final.fit(Xt,yt); dump(final,MODEL_DIR/"model.joblib")
