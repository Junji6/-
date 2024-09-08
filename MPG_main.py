#ライブラリのインポート
from fastapi import FastAPI
from pydantic import BaseModel 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd 
import numpy as np
import pickle

#FastAPIのインスタンス化
app = FastAPI()

#入力するデータ型の定義
class MPG(BaseModel):
    cylinders    : int
    displacement : float
    weight       : float
    modelyear   : int

#学習済みモデルの読み込み
mpg_model = pickle.load(open('mpg_model','rb'))

#トップページ
@app.get("/")
async def index():
    return {"MPG":"MPG_predictions"}

#POSTが送信された時（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def make_predictions(features:MPG):
    return ({'prediction':str(mpg_model.predict([[features.cylinders , features.displacement , features.weight , features.modelyear]])[0])})
    

