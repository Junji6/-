【APIについて】
概要：このAPIは、ユーザーが入力する車両の特性（シリンダー数、排気量、馬力、重量、加速時間、モデル年、原産地）を基に、MPG（Miles Per Gallon: 燃費）を予測します。
HTTPメソッド：POST　
URL："http://127.0.0.1:8000/make_predictions
リクエストボディ: JSON形式　　
・cylinders: エンジンのシリンダー数
・displacement: 排気量（立方インチ）
・weight: 車両の重量（ポンド）
・model_year: 車両のモデル年
・レスポンス: 予測された MPG 値を返します。
・mpg_prediction: 予測された燃費（MPG）

※リクエストサンプル
Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:8000/make_predictions" `
  -Headers @{accept="application/json"; "Content-Type"="application/json"} `
  -Body '{"cylinders": 5, "displacement": 105, "weight": 2230.0, "modelyear": 80}'
