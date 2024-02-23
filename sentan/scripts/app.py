import time
import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

@app.get("/")
def get_html() -> HTMLResponse:
    return HTMLResponse("""
<html>
<head>
    <title>Hello</title>
</head>
<body>
    <h1>Title</h1>
    <br>
    <p>This parragraph.</p>
</body>
</html>
""")

class JsonResponse(BaseModel):
    name: str
    new_name: str

@app.post("/json")
def get_json(name: str) -> JsonResponse:
    return {"name": name, "new_name": f"{name} Pablo"}

class ApiOutput(BaseModel):
    text: str
    prediction: str
    time: float

@app.post("/model")
def model_prediction(text: str) -> ApiOutput:
    t0 = time.time()
    model = joblib.load("model.joblib")
    prediction = str(model.predict([text]).flatten())
    delta_t = time.time() - t0
    return ApiOutput(
            text=text,
            prediction=prediction,
            time=delta_t
            )
