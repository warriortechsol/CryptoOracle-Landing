from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import predict_price
from sentiment import get_sentiment_score
from recommender import get_recommendation

app = FastAPI()

class PredictionRequest(BaseModel):
    symbol: str
    current_price: float

@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        predicted_price = predict_price(req.symbol)
        sentiment_score = get_sentiment_score(req.symbol)
        recommendation = get_recommendation(req.current_price, predicted_price, sentiment_score)

        return {
            "symbol": req.symbol,
            "predicted_price": predicted_price,
            "sentiment_score": sentiment_score,
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
