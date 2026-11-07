from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Dış API'ye gidecek bir değer olduğu için sıkı doğrulama: 1-5 harf, opsiyonel .X / -X (örn. BRK.B)
    symbol: str = Field(..., pattern=r"^[A-Za-z]{1,5}([.-][A-Za-z])?$", examples=["AAPL"])


class ModelRef(BaseModel):
    name: str
    trained_at: Optional[str] = None


class PredictResponse(BaseModel):
    symbol: str
    as_of: date = Field(..., description="Feature'ların hesaplandığı son kapanmış işlem günü")
    horizon_days: int
    probability_up: float = Field(..., ge=0, le=1, description="P(kapanış, horizon gün sonra bugünkünden yüksek)")
    predicted_direction: Literal["up", "down"]
    threshold: float = 0.5
    sentiment_used: bool
    symbol_in_training_set: Optional[bool] = None
    warnings: list[str] = []
    model: ModelRef


class ModelInfo(BaseModel):
    name: str
    model_type: Optional[str] = None
    trained_at: Optional[str] = None
    horizon_days: int
    n_features: int
    uses_sentiment: bool
    train_symbols: list[str] = []
    test_roc_auc: Optional[float] = None
    test_accuracy: Optional[float] = None
    always_up_accuracy: Optional[float] = None
    disclaimer: str = (
        "Eğitim/araştırma projesi; yatırım tavsiyesi değildir. Hisse yönü tahmini çok gürültülüdür; "
        "test AUC'sine ve her zaman 'yükselir' diyen referansa bak."
    )


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
