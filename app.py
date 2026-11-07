"""
FastAPI serving katmanı.

Çalıştır:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000
    # sonra:  curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"symbol":"AAPL"}'

Uçlar:
    GET  /health   canlılık + model yüklü mü (kimlik doğrulama yok, orkestratörler için)
    GET  /model    model metadata'sı (eğitim tarihi, test metrikleri, sembol listesi)
    POST /predict  {"symbol": "AAPL"} -> yükseliş olasılığı

Tasarım:
- Model eksik/bozuksa servis AYAKTA kalır: /health "degraded", /predict 503 döner.
- Uç noktalar `def` (async değil): arkada bloklayan HTTP istemcileri (requests) var,
  FastAPI bunları thread havuzunda çalıştırır; event loop tıkanmaz.
- API_AUTH_TOKEN doluysa istemci `X-API-Key` göndermek zorunda: servis ücretli/kotalı
  veri anahtarlarını kullandığı için internete açık ve kimliksiz bırakma.
"""
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException

from data_pipeline.clients.finnhub_client import FinnhubClient
from data_pipeline.clients.twelve_data_client import TwelveDataClient
from model_training.dataset import InsufficientHistoryError
from serving.feature_service import DataUnavailableError, FeatureService
from serving.model_store import ModelBundle, load_bundle
from serving.schemas import Health, ModelInfo, ModelRef, PredictRequest, PredictResponse
from serving.settings import Settings

logger = logging.getLogger(__name__)
THRESHOLD = 0.5


def build_default_feature_service(settings: Settings) -> FeatureService:
    scorer = None
    if settings.sentiment_scorer == "deepseek":
        from feature_engineering.llm_sentiment import DeepSeekSentimentScorer
        scorer = DeepSeekSentimentScorer()
    return FeatureService(
        price_client=TwelveDataClient(), news_client=FinnhubClient(), scorer=scorer,
        history_rows=settings.history_rows, news_days=settings.news_days, ttl_seconds=settings.cache_ttl_seconds,
    )


def create_app(bundle: Optional[ModelBundle] = None, feature_service: Optional[FeatureService] = None,
               settings: Optional[Settings] = None) -> FastAPI:
    """Bağımlılıklar enjekte edilebilir (testler sahte model/veri sağlayıcı verir)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = bundle or load_bundle(settings.model_path)
        app.state.features = feature_service or build_default_feature_service(settings)
        yield

    app = FastAPI(title="Stock Direction API", version="1.0.0", lifespan=lifespan)

    def require_auth(x_api_key: Optional[str] = Header(default=None)):
        if not settings.auth_token:
            return
        if not x_api_key or not secrets.compare_digest(x_api_key, settings.auth_token):
            raise HTTPException(status_code=401, detail="Geçersiz veya eksik X-API-Key")

    def get_bundle() -> ModelBundle:
        b = app.state.bundle
        if b is None:
            raise HTTPException(status_code=503, detail="Model yüklü değil")
        return b

    @app.get("/health", response_model=Health)
    def health():
        loaded = app.state.bundle is not None
        return Health(status="ok" if loaded else "degraded", model_loaded=loaded)

    @app.get("/model", response_model=ModelInfo, dependencies=[Depends(require_auth)])
    def model_info(b: ModelBundle = Depends(get_bundle)):
        m = b.metadata
        return ModelInfo(
            name=b.name, model_type=m.get("model_type"), trained_at=m.get("trained_at"),
            horizon_days=b.horizon, n_features=len(b.features), uses_sentiment=b.uses_sentiment,
            train_symbols=m.get("train_symbols", []), test_roc_auc=m.get("test_roc_auc"),
            test_accuracy=m.get("test_accuracy"), always_up_accuracy=m.get("always_up_accuracy"),
        )

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_auth)])
    def predict(req: PredictRequest, b: ModelBundle = Depends(get_bundle)):
        symbol = req.symbol.upper()
        try:
            res = app.state.features.get(symbol, b.features, b.horizon, b.uses_sentiment)
        except DataUnavailableError:
            # Bilinmeyen sembol mü, sağlayıcı kesintisi mi ayırt edilemiyor; ayrıntı sızdırma
            raise HTTPException(status_code=502, detail=f"{symbol} için fiyat verisi alınamadı "
                                                        "(bilinmeyen sembol veya veri sağlayıcı sorunu)")
        except InsufficientHistoryError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        prob = float(b.model.predict_proba(res.X)[0, 1])
        train_symbols = b.metadata.get("train_symbols")
        return PredictResponse(
            symbol=symbol, as_of=res.as_of, horizon_days=b.horizon, probability_up=prob,
            predicted_direction="up" if prob >= THRESHOLD else "down", threshold=THRESHOLD,
            sentiment_used=res.sentiment_used,
            symbol_in_training_set=(symbol in train_symbols) if train_symbols else None,
            warnings=res.warnings + ([] if not train_symbols or symbol in train_symbols else
                                     ["Bu sembol eğitim setinde yoktu; tahmin güvenilirliği daha da düşük."]),
            model=ModelRef(name=b.name, trained_at=b.metadata.get("trained_at")),
        )

    return app


app = create_app()
