"""
FastAPI serving testleri. Gerçek API/ağ yok: fiyat ve haber sağlayıcıları sahte.

Çalıştır:  python -m pytest tests -q
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from feature_engineering.build_features import build_feature_dataset
from model_training import experiment as E
from model_training.dataset import make_stationary_features
from serving.app import create_app
from serving.feature_service import MARKET_TZ, FeatureService
from serving.model_store import ModelBundle
from serving.settings import Settings
from tests.test_sentiment_history import make_news, make_prices

FUTURE = lambda: datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)  # "bugün" hiçbir barla çakışmaz
BASE_COLS = ["datetime", "open", "high", "low", "close", "volume", "symbol"]
SETTINGS = Settings(model_path=Path("/nonexistent/model.joblib"))


# --------------------------------------------------------------------------- #
# Sahte sağlayıcılar ve model
# --------------------------------------------------------------------------- #
class FakePrice:
    def __init__(self, df):
        self.df, self.calls = df, 0

    def get_daily_ohlcv(self, symbol, outputsize=365):
        self.calls += 1
        if self.df is None:
            return None
        d = self.df[self.df["symbol"] == symbol]
        return d.tail(outputsize).reset_index(drop=True) if len(d) else None


class FakeNews:
    def __init__(self, news=None, mode="ok"):
        self.news, self.mode, self.calls = news, mode, 0

    def get_company_news_range(self, symbol, start, end, **kw):
        self.calls += 1
        if self.mode == "fail":
            return None
        if self.mode == "raise":
            raise RuntimeError("boom")
        return self.news


@pytest.fixture(scope="module")
def prices():
    return make_prices(n=420, symbols=("AAPL",))[BASE_COLS]


@pytest.fixture(scope="module")
def news(prices):
    heads = ["stock surge on record profit", "shares plunge after lawsuit", "company announces product"]
    d = prices["datetime"].reset_index(drop=True)
    nf = make_news([(d.iloc[i].strftime("%Y-%m-%d") + " 15:00", heads[i % 3]) for i in range(0, len(d), 2)])
    nf["symbol"] = "AAPL"
    return nf


def make_bundle(variant, prices, news, train_symbols=("AAPL",)):
    data = E.build_variant(variant, make_with_extras(prices), news, horizon=1)
    model = XGBClassifier(n_estimators=20, max_depth=2, random_state=0, eval_metric="logloss")
    model.fit(data.X_train, data.y_train)
    return ModelBundle(model=model, features=data.feature_names, horizon=1,
                       metadata={"name": f"xgb_{variant}", "train_symbols": list(train_symbols),
                                 "trained_at": "2026-11-07T00:00:00+00:00", "test_roc_auc": 0.51})


def make_with_extras(prices):
    out = prices.copy()
    out["news_sentiment_score"], out["collected_at"] = 0.0, "x"
    return out


@pytest.fixture(scope="module")
def price_bundle(prices):
    return make_bundle("price_only", prices, None)


@pytest.fixture(scope="module")
def sentiment_bundle(prices, news):
    b = make_bundle("lexicon", prices, news)
    assert b.uses_sentiment
    return b


def client_for(bundle, price_df, news_client=None, settings=SETTINGS, now_fn=FUTURE, ttl=3600):
    svc = FeatureService(FakePrice(price_df), news_client, now_fn=now_fn, ttl_seconds=ttl)
    return TestClient(create_app(bundle=bundle, feature_service=svc, settings=settings)), svc


# --------------------------------------------------------------------------- #
# Temel akış
# --------------------------------------------------------------------------- #
def test_predict_happy_path(price_bundle, prices):
    c, _ = client_for(price_bundle, prices)
    with c:
        r = c.post("/predict", json={"symbol": "aapl"})
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "AAPL" and body["horizon_days"] == 1
    assert 0 <= body["probability_up"] <= 1
    assert body["predicted_direction"] == ("up" if body["probability_up"] >= 0.5 else "down")
    assert body["as_of"] == str(prices["datetime"].max().date())
    assert body["sentiment_used"] is False and body["symbol_in_training_set"] is True
    assert body["model"]["name"] == "xgb_price_only"


def test_health_and_model_info(price_bundle, prices):
    c, _ = client_for(price_bundle, prices)
    with c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": True}
        info = c.get("/model").json()
    assert info["horizon_days"] == 1 and info["uses_sentiment"] is False
    assert info["train_symbols"] == ["AAPL"] and "yatırım tavsiyesi değildir" in info["disclaimer"]


def test_unseen_symbol_is_flagged(price_bundle, prices):
    other = prices.copy()
    other["symbol"] = "AAPL"
    b = ModelBundle(price_bundle.model, price_bundle.features, 1, {**price_bundle.metadata, "train_symbols": ["MSFT"]})
    c, _ = client_for(b, other)
    with c:
        body = c.post("/predict", json={"symbol": "AAPL"}).json()
    assert body["symbol_in_training_set"] is False
    assert any("eğitim setinde yoktu" in w for w in body["warnings"])


# --------------------------------------------------------------------------- #
# Hata yolları
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["AAPL; DROP", "TOOLONGSYM", "", "12345", "AA PL", "../etc"])
def test_symbol_validation(price_bundle, prices, bad):
    c, svc = client_for(price_bundle, prices)
    with c:
        assert c.post("/predict", json={"symbol": bad}).status_code == 422
    assert svc.price_client.calls == 0          # geçersiz sembol dış API'ye hiç gitmedi


def test_price_data_unavailable_returns_502(price_bundle):
    c, _ = client_for(price_bundle, None)
    with c:
        r = c.post("/predict", json={"symbol": "AAPL"})
    assert r.status_code == 502 and "fiyat verisi alınamadı" in r.json()["detail"]


def test_insufficient_history_returns_422(price_bundle, prices):
    c, _ = client_for(price_bundle, prices.head(30))
    with c:
        r = c.post("/predict", json={"symbol": "AAPL"})
    assert r.status_code == 422 and "yeterli geçmiş yok" in r.json()["detail"]


def test_missing_model_degrades_gracefully(prices):
    c, _ = client_for(None, prices)             # model dosyası yok -> servis ayakta
    with c:
        assert c.get("/health").json() == {"status": "degraded", "model_loaded": False}
        assert c.post("/predict", json={"symbol": "AAPL"}).status_code == 503
        assert c.get("/model").status_code == 503


# --------------------------------------------------------------------------- #
# Kimlik doğrulama
# --------------------------------------------------------------------------- #
def test_auth_token_protects_predict_but_not_health(price_bundle, prices):
    secured = Settings(model_path=SETTINGS.model_path, auth_token="your_test_token")
    c, _ = client_for(price_bundle, prices, settings=secured)
    with c:
        assert c.get("/health").status_code == 200
        assert c.post("/predict", json={"symbol": "AAPL"}).status_code == 401
        assert c.post("/predict", json={"symbol": "AAPL"}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/model").status_code == 401
        assert c.post("/predict", json={"symbol": "AAPL"}, headers={"X-API-Key": "your_test_token"}).status_code == 200


# --------------------------------------------------------------------------- #
# Cache ve eşzamanlılık
# --------------------------------------------------------------------------- #
def test_cache_prevents_repeated_provider_calls(price_bundle, prices):
    c, svc = client_for(price_bundle, prices)
    with c:
        for _ in range(3):
            assert c.post("/predict", json={"symbol": "AAPL"}).status_code == 200
    assert svc.price_client.calls == 1


def test_concurrent_requests_for_same_symbol_make_one_call(price_bundle, prices):
    svc = FeatureService(FakePrice(prices), None, now_fn=FUTURE)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda _: svc.get("AAPL", price_bundle.features, 1, False), range(8)))
    assert svc.price_client.calls == 1


def test_cache_expires_after_ttl(price_bundle, prices):
    svc = FeatureService(FakePrice(prices), None, now_fn=FUTURE, ttl_seconds=0)
    svc.get("AAPL", price_bundle.features, 1, False)
    svc.get("AAPL", price_bundle.features, 1, False)
    assert svc.price_client.calls == 2


# --------------------------------------------------------------------------- #
# Yarım gün barı
# --------------------------------------------------------------------------- #
def at_et(day: pd.Timestamp, hh: int, mm: int = 0):
    return lambda: datetime(day.year, day.month, day.day, hh, mm, tzinfo=MARKET_TZ)


def test_incomplete_todays_bar_is_dropped_during_session(price_bundle, prices):
    last = prices["datetime"].max()
    c, _ = client_for(price_bundle, prices, now_fn=at_et(last, 11))
    with c:
        body = c.post("/predict", json={"symbol": "AAPL"}).json()
    assert body["as_of"] == str(prices["datetime"].iloc[-2].date())
    assert any("tamamlanmamış" in w for w in body["warnings"])


def test_bar_kept_after_close_buffer(price_bundle, prices):
    last = prices["datetime"].max()
    c, _ = client_for(price_bundle, prices, now_fn=at_et(last, 17))
    with c:
        body = c.post("/predict", json={"symbol": "AAPL"}).json()
    assert body["as_of"] == str(last.date()) and body["warnings"] == []


# --------------------------------------------------------------------------- #
# Sentiment: haber var / yok
# --------------------------------------------------------------------------- #
def test_sentiment_model_uses_news(sentiment_bundle, prices, news):
    c, svc = client_for(sentiment_bundle, prices, news_client=FakeNews(news))
    with c:
        body = c.post("/predict", json={"symbol": "AAPL"}).json()
    assert body["sentiment_used"] is True and body["warnings"] == []
    assert svc.news_client.calls == 1


@pytest.mark.parametrize("mode", ["fail", "raise"])
def test_news_failure_degrades_to_neutral(sentiment_bundle, prices, mode):
    c, _ = client_for(sentiment_bundle, prices, news_client=FakeNews(None, mode=mode))
    with c:
        r = c.post("/predict", json={"symbol": "AAPL"})
    body = r.json()
    assert r.status_code == 200 and body["sentiment_used"] is False
    assert any("Haber verisi alınamadı" in w for w in body["warnings"])
    assert 0 <= body["probability_up"] <= 1


def test_price_only_model_never_fetches_news(price_bundle, prices, news):
    c, svc = client_for(price_bundle, prices, news_client=FakeNews(news))
    with c:
        c.post("/predict", json={"symbol": "AAPL"})
    assert svc.news_client.calls == 0


# --------------------------------------------------------------------------- #
# EN ÖNEMLİ TEST: eğitim ve servis aynı feature'ı üretiyor mu?
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", ["price_only", "lexicon"])
def test_serving_features_equal_training_features(variant, prices, news):
    bundle = make_bundle(variant, prices, news if variant == "lexicon" else None)

    # Eğitim yolu: tüm geçmiş üzerinde feature'lar, t gününün satırı
    full = build_feature_dataset(make_with_extras(prices), news_df=news if variant == "lexicon" else None,
                                 drop_warmup=False)
    stationary = make_stationary_features(full.sort_values(["symbol", "datetime"]).reset_index(drop=True))
    t = prices["datetime"].iloc[300]
    expected = stationary.loc[stationary["datetime"] == t, bundle.features].to_numpy(dtype=float)

    # Servis yolu: t gününe kadar olan veriyle (gelecek yok), aynı başlangıçtan
    svc = FeatureService(FakePrice(prices.iloc[:301]), FakeNews(news) if variant == "lexicon" else None,
                         now_fn=FUTURE)
    got = svc.get("AAPL", bundle.features, 1, bundle.uses_sentiment).X.to_numpy(dtype=float)

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)
