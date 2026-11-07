"""
Canlı tahmin için bir sembolün feature satırını üretir.

Eğitimle AYNI kod yolu kullanılır (build_feature_dataset + make_stationary_features);
böylece training-serving skew oluşmaz. Servise özgü işler:

  - Yarım gün barı: sağlayıcı, işlem günü sürerken bugünün henüz tamamlanmamış
    barını döndürebilir. Eğitimde yalnızca TAMAMLANMIŞ günler vardı; yarım barla
    hesaplanan indikatörler eğitim dağılımından sapar. Kapanış (+ tampon)
    öncesindeki bugünkü bar atılır.
  - Cache: Twelve Data ücretsiz kotası dar (800/gün, 8/dk). Günlük barlar günde
    bir kez değiştiği için sembol başına feature'lar TTL süresince tutulur;
    aynı anda gelen istekler aynı sembol için tek API çağrısı yapar.
  - Graceful degradation: model sentiment feature'ı kullanıyor ama haber
    alınamıyorsa nötr değerle devam edilir ve cevapta uyarı verilir.
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from feature_engineering.build_features import build_feature_dataset
from model_training.dataset import InsufficientHistoryError, build_inference_frame

logger = logging.getLogger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")
# Kapanıştan (16:00 ET) sonra sağlayıcının günlük barı kesinleştirmesi için tampon
BAR_FINAL_AFTER = dtime(16, 30)
MIN_PRICE_ROWS = 70  # SMA-50 + getiri gecikmeleri + hacim ortalaması için asgari satır


class DataUnavailableError(RuntimeError):
    """Fiyat verisi alınamadı (bilinmeyen sembol ya da sağlayıcı sorunu)."""


@dataclass
class FeatureResult:
    X: pd.DataFrame
    as_of: date
    sentiment_used: bool
    warnings: list = field(default_factory=list)


def drop_incomplete_bar(price: pd.DataFrame, now: datetime) -> tuple:
    """Bugünün (ET) henüz kesinleşmemiş barını atar. Döner: (df, atıldı_mı)."""
    if price.empty:
        return price, False
    now_et = now.astimezone(MARKET_TZ)
    last_day = pd.Timestamp(price["datetime"].max()).date()
    if last_day == now_et.date() and now_et.time() < BAR_FINAL_AFTER:
        return price[pd.to_datetime(price["datetime"]).dt.date < last_day].reset_index(drop=True), True
    return price, False


class FeatureService:
    def __init__(self, price_client, news_client=None, scorer=None, history_rows: int = 365,
                 news_days: int = 45, ttl_seconds: int = 3600, now_fn=None):
        self.price_client = price_client
        self.news_client = news_client
        self.scorer = scorer
        self.history_rows = history_rows
        self.news_days = news_days
        self.ttl = ttl_seconds
        self.now_fn = now_fn or (lambda: datetime.now(MARKET_TZ))
        self._cache: dict = {}
        self._locks: dict = {}
        self._guard = threading.Lock()

    def _lock_for(self, symbol: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(symbol, threading.Lock())

    def get(self, symbol: str, feature_names: list, horizon: int, uses_sentiment: bool) -> FeatureResult:
        symbol = symbol.upper()
        with self._lock_for(symbol):           # aynı sembol için eşzamanlı istekler tek çağrı yapsın
            hit = self._cache.get(symbol)
            if hit and time.monotonic() - hit[0] < self.ttl:
                return hit[1]
            result = self._compute(symbol, feature_names, horizon, uses_sentiment)
            self._cache[symbol] = (time.monotonic(), result)
            return result

    def _compute(self, symbol, feature_names, horizon, uses_sentiment) -> FeatureResult:
        warnings = []
        price = self.price_client.get_daily_ohlcv(symbol, outputsize=self.history_rows)
        if price is None or price.empty:
            raise DataUnavailableError(f"{symbol} için fiyat verisi alınamadı")

        price, dropped = drop_incomplete_bar(price, self.now_fn())
        if dropped:
            warnings.append("Bugünün tamamlanmamış günlük barı kullanılmadı; son kapanmış seans esas alındı.")
        if len(price) < MIN_PRICE_ROWS:
            raise InsufficientHistoryError(f"{symbol} için yeterli geçmiş yok ({len(price)} < {MIN_PRICE_ROWS} gün)")

        news_df, sentiment_used = None, False
        if uses_sentiment:
            news_df = self._fetch_news(symbol)
            sentiment_used = news_df is not None
            if news_df is None:
                warnings.append("Haber verisi alınamadı; sentiment feature'ları nötr kabul edildi.")

        feats = build_feature_dataset(price, news_df=news_df, target_horizon=horizon,
                                      drop_warmup=False, scorer=self.scorer)
        X, filled = build_inference_frame(feats, feature_names)
        if filled and sentiment_used:
            logger.info("%s: haber boş, nötr doldurulan kolonlar: %s", symbol, len(filled))

        as_of = pd.Timestamp(price["datetime"].max()).date()
        return FeatureResult(X=X, as_of=as_of, sentiment_used=sentiment_used, warnings=warnings)

    def _fetch_news(self, symbol: str) -> Optional[pd.DataFrame]:
        if self.news_client is None:
            return None
        end = date.today()
        try:
            return self.news_client.get_company_news_range(symbol, end - timedelta(days=self.news_days), end)
        except Exception as exc:  # haber hatası tahmini düşürmemeli
            logger.warning("Haber çekilemedi (%s): %s", symbol, type(exc).__name__)
            return None
