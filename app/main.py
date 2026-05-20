from fastapi import FastAPI

from app.api.forecast import router as forecast_router
from app.api.report import router as report_router

app = FastAPI(
    title="Smart Home Energy Forecast API",
    description="스마트홈 센서 데이터 기반 전력 사용량 예측 및 절감 리포트 API",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(forecast_router)
app.include_router(report_router)