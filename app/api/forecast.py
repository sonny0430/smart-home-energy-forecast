from fastapi import APIRouter, HTTPException

from app.db.database import get_forecast_logs, insert_forecast_log
from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.predictor import predictor
from app.services.report_generator import generate_energy_report

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.post("", response_model=ForecastResponse)
def forecast_energy(request: ForecastRequest):
    try:
        predicted_usage = predictor.predict(forecast_steps=request.forecast_steps)

        peak_risk, recommendation, report_text = generate_energy_report(
            forecast_steps=request.forecast_steps,
            predicted_usage=predicted_usage,
            current_temperature=request.current_temperature,
            current_humidity=request.current_humidity,
        )

        input_summary = {
            "forecast_steps": request.forecast_steps,
            "current_temperature": request.current_temperature,
            "current_humidity": request.current_humidity,
        }

        insert_forecast_log(
            forecast_steps=request.forecast_steps,
            input_summary=input_summary,
            predicted_usage=predicted_usage,
            peak_risk=peak_risk,
            recommendation=recommendation,
            report_text=report_text,
        )

        return ForecastResponse(
            forecast_steps=request.forecast_steps,
            predicted_usage=predicted_usage,
            peak_risk=peak_risk,
            recommendation=recommendation,
            report_text=report_text,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
def read_forecast_logs(limit: int = 10):
    try:
        return get_forecast_logs(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))