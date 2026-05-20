from fastapi import APIRouter

from app.schemas.forecast import ReportRequest, ReportResponse
from app.services.report_generator import generate_energy_report

router = APIRouter(prefix="/report", tags=["Report"])


@router.post("", response_model=ReportResponse)
def create_report(request: ReportRequest):
    peak_risk, recommendation, report_text = generate_energy_report(
        forecast_steps=request.forecast_steps,
        predicted_usage=request.predicted_usage,
        current_temperature=request.current_temperature,
        current_humidity=request.current_humidity,
    )

    return ReportResponse(
        peak_risk=peak_risk,
        recommendation=recommendation,
        report_text=report_text,
    )