from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    forecast_steps: int = Field(default=6, ge=1, le=24)
    current_temperature: float | None = None
    current_humidity: float | None = None


class ForecastResponse(BaseModel):
    forecast_steps: int
    predicted_usage: list[float]
    peak_risk: str
    recommendation: str
    report_text: str


class ReportRequest(BaseModel):
    forecast_steps: int
    predicted_usage: list[float]
    current_temperature: float | None = None
    current_humidity: float | None = None


class ReportResponse(BaseModel):
    peak_risk: str
    recommendation: str
    report_text: str