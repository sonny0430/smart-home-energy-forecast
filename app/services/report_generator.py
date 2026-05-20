from langchain_core.prompts import PromptTemplate

from app.services.energy_policy import analyze_energy_usage


REPORT_TEMPLATE = """
스마트홈 전력 사용량 예측 결과를 바탕으로 절감 리포트를 작성하세요.

예측 구간: 다음 {forecast_steps}개 시점
예측 전력 사용량: {predicted_usage}
피크 위험도: {peak_risk}
권장사항: {recommendation}
현재 온도: {current_temperature}
현재 습도: {current_humidity}

아래 형식으로 간결하게 작성하세요.

1. 예측 요약
2. 사용량 위험 판단
3. 절감 권장사항
"""


prompt = PromptTemplate.from_template(REPORT_TEMPLATE)


def generate_energy_report(
    forecast_steps: int,
    predicted_usage: list[float],
    current_temperature: float | None = None,
    current_humidity: float | None = None,
) -> tuple[str, str, str]:
    peak_risk, recommendation = analyze_energy_usage(predicted_usage)

    report_text = prompt.format(
        forecast_steps=forecast_steps,
        predicted_usage=predicted_usage,
        peak_risk=peak_risk,
        recommendation=recommendation,
        current_temperature=current_temperature if current_temperature is not None else "입력 없음",
        current_humidity=current_humidity if current_humidity is not None else "입력 없음",
    )

    return peak_risk, recommendation, report_text