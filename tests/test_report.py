from app.services.report_generator import generate_energy_report


def test_generate_energy_report():
    peak_risk, recommendation, report_text = generate_energy_report(
        forecast_steps=6,
        predicted_usage=[80, 90, 120, 160, 130, 100],
        current_temperature=26.0,
        current_humidity=55.0,
    )

    assert peak_risk
    assert recommendation
    assert "예측 요약" in report_text
    assert "절감 권장사항" in report_text