from app.services.energy_policy import analyze_energy_usage


def test_energy_policy_stable():
    peak_risk, recommendation = analyze_energy_usage([50, 55, 52, 54])

    assert peak_risk == "안정"
    assert "안정" in recommendation or "유지" in recommendation


def test_energy_policy_high_average():
    peak_risk, recommendation = analyze_energy_usage([110, 120, 105, 115])

    assert peak_risk == "평균 사용량 높음"
    assert "점검" in recommendation


def test_energy_policy_peak_risk():
    peak_risk, recommendation = analyze_energy_usage([50, 52, 55, 100])

    assert peak_risk == "피크 사용량 증가 위험"
    assert "분산" in recommendation