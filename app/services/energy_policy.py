def analyze_energy_usage(predicted_usage: list[float]) -> tuple[str, str]:
    if not predicted_usage:
        return "예측값 없음", "예측 결과가 없어 권장사항을 생성할 수 없습니다."

    avg_usage = sum(predicted_usage) / len(predicted_usage)
    max_usage = max(predicted_usage)

    if max_usage >= avg_usage * 1.3:
        peak_risk = "피크 사용량 증가 위험"
        recommendation = (
            "예측 구간 중 특정 시점의 전력 사용량이 높게 나타납니다. "
            "고전력 가전 사용 시간을 분산하는 것이 좋습니다."
        )
    elif avg_usage >= 100:
        peak_risk = "평균 사용량 높음"
        recommendation = (
            "예측 구간의 평균 전력 사용량이 높은 편입니다. "
            "냉난방 설정 온도와 대기전력 사용 여부를 점검하는 것이 좋습니다."
        )
    else:
        peak_risk = "안정"
        recommendation = (
            "예측 구간의 전력 사용량은 비교적 안정적입니다. "
            "현재 사용 패턴을 유지해도 무리가 없습니다."
        )

    return peak_risk, recommendation