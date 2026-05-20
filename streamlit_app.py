import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 30


def get_health(api_base_url: str) -> dict[str, Any]:
    response = requests.get(
        f"{api_base_url}/health",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def run_forecast(
    api_base_url: str,
    forecast_steps: int,
    current_temperature: float | None,
    current_humidity: float | None,
) -> dict[str, Any]:
    payload = {
        "forecast_steps": forecast_steps,
        "current_temperature": current_temperature,
        "current_humidity": current_humidity,
    }

    response = requests.post(
        f"{api_base_url}/forecast",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def generate_report(
    api_base_url: str,
    forecast_steps: int,
    predicted_usage_text: str,
    current_temperature: float | None,
    current_humidity: float | None,
) -> dict[str, Any]:
    predicted_usage = [
        float(value.strip())
        for value in predicted_usage_text.split(",")
        if value.strip()
    ]

    payload = {
        "forecast_steps": forecast_steps,
        "predicted_usage": predicted_usage,
        "current_temperature": current_temperature,
        "current_humidity": current_humidity,
    }

    response = requests.post(
        f"{api_base_url}/report",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def fetch_logs(api_base_url: str, limit: int) -> list[dict[str, Any]]:
    response = requests.get(
        f"{api_base_url}/forecast/logs",
        params={"limit": limit},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Smart Home Energy Forecast Demo",
    page_icon="🏠",
    layout="wide",
)

st.title("Smart Home Energy Forecast Demo")
st.caption("FastAPI API를 호출해 예측, 리포트, 로그 조회를 시연하는 최소 Streamlit 화면입니다.")

api_base_url = st.sidebar.text_input(
    "FastAPI Base URL",
    value=DEFAULT_API_BASE_URL,
    help="로컬은 http://127.0.0.1:8000, 배포 환경은 Cloudtype FastAPI URL을 넣습니다.",
)

st.sidebar.markdown("### API 상태 확인")
if st.sidebar.button("Health Check"):
    try:
        health_result = get_health(api_base_url)
        st.sidebar.success(f"연결 성공: {health_result}")
    except Exception as exc:
        st.sidebar.error(f"연결 실패: {exc}")

forecast_tab, report_tab, logs_tab = st.tabs(
    ["Forecast Demo", "Report Demo", "Forecast Logs"]
)

with forecast_tab:
    st.subheader("전력 사용량 예측")
    with st.form("forecast_form"):
        forecast_steps = st.slider("forecast_steps", min_value=1, max_value=24, value=6)
        current_temperature = st.number_input(
            "current_temperature",
            value=23.5,
            step=0.1,
            format="%.1f",
        )
        current_humidity = st.number_input(
            "current_humidity",
            value=48.0,
            step=0.1,
            format="%.1f",
        )
        forecast_submit = st.form_submit_button("예측 실행")

    if forecast_submit:
        try:
            forecast_result = run_forecast(
                api_base_url=api_base_url,
                forecast_steps=forecast_steps,
                current_temperature=current_temperature,
                current_humidity=current_humidity,
            )
            st.success("예측 요청이 성공했습니다.")
            st.json(forecast_result)
        except Exception as exc:
            st.error(f"/forecast 호출 실패: {exc}")

with report_tab:
    st.subheader("리포트 생성")
    st.caption("쉼표로 구분된 predicted_usage를 입력하면 /report를 호출합니다.")
    with st.form("report_form"):
        report_steps = st.slider("report forecast_steps", min_value=1, max_value=24, value=6)
        predicted_usage_text = st.text_area(
            "predicted_usage",
            value="80, 95, 120, 160, 130, 100",
            help="예: 80, 95, 120, 160, 130, 100",
        )
        report_temperature = st.number_input(
            "report current_temperature",
            value=26.0,
            step=0.1,
            format="%.1f",
        )
        report_humidity = st.number_input(
            "report current_humidity",
            value=55.0,
            step=0.1,
            format="%.1f",
        )
        report_submit = st.form_submit_button("리포트 생성")

    if report_submit:
        try:
            report_result = generate_report(
                api_base_url=api_base_url,
                forecast_steps=report_steps,
                predicted_usage_text=predicted_usage_text,
                current_temperature=report_temperature,
                current_humidity=report_humidity,
            )
            st.success("리포트 생성이 성공했습니다.")
            st.json(report_result)
        except Exception as exc:
            st.error(f"/report 호출 실패: {exc}")

with logs_tab:
    st.subheader("예측 로그 조회")
    logs_limit = st.slider("limit", min_value=1, max_value=50, value=10)
    if st.button("최근 로그 조회"):
        try:
            logs_result = fetch_logs(api_base_url=api_base_url, limit=logs_limit)
            if logs_result:
                st.dataframe(logs_result, use_container_width=True)
            else:
                st.info("조회된 로그가 없습니다.")
        except Exception as exc:
            st.error(f"/forecast/logs 호출 실패: {exc}")
