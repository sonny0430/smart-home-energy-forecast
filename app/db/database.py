import json
import os
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_engine() -> Engine:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL 환경변수가 설정되지 않았습니다.")

    return create_engine(DATABASE_URL, pool_pre_ping=True)


def insert_forecast_log(
    forecast_steps: int,
    input_summary: dict[str, Any],
    predicted_usage: list[float],
    peak_risk: str,
    recommendation: str,
    report_text: str,
) -> None:
    engine = get_engine()

    query = text(
        """
        insert into forecast_logs (
            forecast_steps,
            input_summary,
            predicted_usage,
            peak_risk,
            recommendation,
            report_text
        )
        values (
            :forecast_steps,
            cast(:input_summary as jsonb),
            cast(:predicted_usage as jsonb),
            :peak_risk,
            :recommendation,
            :report_text
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "forecast_steps": forecast_steps,
                "input_summary": json.dumps(input_summary, ensure_ascii=False),
                "predicted_usage": json.dumps(predicted_usage, ensure_ascii=False),
                "peak_risk": peak_risk,
                "recommendation": recommendation,
                "report_text": report_text,
            },
        )


def get_forecast_logs(limit: int = 10) -> list[dict[str, Any]]:
    engine = get_engine()

    query = text(
        """
        select
            id,
            forecast_steps,
            input_summary,
            predicted_usage,
            peak_risk,
            recommendation,
            report_text,
            created_at
        from forecast_logs
        order by created_at desc
        limit :limit
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(query, {"limit": limit}).mappings().all()

    return [dict(row) for row in rows]