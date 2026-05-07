from __future__ import annotations

import datetime

from decision_making.ama_data import load_specific_data


def test_current_price_falls_back_to_latest_available_trading_day():
    date = datetime.datetime(2026, 5, 5)

    tsla = load_specific_data(symbol="TSLA", date=date, type="current_price")
    btc = load_specific_data(symbol="BTC", date=date, type="current_price")

    assert tsla == 392.510009765625
    assert btc == 79858.72


def test_payload_price_overrides_local_history_for_current_day():
    payload = {
        "date": "2026-05-05",
        "price": {"TSLA": 400.0},
        "history_price": {
            "TSLA": [
                {"date": "2026-05-03", "price": 390.0},
                {"date": "2026-05-04", "price": 392.51},
            ]
        },
    }

    prices = load_specific_data(
        symbol="TSLA",
        date=datetime.datetime(2026, 5, 5),
        type="price",
        api_payload=payload,
    )
    current_price = load_specific_data(
        symbol="TSLA",
        date=datetime.datetime(2026, 5, 5),
        type="current_price",
        api_payload=payload,
    )

    assert prices.select("date").tail(1).item() == datetime.date(2026, 5, 5)
    assert prices.select("prices").tail(1).item() == 400.0
    assert current_price == 400.0
