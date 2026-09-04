STRATEGY_KEY = "tl_ma_cmf"
STRATEGIES = {
    STRATEGY_KEY: "Trendline Break (LuxAlgo) + MA + CMF",
}
DEFAULT_STRATEGY = STRATEGY_KEY

FACTOR_SPECS = {
    STRATEGY_KEY: [
        ("break", "⚡", "Break", "Break kháng cự Trendline (LuxAlgo)"),
        ("ma", "▲", "MA50", "Close > SMA50"),
        ("cmf", "▮", "CMF", "CMF20 > 0"),
    ]
}
