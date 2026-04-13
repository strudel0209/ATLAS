"""
Weather MCP Server - simulated weather data server for multi-server demos.
Provides a heterogeneous tool set to demonstrate ATLAS cross-server orchestration.
Tools: get_forecast, get_alerts, get_historical_temp
"""

from typing import Any
import random

# Deterministic seed for reproducible demo results
random.seed(42)

# Pre-computed weather data for demo cities
_FORECASTS = {
    "New York": {"temp_f": 45, "temp_c": 7, "condition": "Partly Cloudy", "humidity": 62, "wind_mph": 12},
    "London": {"temp_f": 50, "temp_c": 10, "condition": "Overcast", "humidity": 78, "wind_mph": 8},
    "Tokyo": {"temp_f": 55, "temp_c": 13, "condition": "Clear", "humidity": 45, "wind_mph": 5},
    "Sydney": {"temp_f": 72, "temp_c": 22, "condition": "Sunny", "humidity": 55, "wind_mph": 15},
    "Berlin": {"temp_f": 38, "temp_c": 3, "condition": "Light Rain", "humidity": 85, "wind_mph": 10},
}

_ALERTS = {
    "New York": [{"type": "Wind Advisory", "severity": "moderate", "message": "Winds 25-35 mph expected tonight"}],
    "London": [],
    "Tokyo": [],
    "Sydney": [{"type": "UV Index Warning", "severity": "high", "message": "UV index 11+ expected midday"}],
    "Berlin": [{"type": "Frost Warning", "severity": "low", "message": "Temperatures near freezing overnight"}],
}

_HISTORICAL = {
    "New York": [32, 35, 40, 55, 68, 78, 84, 82, 73, 60, 48, 36],
    "London": [45, 46, 50, 55, 62, 68, 72, 71, 65, 57, 50, 46],
    "Tokyo": [42, 44, 52, 62, 70, 76, 83, 85, 78, 66, 55, 46],
    "Sydney": [78, 78, 75, 69, 63, 58, 56, 59, 64, 69, 73, 76],
    "Berlin": [33, 36, 43, 53, 63, 70, 74, 73, 65, 54, 43, 36],
}


class WeatherMCPServer:
    name = "Weather MCP"
    description = "Weather forecasts, alerts, and historical temperature data for major cities."

    tools = {
        "get_forecast": {
            "description": "Get current weather forecast for a city",
            "params": {
                "city": {"type": "string", "required": True, "description": "City name (e.g. 'New York', 'London')"},
            },
            "returns": "dict with temp_f, temp_c, condition, humidity, wind_mph",
            "example_call": "get_forecast(city='New York')",
            "example_output": "{'temp_f': 45, 'temp_c': 7, 'condition': 'Partly Cloudy', 'humidity': 62, 'wind_mph': 12}",
        },
        "get_alerts": {
            "description": "Get active weather alerts for a city",
            "params": {
                "city": {"type": "string", "required": True, "description": "City name"},
            },
            "returns": "list of alert dicts with type, severity, message",
            "example_call": "get_alerts(city='New York')",
            "example_output": "[{'type': 'Wind Advisory', 'severity': 'moderate', 'message': 'Winds 25-35 mph expected tonight'}]",
        },
        "get_historical_temp": {
            "description": "Get average monthly temperatures (°F) for a city over the past year",
            "params": {
                "city": {"type": "string", "required": True, "description": "City name"},
            },
            "returns": "list of 12 numbers (Jan-Dec average temps in °F)",
            "example_call": "get_historical_temp(city='New York')",
            "example_output": "[32, 35, 40, 55, 68, 78, 84, 82, 73, 60, 48, 36]",
        },
    }

    @staticmethod
    def execute(tool_name: str, args: dict) -> Any:
        city = args.get("city", "")
        if tool_name == "get_forecast":
            if city not in _FORECASTS:
                return f"Error: city '{city}' not found. Available: {list(_FORECASTS.keys())}"
            return _FORECASTS[city]

        elif tool_name == "get_alerts":
            if city not in _ALERTS:
                return f"Error: city '{city}' not found. Available: {list(_ALERTS.keys())}"
            return _ALERTS[city]

        elif tool_name == "get_historical_temp":
            if city not in _HISTORICAL:
                return f"Error: city '{city}' not found. Available: {list(_HISTORICAL.keys())}"
            return _HISTORICAL[city]

        return f"Error: unknown tool '{tool_name}'. Available: get_forecast, get_alerts, get_historical_temp"
