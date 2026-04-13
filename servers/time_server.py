"""
Time MCP Server - mirrors the Time MCP from MCPBench used in ATLAS paper.
Tools: get_current_time, convert_time
(See ATLAS paper Appendix A.7, F.4.3 for exact tool listing and examples)
"""

from datetime import datetime, timezone, timedelta
from typing import Any

# Mapping of common timezone names to UTC offsets (hours)
_TZ_OFFSETS = {
    "UTC": 0,
    "America/New_York": -5,
    "America/Chicago": -6,
    "America/Denver": -7,
    "America/Los_Angeles": -8,
    "Europe/London": 0,
    "Europe/Paris": 1,
    "Europe/Berlin": 1,
    "Asia/Tokyo": 9,
    "Asia/Shanghai": 8,
    "Asia/Kolkata": 5.5,
    "Australia/Sydney": 11,
}


class TimeMCPServer:
    name = "Time MCP"
    description = "Current time retrieval and timezone conversions."

    tools = {
        "get_current_time": {
            "description": "Get the current time in a specified timezone",
            "params": {
                "timezone": {
                    "type": "string",
                    "required": True,
                    "description": "IANA timezone name, e.g. 'America/New_York', 'UTC'",
                },
            },
            "returns": "string (ISO 8601 datetime)",
            "example_call": "get_current_time(timezone='America/New_York')",
            "example_output": "'2026-01-26T04:35:11-05:00'",
        },
        "convert_time": {
            "description": "Convert a datetime from one timezone to another",
            "params": {
                "time": {
                    "type": "string",
                    "required": True,
                    "description": "ISO 8601 datetime string",
                },
                "from_timezone": {
                    "type": "string",
                    "required": True,
                    "description": "Source IANA timezone",
                },
                "to_timezone": {
                    "type": "string",
                    "required": True,
                    "description": "Target IANA timezone",
                },
            },
            "returns": "string (ISO 8601 datetime in target timezone)",
            "example_call": "convert_time(time='2026-01-26T10:00:00', from_timezone='UTC', to_timezone='Asia/Tokyo')",
            "example_output": "'2026-01-26T19:00:00+09:00'",
        },
    }

    @staticmethod
    def execute(tool_name: str, args: dict) -> Any:
        if tool_name == "get_current_time":
            tz_name = args.get("timezone", "UTC")
            offset_hours = _TZ_OFFSETS.get(tz_name)
            if offset_hours is None:
                return f"Error: unknown timezone '{tz_name}'. Supported: {list(_TZ_OFFSETS.keys())}"
            tz = timezone(timedelta(hours=offset_hours))
            now = datetime.now(tz)
            return now.isoformat()

        elif tool_name == "convert_time":
            time_str = args.get("time", "")
            from_tz = args.get("from_timezone", "UTC")
            to_tz = args.get("to_timezone", "UTC")
            from_offset = _TZ_OFFSETS.get(from_tz)
            to_offset = _TZ_OFFSETS.get(to_tz)
            if from_offset is None:
                return f"Error: unknown source timezone '{from_tz}'"
            if to_offset is None:
                return f"Error: unknown target timezone '{to_tz}'"
            try:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone(timedelta(hours=from_offset)))
                target_tz = timezone(timedelta(hours=to_offset))
                converted = dt.astimezone(target_tz)
                return converted.isoformat()
            except ValueError as e:
                return f"Error parsing time '{time_str}': {e}"

        return f"Error: unknown tool '{tool_name}'. Available: get_current_time, convert_time"
