"""
Stateful Tool Calling and In-Memory Database for Vaani Duplex.
Supports:
1. schedule_calendar_meeting
2. create_support_ticket
3. check_order_status
"""

import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger("vaani.tools")

INITIAL_SESSION_STATE = {
    "appointments": [
        {
            "id": "APT-101",
            "title": "Quarterly Performance Review",
            "date": "2026-11-12",
            "time": "14:30",
            "attendee": "Rohan Sharma",
            "status": "CONFIRMED"
        }
    ],
    "tickets": [
        {
            "id": "TCK-802",
            "summary": "UPI Payment failed on checkout",
            "priority": "HIGH",
            "status": "OPEN",
            "created_at": "2026-11-09 10:15"
        }
    ],
    "orders": {
        "ORD-9921": {
            "order_id": "ORD-9921",
            "item": "Sony WH-1000XM5 Wireless Headphones",
            "status": "Out for Delivery",
            "eta": "Today by 6:00 PM",
            "delivery_partner": "Bluedart Express",
            "destination": "Hyderabad, Telangana"
        },
        "ORD-4410": {
            "order_id": "ORD-4410",
            "item": "Apple MacBook Air M3 (16GB RAM)",
            "status": "Shipped",
            "eta": "Tomorrow by 2:00 PM",
            "delivery_partner": "Delhivery Logistics",
            "destination": "Bengaluru, Karnataka"
        }
    }
}

# Live mutable session state
SESSION_STATE: Dict[str, Any] = json.loads(json.dumps(INITIAL_SESSION_STATE))


def reset_session_state() -> Dict[str, Any]:
    """Resets in-memory state back to defaults for clean demoing."""
    global SESSION_STATE
    SESSION_STATE = json.loads(json.dumps(INITIAL_SESSION_STATE))
    return SESSION_STATE


def get_session_state() -> Dict[str, Any]:
    """Returns current live session state."""
    return SESSION_STATE


# OpenAI / Groq Compatible Function Calling Schemas
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "schedule_calendar_meeting",
            "description": "Schedule a new appointment or calendar meeting with a designated attendee.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Topic or purpose of the meeting (e.g., 'Project Sync', 'Design Review')."
                    },
                    "date": {
                        "type": "string",
                        "description": "Date of the meeting (e.g., '2026-11-15' or 'Tomorrow')."
                    },
                    "time": {
                        "type": "string",
                        "description": "Time of the meeting (e.g., '10:00 AM' or '15:30')."
                    },
                    "attendee": {
                        "type": "string",
                        "description": "Name or email of the person attending the meeting."
                    }
                },
                "required": ["title", "date", "time", "attendee"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Log an urgent support or helpdesk ticket into the ticketing system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_summary": {
                        "type": "string",
                        "description": "Concise summary of the issue or problem being reported."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                        "description": "Urgency and impact level of the issue."
                    }
                },
                "required": ["issue_summary", "priority"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Lookup the current shipping, fulfillment, and delivery status of an e-commerce order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The exact Order tracking ID (e.g., 'ORD-9921' or 'ORD-4410')."
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]


def _tool_schedule_meeting(args: Dict[str, Any]) -> Dict[str, Any]:
    meeting_id = f"APT-{random.randint(102, 999)}"
    title = args.get("title", "General Discussion")
    date = args.get("date", "Upcoming")
    time = args.get("time", "TBD")
    attendee = args.get("attendee", "Guest")

    meeting_record = {
        "id": meeting_id,
        "title": title,
        "date": date,
        "time": time,
        "attendee": attendee,
        "status": "CONFIRMED"
    }
    SESSION_STATE["appointments"].append(meeting_record)
    logger.info("📅 Scheduled meeting %s: %s with %s", meeting_id, title, attendee)
    return {
        "success": True,
        "meeting_id": meeting_id,
        "title": title,
        "date": date,
        "time": time,
        "attendee": attendee,
        "message": f"Meeting '{title}' successfully confirmed with {attendee} for {date} at {time}."
    }


def _tool_create_ticket(args: Dict[str, Any]) -> Dict[str, Any]:
    ticket_id = f"TCK-{random.randint(803, 999)}"
    issue = args.get("issue_summary", "General Inconvenience")
    priority = str(args.get("priority", "MEDIUM")).upper()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    ticket_record = {
        "id": ticket_id,
        "summary": issue,
        "priority": priority,
        "status": "OPEN",
        "created_at": now_str
    }
    SESSION_STATE["tickets"].append(ticket_record)
    logger.info("🎫 Created ticket %s [%s]: %s", ticket_id, priority, issue)
    return {
        "success": True,
        "ticket_id": ticket_id,
        "summary": issue,
        "priority": priority,
        "status": "OPEN",
        "message": f"Support ticket {ticket_id} created with {priority} priority."
    }


def _tool_check_order(args: Dict[str, Any]) -> Dict[str, Any]:
    raw_id = str(args.get("order_id", "")).strip().upper()

    # Search in registered orders
    orders_db = SESSION_STATE["orders"]
    if raw_id in orders_db:
        info = orders_db[raw_id]
        logger.info("📦 Fetched order %s: %s", raw_id, info["status"])
        return {
            "found": True,
            "order_id": raw_id,
            "item": info["item"],
            "status": info["status"],
            "eta": info["eta"],
            "delivery_partner": info["delivery_partner"],
            "destination": info["destination"],
            "message": f"Order {raw_id} ({info['item']}) is {info['status']}, arriving {info['eta']} via {info['delivery_partner']}."
        }

    # If ID not found, return available options
    available_keys = list(orders_db.keys())
    return {
        "found": False,
        "order_id": raw_id,
        "message": f"Order ID '{raw_id}' was not found. Active orders in system: {', '.join(available_keys)}."
    }


def execute_tool(name: str, arguments: Any) -> Dict[str, Any]:
    """
    Executes a recognized tool and returns structured result.
    """
    if isinstance(arguments, str):
        try:
            args_dict = json.loads(arguments)
        except Exception:
            args_dict = {}
    elif isinstance(arguments, dict):
        args_dict = arguments
    else:
        args_dict = {}

    if name == "schedule_calendar_meeting":
        return _tool_schedule_meeting(args_dict)
    elif name == "create_support_ticket":
        return _tool_create_ticket(args_dict)
    elif name == "check_order_status":
        return _tool_check_order(args_dict)
    else:
        return {"error": f"Tool '{name}' is not recognized."}
