import json
from datetime import datetime

# -----------------------------------
# VDA 5050–STYLE AGV MESSAGE SCHEMAS
# -----------------------------------

def task_message(agv_id, source, destination, priority=1):
    """
    Task assignment message
    """
    return json.dumps({
        "header": {
            "messageType": "TASK",
            "timestamp": datetime.utcnow().isoformat()
        },
        "agvId": agv_id,
        "task": {
            "type": "MOVE",
            "from": source,
            "to": destination,
            "priority": priority
        }
    })


def status_message(agv_id, x, y, state, battery):
    """
    AGV status update message
    """
    return json.dumps({
        "header": {
            "messageType": "STATUS",
            "timestamp": datetime.utcnow().isoformat()
        },
        "agvId": agv_id,
        "position": {
            "x": x,
            "y": y
        },
        "state": state,
        "battery": battery
    })


def error_message(agv_id, error_code, description):
    """
    AGV error message
    """
    return json.dumps({
        "header": {
            "messageType": "ERROR",
            "timestamp": datetime.utcnow().isoformat()
        },
        "agvId": agv_id,
        "errorCode": error_code,
        "description": description,
        "severity": "WARNING"
    })


def pause_message(agv_id, permission):
    """
    Pause / resume control message
    permission: ALLOW or BLOCK
    """
    return json.dumps({
        "header": {
            "messageType": "CONTROL",
            "timestamp": datetime.utcnow().isoformat()
        },
        "agvId": agv_id,
        "command": permission
    })
