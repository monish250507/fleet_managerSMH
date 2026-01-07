# Member 2: AGV Zone & Traffic Logic

SAFE_TIME_GAP = 3.0  # seconds


def time_to_conflict(distance, speed):
    """Calculate time to reach conflict point"""
    if speed <= 0:
        return float('inf')
    return distance / speed


def can_enter_zone(agv1, agv2):
    """
    Decide whether AGV2 can enter the zone
    based on time separation
    """
    t1 = time_to_conflict(agv1["distance"], agv1["speed"])
    t2 = time_to_conflict(agv2["distance"], agv2["speed"])

    print(f"AGV1 ETA: {t1:.2f}s | AGV2 ETA: {t2:.2f}s")

    if abs(t2 - t1) >= SAFE_TIME_GAP:
        return True
    return False


if __name__ == "__main__":
    # Dummy AGV data (no Gazebo needed)
    agv1 = {"distance": 2.0, "speed": 0.5}
    agv2 = {"distance": 4.0, "speed": 0.5}

    if can_enter_zone(agv1, agv2):
        print("✅ SAFE: Both AGVs allowed in zone")
    else:
        print("⛔ UNSAFE: AGV must wait")
