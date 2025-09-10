from datetime import datetime, timezone
from zoneinfo import ZoneInfo

EUROPE_MADRID = ZoneInfo("Europe/Madrid")

def now_utc():
    return datetime.now(timezone.utc)

def today_utc_date():
    return now_utc().date()

def now_local():
    return now_utc().astimezone(EUROPE_MADRID)

def today_local_date():
    return now_local().date()
