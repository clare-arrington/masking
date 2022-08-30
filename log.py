from datetime import datetime
from dateutil import tz
import time

def convert_to_local(t):
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    t = datetime.utcfromtimestamp(t)
    t = t.replace(tzinfo=from_zone)
    t = t.astimezone(to_zone)

    return datetime.strftime(t, '%H:%M')

def record_time(desc):
    t = convert_to_local(time.time())
    t_str = f'\t  {desc.capitalize()} time : {t}'
    print(t_str)
    
    return t_str
