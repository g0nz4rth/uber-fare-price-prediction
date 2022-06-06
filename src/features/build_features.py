import re
import time
import datetime

def handle_wrong_minutes(travel_time: str):
    """
    This function converts minuts (as int) to hours.
    """
    travel_time_split = travel_time.split(":")
    hours = int(travel_time_split[0])
    minuts = int(travel_time_split[1])
    seconds = int(travel_time_split[-1])
    
    if (minuts > 59):
        # converting time
        hours_to_add = minuts // 60
        minuts_remaining = minuts % 60
        
        # adding time
        hours = str(hours + hours_to_add)
        minuts_remaining = str(minuts_remaining)
        seconds = str(seconds)

        # formatted output
        if (len(hours) == 1) & (len(minuts_remaining) == 1) & (len(seconds) == 1):
            hours = '0'+hours
            minuts_remaining = '0'+minuts_remaining
            seconds = '0'+seconds
            
        return f"{hours}:{minuts_remaining}:{seconds}"
    
    return travel_time

def time_str_2_sec(time_str: str, time_format: str = '%H:%M:%S'):
    """
    This function converts a time string object to its total in seconds.
    """
    time_str = time.strptime(time_str, time_format)
    
    in_seconds = datetime.timedelta(
        hours = time_str.tm_hour,
        minutes = time_str.tm_min,
        seconds = time_str.tm_sec
    ).total_seconds()
    
    return in_seconds

def count_stopovers(stopover_str: str, delimiter: str = ' '*3, version: str = '2015'):
    """
    This function counts the number of stopovers for a given trip.
    """
    if (version == '2015'):
        # returning a default value for trips without stopovers
        if stopover_str == '-':
            return 0

        return len(stopover_str.split(delimiter))
    
    penalty = 2
    return len(stopover_str.split('-')) - penalty

def handle_travel_time(value: str):
    """
    This function replaces problematic data in two columns,
    DEPARTURE_ARRIVAL_TIME and ARRIVAL_DEPARTURE_TIME.
    """
    # setting problematic travel times
    problematic_records_replacement = {
        '76:00': '01:26:59', '72:00': '01:20:00', '120:00': '02:00:00',
        '96:00': handle_wrong_minutes('01:60:00'), '216:00': '03:59:00', '124:00': '02:06:59',
        '80:00': '01:33:33', '68:00': '01:13:33', '130:00': '02:16:59',
        '64:00': '01:06:59', '90:00': '01:50:00','140:00': '02:33:33', 
        '768:00': handle_wrong_minutes('12:80:00'), '91:00': '01:51:59',
        '152:00': '02:53:33', '110:00': handle_wrong_minutes('01:83:33'), '00:30': '00:30:00',
        '100:00': handle_wrong_minutes('01:66:59'), '200:00': '03:33:33', '168:00': handle_wrong_minutes('02:66:59'),
        '70:00': '01:16:59', '528:00': handle_wrong_minutes('08:80:00'), '02:00': '00:02:00',
        '94:00': '01:56:59', '144:00': '02:40:00', '98:00': handle_wrong_minutes('01:63:33'),
        '92:00': '01:53:33', '78:00': '01:30:00'
    }
    
    # checking if time is problematic and return the predefined solution
    if (value in list(problematic_records_replacement.keys())):
        return problematic_records_replacement[value]
    
    # handling wrong time format
    value_splitted = value.split(":")
    
    if (len(value_splitted) < 3):
        # checking if minutes comes first or last
        if (re.findall("[0-9][1-9]", value_splitted[0])):
            return f"00:{value_splitted[0]}:00"
        else:
            return f"00:{value_splitted[-1]}:00"
        
    # handling wrong minutes for observations like '00:00:00'
    return handle_wrong_minutes(value)

def analyze_rare_labels(data: pd.DataFrame, columns: list, ratio: float = 0.01):
    """
    This function calculates the % of observations
    per category in label-based variables.
    """
    # calculating the% of observations per category
    df = data.copy()
    tmp_df = df.groupby(columns)['FUEL_CONSUMPTION'].count() / len(df)
    
    # returning rare categories only (default: less than 1% of observations)
    return tmp_df[tmp_df < ratio]
        