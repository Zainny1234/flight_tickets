from conf.config import DAY_OF_BOOKING
import pandas as pd
from src.data_ingest import load_dataset


def marketshare(df, ms):
    df['Market_Share'] = df['Airline'].map(ms)
    return df


def classshare(df, bk_class):
    df['Booking_Class'] = df['Airline'].map(bk_class)
    return df


def daystodep(df, dayofbk=DAY_OF_BOOKING):
    df1 = df.copy()
    df1['Day_of_Booking'] = dayofbk
    df1['Day_of_Booking'] = pd.to_datetime(df1['Day_of_Booking'], format='%d/%m/%Y')
    df1['Date_of_Journey'] = pd.to_datetime(df1['Date_of_Journey'], format='%d/%m/%Y')
    df1['Days_to_Departure'] = (df1['Date_of_Journey'] - df1['Day_of_Booking']).dt.days
    df['Days_to_Departure'] = df1['Days_to_Departure']
    return df


def get_departure(dep):
    dep = dep.split(':')
    dep = int(dep[0])
    if 6 <= dep < 12:
        return 'Morning'
    elif 12 <= dep < 17:
        return 'Noon'
    elif 17 <= dep < 20:
        return 'Evening'
    else:
        return 'Night'


def timeofday(df):
    df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]
    df['Dep_timeofday'] = df['Dep_Time'].apply(get_departure)
    df['Arr_timeofday'] = df['Arrival_Time'].apply(get_departure)
    return df


def totalstops(df):
    df['Total_Stops'] = df['Total_Stops'].str.replace('non-stop', '0')
    df['Total_Stops'] = df['Total_Stops'].str.replace('stops', '')
    df['Total_Stops'] = df['Total_Stops'].str.replace('stop', '')
    df['Total_Stops'].fillna(0, inplace=True)
    df['Total_Stops'] = df['Total_Stops'].astype(float)
    return df

# def duration(df):
#     df['Hours'] = df['Duration'].str.split(' ').str[0]
#     df['Hours'] = df['Hours'].str.replace('h', '')
#     df['Hours'].fillna(0, inplace=True)
#
#     df['Minutes'] = df['Duration'].str.split(' ').str[1]
#     df['Minutes'] = df['Minutes'].str.replace('m', '').astype(float)
#     df['Minutes'].fillna(0, inplace=True)
#
#     df['Hours'] = df['Hours'] * 60
#     df['Duration'] = df['Hours'] + df['Minutes']
#     return df

if __name__ == "__main__":
    x = load_dataset('train.xlsx')
    ms = load_dataset('ms.json')['market']
    bk_class = load_dataset('class.json')['class']
    df = marketshare(x, ms)
    df = classshare(df, bk_class)
    df = daystodep(df)
    df = timeofday(df)
    df = totalstops(df)
    df = duration(df)
