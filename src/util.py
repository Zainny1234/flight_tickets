from conf.config import DAY_OF_BOOKING
import pandas as pd
from src.data_ingest import load_dataset


def marketshare(df, ms):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be pandas dataframe')

    if not isinstance(ms, dict):
        raise TypeError('Input must be a dictionary')
    df['Market_Share'] = df['Airline'].map(ms)
    return df


def classshare(df, bk_class):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be pandas dataframe')

    if not isinstance(bk_class, dict):
        raise TypeError('Input must be a dictionary')
    df['Booking_Class'] = df['Airline'].map(bk_class)
    return df


def daystodep(df, dayofbk=DAY_OF_BOOKING):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be pandas dataframe')
    df1 = df.copy()
    df1['Day_of_Booking'] = dayofbk
    df1['Day_of_Booking'] = pd.to_datetime(df1['Day_of_Booking'], format='%d/%m/%Y')
    df1['Date_of_Journey'] = pd.to_datetime(df1['Date_of_Journey'], format='%d/%m/%Y')
    df1['Days_to_Departure'] = (df1['Date_of_Journey'] - df1['Day_of_Booking']).dt.days
    df['Days_to_Departure'] = df1['Days_to_Departure']
    return df


def get_departure(dep):
    try:
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
    except ValueError as err:
        print('time must be in the format hr:min')
        raise


def timeofday(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be pandas dataframe')
    df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]
    df['Dep_timeofday'] = df['Dep_Time'].apply(get_departure)
    df['Arr_timeofday'] = df['Arrival_Time'].apply(get_departure)
    return df


def totalstops(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be pandas dataframe')
    df['Total_Stops'] = df['Total_Stops'].str.replace('non-stop', '0')
    df['Total_Stops'] = df['Total_Stops'].str.replace('stops', '')
    df['Total_Stops'] = df['Total_Stops'].str.replace('stop', '')
    df['Total_Stops'].fillna(0, inplace=True)
    df['Total_Stops'] = df['Total_Stops'].astype(float)
    return df


def X(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be pandas dataframe')
    df['Hours'] = df['Duration'].str.split(' ').str[0]
    df['Hours'] = df['Hours'].str.replace('h', '').str.replace('m', '').astype(float)
    df['Hours'].fillna(0, inplace=True)

    df['Minutes'] = df['Duration'].str.split(' ').str[1]
    df['Minutes'] = df['Minutes'].str.replace('m', '').astype(float)
    df['Minutes'].fillna(0, inplace=True)

    df['Hours'] = df['Hours'] * 60
    df['Duration'] = df['Hours'] + df['Minutes']
    df.drop(['Hours', 'Minutes'], axis=1, inplace=True)
    return df


def clean_route(route):
    route = str(route)
    route = route.split(' → ')
    return ' '.join(route)


if __name__ == "__main__":
    x = load_dataset('train.xlsx')
    ms = load_dataset('ms.json')['market']
    bk_class = load_dataset('class.json')['class']
    df = marketshare(x, ms)
    df = classshare(df, bk_class)
    df = daystodep(df)
    df = timeofday(df)
    df = totalstops(df)
    df = X(df)
