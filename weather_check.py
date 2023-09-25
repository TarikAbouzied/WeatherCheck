import requests
import os
import logging
import pandas as pd
import os.path
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook')
import smtplib
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage


api_key = os.environ.get('OPEN_WEATHER_MAP_KEY')
lat = "47.550440"
lon = "-122.393460"
MM_PER_INCH = 25.4
email_address = os.environ.get('EMAIL_ADDRESS')
pword = os.environ.get('GMAIL_PASS')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_weather(api_key, lat, lon):
    try:
        
        if not api_key:
            raise ValueError("API key not found in environment variables.")
            
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=imperial&appid={api_key}"
        
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        weather = response.json()
        logging.info("Successfully fetched weather data.")
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
    
    except ValueError as e:
        logging.error(f"Value error: {e}")
    
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        
    return weather

def transform_to_dataframes(weather):
    
    weather_data = []
    
    for entry in weather['list']:
        dt_pacific = datetime.utcfromtimestamp(entry['dt']).replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Pacific'))
        date = dt_pacific.strftime('%Y-%m-%d')
        high_temp = entry['main']['temp_max']
        low_temp = entry['main']['temp_min']
        rainfall = entry.get('rain', {}).get('3h', 0) / MM_PER_INCH
        main_weather = entry['weather'][0]['main']
        max_windspeed = entry['wind']['speed']
    
        weather_data.append([dt_pacific, date, high_temp, low_temp, rainfall, max_windspeed, main_weather])

    df_3hr_info = pd.DataFrame(weather_data, columns=['dt_pacific', 'date', 'high_temp', 'low_temp', 'rain_3h', 'max_windspeed','main_weather'])
    
    # Aggregate by date
    agg_funcs = {
        'high_temp': 'max',
        'low_temp': 'min',
        'rain_3h': 'sum',
        'max_windspeed': 'max'
    }
    daily_summary = df_3hr_info.groupby('date').agg(agg_funcs).reset_index()
    
    # Custom aggregation for weather hours
    weather_hours = df_3hr_info.groupby('date')['main_weather'].apply(lambda x: x.value_counts()).unstack().fillna(0)
    weather_hours.columns = [f'hours_{col.lower()}' for col in weather_hours.columns]
    weather_hours = weather_hours * 3
    
    # Merge with daily_summary
    daily_summary = pd.merge(daily_summary, weather_hours, left_on='date', right_index=True, how='left').fillna(0)
    daily_summary.set_index('date', inplace=True)
    
    return df_3hr_info, daily_summary

def check_overseed_conditions(daily_summary):

    daily_summary['overseed_today'] = False
    
    for i in range(len(daily_summary) - 3):
        following_three_days = daily_summary.iloc[i+1:i+4, :]
        temps_ok = (following_three_days['high_temp'] <= 75).all() & (following_three_days['low_temp'] >= 50).all()
        rain_ok = (following_three_days['rain_3h'] >= 0.1).all() & (following_three_days['rain_3h'] <= 0.25).all()
        wind_ok = (following_three_days['max_windspeed'] < 10).all()
    
        daily_summary.loc[daily_summary.index.to_list()[i], 'overseed_today'] = (temps_ok & rain_ok & wind_ok)
        
    if any(daily_summary['overseed_today']):
        msg = EmailMessage()
        msg['Subject'] = 'Overseed window Detected!'
        msg['From'] = email_address
        msg['To'] = email_address
        msg_text = f"The following dates are good for overseeding: {daily_summary[daily_summary.overseed_today].index.to_list()}\n\n{daily_summary.to_markdown()}"
        msg.set_content(msg_text)

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(email_address, pword)
        server.send_message(msg)
        server.quit()
        logging.info('Overseed window detected, email sent successfully')
    else:
        logging.info('No overseed window detected')
        
def check_flood_conditions(df_3hr_info):
    
    flood_conditions_2019 = pd.Series({
        'rain_3h': 0.2606299212598425, 
        'rain_6h': 0.5153543307086614, 
        'rain_12h': 1.0059055118110236, 
        'rain_24h': 1.6803149606299215, 
        'rain_48h': 2.078740157480315, 
        'rain_72h': 2.15748031496063, 
        'rain_96h': 2.265748031496063, 
        'rain_120h': 2.5385826771653544
    })

    flood_conditions_2020 = pd.Series({
        'rain_3h': 0.6358267716535433,
        'rain_6h': 1.12244094488189,
        'rain_12h': 1.3748031496062991,
        'rain_24h': 1.6688976377952756,
        'rain_48h': 1.9295275590551182,
        'rain_72h': 2.2791338582677168,
        'rain_96h': 2.604330708661417,
        'rain_120h': 2.9586614173228347
    })

    df_with_cumulative_totals = df_3hr_info.copy().set_index('dt_pacific').drop(['date','high_temp','low_temp','max_windspeed','main_weather'], axis=1)

    for hours in [6, 12, 24, 48, 72, 96, 120]:
        df_with_cumulative_totals[f'rain_{hours}h'] = df_with_cumulative_totals['rain_3h'].rolling(window=int(hours / 3)).sum()

    potential_flood_conditions = {}
    saturation_windows = ['rain_24h', 'rain_48h', 'rain_72h', 'rain_96h', 'rain_120h']
    i = 2019
    for flood_conditions in [flood_conditions_2019, flood_conditions_2020]:
        high_active_rain = (df_with_cumulative_totals['rain_3h'] >= flood_conditions['rain_3h'])
        saturated_soil = (df_with_cumulative_totals[saturation_windows] >= flood_conditions[saturation_windows]).any(axis=1)
    
        potential_flood_conditions[i] = df_with_cumulative_totals[high_active_rain & saturated_soil]
        i += 1

    potential_flood_times = pd.Index(potential_flood_conditions[2019].index).union(potential_flood_conditions[2020].index).drop_duplicates()

    if len(potential_flood_times) > 0:
    
        formatted_index = potential_flood_times.strftime('%a %b %d %I%p')
        formatted_string = '\n'.join(formatted_index)
    
        msg = MIMEMultipart()
        msg['Subject'] = 'Upcoming Flood Conditions Detected!'
        msg['From'] = email_address
        msg['To'] = email_address
    
        msg_text = f"The following times are forecast to meet/exceed flood conditions:\n\n{formatted_string}"
        text_part = MIMEText(msg_text, 'plain')
        msg.attach(text_part)
    
        charts = []
    
        if len(potential_flood_conditions[2019]) > 0:
            for idx, row in potential_flood_conditions[2019].iterrows():
                data = pd.DataFrame({
                    'Conditions': row.index,
                     idx.strftime('%Y-%m-%d %I%p'): row.values,
                    '2019 Flood': flood_conditions_2019.values
                })
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Conditions', y='value', hue='variable', data=pd.melt(data, ['Conditions']))
                plt.xticks(rotation=60)
                plt.xlabel('')
                plt.ylabel('inches')
                plt.legend(title='')
                plt.title(f"{idx.strftime('%a %b %d %I%p')} vs 2019 Flood")
                plt.tight_layout() 
                plt.savefig(f"{idx.strftime('%Y-%m-%d_%I%p')}_v_2019_flood.png", dpi=125)
            
                charts.append(f"{idx.strftime('%Y-%m-%d_%I%p')}_v_2019_flood.png")
            
        if len(potential_flood_conditions[2020]) > 0:
            for idx, row in potential_flood_conditions[2020].iterrows():
                data = pd.DataFrame({
                    'Conditions': row.index,
                     idx.strftime('%Y-%m-%d %I%p'): row.values,
                    '2020 Flood': flood_conditions_2020.values
                })
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Conditions', y='value', hue='variable', data=pd.melt(data, ['Conditions']))
                plt.xticks(rotation=60)
                plt.xlabel('')
                plt.ylabel('inches')
                plt.legend(title='')
                plt.title(f"{idx.strftime('%a %b %d %I%p')} vs 2020 Flood")
                plt.tight_layout() 
                plt.savefig(f"{idx.strftime('%Y-%m-%d_%I%p')}_v_2020_flood.png", dpi=125)
            
                charts.append(f"{idx.strftime('%Y-%m-%d_%I%p')}_v_2020_flood.png")
    
        for chart in charts:
            with open(chart, 'rb') as f:
                image_data = f.read()
            image_part = MIMEImage(image_data, name=chart)
            msg.attach(image_part)
    
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(email_address, pword)
        server.send_message(msg)
        server.quit()
        logging.info('Potential flood conditions detected, email sent successfully')
    else:
        logging.info('No potential flood conditions detected')
        
        
def main():
    weather_data = fetch_weather(api_key, lat, lon)
    df_3hr_info, daily_summary = transform_to_dataframes(weather_data)
    check_overseed_conditions(daily_summary)
    check_flood_conditions(df_3hr_info)

if __name__ == "__main__":
    main()
