import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
from twilio.rest import Client

warnings.filterwarnings("ignore")

# ---------------- Twilio Credentials ----------------

TWILIO_AUTH_TOKEN = "XXXXXXXXXXXXXXXXXXXXXXXXXXX"
TWILIO_FROM_NUMBER = "+15673XXXXXX"  # Twilio number
TWILIO_TO_NUMBER = "+91XXXXXXXXX"    # recipient number

# ----------- Load Data Function -----------  
@st.cache_data
def load_data():
    df = pd.read_csv("combined_aqi_with_aqi_column.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    df = df.dropna(subset=['AQI'])
    return df

df = load_data()
all_cities = df['City'].unique().tolist()

# ----------- Sidebar Navigation -----------  
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose App", ["Home", "Forecast + SMS Alert", "Predict Future AQI"])

# ----------- Home Screen -----------  
if app_mode == "Home":
    st.title("🌫️ AQI Multi-Tool Dashboard")
    st.markdown("Welcome! Choose an option from the sidebar to either forecast AQI with SMS alerts or make single/range AQI predictions using Prophet.")
    st.image("https://www.airvisual.com/images/icons/air-quality-dashboard-graphic.png", use_column_width=True)
# ----------- Forecast + SMS Alert (Script 1) -----------  
elif app_mode == "Forecast + SMS Alert":
    st.title("🌫️ AQI Forecast & SMS Alert")
    st.markdown("Predict AQI and send an SMS alert if AQI > 150.")

    # City selection  
    city = st.selectbox("Select a city", all_cities)

    # Date ranges  
    today = datetime.now().date()
    week1_start = today + timedelta(days=1)
    week1_end   = today + timedelta(days=8)
    week2_start = today + timedelta(days=9)
    week2_end   = today + timedelta(days=16)
    month_start = today + timedelta(days=17)
    month_end   = month_start + timedelta(days=31)

    # Forecast function  
    def forecast_with_alert(city_name):
        city_df = df[df['City'].str.lower() == city_name.lower()][['Timestamp', 'AQI']]
        if city_df.empty:
            st.error("No data available for this city.")
            return None, None
        city_df = city_df.rename(columns={'Timestamp':'ds','AQI':'y'})
        model = Prophet(daily_seasonality=True)
        try:
            model.fit(city_df)
            future = model.make_future_dataframe(periods=200)
            forecast = model.predict(future)
            forecast['date'] = forecast['ds'].dt.date
            return city_df, forecast
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            return None, None

    # SMS alert function  
    def send_alert(city_name, aqi_val):
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        body = f"⚠️ AQI Alert for {city_name}: Predicted AQI = {aqi_val} (above 150). Stay safe!"
        try:
            message = client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=TWILIO_TO_NUMBER)
            return message
        except Exception as e:
            st.error(f"Failed to send SMS: {e}")
            return None

    city_df, forecast = forecast_with_alert(city)
    if forecast is not None:
        # Today's prediction  
        today_row = forecast.iloc[(forecast['date'] - today).abs().argsort()[:1]]
        today_aqi = round(today_row['yhat'].values[0], 2)

        # Automatically send alert if AQI > 150  
        if today_aqi > 150:
            try:
                message = send_alert(city, today_aqi)
                if message:
                    st.success("✅ Automatic SMS alert sent! The predicted AQI is above 150.")
            except Exception as e:
                st.error(f"Failed to send SMS alert: {e}")

        # Aggregates  
        week1 = forecast[(forecast['date']>=week1_start)&(forecast['date']<=week1_end)]
        week2 = forecast[(forecast['date']>=week2_start)&(forecast['date']<=week2_end)]
        month = forecast[(forecast['date']>=month_start)&(forecast['date']<=month_end)]
        week1_avg = round(week1['yhat'].mean(),2) if not week1.empty else None
        week2_avg = round(week2['yhat'].mean(),2) if not week2.empty else None
        month_avg = round(month['yhat'].mean(),2) if not month.empty else None

        # Display  
        st.metric("📈 Predicted AQI Today", today_aqi)
        st.metric(f"📊 Week 1 Avg ({week1_start} to {week1_end})", week1_avg)
        st.metric(f"📊 Week 2 Avg ({week2_start} to {week2_end})", week2_avg)
        st.metric(f"📊 Month Avg ({month_start} to {month_end})", month_avg)

        # Forecast plot  
        st.subheader("Forecast Plot")
        fig, ax = plt.subplots()
        ax.plot(city_df['ds'], city_df['y'], label="Historical")
        ax.plot(forecast['ds'], forecast['yhat'], label="Forecast", color='orange')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
        ax.set_xlabel("Date"); ax.set_ylabel("AQI"); ax.legend()
        st.pyplot(fig)

	

# ----------- Prediction with Prophet (Script 2) -----------  
elif app_mode == "Predict Future AQI":
    st.title("🌫️ Predict future AQI")
    #st.markdown("Forecast AQI for a single date or average over a date range. Made by Abhishek Kumar Gupta")

    # Prophet forecast cache  
    @st.cache_data
    def prophet_forecast(city_name):
        city_df = df[df['City']==city_name][['Timestamp','AQI']].rename(columns={'Timestamp':'ds','AQI':'y'})
        model = Prophet(daily_seasonality=True)
        model.fit(city_df)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        return forecast[['ds','yhat']]

    # Function to suggest outdoor activities based on AQI
    def suggest_activity(aqi):
        if aqi <= 50:
            return "🌞 Excellent air quality! Great for outdoor activities like walking, cycling, or hiking."
        elif aqi <= 100:
            return "😊 Good air quality! You can enjoy outdoor activities like jogging, cycling, or a walk in the park."
        elif aqi <= 150:
            return "😷 Moderate air quality. Outdoor activities like walking or light jogging are fine, but avoid strenuous exercises."
        elif aqi <= 200:
            return "⚠️ Unhealthy air quality. Consider staying indoors or limit outdoor activities, especially strenuous ones."
        elif aqi <= 300:
            return "🚫 Very unhealthy air quality. Stay indoors as much as possible. If you must go outside, take precautions like wearing a mask."
        else:
            return "❌ Hazardous air quality. Stay indoors and avoid outdoor activities."

    city_choice = st.selectbox("Select City", all_cities)
    mode = st.radio("Mode", ["Single Date","Date Range"])

    if mode == "Single Date":
        date_inp = st.date_input("Select Date")
        if st.button("Predict AQI"):
            val = prophet_forecast(city_choice)
            res = val[val['ds']==pd.to_datetime(date_inp)]['yhat']
            if res.empty:
                st.error(f"No data for {date_inp}")
            else:
                predicted_aqi = round(res.values[0], 2)
                st.success(f"Predicted AQI on {date_inp} = {predicted_aqi}")
                
                # Suggest outdoor activity based on predicted AQI
                activity_suggestion = suggest_activity(predicted_aqi)
                st.markdown(f"### Outdoor Activity Suggestion: {activity_suggestion}")
    else:
        start = st.date_input("Start Date", key='s')
        end   = st.date_input("End Date", key='e')
        if start>end:
            st.warning("Start must be before End.")
        elif st.button("Predict Avg AQI"):
            val = prophet_forecast(city_choice)
            mask = (val['ds']>=pd.to_datetime(start))&(val['ds']<=pd.to_datetime(end))
            subset = val.loc[mask]
            if subset.empty:
                st.error("No data in range.")
            else:
                avg_aqi = round(subset['yhat'].mean(), 2)
                st.success(f"Avg AQI from {start} to {end} = {avg_aqi}")
                
                # Suggest outdoor activity based on average AQI
                activity_suggestion = suggest_activity(avg_aqi)
                st.markdown(f"### Outdoor Activity Suggestion: {activity_suggestion}")

st.markdown("Made by Anshu Kumar")