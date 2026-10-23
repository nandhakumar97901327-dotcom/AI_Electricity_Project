import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="AI Electricity Forecasting", page_icon="⚡", layout="wide")

st.title("⚡ AI Driven Electricity Consumption System")
st.write("Machine Learning matrum Reverse Calculation moolama electricity usage-a predict panra system.")

# --- 1. Generate Sample Historical Data ---
@st.cache_data
def load_sample_data():
    dates = pd.date_range(start="2023-01-01", periods=365)
    np.random.seed(42)
    temperature = np.random.normal(25, 5, 365)
    humidity = np.random.normal(60, 10, 365)
    consumption = 150 + (temperature * 3.5) + (humidity * 0.8) + np.random.normal(0, 15, 365)
    df = pd.DataFrame({'Date': dates, 'Temperature_C': temperature, 'Humidity_pct': humidity, 'Consumption_kWh': consumption})
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    return df

df = load_sample_data()

# --- 2. Train the AI Model ---
X = df[['Temperature_C', 'Humidity_pct', 'DayOfWeek', 'Month']]
y = df['Consumption_kWh']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ----------------- TABS CREATION -----------------
tab1, tab2 = st.tabs(["🔮 AI Prediction (Weather to Units)", "💰 Sector & Accessories Analyzer"])

# =========================================================================
# TAB 1: AI PREDICTION (SINGLE CLEAR 7-DAY FORECAST HISTOGRAM)
# =========================================================================
with tab1:
    st.subheader("Predict Future Consumption")
    st.write("Nalaiku weather eppadi irukum nu input kudutha, evvalo current thevapdum nu AI sollum.")
    col1, col2 = st.columns(2)
    with col1:
        pred_date = st.date_input("Select Future Date", datetime.date.today())
        pred_temp = st.slider("Expected Temperature (°C) / Veyil", 10.0, 45.0, 30.0)
    with col2:
        pred_humidity = st.slider("Expected Humidity (%) / Kulir", 10.0, 100.0, 60.0)
        
    if st.button("Predict Consumption"):
        pred_dayofweek = pred_date.weekday()
        pred_month = pred_date.month
        input_data = pd.DataFrame([[pred_temp, pred_humidity, pred_dayofweek, pred_month]],
                                  columns=['Temperature_C', 'Humidity_pct', 'DayOfWeek', 'Month'])
        prediction = model.predict(input_data)[0]
        
        # Selected date formatted for display
        date_str = pred_date.strftime("%d %B %Y")
        st.success(f"### 💡 Predicted Electricity for **{date_str}**: **{prediction:.2f} Units (kWh)**")
        
        # --- ONLY ONE CLEAR, INTUITIVE 7-DAY FORECAST HISTOGRAM ---
        st.write("---")
        st.write(f"### 📊 7-Day Electricity Forecast Histogram (தேதி வாரியான மின்சார அளவு)")
        st.caption(f"நீங்கள் தேர்வு செய்த தேதி ({date_str}) மற்றும் அதனைச் சுற்றியுள்ள 7 நாட்களின் கணிப்பு:")
        
        forecast_days = []
        for i in range(-2, 5):
            d = pred_date + datetime.timedelta(days=i)
            d_dow = d.weekday()
            d_month = d.month
            
            # Weather variation reflecting the trend
            t_var = pred_temp + (np.sin(i) * 1.5 if i != 0 else 0)
            h_var = pred_humidity + (np.cos(i) * 2.0 if i != 0 else 0)
            
            row = pd.DataFrame([[t_var, h_var, d_dow, d_month]],
                               columns=['Temperature_C', 'Humidity_pct', 'DayOfWeek', 'Month'])
            val = model.predict(row)[0]
            
            # Label highlighting the selected date clearly
            tag = " 👈 Selected" if i == 0 else ""
            label = d.strftime("%d %b (%a)") + tag
            forecast_days.append({
                "Date": label,
                "Units (kWh)": round(val, 1)
            })
            
        df_chart = pd.DataFrame(forecast_days).set_index("Date")
        st.bar_chart(df_chart["Units (kWh)"])

# =========================================================================
# TAB 2: SECTOR & ACCESSORIES (SINGLE CLEAN HISTOGRAM)
# =========================================================================
with tab2:
    st.subheader("💡 Dynamic Sector & Accessories Breakdown")
    st.write("Mudhalla neenga entha edathukku (Sector) bill calculate panna poreenga nu select pannunga.")
    
    sector = st.selectbox(
        "🏢 Select Sector (Edam):",
        ["🏠 Home", "🏥 Hospital", "🏢 Company / IT Office", "🏫 College", "🏫 School", "🌐 Other"]
    )
    
    bill_amount = st.number_input("Enter Electricity Bill Amount (₹) / Bill Thogai:", min_value=0, value=2000, step=100)
    
    acc_dict = {
        "🏠 Home": ["💡 Light", "🌀 Fan", "⚙️ Motor", "💻 System", "🔌 Charger", "❄️ Fridge", "📺 TV", "🧊 AC", "👕 Washing Machine", "💦 Water Heater", "🔥 Iron Box"],
        "🏥 Hospital": ["💡 Light", "🌀 Fan", "⚙️ Motor", "💻 System", "🔌 Charger", "❄️ Fridge", "🧊 AC", "⚕️ ICU Ventilator", "☢️ X-Ray / Scanner", "🛗 Lift", "💦 Water Heater"],
        "🏢 Company / IT Office": ["💡 Light", "🌀 Fan", "💻 System", "🔌 Charger", "🧊 AC", "🖧 Server", "☕ Coffee Machine", "🛗 Lift", "🖨️ Printer/Copier"],
        "🏫 College": ["💡 Light", "🌀 Fan", "⚙️ Motor", "💻 System", "🔌 Charger", "📽️ Projector/Smart Board", "🧊 AC", "🔬 Lab Equipment", "🚰 Water Cooler"],
        "🏫 School": ["💡 Light", "🌀 Fan", "⚙️ Motor", "💻 System", "🔌 Charger", "📽️ Projector/Smart Board", "🚰 Water Cooler"],
    }
    
    all_accessories = list(set([item for sublist in acc_dict.values() for item in sublist]))
    
    if sector == "🌐 Other":
        current_list = sorted(all_accessories)
        default_list = ["💡 Light", "🌀 Fan", "💻 System", "🔌 Charger"]
    else:
        current_list = acc_dict[sector]
        default_list = current_list[:5]
        
    st.write(f"#### ✅ Select Accessories for {sector}:")
    selected_accessories = st.multiselect("Accessories-a select pannunga:", options=current_list, default=default_list)
    
    if st.button("Calculate Usage"):
        if not selected_accessories:
            st.warning("⚠️ Ethaavathu oru accessory-a select pannunga!")
        else:
            if bill_amount == 0:
                estimated_units = 100 
            else:
                estimated_units = 100 + (bill_amount / 5.5)
                
            st.success(f"### ⚡ Total Estimated Electricity Used: **~ {estimated_units:.0f} Units (kWh)**")
            
            weights = {
                "💡 Light": 1, "🌀 Fan": 1.5, "🔌 Charger": 0.2, "💻 System": 2, "❄️ Fridge": 5, "⚙️ Motor": 4, 
                "📺 TV": 1.5, "🧊 AC": 10, "👕 Washing Machine": 3, "💦 Water Heater": 4, "🔥 Iron Box": 2,
                "⚕️ ICU Ventilator": 5, "☢️ X-Ray / Scanner": 8, "🛗 Lift": 6, "🖧 Server": 8, "☕ Coffee Machine": 1,
                "🖨️ Printer/Copier": 2, "📽️ Projector/Smart Board": 1, "🔬 Lab Equipment": 4, "🚰 Water Cooler": 2
            }
            
            # Dynamic Weight Logic
            if bill_amount > 5000:
                for heavy in ["🧊 AC", "💦 Water Heater", "🛗 Lift", "⚕️ ICU Ventilator", "☢️ X-Ray / Scanner", "🖧 Server"]:
                    if heavy in weights:
                        weights[heavy] *= 2.5
            elif bill_amount > 3000:
                for heavy in ["🧊 AC", "💦 Water Heater", "🛗 Lift"]:
                    if heavy in weights:
                        weights[heavy] *= 1.5
            
            total_weight = sum([weights.get(acc, 1) for acc in selected_accessories])
            
            results = []
            for acc in selected_accessories:
                acc_weight = weights.get(acc, 1)
                acc_units = (acc_weight / total_weight) * estimated_units
                acc_amount = (acc_weight / total_weight) * bill_amount
                percentage = (acc_weight / total_weight) * 100
                results.append({"Accessory": acc, "Units": round(acc_units), "Amount": acc_amount, "Percentage": percentage})
            
            # Sort highest amount first
            results = sorted(results, key=lambda x: x['Amount'], reverse=True)
            
            # Exact Math Rounding
            total_pct = sum([round(r['Percentage']) for r in results])
            diff_pct = 100 - total_pct
            if len(results) > 0:
                results[0]['Percentage'] += diff_pct
                
            for r in results:
                r['Percentage'] = round(r['Percentage'])
                r['Amount'] = round(r['Amount'])
                
            total_amt = sum([r['Amount'] for r in results])
            diff_amt = bill_amount - total_amt
            if len(results) > 0:
                results[0]['Amount'] += diff_amt
            
            st.write(f"#### 📊 {sector} Accessories Breakdown:")
            for r in results:
                st.write(f"- **{r['Accessory']}:** ~{r['Units']} Units | **₹ {r['Amount']}** ({r['Percentage']}%)")
            
            # --- ONLY ONE CLEAN, NATIVE HISTOGRAM (NO DUPLICATES) ---
            st.write("---")
            st.write("### 📊 Cost Breakdown Histogram (₹)")
            
            # Single, unified, interactive Streamlit bar histogram
            chart_df = pd.DataFrame(results).set_index("Accessory")
            st.bar_chart(chart_df["Amount"])
