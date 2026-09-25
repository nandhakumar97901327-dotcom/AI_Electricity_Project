import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime

# Page Configuration
st.set_page_config(page_title="AI Electricity Forecasting", page_icon="⚡", layout="wide")

st.title("⚡ AI Driven Electricity Consumption System")
st.write("Machine Learning matrum Reverse Calculation moolama electricity usage-a predict panra system.")

# 1. Generate Sample Historical Data
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

# 2. Train the AI Model
X = df[['Temperature_C', 'Humidity_pct', 'DayOfWeek', 'Month']]
y = df['Consumption_kWh']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ----------------- TABS CREATION -----------------
tab1, tab2 = st.tabs(["🔮 AI Prediction (Weather to Units)", "💰 Sector & Accessories Analyzer"])

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
        
        st.info(f"### 💡 Predicted Electricity Consumption: **{prediction:.2f} Units (kWh)**")
        
        st.line_chart(df.set_index('Date')['Consumption_kWh'])

with tab2:
    st.subheader("💡 Dynamic Sector & Accessories Breakdown")
    st.write("Mudhalla neenga entha edathukku (Sector) bill calculate panna poreenga nu select pannunga. Apram antha edathula irukka Accessories-a select pannunga.")
    
    # 1. Sector Selection
    sector = st.selectbox(
        "🏢 Select Sector (Edam):",
        ["🏠 Home", "🏥 Hospital", "🏢 Company / IT Office", "🏫 College", "🏫 School", "🌐 Other"]
    )
    
    bill_amount = st.number_input("Enter Electricity Bill Amount (₹) / Bill Thogai:", min_value=0, value=2000, step=100)
    
    # 2. Accessories lists based on sector
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
        default_list = current_list[:5] # Select first 5 by default
        
    st.write(f"#### ✅ Select Accessories for {sector}:")
    selected_accessories = st.multiselect("Accessories-a select pannunga:", options=current_list, default=default_list)
    
    if st.button("Calculate Usage"):
        if not selected_accessories:
            st.warning("⚠️ Ethaavathu oru accessory-a select pannunga!")
        else:
            # Estimation Logic
            if bill_amount == 0:
                estimated_units = 100 
            else:
                estimated_units = 100 + (bill_amount / 5.5)
                
            st.success(f"### ⚡ Total Estimated Electricity Used: **~ {estimated_units:.0f} Units (kWh)**")
            
            st.write(f"#### 📊 {sector} Accessories Breakdown:")
            
            # Master Weights for all accessories
            weights = {
                "💡 Light": 1, "🌀 Fan": 1.5, "🔌 Charger": 0.2, "💻 System": 2, "❄️ Fridge": 5, "⚙️ Motor": 4, 
                "📺 TV": 1.5, "🧊 AC": 10, "👕 Washing Machine": 3, "💦 Water Heater": 4, "🔥 Iron Box": 2,
                "⚕️ ICU Ventilator": 5, "☢️ X-Ray / Scanner": 8, "🛗 Lift": 6, "🖧 Server": 8, "☕ Coffee Machine": 1,
                "🖨️ Printer/Copier": 2, "📽️ Projector/Smart Board": 1, "🔬 Lab Equipment": 4, "🚰 Water Cooler": 2
            }
            
            total_weight = sum([weights.get(acc, 1) for acc in selected_accessories])
            
            for acc in selected_accessories:
                acc_weight = weights.get(acc, 1)
                acc_units = (acc_weight / total_weight) * estimated_units
                acc_amount = (acc_weight / total_weight) * bill_amount  # Calculate Amount
                percentage = int((acc_weight / total_weight) * 100)
                
                # Update text to show Units and Amount
                st.write(f"- **{acc}:** ~{acc_units:.0f} Units | **₹ {acc_amount:.0f}** ({percentage}%)")
                st.progress(percentage)
                
            st.info("💡 (Note: Ithu ovvoru accessory-oda general power usage-a vechu AI calculate panna model.)")
