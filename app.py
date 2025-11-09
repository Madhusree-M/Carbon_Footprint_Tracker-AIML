# import base64
# import streamlit as st
# import pandas as pd
# import joblib
# import math

# # Load trained model
# model = joblib.load("carbon_model.pkl")

# # Page setup
# st.set_page_config(page_title="Carbon Footprint Calculator", page_icon="🌱", layout="centered")

# # === Background setup ===
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as file:
#         encoded_string = base64.b64encode(file.read()).decode()
#     st.markdown(
#         f"""
#         <style>
#         [data-testid="stAppViewContainer"] {{
#             background-image:
#                 linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)),
#                 url("data:image/png;base64,{encoded_string}");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}

#         /* Text colors and readability */
#         h1, h2, h3, h4, h5, h6, p, label, span {{
#             color: #ffffff !important;
#             text-shadow: 0px 0px 6px rgba(0,0,0,0.6);
#         }}

#         [data-testid="stHeader"], [data-testid="stToolbar"] {{
#             background: rgba(0,0,0,0);
#         }}

#         /* Style input boxes */
#         .stSelectbox div[data-baseweb="select"], input, textarea {{
#             background-color: rgba(255, 255, 255, 0.9) !important;
#             border-radius: 10px;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # 👉 change this image to your own background file if needed
# add_bg_from_local("bg.jpg")

# # === Page title ===
# st.title("🌎 AI-Based Carbon Footprint Calculator")
# st.write("Estimate your carbon emission based on your lifestyle choices below:")

# # === User inputs ===
# st.subheader("\nPersonal details : ")
# body = st.selectbox("🏋️ Body Type", ["underweight", "normal", "overweight", "obese"])
# sex = st.selectbox("🧍 Gender", ["male", "female"])
# diet = st.selectbox("🥗 Diet", ["vegan", "vegetarian", "pescatarian", "omnivore"])
# activity = st.selectbox("🎉 Social Activity", ["never", "sometimes", "often", "frequently", "very frequently"])

# st.subheader("\nTravel details : ")
# transport = st.selectbox("🚗 Transport Type", ["walk/bicycle", "public", "private"])
# vehicle = st.selectbox("🚘 Vehicle Type", ["", "petrol", "diesel", "hybrid", "lpg"])
# distance = st.number_input("🛣️ Vehicle Monthly Distance (in km)", min_value=0)
# flight = st.selectbox("✈️ Frequency of Traveling by Air", ["never", "rarely", "frequently", "very frequently"])

# st.subheader("\nWaste management : ")
# bag_size = st.selectbox("🗑️ Waste Bag Size", ["small", "medium", "large", "extra large"])
# bag_count = st.number_input("♻️ Waste Bag Weekly Count", min_value=0)
# recycling = st.multiselect("♻️ Recycling Materials", ["Paper", "Plastic", "Glass", "Metal"])

# st.subheader("\nEnergy management : ")
# heating = st.selectbox("🔥 Heating Energy Source", ["coal", "wood", "natural gas", "electricity"])
# cooking = st.multiselect("🍳 Cooking Methods", ["Stove", "Oven", "Microwave", "Grill", "Airfryer"])
# efficiency = st.selectbox("⚡ Energy Efficiency", ["Yes", "No", "Sometimes"])
# screen_time = st.number_input("💻 How Long TV/PC Daily (hours)", min_value=0)
# internet = st.number_input("🌐 Internet Usage Daily (hours)", min_value=0)

# st.subheader("\nConsumption : ")
# shower = st.selectbox("🚿 How Often Do You Shower", ["less frequently", "daily", "more frequently", "twice a day"])
# grocery = st.number_input("🛒 Monthly Grocery Bill (in ₹)", min_value=0)
# clothes = st.number_input("👕 How Many New Clothes Monthly", min_value=0)

# # === Prediction ===
# if st.button("Calculate My Carbon Footprint 🌱"):
#     # Create dataframe
#     input_data = pd.DataFrame([{
#         'Body Type': body,
#         'Sex': sex,
#         'Diet': diet,
#         'How Often Shower': shower,
#         'Heating Energy Source': heating,
#         'Transport': transport,
#         'Vehicle Type': vehicle,
#         'Social Activity': activity,
#         'Monthly Grocery Bill': grocery,
#         'Frequency of Traveling by Air': flight,
#         'Vehicle Monthly Distance Km': distance,
#         'Waste Bag Size': bag_size,
#         'Waste Bag Weekly Count': bag_count,
#         'How Long TV PC Daily Hour': screen_time,
#         'How Many New Clothes Monthly': clothes,
#         'How Long Internet Daily Hour': internet,
#         'Energy efficiency': efficiency,
#         'Recycling': str(recycling),
#         'Cooking_With': str(cooking)
#     }])

#     # Predict emission
#     emission = model.predict(input_data)[0]
#     st.success(f"Your estimated Carbon Emission is **{emission:.2f} kg CO₂ per month** 🌍")

#     # Emission category
#     if emission < 1500:
#         st.info("🌿 Low carbon footprint — you're eco-conscious!")
#     elif emission < 3000:
#         st.warning("⚠️ Medium carbon footprint — room for improvement.")
#     else:
#         st.error("🔥 High carbon footprint — consider sustainable changes.")

#     # === Tree calculation (fixed realistic range) ===
#     sequestration_per_tree_per_year = 30.0     # kg CO₂ per year per tree
#     time_horizon_years = 40                    # number of years tree absorbs carbon
#     annual_emission = emission * 12.0
#     total_absorption = sequestration_per_tree_per_year * time_horizon_years

#     trees_needed = math.ceil(annual_emission / total_absorption)

#     # === Display result ===
#     st.markdown(
#         f"""
#         <div style="
#             background: rgba(0, 100, 0, 0.45);
#             border-radius: 15px;
#             padding: 20px;
#             margin-top: 20px;
#             text-align: center;
#             font-size: 24px;
#             color: #ADFF2F;
#             font-weight: 700;
#             text-shadow: 1px 1px 6px black;
#         ">
#             🌳 To offset your <b>{annual_emission:.0f} kg CO₂/year</b>,
#             you would need to plant approximately
#             <span style='color:#00ffcc; font-size:30px;'>{trees_needed}</span> trees
#             (assuming ~{sequestration_per_tree_per_year:.0f} kg CO₂/year per tree over {time_horizon_years} years).
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import math

app = Flask(__name__)
app.secret_key = 'carbon_secret_key'

# Load your trained ML model
model = joblib.load(open('carbon_model.pkl', 'rb'))

# ---------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/personal', methods=['GET', 'POST'])
def personal():
    if request.method == 'POST':
        session['body'] = request.form['body']
        session['sex'] = request.form['sex']
        session['diet'] = request.form['diet']
        session['activity'] = request.form['activity']
        return redirect(url_for('travel'))
    return render_template('personal.html')


@app.route('/travel', methods=['GET', 'POST'])
def travel():
    if request.method == 'POST':
        session['transport'] = request.form['transport']
        session['vehicle'] = request.form['vehicle']
        session['distance'] = request.form['distance']
        session['flight'] = request.form['flight']
        return redirect(url_for('waste'))
    return render_template('travel.html')


@app.route('/waste', methods=['GET', 'POST'])
def waste():
    if request.method == 'POST':
        session['bag_size'] = request.form['bag_size']
        session['bag_count'] = request.form['bag_count']
        session['recycling'] = request.form.getlist('recycling')
        return redirect(url_for('energy'))
    return render_template('waste.html')


@app.route('/energy', methods=['GET', 'POST'])
def energy():
    if request.method == 'POST':
        session['heating'] = request.form['heating']
        session['cooking'] = request.form.getlist('cooking')
        session['efficiency'] = request.form['efficiency']
        session['screen_time'] = request.form['screen_time']
        session['internet'] = request.form['internet']
        return redirect(url_for('consumption'))
    return render_template('energy.html')


@app.route('/consumption', methods=['GET', 'POST'])
def consumption():
    if request.method == 'POST':
        session['shower'] = request.form['shower']
        session['grocery'] = request.form['grocery']
        session['clothes'] = request.form['clothes']

        # Prepare data for prediction
        data = pd.DataFrame([{
            'Body Type': session.get('body'),
            'Sex': session.get('sex'),
            'Diet': session.get('diet'),
            'How Often Shower': session.get('shower'),
            'Heating Energy Source': session.get('heating'),
            'Transport': session.get('transport'),
            'Vehicle Type': session.get('vehicle'),
            'Social Activity': session.get('activity'),
            'Monthly Grocery Bill': session.get('grocery'),
            'Frequency of Traveling by Air': session.get('flight'),
            'Vehicle Monthly Distance Km': session.get('distance'),
            'Waste Bag Size': session.get('bag_size'),
            'Waste Bag Weekly Count': session.get('bag_count'),
            'How Long TV PC Daily Hour': session.get('screen_time'),
            'How Many New Clothes Monthly': session.get('clothes'),
            'How Long Internet Daily Hour': session.get('internet'),
            'Energy efficiency': session.get('efficiency'),
            'Recycling': str(session.get('recycling')),
            'Cooking_With': str(session.get('cooking'))
        }])

        # Model prediction
        emission_value = float(model.predict(data)[0])

        # Trees needed calculation
        sequestration_per_tree_per_year = 30.0
        time_horizon_years = 40
        annual_emission = emission_value * 12.0
        total_absorption = sequestration_per_tree_per_year * time_horizon_years
        trees_needed = math.ceil(annual_emission / total_absorption)

        return redirect(url_for('result', emission=emission_value, trees=trees_needed))

    return render_template('consumption.html')


@app.route('/result')
def result():
    emission = float(request.args.get('emission', 0))
    trees = int(request.args.get('trees', 0))
    return render_template('result.html', emission=emission, trees=trees)


if __name__ == '__main__':
    app.run(debug=True)
