import streamlit as st
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------- Data + Model Setup --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("vehicle_recalls.csv")
    df['report_received_date'] = pd.to_datetime(df['report_received_date'], errors='coerce')
    df['year'] = df['report_received_date'].dt.year
    df.dropna(subset=['year'], inplace=True)
    df['year'] = df['year'].astype(int)

    df['completion_rate'].fillna(df['completion_rate'].median(), inplace=True)
    df['potentially_affected'].fillna(df['potentially_affected'].median(), inplace=True)

    text_cols = ['consequence_summary', 'corrective_action', 'defect_summary']
    df[text_cols] = df[text_cols].fillna('Unknown')

    le_manufacturer = LabelEncoder()
    df['manufacturer_encoded'] = le_manufacturer.fit_transform(df['manufacturer'].astype(str))

    le_component = LabelEncoder()
    df['component_encoded'] = le_component.fit_transform(df['component'].astype(str))

    df['recall_severity'] = df['potentially_affected'] * (1 - df['completion_rate'])

    return df

@st.cache_resource
def train_model(df):
    features = df.groupby('year').agg({
        'potentially_affected': 'sum',
        'recall_severity': 'sum',
        'manufacturer_encoded': 'nunique',
        'component_encoded': 'nunique'
    }).reset_index()

    X = features[['year', 'recall_severity', 'manufacturer_encoded', 'component_encoded']]
    y = features['potentially_affected']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X

# -------------------- NLQ Functions --------------------
def extract_entities(query, df):
    query = query.lower()
    entities = {"year": None, "component": None, "intent": None}

    year_match = re.search(r'\b(20\d{2})\b', query)
    if year_match:
        entities["year"] = int(year_match.group())

    component_keywords = set(df['component'].str.lower().unique())
    for word in query.split():
        if word in component_keywords:
            entities["component"] = word

    if "forecast" in query or "demand" in query:
        entities["intent"] = "forecast"
    elif "trend" in query or "recalls over time" in query:
        entities["intent"] = "trend"
    elif "highest recalls" in query or "top manufacturers" in query:
        entities["intent"] = "top_manufacturer"
    elif "how many recalls" in query or "total recalls" in query:
        entities["intent"] = "recall_count"

    return entities

def process_nlq(query, df, model, X):
    entities = extract_entities(query, df)
    year, component, intent = entities["year"], entities["component"], entities["intent"]

    if intent == "forecast":
        if year:
            result = model.predict([[year, X['recall_severity'].mean(),
                                     X['manufacturer_encoded'].mean(),
                                     X['component_encoded'].mean()]])
            return f"🔮 Predicted spare part demand for {year}: **{int(result[0]):,}** units."
        else:
            return "📅 Please specify a year for demand forecasting."

    elif intent == "top_manufacturer":
        if year:
            top_mfr = df[df["year"] == year].groupby("manufacturer")["potentially_affected"].sum().idxmax()
            return f"🏆 The manufacturer with the highest recalls in {year} is **{top_mfr}**."
        else:
            top_mfr = df.groupby("manufacturer")["potentially_affected"].sum().idxmax()
            return f"🏆 The manufacturer with the highest recalls overall is **{top_mfr}**."

    elif intent == "trend":
        st.line_chart(df.groupby("year")["potentially_affected"].sum())
        return "📈 Showing recall trend over the years."

    elif intent == "recall_count":
        if year and component:
            count = df[(df["year"] == year) & (df["component"].str.lower() == component)]["potentially_affected"].sum()
            return f"🚗 Total recalls for **{component}** in {year}: **{int(count):,}** vehicles."
        elif year:
            count = df[df["year"] == year]["potentially_affected"].sum()
            return f"🚗 Total recalls in {year}: **{int(count):,}** vehicles."
        else:
            return "📅 Please specify a year to get recall count information."

    return "🤖 Sorry, I couldn't understand your query. Try asking about forecast, top manufacturer, trends, or recall count."

# -------------------- Streamlit App --------------------
st.title("🔧 Vehicle Recall Chatbot")
st.markdown("Ask me questions about vehicle recalls — forecasts, trends, top manufacturers, and more!")

# Load everything before chatbot UI
df = load_data()
model, X = train_model(df)

# Init chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show prior messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user message
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = process_nlq(user_input, df, model, X)
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
