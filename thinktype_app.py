import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ThinkType - Typing Personality", layout="centered")
st.title("🧠 ThinkType – Typing Personality Detector")

st.markdown("Type the following sentence below and we'll analyze your typing style:")

sample_text = "The quick brown fox jumps over the lazy dog."
st.info(f"👉 Type this: **{sample_text}**")

text_input = st.text_input("Start typing here...")

# Timing logic
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "hesitation_count" not in st.session_state:
    st.session_state.hesitation_count = 0
if "last_time" not in st.session_state:
    st.session_state.last_time = None

if text_input and st.session_state.start_time is None:
    st.session_state.start_time = time.time()

# Count hesitation (pause > 2 seconds between keystrokes)
if text_input:
    now = time.time()
    if st.session_state.last_time and now - st.session_state.last_time > 2:
        st.session_state.hesitation_count += 1
    st.session_state.last_time = now

# ML model training on-the-fly
@st.cache_resource
def train_model():
    data = {
        "avg_key_delay": [0.15, 0.35, 0.25, 0.10, 0.5, 0.18],
        "total_time": [20, 45, 30, 15, 50, 22],
        "hesitation_count": [1, 5, 3, 0, 6, 2],
        "prediction": ["Focused", "Stressed", "Neutral", "Focused", "Stressed", "Focused"]
    }
    df = pd.DataFrame(data)
    X = df[["avg_key_delay", "total_time", "hesitation_count"]]
    y = df["prediction"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

# When done
if text_input == sample_text:
    total_time = time.time() - st.session_state.start_time
    avg_key_delay = total_time / len(text_input)
    hesitation_count = st.session_state.hesitation_count

    st.success("✅ Typing complete!")
    st.write(f"⏱️ Total Time: {total_time:.2f} seconds")
    st.write(f"⌨️ Average Keystroke Delay: {avg_key_delay:.3f} seconds")
    st.write(f"😬 Hesitation Count: {hesitation_count}")

    # Predict
    features = np.array([[avg_key_delay, total_time, hesitation_count]])
    prediction = model.predict(features)[0]
    st.markdown(f"### 🧠 Your Typing Personality: **{prediction}**")

    # Reset
    st.session_state.start_time = None
    st.session_state.hesitation_count = 0
    st.session_state.last_time = None
