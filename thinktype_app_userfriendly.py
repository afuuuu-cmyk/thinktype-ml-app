import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ThinkType - Typing Personality", layout="centered")
st.title("🧠 ThinkType – Typing Personality Detector")

sample_text = "The quick brown fox jumps over the lazy dog."
st.markdown("### 👇 Start the test to analyze your typing behavior")

if "started" not in st.session_state:
    st.session_state.started = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "hesitation_count" not in st.session_state:
    st.session_state.hesitation_count = 0
if "last_time" not in st.session_state:
    st.session_state.last_time = None
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

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

# UI Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Typing Test"):
        st.session_state.started = True
        st.session_state.start_time = time.time()
        st.session_state.text_input = ""
        st.session_state.hesitation_count = 0
        st.session_state.last_time = None

with col2:
    if st.button("🔁 Reset"):
        st.session_state.started = False
        st.session_state.start_time = None
        st.session_state.hesitation_count = 0
        st.session_state.last_time = None
        st.session_state.text_input = ""

# Typing input box
if st.session_state.started:
    st.info(f"✍️ Type this exact sentence below:

**{sample_text}**")
    typed = st.text_input("Start typing here...", value=st.session_state.text_input)
    st.session_state.text_input = typed

    # Hesitation tracking
    now = time.time()
    if st.session_state.last_time and now - st.session_state.last_time > 2:
        st.session_state.hesitation_count += 1
    st.session_state.last_time = now

    # Submit button
    if st.button("✅ Submit Typing"):
        if typed.strip() == sample_text:
            total_time = time.time() - st.session_state.start_time
            avg_key_delay = total_time / len(sample_text)
            hesitation_count = st.session_state.hesitation_count

            st.success("✅ Typing captured successfully!")
            st.markdown(f"""
                #### 📊 Typing Analysis
                - ⏱️ **Total Time:** {total_time:.2f} sec
                - ⌨️ **Avg Keystroke Delay:** {avg_key_delay:.3f} sec
                - 😬 **Hesitations:** {hesitation_count}
            """)

            # Predict
            features = np.array([[avg_key_delay, total_time, hesitation_count]])
            prediction = model.predict(features)[0]

            desc = {
                "Focused": "💡 You were consistent and quick — a sign of sharp focus.",
                "Stressed": "⚠️ Irregular typing with multiple pauses — may reflect stress or distraction.",
                "Neutral": "😐 Balanced typing behavior — neither stressed nor fully focused."
            }

            st.markdown(f"### 🧠 Personality: **{prediction}**")
            st.info(desc.get(prediction, "Interesting style!"))

            # Reset after submission
            st.session_state.started = False
        else:
            st.error("❌ Please type the exact sentence, including punctuation and capitalization.")
