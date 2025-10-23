import time
import random
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="NeuroFocus Lab", page_icon="🧠", layout="centered")
st.title("🧠 NeuroFocus Lab — Stroop + ML Lapse Predictor")

# Session state
# Streamlit re runs the file on every button click
# The "st.session_state" dict keeps values persistent
if "data" not in st.session_state:
    st.session_state.data = []  # list of dict rows
if "trial_idx" not in st.session_state:
    st.session_state.trial_idx = 0
if "trial" not in st.session_state:
    st.session_state.trial = None
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False
if "t_start" not in st.session_state:
    st.session_state.t_start = None

# Shows the instructions with bold formatting (markdown)
st.markdown("""
This app runs a mini **Stroop** task. Press the button matching the **INK COLOR** (ignore the word).
We log Reaction Time (RT) & accuracy. After a block, we train a small model to predict **next-trial error**.
""")

# Constants + sidebar controls
colors = ["RED", "GREEN", "BLUE", "YELLOW"]
color_to_hex = {"RED":"#e74c3c","GREEN":"#27ae60","BLUE":"#2980b9","YELLOW":"#f1c40f"}

block_len = st.sidebar.slider("Trials per block", 10, 80, 20, 5)
jitter_ms = st.sidebar.slider("Inter-trial interval (ms)", 300, 1500, 600, 100)
st.sidebar.write("Tip: Try 2–3 blocks, then train the model.")

# Making a new trial (first function)
def new_trial():
    word = random.choice(colors)
    ink = random.choice(colors)
    congruent = int(word == ink)
    return {"word": word, "ink": ink, "congruent": congruent}

# Start/next trial
# Draws a button
# Disabled=... keeps it disabled while waiting for a repsonse so multiple trials arent spawned accidently
if st.button("Start / Next Trial", disabled=st.session_state.awaiting_response):
    st.session_state.trial = new_trial()
    st.session_state.awaiting_response = True
    st.session_state.t_start = time.perf_counter()

# Show the stimulus (only if we are waiting)
# unsafe_allow_html = True lets HTML render
if st.session_state.awaiting_response and st.session_state.trial:
    t = st.session_state.trial
    st.markdown(
        f"<h1 style='text-align:center;color:{color_to_hex[t['ink']]};font-size:64px;'>{t['word']}</h1>",
        unsafe_allow_html=True
    )

    # Answer buttons + click handler
    cols = st.columns(4)
    choices = ["RED","GREEN","BLUE","YELLOW"]

    # Handle(choice) runs when you click one of the colro buttons
    # Safety check (if not awaiting response, ignore)
    # Correct == 1 if click equals the ink color (goal of stroop)
    # Row dict captures everything about this trial
    # Append the row to data, increment trial_idx, mark response as finished
    def handle(choice):
        if not st.session_state.awaiting_response:
            return
        rt = (time.perf_counter() - st.session_state.t_start) * 1000.0
        correct = int(choice == st.session_state.trial["ink"])
        row = {
            "trial": st.session_state.trial_idx,
            "word": st.session_state.trial["word"],
            "ink": st.session_state.trial["ink"],
            "congruent": st.session_state.trial["congruent"],
            "choice": choice,
            "correct": correct,
            "rt_ms": rt
        }
        st.session_state.data.append(row)
        st.session_state.trial_idx += 1
        st.session_state.awaiting_response = False
        time.sleep(jitter_ms/1000.0)

    # Draws the 4 response buttons in 4 columns, on click calls the handle(...)
    with cols[0]:
        if st.button("RED"): handle("RED")
    with cols[1]:
        if st.button("GREEN"): handle("GREEN")
    with cols[2]:
        if st.button("BLUE"): handle("BLUE")
    with cols[3]:
        if st.button("YELLOW"): handle("YELLOW")

# Turn rows into a table + download
# df == whole dataset (every trial so far)
# If no trials yet, show message
# Else, show the last 10 rows and a button to download all data as CSV
# Dataframe + save
df = pd.DataFrame(st.session_state.data)
st.subheader("Collected data")
if df.empty:
    st.info("No data yet. Click **Start / Next Trial** and respond.")
else:
    st.dataframe(df.tail(10), use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "stroop_data.csv", "text/csv")

# If df has any rows, plot the last 50 RTs as line chart
# Quick charts
if not df.empty and "rt_ms" in df:
    st.subheader("Reaction time (ms) — recent")
    st.line_chart(df["rt_ms"].tail(50))

    # Shows a boxplot of RT grouped by congruent (1) vs incongruent (0). If Altair is not installed, print small note
    st.subheader("Congruent vs Incongruent RTs")
    try:
        import altair as alt
        chart = alt.Chart(df).mark_boxplot().encode(
            x=alt.X("congruent:N", title="Congruent (=1) vs Incongruent (=0)"),
            y=alt.Y("rt_ms:Q", title="RT (ms)")
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.caption("Altair not available for boxplot.")

# Train simple ML to predict next-trial error from recent history
st.subheader("Train ML lapse predictor")
if len(df) < 30:
    st.warning("Collect at least ~30 trials for a decent toy model.")
elif "correct" in df and "rt_ms" in df:
    # Build temporal features using a rolling window of the past 5 trials
    # W = 5: how far back to look
    # df_feat: copy of data to add feature columns
    W = 5
    df_feat = df.copy()

    # rt_z is a z score of RT vs recent (rolling) mean over a 20 trial window
    # Subtract your recent average and divide by your recent standard deviation
    # Result will tell you if this trial was faster/slower than your usual lately
    df_feat["rt_z"] = (df_feat["rt_ms"] - df_feat["rt_ms"].rolling(20, min_periods=5).mean()) / \
                      (df_feat["rt_ms"].rolling(20, min_periods=5).std().replace(0,np.nan))

    # .shift(k) means "the value k trials ago"
    # creates 15 columns total here: for k = 1..5, and 3 variables for each (correct, congruent, rt)
    for k in range(1, W+1):
        df_feat[f"prev{k}_correct"] = df_feat["correct"].shift(k)
        df_feat[f"prev{k}_congruent"] = df_feat["congruent"].shift(k)
        df_feat[f"prev{k}_rt"] = df_feat["rt_ms"].shift(k)
    # Target: whether *this* trial is an error (1 = error)
    # Target = 1 when trial is wrong and 0 when correct
    df_feat["error"] = 1 - df_feat["correct"]
    df_feat = df_feat.dropna().reset_index(drop=True)

    # pick inputs...
    # all columns that start with "prev" (the 15 history features)
    # plus rt_z (current trial's relative speed)
    # plus congruent (wether this trial was congruent)
    # X is the matrix of inputs, y is the vector of targets
    if len(df_feat) >= 10:
        feature_cols = [c for c in df_feat.columns if c.startswith("prev")] + ["rt_z", "congruent"]
        X = df_feat[feature_cols].values
        y = df_feat["error"].values

        # Splits the time ordered data into: first 70% is training, and the last 30% is testing
        # Shuffle = False prevents mixing time order
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        
        # Creates a logistic regression model and teaches it using the training data
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # y_pred: 0/1 predictions for the test set
        # y_proba: the models confidence that each row is a error (a num between 0 and 1)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        
        # acc: percent of test samples correctly predicted
        # auc: measures how weel the model ranks errors vs non errors
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        # Prints the scores and shows which columns went into the model
        st.write(f"**Test Accuracy:** {acc:.3f} | **ROC-AUC:** {auc:.3f}")
        st.caption("Tiny, noisy dataset—scores vary. Point is data → features → model → metric.")
        st.markdown("**Feature columns used:**")
        st.code(feature_cols)
    else:
        st.info("Not enough rows after feature building. Do a few more trials.")