import streamlit as st
import numpy as np
import pickle
import html

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NextWord",
    page_icon="N",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PROFESSIONAL LIGHT UI
# =========================================================

st.html("""
<style>

    /* -----------------------------------------------------
       GLOBAL
    ----------------------------------------------------- */

    .stApp {
        background: #f7f8fa;
        color: #172033;
    }

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 900px;
        padding-top: 30px;
        padding-bottom: 50px;
    }


    /* -----------------------------------------------------
       HEADER
    ----------------------------------------------------- */

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;

        padding-bottom: 22px;

        border-bottom: 1px solid #e5e7eb;
    }

    .brand-section {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .brand-logo {
        width: 34px;
        height: 34px;

        display: flex;
        align-items: center;
        justify-content: center;

        background: #243b64;
        color: white;

        border-radius: 8px;

        font-size: 15px;
        font-weight: 700;
    }

    .brand-name {
        color: #172033;

        font-size: 18px;
        font-weight: 700;

        letter-spacing: -0.3px;
    }

    .brand-description {
        color: #7b8494;

        font-size: 11px;

        margin-top: 1px;
    }

    .model-status {
        color: #64748b;

        font-size: 11px;

        display: flex;
        align-items: center;

        gap: 6px;
    }

    .status-dot {
        width: 6px;
        height: 6px;

        border-radius: 50%;

        background: #16a34a;
    }


    /* -----------------------------------------------------
       INTRO
    ----------------------------------------------------- */

    .intro {
        margin-top: 52px;
        margin-bottom: 28px;
    }

    .intro-title {
        color: #172033;

        font-size: 30px;
        font-weight: 700;

        letter-spacing: -0.8px;
    }

    .intro-description {
        color: #6b7280;

        font-size: 14px;

        margin-top: 7px;

        line-height: 1.6;
    }


    /* -----------------------------------------------------
       INPUT LABEL
    ----------------------------------------------------- */

    .input-label {
        color: #374151;

        font-size: 13px;

        font-weight: 600;

        margin-bottom: 8px;
    }


    /* -----------------------------------------------------
       INPUT
    ----------------------------------------------------- */

    div[data-testid="stTextInput"] {
        margin-bottom: 0;
    }

    div[data-testid="stTextInput"] input {

        background: white !important;

        color: #172033 !important;

        border: 1px solid #d9dee7 !important;

        border-radius: 9px !important;

        height: 58px !important;

        padding-left: 16px !important;
        padding-right: 16px !important;

        font-size: 16px !important;

        box-shadow: 0 1px 2px rgba(0,0,0,0.03);

        transition: border-color 0.15s ease,
                    box-shadow 0.15s ease;
    }

    div[data-testid="stTextInput"] input:hover {
        border-color: #b9c2d0 !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #4769a1 !important;

        box-shadow:
            0 0 0 2px rgba(71,105,161,0.10) !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #9aa3b2 !important;
    }


    /* -----------------------------------------------------
       BUTTON
    ----------------------------------------------------- */

    div.stButton {
        margin-top: 12px;
    }

    div.stButton > button {

        width: 100%;

        height: 45px;

        background: #243b64 !important;

        color: white !important;

        border: 1px solid #243b64 !important;

        border-radius: 8px !important;

        font-size: 13px !important;

        font-weight: 600 !important;

        transition: background 0.15s ease;
    }

    div.stButton > button:hover {
        background: #1e3153 !important;

        border-color: #1e3153 !important;
    }


    /* -----------------------------------------------------
       RESULT SECTION
    ----------------------------------------------------- */

    .section-label {
        color: #6b7280;

        font-size: 11px;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 0.8px;

        margin-top: 42px;

        margin-bottom: 10px;
    }


    /* -----------------------------------------------------
       MAIN PREDICTION
    ----------------------------------------------------- */

    .prediction {

        background: white;

        border: 1px solid #e1e5eb;

        border-radius: 10px;

        padding: 20px;

        box-shadow:
            0 1px 3px rgba(0,0,0,0.04);
    }

    .prediction-top {
        display: flex;

        align-items: center;
        justify-content: space-between;
    }

    .prediction-word {
        color: #243b64;

        font-size: 25px;

        font-weight: 700;

        letter-spacing: -0.4px;
    }

    .confidence {
        color: #667085;

        font-size: 12px;
    }

    .confidence strong {
        color: #344054;
    }


    /* -----------------------------------------------------
       PROGRESS
    ----------------------------------------------------- */

    .progress-background {

        height: 4px;

        background: #edf0f4;

        border-radius: 5px;

        margin-top: 15px;

        overflow: hidden;
    }

    .progress {

        height: 100%;

        background: #5878ae;

        border-radius: 5px;
    }


    /* -----------------------------------------------------
       SENTENCE PREVIEW
    ----------------------------------------------------- */

    .sentence {

        margin-top: 12px;

        background: #f9fafb;

        border: 1px solid #eaecf0;

        border-radius: 8px;

        padding: 14px 16px;

        color: #475467;

        font-size: 14px;

        line-height: 1.6;
    }

    .sentence-word {
        color: #243b64;

        font-weight: 650;
    }


    /* -----------------------------------------------------
       SUGGESTIONS
    ----------------------------------------------------- */

    .suggestions {

        display: grid;

        grid-template-columns:
            repeat(5, 1fr);

        gap: 8px;

        margin-top: 10px;
    }

    .suggestion {

        background: white;

        border: 1px solid #e1e5eb;

        border-radius: 8px;

        padding: 11px 8px;

        text-align: center;
    }

    .suggestion-word {
        color: #344054;

        font-size: 13px;

        font-weight: 600;

        white-space: nowrap;

        overflow: hidden;

        text-overflow: ellipsis;
    }

    .suggestion-probability {
        color: #98a2b3;

        font-size: 10px;

        margin-top: 3px;
    }


    /* -----------------------------------------------------
       EMPTY STATE
    ----------------------------------------------------- */

    .empty {

        margin-top: 42px;

        padding: 30px;

        text-align: center;

        border: 1px dashed #d8dde6;

        border-radius: 10px;

        background: rgba(255,255,255,0.5);
    }

    .empty-title {
        color: #475467;

        font-size: 13px;

        font-weight: 600;
    }

    .empty-description {
        color: #98a2b3;

        font-size: 11px;

        margin-top: 5px;
    }


    /* -----------------------------------------------------
       FOOTER
    ----------------------------------------------------- */

    .footer {

        border-top: 1px solid #e5e7eb;

        margin-top: 55px;

        padding-top: 18px;

        text-align: center;

        color: #98a2b3;

        font-size: 10px;
    }


    /* -----------------------------------------------------
       MOBILE
    ----------------------------------------------------- */

    @media (max-width: 650px) {

        .block-container {
            padding-left: 18px;
            padding-right: 18px;
        }

        .intro-title {
            font-size: 26px;
        }

        .model-status {
            display: none;
        }

        .suggestions {
            grid-template-columns:
                repeat(2, 1fr);
        }
    }

</style>
""")


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_resources():

    model = load_model(
        "lstm_model.h5",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open("config.pkl", "rb") as file:
        config = pickle.load(file)

    return model, tokenizer, config


model, tokenizer, config = load_resources()

max_sequence_len = config["max_sequence_len"]
vocab_size = config["vocab_size"]


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_top_words(text, top_k=5):

    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return []

    sequence = pad_sequences(
        [sequence],
        maxlen=max_sequence_len,
        padding="pre",
        truncating="pre"
    )

    probabilities = model.predict(
        sequence,
        verbose=0
    )[0]

    top_indices = np.argsort(
        probabilities
    )[-top_k:][::-1]

    results = []

    for index in top_indices:

        index = int(index)

        word = tokenizer.index_word.get(index)

        if word:

            probability = float(
                probabilities[index]
            )

            results.append(
                (word, probability)
            )

    return results


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="header">

    <div class="brand-section">

        <div class="brand-logo">
            N
        </div>

        <div>

            <div class="brand-name">
                NextWord
            </div>

            <div class="brand-description">
                Next word prediction
            </div>

        </div>

    </div>

    <div class="model-status">

        <span class="status-dot"></span>

        Model ready

    </div>

</div>
""")


# =========================================================
# INTRO
# =========================================================

st.html("""
<div class="intro">

    <div class="intro-title">
        Complete your sentence
    </div>

    <div class="intro-description">
        Enter some text and let the model predict
        what word is most likely to come next.
    </div>

</div>
""")


# =========================================================
# INPUT
# =========================================================

st.html("""
<div class="input-label">
    Your text
</div>
""")


text = st.text_input(
    "Your text",
    placeholder="For example: I love machine...",
    label_visibility="collapsed"
)


# =========================================================
# BUTTON
# =========================================================

predict_clicked = st.button(
    "Predict next word"
)


# =========================================================
# RESULT
# =========================================================

if predict_clicked:

    if not text.strip():

        st.warning(
            "Please enter some text before predicting."
        )

    else:

        predictions = predict_top_words(
            text,
            top_k=5
        )

        if predictions:

            best_word = predictions[0][0]

            best_probability = predictions[0][1]

            safe_text = html.escape(text)

            safe_word = html.escape(
                best_word
            )


            # -------------------------------------------------
            # PREDICTION LABEL
            # -------------------------------------------------

            st.html("""
            <div class="section-label">
                Prediction
            </div>
            """)


            # -------------------------------------------------
            # MAIN PREDICTION
            # -------------------------------------------------

            confidence_percent = (
                best_probability * 100
            )

            st.html(f"""
            <div class="prediction">

                <div class="prediction-top">

                    <div class="prediction-word">
                        {safe_word}
                    </div>

                    <div class="confidence">
                        Confidence:
                        <strong>
                            {confidence_percent:.2f}%
                        </strong>
                    </div>

                </div>

                <div class="progress-background">

                    <div
                        class="progress"
                        style="width:
                        {confidence_percent:.2f}%"
                    >
                    </div>

                </div>

            </div>
            """)


            # -------------------------------------------------
            # SENTENCE PREVIEW
            # -------------------------------------------------

            st.html(f"""
            <div class="sentence">

                {safe_text}

                <span class="sentence-word">
                    {safe_word}
                </span>

            </div>
            """)


            # -------------------------------------------------
            # OTHER SUGGESTIONS
            # -------------------------------------------------

            st.html("""
            <div class="section-label">
                Other suggestions
            </div>
            """)


            suggestions_html = ""

            for word, probability in predictions:

                safe_word = html.escape(word)

                suggestions_html += f"""
                <div class="suggestion">

                    <div class="suggestion-word">
                        {safe_word}
                    </div>

                    <div class="suggestion-probability">
                        {probability * 100:.2f}%
                    </div>

                </div>
                """


            st.html(
                f"""
                <div class="suggestions">
                    {suggestions_html}
                </div>
                """
            )


        else:

            st.warning(
                "The model could not generate a prediction."
            )


# =========================================================
# EMPTY STATE
# =========================================================

if not predict_clicked:

    st.html("""
    <div class="empty">

        <div class="empty-title">
            Ready to predict
        </div>

        <div class="empty-description">
            Enter your text above and click
            "Predict next word".
        </div>

    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">
    NextWord · Natural Language Processing
</div>
""")