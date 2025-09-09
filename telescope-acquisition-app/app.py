import streamlit as st
import numpy as np
from ModelBuilder import ModelBuilder
from ModelBuilder import ModelBuilderNN
from Parsing import TextParser

# -- Page configuration --
st.set_page_config(page_title="FLI-Point Model Trainer", layout="wide")

# -- Session state initialization --
for key, default in {
    'uploaded': False,
    'data': None,
    'lon': None,
    'lat': None,
    'headers': None,
    'builder': None,
    'modelBuilt': False,
    'step': 'predict',
    'last_prediction': None,
    'predictions': [],
    'prev_outputs': [],
    'eval_done': False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -- Cache for file parsing --
@st.cache_data(show_spinner=False)
def load_and_parse(file):
    parser = TextParser.TextParser(file)
    return parser.parseFile()  # returns data, lon, lat, headers

# -- Sidebar: Upload & Train Model --
st.sidebar.header("Upload & Train Model")
uploaded_file = st.sidebar.file_uploader("Upload .txt data file", type=["txt"])
model_name    = st.sidebar.text_input("Model name")
units         = st.sidebar.text_input("Coordinate units", value="degrees")

if uploaded_file and not st.session_state.uploaded:
    data, lon, lat, headers = load_and_parse(uploaded_file)
    st.session_state.data     = data
    st.session_state.lon      = lon
    st.session_state.lat      = lat
    st.session_state.headers  = headers
    st.session_state.uploaded = True
    st.sidebar.success("Data uploaded and parsed")

if st.session_state.uploaded:
    st.sidebar.subheader("Training parameters")
    n_estimators   = st.sidebar.number_input("Number of estimators", 10, 10000, 100)
    learning_rate  = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01, step=0.0001)
    max_depth      = st.sidebar.slider("Max depth", 1, 100, 8, step=1)
    early_stopping = st.sidebar.number_input("Early stopping rounds", 1, n_estimators, 10)

    extra_augments = st.sidebar.checkbox("Enable additional feature engineering", value=True)
    split_by_day   = st.sidebar.checkbox("Split train/test by date", value=True)

    st.sidebar.subheader("Data splits (%)")
    train_pct = st.sidebar.slider("Train (%)",     0, 100, 70)
    val_pct   = st.sidebar.slider("Validate (%)",  0, 100 - train_pct, 15)
    test_pct  = 100 - train_pct - val_pct
    st.sidebar.markdown(f"Test: **{test_pct}%**")

    show_cdf   = st.sidebar.checkbox("Show CDF plot", value=True)
    show_stats = st.sidebar.checkbox("Show stats",    value=True)
    use_NN = st.sidebar.checkbox("Use Neural Network Architecture",    value=False)

    if st.sidebar.button("Train Model", disabled=not model_name):
        with st.spinner("Training model..."):
            
            if use_NN:
                ## NEED TO FIX THE PARAMS HERE
                builder = ModelBuilderNN.ModelBuilderNN(
                {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'early_stopping_rounds': early_stopping
                },
                st.session_state.data,
                model_name,
                extra_augments
                )

            else:
                builder = ModelBuilder.ModelBuilder(
                    {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'early_stopping_rounds': early_stopping
                    },
                    st.session_state.data,
                    model_name,
                    extra_augments
                )
            builder.TrainTestSplit(
                split_by_day,
                train_pct / 100,
                (train_pct + val_pct) / 100
            )
            builder.createModel()
            st.session_state.builder    = builder
            st.session_state.modelBuilt = True
            st.session_state.eval_done  = False
        st.sidebar.success("Model trained")

# -- Main area --
st.title("FLI-Point Model Trainer")

if not st.session_state.uploaded:
    st.info("Please upload and parse your data to begin.")
elif not st.session_state.modelBuilt:
    st.info("Configure training parameters and click **Train Model** in the sidebar.")
else:
    tabs = st.tabs(["Evaluate", "Predict & Augment", "History"])

    # -- Evaluation tab --
    with tabs[0]:
        st.header("Model Evaluation")
        if not st.session_state.eval_done:
            st.session_state.builder.evaluateModel()
            st.session_state.eval_done = True

        if show_stats:
            st.subheader("Statistics on Test Set")
            offsetAz, offsetAlt, errorAz, errorAlt = st.session_state.builder.displayStatistics()
            st.write("Median Az offset:",    np.median(abs(offsetAz)), units)
            st.write("Median Az error:",     np.median(abs(errorAz)), units)
            st.write("90th % Az offset:",    np.percentile(abs(offsetAz), 90), units)
            st.write("90th % Az error:",     np.percentile(abs(errorAz), 90), units)
            st.write("Median Alt offset:",   np.median(abs(offsetAlt)), units)
            st.write("Median Alt error:",    np.median(abs(errorAlt)), units)
            st.write("90th % Alt offset:",   np.percentile(abs(offsetAlt), 90), units)
            st.write("90th % Alt error:",    np.percentile(abs(errorAlt), 90), units)

        if show_cdf:
            st.subheader("CDF of Errors")
            fig = st.session_state.builder.createCDF()
            st.pyplot(fig)

    # -- Predict & Augment tab --
    with tabs[1]:
        st.header("Predict & Augment Cycle")

        # Step 1: Predict Offset
        if st.session_state.step == 'predict':
            with st.form("predict_form", clear_on_submit=True):
                st.write("Enter inputs and click **Predict Offset**")
                mountAz  = st.number_input("Mount Azimuth",  value=150.0)
                mountAlt = st.number_input("Mount Altitude", value=30.0)
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    year   = st.number_input("Year",   value=2025)
                with c2:
                    month  = st.number_input("Month",  value=6)
                with c3:
                    day    = st.number_input("Day",    value=7)
                with c4:
                    hour   = st.number_input("Hour",   value=22)
                with c5:
                    minute = st.number_input("Minute", value=24)
                with c6:
                    second = st.number_input("Second", value=26)
                predict = st.form_submit_button("Predict Offset")
            if predict:
                pred = st.session_state.builder.makePrediction(
                    mountAz, mountAlt,
                    second, minute, hour, day, month, year,
                    st.session_state.predictions,
                    extra_augments
                )
                st.session_state.last_prediction = pred
                a1, a2 = st.columns(2)
                with a1:
                    st.write("Az Offset", pred[0][0])
                with a2:
                    st.write("Alt Offset", pred[0][1])

                st.session_state.prev_outputs.append(
                    (pred[0][0], pred[0][1])
                )

                if extra_augments:
                    st.session_state.step = 'augment'

        # Step 2: Add Additional Info
        elif st.session_state.step == 'augment':
            with st.form("augment_form", clear_on_submit=True):
                st.markdown("#### Additional Information")
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    prev_acq_az  = st.number_input("Prev Acq Error Az",  value=2.0)
                with a2:
                    prev_acq_alt = st.number_input("Prev Acq Error Alt", value=2.0)
                with a3:
                    prev_az      = st.number_input("Prev Mount Az",     value=150.0)
                with a4:
                    prev_alt     = st.number_input("Prev Mount Alt",    value=30.0)
                add_info = st.form_submit_button("Add Additional Info")
            if add_info:
                st.session_state.predictions.append(
                    (prev_acq_az, prev_acq_alt, prev_az, prev_alt)
                )
                st.session_state.step = 'predict'

    # -- History tab --
    with tabs[2]:
        st.header("Additional Info History")
        if st.session_state.predictions:
            for i, rec in enumerate(st.session_state.predictions, start=1):
                a0, a1, a2, a3, a4 = st.columns(5)
                with a0:
                    st.write("History List Input " + str(i) + ": ")
                with a1:
                    st.write("Prev Acq Error Az", rec[0])
                with a2:
                    st.write("Prev Acq Error Alt", rec[1])
                with a3:
                    st.write("Prev Mount Az", rec[2])
                with a4:
                    st.write("Prev Mount Alt", rec[3])
            for i, rec in enumerate(st.session_state.prev_outputs, start=1):
                a0, a1, a2, = st.columns(3)
                with a0:
                    st.write("Previous Modle Output " + str(i) + ": ")
                with a1:
                    st.write("Prev Az", rec[0])
                with a2:
                    st.write("Prev Alt", rec[1])

        else:
            st.write("No additional info records yet.")
