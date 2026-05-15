import streamlit as st
import numpy as np
import pandas as pd


# -- Page configuration --
st.set_page_config(page_title="FLI-Point Acquisition Correction", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1320px; }
    [data-testid="stSidebar"] { background: #f7f8fa; }
    div[data-testid="stMetric"] {
        border: 1px solid #e7e9ee;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        background: #fff;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 0.5rem 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    'last_input': None,
    'eval_done': False,
    'upload_id': None,
    'model_kind': 'XGBoost',
    'flash_message': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -- Cache for file parsing --
@st.cache_data(show_spinner=False)
def load_and_parse(file):
    from Parsing.TextParser import TextParser
    parser = TextParser(file)
    return parser.parseFile()  # returns data, lon, lat, headers

def _field_key(prefix, label):
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(label))
    return f"{prefix}_{safe}".lower()

def _make_training_progress(label):
    progress = st.sidebar.progress(0, text=f"{label}: preparing...")
    status = st.sidebar.empty()

    def update(current, total, logs=None):
        total = max(int(total), 1)
        current = min(max(float(current), 0.0), float(total))
        pct = current / total
        message = f"{label}: {current:.1f}/{total}" if current % 1 else f"{label}: {int(current)}/{total}"

        if logs:
            if "status" in logs:
                message = f"{label}: {logs['status']}"
            val_loss = logs.get("validation_0", {}).get("mae")
            if isinstance(val_loss, list) and val_loss:
                message += f" | validation MAE {val_loss[-1]:.4g}"
            elif "val_loss" in logs:
                message += f" | val loss {logs['val_loss']:.4g}"
            elif "loss" in logs:
                message += f" | loss {logs['loss']:.4g}"

        progress.progress(pct, text=message)
        status.caption(message)

    return update, progress, status

# -- Sidebar: Upload & Train Model --
st.sidebar.header("Upload & Train")
uploaded_file = st.sidebar.file_uploader("Upload .txt data file", type=["txt"])
model_name    = st.sidebar.text_input("Model name", value="acquisition-correction")
units         = st.sidebar.text_input("Coordinate units", value="degrees")

with st.sidebar.expander("Required file format", expanded=False):
    st.markdown(
        """
        Each file must start with site coordinates, then a header row.

        Required columns:
        `Date Time MountRA MountDec ActualRA ActualDec`

        Extra columns are optional. Any additional numeric or boolean columns
        are used automatically as model features.

        YMD example:
        ```text
        Longitude = -110.95deg Latitude = 32.18deg
        Date Time MountRA MountDec ActualRA ActualDec Pressure Count Temperature
        2021-01-01 00:00:00 188.1 44.7 188.3 44.7 48.6 True 11
        ```

        UNIX example:
        ```text
        Longitude = -110.95deg Latitude = 32.18deg
        Date Time MountRA MountDec ActualRA ActualDec Pressure Count Temperature
        1609459200 0 188.1 44.7 188.3 44.7 48.6 True 11
        ```
        """
    )

if uploaded_file:
    upload_id = f"{uploaded_file.name}:{uploaded_file.size}"
    if upload_id != st.session_state.upload_id:
        try:
            data, lon, lat, headers = load_and_parse(uploaded_file)
            st.session_state.data     = data
            st.session_state.lon      = lon
            st.session_state.lat      = lat
            st.session_state.headers  = headers
            st.session_state.uploaded = True
            st.session_state.upload_id = upload_id
            st.session_state.builder = None
            st.session_state.modelBuilt = False
            st.session_state.eval_done = False
            st.session_state.predictions = []
            st.session_state.prev_outputs = []
            st.session_state.last_input = None
            st.session_state.step = 'predict'
            st.sidebar.success(f"Parsed {data.shape[0]:,} observations")
        except Exception as exc:
            st.session_state.uploaded = False
            st.sidebar.error(f"Could not parse file: {exc}")

if st.session_state.uploaded:
    st.sidebar.caption(
        f"Site: lon {st.session_state.lon:.4f}, lat {st.session_state.lat:.4f} | "
        f"{st.session_state.data.shape[1]} columns"
    )
    st.sidebar.subheader("Training parameters")
    model_kind = st.sidebar.radio("Model family", ["XGBoost", "Neural network"], horizontal=True)
    learning_rate  = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.01, step=0.0001, format="%.4f")
    if model_kind == "XGBoost":
        n_estimators   = st.sidebar.number_input("Number of estimators", 10, 10000, 300)
        max_depth      = st.sidebar.slider("Max depth", 1, 30, 8, step=1)
        early_stopping = st.sidebar.number_input("Early stopping rounds", 1, int(n_estimators), 25)
        nn_epochs = 200
        nn_batch_size = 16
        nn_hidden_units = 128
        nn_alpha = 0.0001
    else:
        n_estimators = 300
        max_depth = 8
        early_stopping = st.sidebar.number_input("Patience", 1, 200, 25)
        nn_epochs = st.sidebar.number_input("Max epochs", 10, 2000, 300)
        nn_batch_size = st.sidebar.number_input("Batch size", 1, 512, 16)
        nn_hidden_units = st.sidebar.slider("Hidden units", 16, 512, 128, step=16)
        nn_alpha = st.sidebar.number_input("L2 regularization", 0.0, 1.0, 0.0001, step=0.0001, format="%.4f")

    extra_augments = st.sidebar.checkbox("Enable additional feature engineering", value=True)
    split_by_day   = st.sidebar.checkbox("Split train/test by date", value=True)

    st.sidebar.subheader("Data splits (%)")
    train_pct = st.sidebar.slider("Train (%)",     0, 100, 70)
    val_pct   = st.sidebar.slider("Validate (%)",  0, 100 - train_pct, 15)
    test_pct  = 100 - train_pct - val_pct
    st.sidebar.markdown(f"Test: **{test_pct}%**")

    show_cdf   = st.sidebar.checkbox("Show CDF plot", value=True)
    show_stats = st.sidebar.checkbox("Show stats",    value=True)
    if st.sidebar.button("Train Model", disabled=not model_name):
        with st.spinner("Training model..."):
            try:
                progress_update, progress_bar, progress_status = _make_training_progress(model_kind)
                params = {
                    'n_estimators': int(n_estimators),
                    'learning_rate': float(learning_rate),
                    'max_depth': int(max_depth),
                    'early_stopping_rounds': int(early_stopping),
                    'epochs': int(nn_epochs),
                    'batch_size': int(nn_batch_size),
                    'hidden_units': int(nn_hidden_units),
                    'alpha': float(nn_alpha),
                }

                if model_kind == "Neural network":
                    from ModelBuilder.ModelBuilderNN import ModelBuilderNN
                    builder = ModelBuilderNN(
                        params,
                        np.copy(st.session_state.data),
                        model_name,
                        extra_augments,
                        st.session_state.headers,
                    )
                else:
                    from ModelBuilder.ModelBuilder import ModelBuilder
                    builder = ModelBuilder(
                        params,
                        np.copy(st.session_state.data),
                        model_name,
                        extra_augments,
                        st.session_state.headers,
                    )
                builder.TrainTestSplit(
                    split_by_day,
                    train_pct / 100,
                    (train_pct + val_pct) / 100
                )
                progress_update(0, int(nn_epochs if model_kind == "Neural network" else n_estimators))
                builder.createModel(progress_callback=progress_update)
                st.session_state.builder    = builder
                st.session_state.modelBuilt = True
                st.session_state.eval_done  = False
                st.session_state.model_kind = model_kind
                st.session_state.predictions = []
                st.session_state.prev_outputs = []
                st.session_state.last_input = None
                st.session_state.step = 'predict'
                progress_bar.progress(1.0, text=f"{model_kind}: training complete")
                progress_status.caption("Training complete")
                st.sidebar.success("Model trained")
            except Exception as exc:
                st.session_state.modelBuilt = False
                st.sidebar.error(f"Training failed: {exc}")

# -- Main area --
st.title("FLI-Point Acquisition Correction")
st.caption("Train from prior observations, evaluate correction quality, then run an autoregressive prediction loop.")
def _summ_stats(x, p=90):
    x = np.asarray(x, dtype=float)
    x = np.abs(x[np.isfinite(x)])
    return dict(
        count=int(x.size),
        median=float(np.median(x)) if x.size else np.nan,
        pctl=float(np.percentile(x, p)) if x.size else np.nan,
        mean=float(np.mean(x)) if x.size else np.nan,
        std=float(np.std(x)) if x.size else np.nan,
    )

def _fmt(x, units):
    if x is None or np.isnan(x):
        return "—"
    return f"{x:.3g} {units}"

    
if not st.session_state.uploaded:
    st.info("Please upload and parse your data to begin.")
elif not st.session_state.modelBuilt:
    st.subheader("Parsed Observation File")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{st.session_state.data.shape[0]:,}")
    c2.metric("Features", f"{st.session_state.data.shape[1] - 2}")
    c3.metric("Longitude", f"{st.session_state.lon:.4f}")
    c4.metric("Latitude", f"{st.session_state.lat:.4f}")
    st.dataframe(
        pd.DataFrame(st.session_state.data, columns=st.session_state.headers).head(25),
        use_container_width=True,
    )
    st.info("Configure training parameters and click **Train Model** in the sidebar.")
else:
    tabs = st.tabs(["Evaluate", "Predict & Augment", "History"])
    if st.session_state.flash_message:
        st.success(st.session_state.flash_message)
        st.session_state.flash_message = None

    # -- Evaluation tab --
    with tabs[0]:
        st.header("Model Evaluation")
        st.caption(f"{st.session_state.model_kind} | {st.session_state.builder.title}")
        if not st.session_state.eval_done:
            with st.spinner("Evaluating held-out observations..."):
                st.session_state.builder.evaluateModel()
                st.session_state.eval_done = True

        if show_stats:
            st.subheader("Statistics on Test Set")

            # Pull arrays
            offsetRA, offsetDec, errorRA, errorDec = st.session_state.builder.displayStatistics()

            # Optional: let the user choose the percentile (default 90th)
            p = st.slider("Percentile to display", min_value=50, max_value=99, value=90, step=1)

            # Compute summaries
            RA_off = _summ_stats(offsetRA, p=p)
            RA_err = _summ_stats(errorRA, p=p)
            Dec_off = _summ_stats(offsetDec, p=p)
            Dec_err = _summ_stats(errorDec, p=p)

            # Tabs for RA / Dec
            tab_RA, tab_Dec = st.tabs(["RA", "Dec"])

            with tab_RA:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Offset (RA)")
                    st.metric("Median", _fmt(RA_off["median"], units))
                    st.metric(f"{p}th Percentile", _fmt(RA_off["pctl"], units))
                with c2:
                    st.caption("Prediction Error (RA)")
                    st.metric("Median", _fmt(RA_err["median"], units))
                    st.metric(f"{p}th Percentile", _fmt(RA_err["pctl"], units))

                # Small table below
                RA_df = pd.DataFrame(
                    {
                        "Count":   [RA_off["count"], RA_err["count"]],
                        "Mean":    [RA_off["mean"],  RA_err["mean"]],
                        "Std Dev": [RA_off["std"],   RA_err["std"]],
                        f"Median": [RA_off["median"], RA_err["median"]],
                        f"P{p}":   [RA_off["pctl"],   RA_err["pctl"]],
                    },
                    index=["Offset RA", "Error RA"],
                )
                st.dataframe(
                    RA_df.style.format(
                        {
                            "Mean":    lambda x: _fmt(x, units),
                            "Std Dev": lambda x: _fmt(x, units),
                            "Median":  lambda x: _fmt(x, units),
                            f"P{p}":   lambda x: _fmt(x, units),
                        }
                    ),
                    use_container_width=True,
                    hide_index=False,
                )

            with tab_Dec:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Offset (Dec)")
                    st.metric("Median", _fmt(Dec_off["median"], units))
                    st.metric(f"{p}th Percentile", _fmt(Dec_off["pctl"], units))
                with c2:
                    st.caption("Prediction Error (Dec)")
                    st.metric("Median", _fmt(Dec_err["median"], units))
                    st.metric(f"{p}th Percentile", _fmt(Dec_err["pctl"], units))

                Dec_df = pd.DataFrame(
                    {
                        "Count":   [Dec_off["count"], Dec_err["count"]],
                        "Mean":    [Dec_off["mean"],  Dec_err["mean"]],
                        "Std Dev": [Dec_off["std"],   Dec_err["std"]],
                        f"Median": [Dec_off["median"], Dec_err["median"]],
                        f"P{p}":   [Dec_off["pctl"],   Dec_err["pctl"]],
                    },
                    index=["Offset Dec", "Error Dec"],
                )
                st.dataframe(
                    Dec_df.style.format(
                        {
                            "Mean":    lambda x: _fmt(x, units),
                            "Std Dev": lambda x: _fmt(x, units),
                            "Median":  lambda x: _fmt(x, units),
                            f"P{p}":   lambda x: _fmt(x, units),
                        }
                    ),
                    use_container_width=True,
                    hide_index=False,
                )
##### END
        if show_cdf:
            st.subheader("CDF of Errors")
            fig = st.session_state.builder.createCDF()
            st.pyplot(fig)

    # -- Predict & Augment tab --
    with tabs[1]:
        st.header("Predict & Augment Cycle")
        builder = st.session_state.builder
        extra_augments = builder.extraAugments

        # Step 1: Predict Offset
        if st.session_state.step == 'predict':
            with st.form("predict_form", clear_on_submit=False):
                st.write("Enter the requested mount position and observation context.")
                mountRA  = st.number_input("Mount RA", value=150.0, key="predict_mount_ra")
                mountDec = st.number_input("Mount Dec", value=30.0, key="predict_mount_dec")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    year   = st.number_input("Year", value=2025, key="predict_year")
                with c2:
                    month  = st.number_input("Month", value=6, key="predict_month")
                with c3:
                    day    = st.number_input("Day", value=7, key="predict_day")
                with c4:
                    hour   = st.number_input("Hour", value=22, key="predict_hour")
                with c5:
                    minute = st.number_input("Minute", value=24, key="predict_minute")
                with c6:
                    second = st.number_input("Second", value=26, key="predict_second")
                feature_values = {}
                if builder.custom_feature_names:
                    st.markdown("#### Observation features")
                    feature_cols = st.columns(min(3, len(builder.custom_feature_names)))
                    for idx, name in enumerate(builder.custom_feature_names):
                        with feature_cols[idx % len(feature_cols)]:
                            feature_values[name] = st.number_input(
                                name,
                                value=float(builder.feature_defaults.get(name, 0.0)),
                                key=_field_key("predict_feature", name),
                            )
                predict = st.form_submit_button("Predict Offset")
            if predict:
                try:
                    pred = builder.makePrediction(
                        mountRA, mountDec,
                        second, minute, hour, day, month, year,
                        st.session_state.predictions,
                        extra_augments,
                        feature_values,
                    )
                    ra_offset = float(pred[0][0])
                    dec_offset = float(pred[0][1])
                    st.session_state.last_prediction = pred
                    st.session_state.last_input = {
                        "mountRA": float(mountRA),
                        "mountDec": float(mountDec),
                        "timestamp": f"{int(year):04d}-{int(month):02d}-{int(day):02d} "
                                     f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}",
                        "features": feature_values,
                    }
                    st.session_state.prev_outputs.append((ra_offset, dec_offset))

                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("RA offset", _fmt(ra_offset, units))
                    a2.metric("Dec offset", _fmt(dec_offset, units))
                    a3.metric("Corrected RA", _fmt(float(mountRA) - ra_offset, units))
                    a4.metric("Corrected Dec", _fmt(float(mountDec) - dec_offset, units))

                    if extra_augments:
                        st.session_state.step = 'augment'
                        st.rerun()
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        # Step 2: Add Additional Info
        elif st.session_state.step == 'augment':
            with st.form("augment_form", clear_on_submit=False):
                st.markdown("#### Record the actual acquired position")
                if st.session_state.last_input:
                    st.caption(
                        f"Previous request: RA {st.session_state.last_input['mountRA']:.6g}, "
                        f"Dec {st.session_state.last_input['mountDec']:.6g} at "
                        f"{st.session_state.last_input['timestamp']}"
                    )
                if st.session_state.last_prediction is not None:
                    last_pred = st.session_state.last_prediction
                    ra_offset = float(last_pred[0][0])
                    dec_offset = float(last_pred[0][1])
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Predicted RA offset", _fmt(ra_offset, units))
                    a2.metric("Predicted Dec offset", _fmt(dec_offset, units))
                    if st.session_state.last_input:
                        a3.metric("Corrected RA", _fmt(st.session_state.last_input["mountRA"] - ra_offset, units))
                        a4.metric("Corrected Dec", _fmt(st.session_state.last_input["mountDec"] - dec_offset, units))
                a1, a2 = st.columns(2)
                with a1:
                    actualRA  = st.number_input("Actual RA", value=150.0, key="actual_ra")
                with a2:
                    actualDec = st.number_input("Actual Dec", value=30.0, key="actual_dec")
                add_info = st.form_submit_button("Add Observation")
            if add_info:
                last_input = st.session_state.last_input or {"mountRA": 0.0, "mountDec": 0.0}
                prev_acq_RA = float(last_input["mountRA"]) - float(actualRA)
                prev_acq_Dec = float(last_input["mountDec"]) - float(actualDec)
                st.session_state.predictions.append(
                    (prev_acq_RA, prev_acq_Dec, float(last_input["mountRA"]), float(last_input["mountDec"]))
                )
                st.session_state.step = 'predict'
                st.session_state.flash_message = "Observation added to autoregressive history."
                st.rerun()

    # -- History tab --
    with tabs[2]:
        st.header("Prediction History")
        if st.session_state.prev_outputs:
            output_df = pd.DataFrame(
                st.session_state.prev_outputs,
                columns=["Predicted RA Offset", "Predicted Dec Offset"],
            )
            output_df.index = output_df.index + 1
            st.dataframe(output_df, use_container_width=True)
        else:
            st.write("No predictions yet.")

        st.header("Autoregressive Observation History")
        if st.session_state.predictions:
            history_df = pd.DataFrame(
                st.session_state.predictions,
                columns=["Previous RA Error", "Previous Dec Error", "Previous Mount RA", "Previous Mount Dec"],
            )
            history_df.index = history_df.index + 1
            st.dataframe(history_df, use_container_width=True)
        else:
            st.write("No observations have been added to the autoregressive history.")
