"""
frontend.py
-----------
Streamlit frontend for the Bengaluru House Price Prediction app.
Communicates with the Flask backend (app.py) via REST API calls.

Run:
  streamlit run frontend.py
"""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Config ──────────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:5000/api"

st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 700;
            color: #1f2328;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #57606a;
            margin-bottom: 1.5rem;
        }
        .price-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1.8rem 2rem;
            text-align: center;
            color: white;
        }
        .price-box h1 {
            font-size: 3rem;
            margin: 0;
            font-weight: 800;
        }
        .price-box p {
            font-size: 1rem;
            margin: 0.4rem 0 0;
            opacity: 0.85;
        }
        .metric-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            text-align: center;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #1f2328;
            margin-bottom: 0.6rem;
        }
        .stButton > button {
            width: 100%;
            background-color: #3b82d4;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.65rem 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #2563ba;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Helper: fetch locations from API ────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_locations():
    try:
        resp = requests.get(f"{API_BASE}/locations", timeout=5)
        resp.raise_for_status()
        return resp.json().get("locations", [])
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return []


def call_predict(location, total_sqft, bhk, bath):
    payload = {
        "location":   location,
        "total_sqft": total_sqft,
        "bhk":        bhk,
        "bath":       bath,
    }
    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
    return resp.json(), resp.status_code


# ─── Sidebar: inputs ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Details")
    st.markdown("---")

    locations = fetch_locations()

    if locations is None:
        st.error(
            "⚠️ Cannot connect to backend.\n\n"
            "Please start the Flask server:\n```\npython app.py\n```"
        )
        st.stop()

    selected_location = st.selectbox(
        "📍 Location",
        options=locations,
        help="Select the locality in Bengaluru",
    )

    total_sqft = st.number_input(
        "📐 Total Area (sq ft)",
        min_value=300,
        max_value=15000,
        value=1200,
        step=50,
        help="Total carpet / built-up area in square feet",
    )

    col1, col2 = st.columns(2)
    with col1:
        bhk = st.selectbox("🛏 BHK", options=[1, 2, 3, 4, 5, 6], index=1)
    with col2:
        bath = st.selectbox("🚿 Bathrooms", options=[1, 2, 3, 4, 5, 6], index=1)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Price")


# ─── Main content ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🏠 Bengaluru House Price Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Powered by Machine Learning · Bengaluru Real Estate Dataset</div>',
    unsafe_allow_html=True,
)

# Tabs
tab_predict, tab_explore, tab_about = st.tabs(
    ["🔮 Prediction", "📊 Data Insights", "ℹ️ About"]
)

# ─── Tab 1: Prediction ────────────────────────────────────────────────────────
with tab_predict:
    if predict_btn:
        with st.spinner("Calculating price ..."):
            try:
                result, status = call_predict(selected_location, total_sqft, bhk, bath)
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the backend. Is `python app.py` running?")
                st.stop()

        if status == 200:
            price = result["predicted_price_lakhs"]

            st.markdown(
                f"""
                <div class="price-box">
                    <h1>₹ {price:.2f} Lakhs</h1>
                    <p>Estimated market price for your property in <strong>{selected_location}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            price_per_sqft = round((price * 1e5) / total_sqft, 2)
            m1.metric("📍 Location", selected_location)
            m2.metric("📐 Area", f"{total_sqft} sq ft")
            m3.metric("🛏 BHK / 🚿 Bath", f"{bhk} / {bath}")
            m4.metric("💰 Price / sq ft", f"₹{price_per_sqft:,.0f}")

            # Price band gauge chart
            st.markdown("### Price Range Context")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=price,
                    title={"text": "Predicted Price (Lakhs ₹)", "font": {"size": 16}},
                    delta={"reference": 80, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
                    gauge={
                        "axis": {"range": [0, 500], "tickwidth": 1},
                        "bar":  {"color": "#3b82d4"},
                        "steps": [
                            {"range": [0,   80],  "color": "#dcfce7"},
                            {"range": [80,  200], "color": "#fef9c3"},
                            {"range": [200, 500], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "#ef4444", "width": 3},
                            "thickness": 0.75,
                            "value": price,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")

    else:
        st.info("👈 Fill in the house details in the sidebar and click **Predict Price**.")

        # Show example cards
        st.markdown("### Sample Price Estimates")
        examples = [
            {"location": "Whitefield",    "sqft": 1200, "bhk": 2, "bath": 2},
            {"location": "Hebbal",        "sqft": 1800, "bhk": 3, "bath": 3},
            {"location": "Marathahalli",  "sqft": 1000, "bhk": 2, "bath": 1},
            {"location": "Rajaji Nagar",  "sqft": 2500, "bhk": 4, "bath": 4},
        ]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            with col:
                try:
                    res, s = call_predict(ex["location"], ex["sqft"], ex["bhk"], ex["bath"])
                    ep = res.get("predicted_price_lakhs", "N/A")
                    col.metric(
                        label=f"🏡 {ex['location']}",
                        value=f"₹{ep} L" if isinstance(ep, float) else ep,
                        delta=f"{ex['bhk']} BHK · {ex['sqft']} sqft",
                    )
                except Exception:
                    col.write(f"**{ex['location']}** – N/A")


# ─── Tab 2: Data Insights ─────────────────────────────────────────────────────
with tab_explore:
    st.markdown("### 📊 Dataset Insights")

    try:
        import os
        DATA_PATH = os.path.join(os.path.dirname(__file__), "Bengaluru_House_Data.csv")
        df_raw = pd.read_csv(DATA_PATH)

        # ── KPIs ─────────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Records",     f"{len(df_raw):,}")
        k2.metric("Unique Locations",  f"{df_raw['location'].nunique():,}")
        k3.metric("Avg Price (L)",     f"₹{df_raw['price'].mean():.1f}")
        k4.metric("Max Price (L)",     f"₹{df_raw['price'].max():.0f}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        # ── Price distribution ────────────────────────────────────────────────
        with c1:
            st.markdown("#### Price Distribution (< 300 Lakhs)")
            filtered = df_raw[df_raw["price"] < 300]
            fig_hist = px.histogram(
                filtered, x="price", nbins=60,
                labels={"price": "Price (Lakhs)"},
                color_discrete_sequence=["#3b82d4"],
            )
            fig_hist.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── BHK breakdown ─────────────────────────────────────────────────────
        with c2:
            st.markdown("#### Property Type by BHK")
            df_raw["bhk"] = df_raw["size"].apply(
                lambda x: str(x).split()[0] if pd.notnull(x) else "Unknown"
            )
            bhk_counts = df_raw["bhk"].value_counts().reset_index()
            bhk_counts.columns = ["BHK", "Count"]
            fig_pie = px.pie(
                bhk_counts, values="Count", names="BHK",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Top 15 locations by average price ────────────────────────────────
        st.markdown("#### Top 15 Locations by Average Price")
        top_locs = (
            df_raw.groupby("location")["price"]
            .mean()
            .nlargest(15)
            .reset_index()
        )
        top_locs.columns = ["Location", "Avg Price (Lakhs)"]
        fig_bar = px.bar(
            top_locs,
            x="Avg Price (Lakhs)",
            y="Location",
            orientation="h",
            color="Avg Price (Lakhs)",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=420,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Sqft vs Price scatter ─────────────────────────────────────────────
        st.markdown("#### Area vs Price (sample 2 000 rows)")
        sample = df_raw.dropna(subset=["total_sqft", "price"]).copy()
        sample["total_sqft"] = pd.to_numeric(sample["total_sqft"], errors="coerce")
        sample = sample.dropna(subset=["total_sqft"])
        sample = sample[(sample["total_sqft"] < 6000) & (sample["price"] < 400)].sample(
            min(2000, len(sample)), random_state=42
        )
        fig_scatter = px.scatter(
            sample, x="total_sqft", y="price",
            color="size",
            labels={"total_sqft": "Total Sqft", "price": "Price (Lakhs)", "size": "Size"},
            opacity=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_scatter.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=380,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    except FileNotFoundError:
        st.warning("Dataset file not found next to frontend.py. Place `Bengaluru_House_Data.csv` in the same folder.")


# ─── Tab 3: About ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        ## About This App

        This application predicts residential property prices in **Bengaluru, India**
        using a machine-learning model trained on real estate listing data.

        ### How it works
        | Step | Detail |
        |------|--------|
        | 1. Data Cleaning | Handles missing values, range-format sqft, rare locations |
        | 2. Outlier Removal | Per-location price-per-sqft ± 1 std, BHK anomaly filter |
        | 3. Feature Engineering | One-hot encodes 200+ Bengaluru localities |
        | 4. Model Selection | Cross-validates Linear Regression, Lasso, Random Forest |
        | 5. Serving | Flask REST API → Streamlit frontend |

        ### Tech Stack
        | Layer | Technology |
        |-------|-----------|
        | Data  | Pandas, NumPy |
        | Model | Scikit-learn (Random Forest / Linear Regression) |
        | API   | Flask + Flask-CORS |
        | UI    | Streamlit + Plotly |

        ### Dataset
        - **Source:** Bengaluru House Data (Kaggle)
        - **Size:** ~13 000 property listings
        - **Features:** location, total_sqft, BHK, bathrooms, price (Lakhs ₹)

        ### Disclaimer
        Predictions are estimates based on historical listing data and should not
        be used as the sole basis for financial decisions.
        """
    )
