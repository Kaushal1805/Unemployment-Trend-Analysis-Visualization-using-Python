import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ─── AUTO PATH FIX ─────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="India Unemployment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0A0E1A;
    color: #E2E8F0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem; max-width: 1400px; }

.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40%; right: -5%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #60A5FA, #34D399, #60A5FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.15;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem; color: #64748B;
    margin-top: 0.6rem; letter-spacing: 0.08em;
}
.tags { display: flex; gap: 0.6rem; margin-top: 1.4rem; flex-wrap: wrap; }
.tag  { padding: 0.3rem 0.9rem; border-radius: 20px; font-size: 0.72rem; font-family: 'Space Mono', monospace; border: 1px solid; }
.t-b { background:rgba(59,130,246,0.1);  border-color:rgba(59,130,246,0.4);  color:#60A5FA; }
.t-g { background:rgba(16,185,129,0.1);  border-color:rgba(16,185,129,0.4);  color:#34D399; }
.t-r { background:rgba(239,68,68,0.1);   border-color:rgba(239,68,68,0.4);   color:#F87171; }
.t-p { background:rgba(139,92,246,0.1);  border-color:rgba(139,92,246,0.4);  color:#A78BFA; }

.m-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
.m-card {
    background: #111827; border: 1px solid #1F2937;
    border-radius: 14px; padding: 1.4rem 1.6rem;
    position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.m-card:hover { transform: translateY(-3px); border-color: #374151; }
.m-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 14px 14px 0 0; }
.mc-b::before { background: linear-gradient(90deg, #2563EB, #60A5FA); }
.mc-g::before { background: linear-gradient(90deg, #059669, #34D399); }
.mc-r::before { background: linear-gradient(90deg, #DC2626, #F87171); }
.mc-p::before { background: linear-gradient(90deg, #7C3AED, #A78BFA); }
.m-icon  { font-size: 1.5rem; margin-bottom: 0.5rem; display: block; }
.m-label { font-family:'Space Mono',monospace; font-size:0.67rem; color:#6B7280; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; }
.m-val   { font-size: 2.1rem; font-weight: 800; color: #F1F5F9; line-height: 1; }
.m-sub   { font-family:'Space Mono',monospace; font-size:0.7rem; margin-top:0.4rem; opacity:0.75; }

.sec { display: flex; align-items: center; gap: 0.8rem; margin: 2.2rem 0 1.2rem 0; padding-bottom: 0.8rem; border-bottom: 1px solid #1F2937; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.d-b { background:#3B82F6; box-shadow:0 0 8px rgba(59,130,246,0.6); }
.d-g { background:#10B981; box-shadow:0 0 8px rgba(16,185,129,0.6); }
.d-r { background:#EF4444; box-shadow:0 0 8px rgba(239,68,68,0.6); }
.d-p { background:#8B5CF6; box-shadow:0 0 8px rgba(139,92,246,0.6); }
.sec-title { font-size:1.15rem; font-weight:700; color:#E2E8F0; margin:0; }

.chart { background: #111827; border: 1px solid #1F2937; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.2rem; }

[data-testid="stSidebar"] { background: #0D1117 !important; border-right: 1px solid #1F2937 !important; }
.sb-brand { font-size: 1.1rem; font-weight: 800; color: #60A5FA; padding-bottom: 1rem; border-bottom: 1px solid #1F2937; margin-bottom: 1.5rem; }
.sb-lbl { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #4B5563; text-transform: uppercase; letter-spacing: 0.1em; margin: 1.2rem 0 0.4rem 0; }
.sb-box { background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 1rem; font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #6B7280; line-height: 2.1; }

/* Prediction Card */
.pred-wrap { background: #111827; border: 1px solid #1F2937; border-radius: 16px; padding: 2rem; }
.pred-select-box {
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.pred-select-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
}
.res-box { text-align: center; padding: 2.5rem; border-radius: 14px; margin-top: 1.5rem; }
.res-h { background:rgba(239,68,68,0.1);  border:1px solid rgba(239,68,68,0.3); }
.res-m { background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); }
.res-n { background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); }
.res-rate  { font-size:4rem; font-weight:800; line-height:1; }
.res-label { font-family:'Space Mono',monospace; font-size:0.88rem; margin-top:0.6rem; opacity:0.85; }
.res-emoji { font-size:2rem; margin-bottom:0.5rem; }
.res-meta  { font-family:'Space Mono',monospace; font-size:0.7rem; color:#6B7280; margin-top:1rem; }

/* State Area Cards */
.state-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin-top: 1rem; }
.state-card {
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.state-name  { font-size: 0.85rem; font-weight: 700; color: #E2E8F0; margin-bottom: 0.3rem; }
.state-rate  { font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700; }
.state-area  { font-family:'Space Mono',monospace; font-size:0.65rem; color:#6B7280; margin-top:0.2rem; }

.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1.05rem !important;
    padding: 0.8rem 2rem !important; width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 30px rgba(37,99,235,0.4) !important; }

.ins-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem; }
.ins-card { background: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 1.2rem 1.4rem; transition: border-color 0.2s, transform 0.2s; }
.ins-card:hover { border-color: #374151; transform: translateY(-2px); }
.ins-num  { font-family:'Space Mono',monospace; font-size:0.68rem; color:#3B82F6; margin-bottom:0.5rem; }
.ins-text { font-size:0.88rem; color:#CBD5E1; line-height:1.55; }

.footer { text-align: center; padding: 2rem; font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #374151; border-top: 1px solid #1F2937; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# LOAD PICKLE MODEL
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("le_area.pkl", "rb") as f:
        le_area = pickle.load(f)
    with open("le_region.pkl", "rb") as f:
        le_region = pickle.load(f)
    return model, le_area, le_region

try:
    model, le_area, le_region = load_model()
except FileNotFoundError as e:
    st.error(f"❌ Pickle file nahi mili: {e}")
    st.stop()

# ── Model ke features automatically detect karo ──
try:
    model_features = model.feature_names_in_.tolist()
except AttributeError:
    # Agar feature names nahi hain toh manually set karo
    model_features = [
        'Month', 'Year', 'Quarter', 'Is_Covid',
        'Area_Encoded', 'Region_Encoded', 'Zone_Encoded',
        'Labour_Participation_Rate', 'Longitude', 'Latitude',
        'Prev_Month_Rate', 'Rolling_Avg_3', 'Labour_Change'
    ]


# ═══════════════════════════════════════════════════════════════
# LOAD & MERGE DONO CSV FILES
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    # ── File 1 ──
    df1 = pd.read_csv("Unemployment_Rate_upto_11_2020[2].csv")
    df1.columns = ['Region', 'Date', 'Frequency',
                   'Unemployment_Rate', 'Employed',
                   'Labour_Participation_Rate',
                   'Region_Zone', 'Longitude', 'Latitude']
    df1['Region'] = df1['Region'].str.strip()
    df1['Date']   = pd.to_datetime(
                    df1['Date'].str.strip(), dayfirst=True)
    df1 = df1.dropna().drop_duplicates()

    # ── File 2 ──
    df2 = pd.read_csv("Unemployment_in_India[1].csv")
    df2.columns = ['Region', 'Date', 'Frequency',
                   'Unemployment_Rate', 'Employed',
                   'Labour_Participation_Rate', 'Area']
    df2['Region'] = df2['Region'].str.strip()
    df2['Area']   = df2['Area'].str.strip()
    df2['Date']   = pd.to_datetime(
                    df2['Date'].str.strip(), dayfirst=True)
    df2 = df2.dropna().drop_duplicates()

    # ── Merge ──
    df = pd.merge(df2,
                  df1[['Region', 'Date', 'Longitude',
                       'Latitude', 'Region_Zone']],
                  on=['Region', 'Date'], how='left')

    # ── Fill NaN ──
    for col in ['Longitude', 'Latitude']:
        df[col] = df.groupby('Region')[col].transform(
                  lambda x: x.fillna(x.mean()))
    df['Region_Zone'] = df.groupby('Region')['Region_Zone'].transform(
                        lambda x: x.fillna(
                        x.mode()[0] if len(x.mode()) > 0 else 'Unknown'))

    df = df.dropna(subset=['Longitude', 'Latitude'])

    # ── Features ──
    df['Month']    = df['Date'].dt.month
    df['Year']     = df['Date'].dt.year
    df['Quarter']  = df['Date'].dt.quarter
    df['Is_Covid'] = (df['Date'] >= '2020-03-01').astype(int)
    df['Season']   = df['Month'].map({
        12:'Winter', 1:'Winter',  2:'Winter',
        3:'Summer',  4:'Summer',  5:'Summer',
        6:'Monsoon', 7:'Monsoon', 8:'Monsoon',
        9:'Autumn',  10:'Autumn', 11:'Autumn'
    })

    # ── Sort for lag features ──
    df = df.sort_values(['Region', 'Area', 'Date'])

    # ── Lag Features ──
    df['Prev_Month_Rate'] = df.groupby(
        ['Region', 'Area'])['Unemployment_Rate'].shift(1)
    df['Rolling_Avg_3']   = df.groupby(
        ['Region', 'Area'])['Unemployment_Rate'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    df['Labour_Change']   = df.groupby(
        ['Region', 'Area'])['Labour_Participation_Rate'].diff()

    df['Prev_Month_Rate'] = df['Prev_Month_Rate'].fillna(
                            df['Unemployment_Rate'])
    df['Labour_Change']   = df['Labour_Change'].fillna(0)

    # ── Encode ──
    df['Area_Encoded']   = le_area.transform(df['Area'])
    df['Region_Encoded'] = le_region.transform(df['Region'])

    # ── Zone Encode ──
    from sklearn.preprocessing import LabelEncoder
    le_zone = LabelEncoder()
    df['Zone_Encoded'] = le_zone.fit_transform(df['Region_Zone'])

    return df, le_zone

try:
    df, le_zone = load_data()
except FileNotFoundError as e:
    st.error(f"❌ CSV file nahi mili: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# PREDICTION FUNCTION — CORRECT FEATURES
# ═══════════════════════════════════════════════════════════════
def make_prediction(region, area, month, year, is_covid, labour):
    """
    State + Area ke basis pe prediction karo
    Sahi features automatically use karta hai
    """
    # Region data se values lo
    region_data = df[
        (df['Region'] == region) &
        (df['Area'] == area)
    ].sort_values('Date')

    # Agar data hai toh actual values use karo
    if len(region_data) > 0:
        last_row      = region_data.iloc[-1]
        prev_rate     = last_row['Unemployment_Rate']
        rolling_avg   = region_data['Unemployment_Rate'].tail(3).mean()
        labour_change = region_data['Labour_Change'].mean()
        longitude     = last_row['Longitude']
        latitude      = last_row['Latitude']
        zone_enc      = last_row['Zone_Encoded']
        region_enc    = last_row['Region_Encoded']
        area_enc      = last_row['Area_Encoded']
    else:
        prev_rate     = df['Unemployment_Rate'].mean()
        rolling_avg   = prev_rate
        labour_change = 0
        longitude     = df['Longitude'].mean()
        latitude      = df['Latitude'].mean()
        zone_enc      = 0
        region_enc    = le_region.transform([region])[0]
        area_enc      = le_area.transform([area])[0]

    # ── Input DataFrame banao with ALL correct features ──
    input_dict = {
        'Month':                     month,
        'Year':                      year,
        'Quarter':                   (month - 1) // 3 + 1,
        'Is_Covid':                  1 if is_covid == "Yes" else 0,
        'Area_Encoded':              area_enc,
        'Region_Encoded':            region_enc,
        'Zone_Encoded':              zone_enc,
        'Labour_Participation_Rate': labour,
        'Longitude':                 longitude,
        'Latitude':                  latitude,
        'Prev_Month_Rate':           prev_rate,
        'Rolling_Avg_3':             rolling_avg,
        'Labour_Change':             labour_change,
    }

    # ── Sirf model ke features use karo ──
    input_df = pd.DataFrame([{
        k: input_dict[k]
        for k in model_features
        if k in input_dict
    }])

    return model.predict(input_df)[0]


# ═══════════════════════════════════════════════════════════════
# MATPLOTLIB DARK
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":   "#111827",
    "axes.facecolor":     "#111827",
    "axes.labelcolor":    "#9CA3AF",
    "axes.titlecolor":    "#E2E8F0",
    "xtick.color":        "#6B7280",
    "ytick.color":        "#6B7280",
    "grid.color":         "#1F2937",
    "grid.linestyle":     "--",
    "grid.alpha":         0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "axes.titlesize":     12,
    "axes.labelsize":     10,
})


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-brand">📊 UnemploymentAI</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sb-lbl">Filter by Region</div>',
                unsafe_allow_html=True)
    regions = ["All"] + sorted(df["Region"].unique().tolist())
    sel_region = st.selectbox("", regions,
                              label_visibility="collapsed")

    st.markdown('<div class="sb-lbl">Filter by Area</div>',
                unsafe_allow_html=True)
    sel_area = st.radio("", ["All", "Urban", "Rural"],
                        label_visibility="collapsed")

    st.markdown('<div class="sb-lbl">Filter by Zone</div>',
                unsafe_allow_html=True)
    zones = ["All"] + sorted(df["Region_Zone"].unique().tolist())
    sel_zone = st.selectbox("Zone", zones,
                            label_visibility="collapsed")

    best  = df.groupby("Region")["Unemployment_Rate"].mean().idxmin()
    worst = df.groupby("Region")["Unemployment_Rate"].mean().idxmax()

    st.markdown(f"""
    <div class="sb-lbl">Dataset Info</div>
    <div class="sb-box">
        Records &nbsp;: {len(df)}<br>
        Regions &nbsp;: {df['Region'].nunique()}<br>
        Period &nbsp;&nbsp;: 2019 – 2020<br>
        Model &nbsp;&nbsp;&nbsp;: Gradient Boost<br>
        Accuracy : 84% ✅
    </div>
    <div class="sb-lbl">Regional Highlights</div>
    <div class="sb-box">
        🔴 Highest : {worst}<br>
        🟢 Lowest &nbsp;: {best}
    </div>
    """, unsafe_allow_html=True)

# ─── Filters Apply ─────────────────────────────────────────────
df_f = df.copy()
if sel_region != "All":
    df_f = df_f[df_f["Region"] == sel_region]
if sel_area != "All":
    df_f = df_f[df_f["Area"] == sel_area]
if sel_zone != "All":
    df_f = df_f[df_f["Region_Zone"] == sel_zone]


# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1 class="hero-title">India Unemployment<br>Analysis Dashboard</h1>
    <p class="hero-sub">
        GRADIENT BOOSTING MODEL &nbsp;·&nbsp;
        84% ACCURACY &nbsp;·&nbsp; COVID-19 IMPACT STUDY
    </p>
    <div class="tags">
        <span class="tag t-b">📅 2019 – 2020</span>
        <span class="tag t-g">✅ 84% Accuracy</span>
        <span class="tag t-r">🦠 Covid-19 Impact</span>
        <span class="tag t-p">🗺️ 28 Indian States</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# KEY METRICS
# ═══════════════════════════════════════════════════════════════
before = df[df["Is_Covid"] == 0]["Unemployment_Rate"].mean()
after  = df[df["Is_Covid"] == 1]["Unemployment_Rate"].mean()
avg    = df_f["Unemployment_Rate"].mean()
urban  = df[df["Area"] == "Urban"]["Unemployment_Rate"].mean()
rural  = df[df["Area"] == "Rural"]["Unemployment_Rate"].mean()

st.markdown(f"""
<div class="m-grid">
    <div class="m-card mc-b">
        <span class="m-icon">📊</span>
        <div class="m-label">Average Rate</div>
        <div class="m-val">{avg:.1f}%</div>
        <div class="m-sub" style="color:#60A5FA;">{len(df_f)} records</div>
    </div>
    <div class="m-card mc-g">
        <span class="m-icon">🕊️</span>
        <div class="m-label">Before Covid</div>
        <div class="m-val">{before:.1f}%</div>
        <div class="m-sub" style="color:#34D399;">Pre March 2020</div>
    </div>
    <div class="m-card mc-r">
        <span class="m-icon">🦠</span>
        <div class="m-label">During Covid</div>
        <div class="m-val">{after:.1f}%</div>
        <div class="m-sub" style="color:#F87171;">↑ +{after-before:.1f}% spike</div>
    </div>
    <div class="m-card mc-p">
        <span class="m-icon">🏙️</span>
        <div class="m-label">Urban Rate</div>
        <div class="m-val">{urban:.1f}%</div>
        <div class="m-sub" style="color:#A78BFA;">Rural: {rural:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# GRAPH 1 — TREND
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec"><div class="dot d-b"></div>
<h2 class="sec-title">Unemployment Trend Over Time</h2></div>
""", unsafe_allow_html=True)

monthly = df_f.groupby("Date")["Unemployment_Rate"].mean()
fig1, ax1 = plt.subplots(figsize=(14, 4.5))
ax1.plot(monthly.index, monthly.values,
         color="#3B82F6", linewidth=2.5, zorder=3)
ax1.fill_between(monthly.index, monthly.values,
                 alpha=0.12, color="#3B82F6")
covid_m = monthly.index >= pd.Timestamp("2020-03-01")
if covid_m.any():
    ax1.fill_between(monthly.index, monthly.values,
                     where=covid_m, alpha=0.22, color="#EF4444")
ax1.axvline(x=pd.Timestamp("2020-03-01"),
            color="#EF4444", linestyle="--",
            linewidth=1.8, label="Covid-19 Starts (Mar 2020)")
ax1.axhline(y=monthly.mean(), color="#4B5563",
            linestyle=":", linewidth=1.2,
            label=f"Mean: {monthly.mean():.1f}%")
ax1.set_xlabel("Date", labelpad=10)
ax1.set_ylabel("Unemployment Rate (%)", labelpad=10)
ax1.legend(fontsize=9, facecolor="#1F2937",
           edgecolor="#374151", labelcolor="#9CA3AF")
ax1.yaxis.grid(True)
st.markdown('<div class="chart">', unsafe_allow_html=True)
st.pyplot(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
plt.close()


# ═══════════════════════════════════════════════════════════════
# GRAPH 2 & 3
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec"><div class="dot d-g"></div>
<h2 class="sec-title">Urban vs Rural & Covid-19 Impact</h2></div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")
with c1:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    area_avg = df.groupby("Area")["Unemployment_Rate"].mean()
    b2 = ax2.bar(area_avg.index, area_avg.values,
                 color=["#F87171", "#60A5FA"],
                 edgecolor="#1F2937", width=0.5)
    for bar, val in zip(b2, area_avg.values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + 0.3, f"{val:.1f}%",
                 ha="center", fontsize=13,
                 fontweight="bold", color="#E2E8F0")
    ax2.set_title("Urban vs Rural Average Rate", pad=12)
    ax2.set_ylabel("Rate (%)")
    ax2.yaxis.grid(True)
    st.markdown('<div class="chart">', unsafe_allow_html=True)
    st.pyplot(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

with c2:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    b3 = ax3.bar(["Before Covid", "During Covid"],
                 [before, after],
                 color=["#34D399", "#F87171"],
                 edgecolor="#1F2937", width=0.5)
    for bar, val in zip(b3, [before, after]):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 val + 0.3, f"{val:.1f}%",
                 ha="center", fontsize=13,
                 fontweight="bold", color="#E2E8F0")
    ax3.set_title("Covid-19 Impact Comparison", pad=12)
    ax3.set_ylabel("Rate (%)")
    ax3.yaxis.grid(True)
    st.markdown('<div class="chart">', unsafe_allow_html=True)
    st.pyplot(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# GRAPH 4 — TOP 10 STATES
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec"><div class="dot d-r"></div>
<h2 class="sec-title">Top 10 States by Unemployment Rate</h2></div>
""", unsafe_allow_html=True)

reg_avg = (df.groupby("Region")["Unemployment_Rate"]
           .mean().sort_values(ascending=False).head(10))
fig4, ax4 = plt.subplots(figsize=(14, 4.5))
clrs = ["#EF4444" if i == 0 else
        "#F97316" if i <= 2 else
        "#60A5FA" for i in range(len(reg_avg))]
b4 = ax4.bar(reg_avg.index, reg_avg.values,
             color=clrs, edgecolor="#1F2937", width=0.6)
for bar, val in zip(b4, reg_avg.values):
    ax4.text(bar.get_x() + bar.get_width()/2,
             val + 0.2, f"{val:.1f}%",
             ha="center", fontsize=9,
             color="#E2E8F0", fontweight="bold")
ax4.set_ylabel("Average Rate (%)")
plt.xticks(rotation=35, ha="right", fontsize=9)
ax4.yaxis.grid(True)
st.markdown('<div class="chart">', unsafe_allow_html=True)
st.pyplot(fig4, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
plt.close()


# ═══════════════════════════════════════════════════════════════
# GRAPH 5 — SEASONAL
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec"><div class="dot d-p"></div>
<h2 class="sec-title">Seasonal Unemployment Pattern</h2></div>
""", unsafe_allow_html=True)

sea = (df.groupby("Season")["Unemployment_Rate"]
       .mean().reindex(["Winter","Summer","Monsoon","Autumn"]))
fig5, ax5 = plt.subplots(figsize=(14, 4))
b5 = ax5.bar(sea.index, sea.values,
             color=["#60A5FA","#FBBF24","#34D399","#F87171"],
             edgecolor="#1F2937", width=0.5)
for bar, val in zip(b5, sea.values):
    ax5.text(bar.get_x() + bar.get_width()/2,
             val + 0.2, f"{val:.1f}%",
             ha="center", fontsize=14,
             fontweight="bold", color="#E2E8F0")
ax5.set_ylabel("Average Rate (%)")
ax5.yaxis.grid(True)
st.markdown('<div class="chart">', unsafe_allow_html=True)
st.pyplot(fig5, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
plt.close()


# ═══════════════════════════════════════════════════════════════
# STATE + AREA WISE ACTUAL DATA TABLE
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec"><div class="dot d-g"></div>
<h2 class="sec-title">📍 State + Area Wise Unemployment</h2></div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")
with c1:
    view_region = st.selectbox(
        "Select State",
        sorted(df["Region"].unique().tolist()),
        key="view_region"
    )
with c2:
    view_area = st.radio(
        "Select Area",
        ["Both", "Urban", "Rural"],
        horizontal=True,
        key="view_area"
    )

# Filter state data
state_df = df[df["Region"] == view_region].copy()
if view_area != "Both":
    state_df = state_df[state_df["Area"] == view_area]

# Show trend graph for selected state
if len(state_df) > 0:
    fig_s, ax_s = plt.subplots(figsize=(14, 4))

    for area_type, color in zip(["Urban", "Rural"],
                                 ["#60A5FA", "#34D399"]):
        area_data = state_df[state_df["Area"] == area_type]
        if len(area_data) > 0:
            area_monthly = area_data.groupby("Date")[
                "Unemployment_Rate"].mean()
            ax_s.plot(area_monthly.index, area_monthly.values,
                      color=color, linewidth=2.5,
                      marker='o', markersize=4,
                      label=f"{area_type}")
            ax_s.fill_between(area_monthly.index,
                              area_monthly.values,
                              alpha=0.08, color=color)

    ax_s.axvline(x=pd.Timestamp("2020-03-01"),
                 color="#EF4444", linestyle="--",
                 linewidth=1.5, label="Covid-19")
    ax_s.set_title(f"{view_region} — Unemployment Trend",
                   pad=12)
    ax_s.set_xlabel("Date")
    ax_s.set_ylabel("Rate (%)")
    ax_s.legend(fontsize=9, facecolor="#1F2937",
                edgecolor="#374151", labelcolor="#9CA3AF")
    ax_s.yaxis.grid(True)

    st.markdown('<div class="chart">', unsafe_allow_html=True)
    st.pyplot(fig_s, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close()

    # Show stats
    col1, col2, col3 = st.columns(3)
    state_avg    = state_df["Unemployment_Rate"].mean()
    state_before = state_df[state_df["Is_Covid"]==0]["Unemployment_Rate"].mean()
    state_after  = state_df[state_df["Is_Covid"]==1]["Unemployment_Rate"].mean()

    col1.metric("State Average", f"{state_avg:.2f}%")
    col2.metric("Before Covid",  f"{state_before:.2f}%")
    col3.metric("During Covid",  f"{state_after:.2f}%",
                f"+{state_after - state_before:.2f}%")


# ═══════════════════════════════════════════════════════════════
# ML PREDICTION — STATE + AREA WISE
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec"><div class="dot d-b"></div>
<h2 class="sec-title">🤖 ML Prediction — State + Area Wise</h2></div>
""", unsafe_allow_html=True)

st.markdown('<div class="pred-wrap">', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown("**🗺️ Select State & Area**")
    pred_region = st.selectbox(
        "State",
        sorted(df["Region"].unique().tolist()),
        key="pred_region"
    )
    pred_area = st.radio(
        "Area",
        ["Urban", "Rural"],
        key="pred_area"
    )

with c2:
    st.markdown("**📅 Time Period**")
    pred_month    = st.selectbox("Month", range(1, 13),
                                  key="pred_month")
    pred_year     = st.selectbox("Year", [2019, 2020, 2021],
                                  key="pred_year")
    pred_is_covid = st.radio("Covid Period?",
                              ["No", "Yes"],
                              key="pred_covid")

with c3:
    st.markdown("**📈 Labour Indicator**")
    # Auto-fill from actual data
    region_labour = df[
        (df["Region"] == pred_region) &
        (df["Area"] == pred_area)
    ]["Labour_Participation_Rate"].mean()

    if pd.isna(region_labour):
        region_labour = 43.0

    pred_labour = st.slider(
        "Labour Participation Rate (%)",
        30.0, 60.0,
        float(round(region_labour, 1)),
        step=0.5,
        key="pred_labour"
    )

    # Show current actual rate
    actual_rate = df[
        (df["Region"] == pred_region) &
        (df["Area"] == pred_area)
    ]["Unemployment_Rate"].mean()

    if not pd.isna(actual_rate):
        st.markdown(f"""
        <div style="background:#1F2937;border:1px solid #374151;
                    border-radius:8px;padding:0.8rem 1rem;margin-top:0.5rem;">
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                        color:#6B7280;margin-bottom:0.3rem;">ACTUAL AVG RATE</div>
            <div style="font-size:1.4rem;font-weight:800;color:#FBBF24;">
                {actual_rate:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Unemployment Rate"):
    try:
        with st.spinner("Predicting..."):
            pred = make_prediction(
                pred_region, pred_area,
                pred_month, pred_year,
                pred_is_covid, pred_labour
            )

        if pred > 15:
            cls, color = "res-h", "#F87171"
            emoji, status = "⚠️", "HIGH ALERT"
            advice = "Immediate government intervention needed!"
        elif pred > 8:
            cls, color = "res-m", "#FBBF24"
            emoji, status = "🔶", "MODERATE"
            advice = "Monitor closely — policy action recommended."
        else:
            cls, color = "res-n", "#34D399"
            emoji, status = "✅", "NORMAL"
            advice = "Unemployment within acceptable range."

        st.markdown(f"""
        <div class="res-box {cls}">
            <div class="res-emoji">{emoji}</div>
            <div class="res-rate" style="color:{color};">{pred:.2f}%</div>
            <div class="res-label" style="color:{color};">
                {status} — {advice}
            </div>
            <div class="res-meta">
                {pred_region} &nbsp;·&nbsp;
                {pred_area} area &nbsp;·&nbsp;
                {'Covid Period' if pred_is_covid=='Yes' else 'Normal Period'}
                &nbsp;·&nbsp; Labour: {pred_labour}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# POLICY INSIGHTS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec"><div class="dot d-b"></div>
<h2 class="sec-title">💡 Policy Recommendations</h2></div>
<div class="ins-grid">
    <div class="ins-card">
        <div class="ins-num">01 / REGIONAL</div>
        <div class="ins-text">Focus job creation programs in high unemployment states like Tripura and Haryana on priority basis.</div>
    </div>
    <div class="ins-card">
        <div class="ins-num">02 / URBAN</div>
        <div class="ins-text">Urban areas need stronger post-Covid industry support and targeted skill development programs.</div>
    </div>
    <div class="ins-card">
        <div class="ins-num">03 / RURAL</div>
        <div class="ins-text">Rural areas need seasonal employment schemes especially during Monsoon and Autumn periods.</div>
    </div>
    <div class="ins-card">
        <div class="ins-num">04 / COVID</div>
        <div class="ins-text">Covid caused +3.69% spike. Emergency employment funds must be pre-planned for future crises.</div>
    </div>
    <div class="ins-card">
        <div class="ins-num">05 / SKILLS</div>
        <div class="ins-text">Vocational training and skill development urgently needed in states with consistently high rates.</div>
    </div>
    <div class="ins-card">
        <div class="ins-num">06 / MONITOR</div>
        <div class="ins-text">Real-time monitoring systems needed so government can detect and respond to spikes faster.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    INDIA UNEMPLOYMENT ANALYSIS &nbsp;·&nbsp;
    GRADIENT BOOSTING 84% &nbsp;·&nbsp;
    DATA: 2019 – 2020
</div>
""", unsafe_allow_html=True)
