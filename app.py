import io, json, numpy as np, pandas as pd, streamlit as st, altair as alt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import savgol_filter, find_peaks, peak_widths
    from scipy.io import loadmat
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

from batterylab_recipe_engine import ElectrodeSpec, CellDesignInput, design_pouch

# ============ PAGE CONFIG ============ #
st.set_page_config(page_title="BatteryLAB Prototype", layout="wide")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🔋 BatteryLAB — Prototype")
st.caption(
    "Unified AI-driven platform for electrode design, performance prediction, and data analytics."
)
st.markdown("---")

# ============ HELPER FUNCTIONS ============ #
def _standardize_columns(df: pd.DataFrame):
    cols_l = [c.lower() for c in df.columns]
    vcol = next((df.columns[i] for i, c in enumerate(cols_l) if c in ["v", "volt", "voltage"]), None)
    qcol = next((df.columns[i] for i, c in enumerate(cols_l) if "capacity" in c or c in ["q", "ah", "mah"]), None)
    cyc = next((df.columns[i] for i, c in enumerate(cols_l) if c in ["cycle", "label", "group"]), None)
    return vcol, qcol, cyc

def _prep_series(voltage, capacity):
    V = pd.to_numeric(voltage, errors="coerce").to_numpy()
    Qraw = pd.to_numeric(capacity, errors="coerce").to_numpy()
    Q = Qraw / 1000.0 if np.nanmax(Qraw) > 100 else Qraw
    mask = np.isfinite(V) & np.isfinite(Q)
    V, Q = V[mask], Q[mask]
    order = np.argsort(V)
    return V[order], Q[order]

def _extract_features(V, Q):
    if SCIPY_OK and len(Q) >= 11:
        try: Qs = savgol_filter(Q, 11, 3)
        except Exception: Qs = Q
    else: Qs = Q
    dQdV = np.gradient(Qs, V, edge_order=2)
    peak_info = {"n_peaks":0,"voltages":[],"widths_V":[]}
    if SCIPY_OK and np.isfinite(dQdV).any():
        try:
            prom = np.nanmax(np.abs(dQdV))*0.05 if np.nanmax(np.abs(dQdV))>0 else 0.0
            peaks,_=find_peaks(dQdV,prominence=prom)
            peak_info["n_peaks"]=int(len(peaks))
            peak_info["voltages"]=[float(V[p]) for p in peaks]
            if len(peaks)>0:
                widths,_,_,_=peak_widths(dQdV,peaks,rel_height=0.5)
                if len(V)>1:
                    dv=np.mean(np.diff(V))
                    peak_info["widths_V"]=[float(w*dv) for w in widths]
        except Exception: pass
    try:
        dVdQ=np.gradient(V,Qs,edge_order=2)
        dVdQ_med=float(np.nanmedian(np.abs(dVdQ)))
    except Exception: dVdQ_med=float("nan")
    return dQdV, peak_info, dVdQ_med

def _compare_two_sets(name_a, feat_a, name_b, feat_b):
    interp=[]
    cap_a=feat_a.get("cap_range_Ah",[np.nan,np.nan])[1]
    cap_b=feat_b.get("cap_range_Ah",[np.nan,np.nan])[1]
    if np.isfinite(cap_a) and np.isfinite(cap_b) and cap_a>0:
        fade_pct=100.0*(cap_a-cap_b)/cap_a
        if abs(fade_pct)>=3:
            interp.append(f"Capacity change {fade_pct:.1f}% ({name_a}→{name_b}).")
    Va,Vb=feat_a.get("ica_peak_voltages_V",[]),feat_b.get("ica_peak_voltages_V",[])
    if Va and Vb:
        n=min(len(Va),len(Vb))
        if n>=1:
            shift=1000*np.nanmean(np.array(Vb[:n])-np.array(Va[:n]))
            if abs(shift)>=5:
                direction="↑" if shift>0 else "↓"
                interp.append(f"ICA peak shift {direction}{abs(shift):.0f} mV.")
    Wa,Wb=feat_a.get("ica_peak_widths_V",[]),feat_b.get("ica_peak_widths_V",[])
    if Wa and Wb:
        n=min(len(Wa),len(Wb))
        if n>=1:
            broad=1000*np.nanmean(np.array(Wb[:n])-np.array(Wa[:n]))
            if broad>2: interp.append(f"ICA peak broadening ~{broad:.0f} mV.")
    return interp or [f"No strong difference between {name_a} and {name_b}."]

def _plot_to_bytes(df_all,ycol,title):
    fig,ax=plt.subplots(figsize=(6,3.8),dpi=150)
    for c in df_all["Cycle"].unique():
        sub=df_all[df_all["Cycle"]==c]
        ax.plot(sub["Voltage"],sub[ycol],label=str(c))
    ax.set_xlabel("Voltage (V)"); ax.set_ylabel(ycol); ax.set_title(title)
    ax.legend(fontsize=8); fig.tight_layout()
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150); plt.close(fig); buf.seek(0)
    return buf.getvalue()

# ---------- PDF builders (recipe + analytics) ----------
def generate_pdf_report(features_by_group, richness_notes, suggestions, interps, vc_all, ica_all):
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=24,leftMargin=24,topMargin=24,bottomMargin=24)
    s=getSampleStyleSheet(); el=[]
    el.append(Paragraph("BatteryLAB Analytics Report",s["Title"])); el.append(Spacer(1,8))
    el.append(Paragraph("Dataset Quality & Richness",s["Heading2"]))
    for r in richness_notes: el.append(Paragraph("- "+r,s["Normal"]))
    el.append(Spacer(1,6)); el.append(Paragraph("Next-Step Suggestions",s["Heading2"]))
    for r in suggestions: el.append(Paragraph("- "+r,s["Normal"]))
    el.append(Spacer(1,6)); el.append(Paragraph("Key Features by Curve",s["Heading2"]))
    tdata=[["Curve","n","V range","Cap range","ICA peaks","V peaks","Widths","|dV/dQ| med"]]
    for g,f in features_by_group.items():
        tdata.append([
            g,f['n_samples'],
            f"{f['voltage_range_V'][0]:.2f}–{f['voltage_range_V'][1]:.2f}",
            f"{f['cap_range_Ah'][0]:.2f}–{f['cap_range_Ah'][1]:.2f}",
            f['ica_peaks_count'],
            ", ".join([f"{v:.3f}" for v in f['ica_peak_voltages_V']]) or "—",
            ", ".join([f"{w:.3f}" for w in f['ica_peak_widths_V']]) or "—",
            f"{f['dVdQ_median_abs']:.4f}" if np.isfinite(f['dVdQ_median_abs']) else "—"])
    tbl=Table(tdata,repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.25,colors.black),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9)]))
    el.append(tbl); el.append(Spacer(1,6))
    el.append(Paragraph("AI Interpretations",s["Heading2"]))
    for i in interps: el.append(Paragraph("- "+i,s["Normal"]))
    el.append(Spacer(1,6))
    if vc_all is not None and len(vc_all)>0:
        el.append(Paragraph("Voltage vs Capacity",s["Heading3"]))
        el.append(Image(io.BytesIO(_plot_to_bytes(vc_all,"Capacity_Ah","V-Q")),width=420,height=270))
    if ica_all is not None and len(ica_all)>0:
        el.append(Paragraph("dQ/dV vs Voltage",s["Heading3"]))
        el.append(Image(io.BytesIO(_plot_to_bytes(ica_all,"dQdV","ICA")),width=420,height=270))
    doc.build(el); buffer.seek(0); return buffer.getvalue()
# ========================= TAB STRUCTURE ========================= #
tab1, tab2 = st.tabs(["⚙️  Recipe → Performance", "📊  Data Analytics"])

# ------------------------------------------------------------
#  TAB 1: Recipe → Performance
# ------------------------------------------------------------
with tab1:
    st.subheader("Electrode & Cell Design")
    c1, c2, c3 = st.columns(3)

    with c1:
        cathode = st.text_input("Cathode material", "LFP")
        anode = st.text_input("Anode material", "Graphite")
    with c2:
        separator = st.text_input("Separator", "Celgard 2400")
        electrolyte = st.text_input("Electrolyte", "1 M LiPF₆ in EC:DEC")
    with c3:
        ambient_temp = st.number_input("Ambient Temp (°C)", 0, 80, 25)
        form_factor = st.selectbox("Cell Form Factor", ["Pouch", "Cylindrical", "Coin"])
        num_layers = st.slider("No. of Electrode Layers", 1, 30, 8)

    st.markdown("#### Results & AI Suggestions")

    spec = CellDesignInput(
        cathode=cathode,
        anode=anode,
        separator=separator,
        electrolyte=electrolyte,
        layers=num_layers,
        form=form_factor,
        ambient=ambient_temp,
    )
    results, ai_suggestion = design_pouch(spec)

    colA, colB = st.columns(2)
    with colA:
        st.metric("Predicted Nominal Voltage (V)", f"{results['voltage']:.2f}")
        st.metric("Estimated Specific Energy (Wh/kg)", f"{results['energy_density']:.1f}")
        st.metric("Expected Cycle Life @ 25 °C", f"{results['life_cycles']:.0f}")
    with colB:
        st.metric("Active Mass Ratio (%)", f"{results['mass_ratio']:.1f}")
        st.metric("Internal Resistance (mΩ)", f"{results['ir']:.2f}")
        st.metric("Thermal Rise @ C-rate (°C)", f"{results['thermal_rise']:.1f}")

    st.info(ai_suggestion)

    # ---- Temperature Advisory ----
    st.markdown("### 🌡️ Temperature Advisory")
    if ambient_temp >= 45:
        st.warning(
            f"High ambient T = {ambient_temp} °C may reduce cycle life by ≈ {(ambient_temp-25)*1.5:.0f}% and increase IR. "
            "Consider active cooling or electrolyte with additives for high-T stability."
        )
    elif ambient_temp <= 10:
        st.info(
            "Low ambient temperature detected — ionic mobility and rate capability may drop. "
            "Consider pre-heating or low-viscosity solvent mixes."
        )
    else:
        st.success("Temperature range is optimal for most LFP and NMC systems.")

    # ---- PDF Download ----
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("BatteryLAB Design Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Cathode: {cathode}, Anode: {anode}", styles["Normal"]),
        Paragraph(f"Separator: {separator}, Electrolyte: {electrolyte}", styles["Normal"]),
        Paragraph(f"Ambient Temp: {ambient_temp} °C", styles["Normal"]),
        Spacer(1, 6),
        Paragraph(f"Nominal Voltage: {results['voltage']:.2f} V", styles["Normal"]),
        Paragraph(f"Energy Density: {results['energy_density']:.1f} Wh/kg", styles["Normal"]),
        Paragraph(f"Cycle Life (25 °C): {results['life_cycles']:.0f}", styles["Normal"]),
        Spacer(1, 6),
        Paragraph("AI Suggestions:", styles["Heading2"]),
        Paragraph(ai_suggestion, styles["Normal"]),
    ]
    doc.build(story)
    st.download_button("📄 Download Design Report (PDF)", pdf_buffer.getvalue(),
                       file_name="BatteryLAB_Design_Report.pdf",
                       mime="application/pdf")

# ------------------------------------------------------------
#  TAB 2: Data Analytics
# ------------------------------------------------------------
with tab2:
    st.subheader("Battery Dataset Analysis & AI Interpretation")

    uploaded = st.file_uploader("Upload .csv or .mat dataset", type=["csv", "mat"])
    if uploaded:
        if uploaded.name.endswith(".mat"):
            if not SCIPY_OK:
                st.error("scipy not available for .mat support.")
            else:
                mat = loadmat(uploaded)
                key = next(k for k in mat.keys() if not k.startswith("__"))
                df = pd.DataFrame(mat[key])
        else:
            df = pd.read_csv(uploaded)

        st.success(f"Loaded {uploaded.name} with {len(df)} rows × {len(df.columns)} cols.")
        vcol, qcol, cyc = _standardize_columns(df)
        if not vcol or not qcol:
            st.error("Could not detect Voltage/Capacity columns.")
        else:
            st.info(f"Using V → {vcol}, Q → {qcol}")
            proceed = st.button("⚡ Analyze Dataset")
            if proceed:
                # --- Analysis logic simplified for clarity ---
                groups = [str(x) for x in df[cyc].unique()] if cyc else ["Curve"]
                features_by_group = {}
                for g in groups:
                    sub = df[df[cyc]==g] if cyc else df
                    V, Q = _prep_series(sub[vcol], sub[qcol])
                    dQdV, peaks, dVdQ_med = _extract_features(V, Q)
                    features_by_group[g] = {
                        "n_samples": len(V),
                        "voltage_range_V": (np.nanmin(V), np.nanmax(V)),
                        "cap_range_Ah": (np.nanmin(Q), np.nanmax(Q)),
                        "ica_peaks_count": peaks["n_peaks"],
                        "ica_peak_voltages_V": peaks["voltages"],
                        "ica_peak_widths_V": peaks["widths_V"],
                        "dVdQ_median_abs": dVdQ_med,
                    }

                richness_notes = ["Multiple curves detected → trend analysis enabled."
                                  if len(groups) > 1 else
                                  "Single curve → add aged/baseline for richer insight."]
                suggestions = ["Generate ICA plots (dQ/dV) and compare peaks vs cycle.",
                               "Correlate IR and capacity fade for diagnostics."]

                interps = []
                if len(groups) >= 2:
                    interps = _compare_two_sets(groups[0], features_by_group[groups[0]],
                                                groups[-1], features_by_group[groups[-1]])

                st.write("### Dataset Richness")
                for r in richness_notes:
                    st.write("•", r)
                st.write("### AI Suggestions")
                for s_ in suggestions:
                    st.write("•", s_)
                st.write("### Interpretations")
                for i in interps:
                    st.write("•", i)

                pdf_bytes = generate_pdf_report(
                    features_by_group, richness_notes, suggestions, interps,
                    pd.DataFrame(columns=["Voltage","Capacity_Ah","Cycle"]),
                    pd.DataFrame(columns=["Voltage","dQdV","Cycle"]),
                )
                st.download_button("📄 Download Analytics Report (PDF)",
                                   pdf_bytes, file_name="BatteryLAB_Analytics_Report.pdf",
                                   mime="application/pdf")

import time, random

# ==============================================================
#  💬  Permanent Right-Side Copilot  (Shared Context with Typing Effect)
# ==============================================================
st.markdown(
    """
    <style>
    .copilot-container {
        position: fixed;
        top: 3.5rem;
        right: 0;
        width: 27%;
        height: 93%;
        background-color: #f5f5f5;
        border-left: 1px solid #d3d3d3;
        padding: 10px 16px;
        overflow-y: auto;
        font-size: 15px;
    }
    .copilot-msg {
        margin-bottom: 10px;
        padding: 8px 10px;
        border-radius: 10px;
        line-height: 1.4;
    }
    .user-msg {
        background-color: #DCF8C6;
        text-align: right;
    }
    .bot-msg {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="copilot-container">', unsafe_allow_html=True)
    st.markdown("### 🤖 BatteryLAB Copilot")
    st.caption("Your friendly AI lab partner — ready to brainstorm, analyse, or chat.")

    user_msg = st.text_input("💬 Type your message:", key="copilot_input")
    send = st.button("Send →", key="copilot_send")

    if send and user_msg.strip():
        st.session_state.chat_history.append({"role": "user", "msg": user_msg})

        with st.spinner("Copilot is thinking..."):
            time.sleep(random.uniform(0.6, 1.3))
        if "design" in user_msg.lower():
            reply = "That’s an interesting design tweak! Let's estimate its impact on voltage or energy density next."
        elif "plot" in user_msg.lower() or "curve" in user_msg.lower():
            reply = "Got it — I’ll update the visualization accordingly in future versions."
        elif "temperature" in user_msg.lower():
            reply = "Good question — thermal factors can greatly affect ionic mobility and cycle life. Want me to suggest additives for high-T operation?"
        elif "dataset" in user_msg.lower() or "data" in user_msg.lower():
            reply = "Looks like you’re diving into analytics — you can explore correlations between IR and capacity fade next!"
        else:
            reply = random.choice([
                "Hmm interesting — I’ll note that for the next run.",
                "Got it! I'll help refine that in the next iteration.",
                "Cool idea — want me to simulate that parameter change?"
            ])
        st.session_state.chat_history.append({"role": "assistant", "msg": reply})

    for h in st.session_state.chat_history[-12:]:
        role = "🧑‍🔬 You" if h["role"] == "user" else "🤖 Copilot"
        css_class = "user-msg" if h["role"] == "user" else "bot-msg"
        st.markdown(f'<div class="copilot-msg {css_class}"><b>{role}:</b> {h["msg"]}</div>',
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
