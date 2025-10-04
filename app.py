import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# For PDF export
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit cloud
import matplotlib.pyplot as plt

# Optional (for analytics on MAT + feature extraction)
try:
    from scipy.signal import savgol_filter, find_peaks, peak_widths
    from scipy.io import loadmat
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

from batterylab_recipe_engine import ElectrodeSpec, CellDesignInput, design_pouch

st.set_page_config(page_title="BatteryLab Prototype", page_icon="🔋", layout="wide")
st.title("🔋 BatteryLab — Prototype")
st.caption(
    "Core promise: Enter an electrode recipe → get performance + feasibility + AI suggestions. "
    "Upload a dataset → get analysis first (richness & next steps), then (on click) plots with dynamic interpretations. "
    "Temperature-aware guidance included. (Prototype)"
)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Recipe → Performance", "Data Analytics (CSV/MAT)"])

# =========================
# TAB 1: Recipe → Performance (temperature-aware)
# =========================
with tab1:
    PRESETS = {
        "LFP 2.5Ah (demo)": {
            "cathode": {"material": "LFP", "thk": 70, "por": 0.35},
            "anode":   {"material": "Graphite", "thk": 85, "por": 0.40, "si": 0.00},
            "geom":    {"mode": "area", "area_cm2": 100.0, "n_layers": 36},
            "sep":     {"thk": 20, "por": 0.45}, "foil": {"al": 15, "cu": 8},
            "np": 1.10, "elyte": "1M LiPF6 in EC:EMC 3:7", "amb": 25
        },
        "NMC811 3Ah (demo)": {
            "cathode": {"material": "NMC811", "thk": 75, "por": 0.33},
            "anode":   {"material": "Graphite", "thk": 90, "por": 0.40, "si": 0.05},
            "geom":    {"mode": "area", "area_cm2": 110.0, "n_layers": 32},
            "sep":     {"thk": 20, "por": 0.45}, "foil": {"al": 15, "cu": 8},
            "np": 1.08, "elyte": "1M LiPF6 + 2% VC in EC:DEC", "amb": 25
        }
    }

    with st.sidebar:
        st.header("Quick Start")
        preset = st.selectbox("🔖 Preset", ["— none —"] + list(PRESETS.keys()))

        defaults = {
            "geom_mode": "area",
            "area_cm2": 100.0, "width_mm": 70.0, "height_mm": 100.0,
            "n_layers": 36, "n_p_ratio": 1.10,
            "electrolyte": "1M LiPF6 in EC:EMC 3:7", "ambient_C": 25,
            "cath_mat": "LFP", "cath_thk": 70, "cath_por": 0.35,
            "anode_mat": "Graphite", "anode_thk": 85, "anode_por": 0.40, "anode_si": 0.00,
            "sep_thk": 20, "sep_por": 0.45, "foil_al": 15, "foil_cu": 8,
        }

        if preset != "— none —":
            p = PRESETS[preset]
            defaults.update({
                "geom_mode": p["geom"]["mode"],
                "area_cm2": p["geom"]["area_cm2"], "n_layers": p["geom"]["n_layers"],
                "n_p_ratio": p["np"], "electrolyte": p["elyte"], "ambient_C": p["amb"],
                "cath_mat": p["cathode"]["material"], "cath_thk": p["cathode"]["thk"], "cath_por": p["cathode"]["por"],
                "anode_mat": p["anode"]["material"], "anode_thk": p["anode"]["thk"],
                "anode_por": p["anode"]["por"], "anode_si": p["anode"]["si"],
                "sep_thk": p["sep"]["thk"], "sep_por": p["sep"]["por"],
                "foil_al": p["foil"]["al"], "foil_cu": p["foil"]["cu"],
            })

        st.header("Cell Geometry")
        area_mode = st.radio("Area Input Mode", ["Direct area (cm²)", "Width × Height (mm)"],
                             index=0 if defaults["geom_mode"] == "area" else 1)

        if area_mode == "Direct area (cm²)":
            area_cm2 = st.number_input("Layer area (cm²)", min_value=10.0, value=float(defaults["area_cm2"]), step=5.0)
            dims = {"area_cm2": area_cm2}
        else:
            width_mm = st.number_input("Width (mm)", min_value=10.0, value=float(defaults["width_mm"]), step=1.0)
            height_mm = st.number_input("Height (mm)", min_value=10.0, value=float(defaults["height_mm"]), step=1.0)
            dims = {"width_mm": width_mm, "height_mm": height_mm}

        n_layers = st.number_input("# Layers", min_value=2, value=int(defaults["n_layers"]), step=2)
        n_p_ratio = st.slider("N/P ratio", 1.00, 1.30, float(defaults["n_p_ratio"]), 0.01)
        electrolyte = st.text_input("Electrolyte (free text)", defaults["electrolyte"])
        ambient_C = st.slider("Ambient Temp (°C)", -20, 60, int(defaults["ambient_C"]), 1)

    st.subheader("Cathode")
    cathode_material = st.selectbox("Material (Cathode)", ["LFP", "NMC811"],
                                    index=(0 if defaults["cath_mat"] == "LFP" else 1))
    cathode_thk = st.slider("Cathode thickness (µm)", 20, 140, int(defaults["cath_thk"]), 1)
    cathode_por = st.slider("Cathode porosity", 0.20, 0.60, float(defaults["cath_por"]), 0.01)

    st.subheader("Anode")
    anode_material = st.selectbox("Material (Anode)", ["Graphite"], index=0)
    anode_thk = st.slider("Anode thickness (µm)", 20, 140, int(defaults["anode_thk"]), 1)
    anode_por = st.slider("Anode porosity", 0.20, 0.60, float(defaults["anode_por"]), 0.01)
    anode_si = st.slider("Anode silicon fraction (0–1)", 0.0, 0.20, float(defaults["anode_si"]), 0.01)

    st.subheader("Separator & Foils")
    sep_thk = st.slider("Separator thickness (µm)", 10, 40, int(defaults["sep_thk"]), 1)
    sep_por = st.slider("Separator porosity", 0.20, 0.70, float(defaults["sep_por"]), 0.01)
    foil_al = st.slider("Cathode Al foil (µm)", 8, 20, int(defaults["foil_al"]), 1)
    foil_cu = st.slider("Anode Cu foil (µm)", 4, 15, int(defaults["foil_cu"]), 1)

    if cathode_thk < 20 or anode_thk < 20:
        st.warning("Very thin coatings may make predictions less reliable.")
    if not (0.2 <= cathode_por <= 0.6 and 0.2 <= anode_por <= 0.6):
        st.warning("Porosity outside typical ranges may reduce accuracy.")

    run = st.button("Compute Performance →", type="primary")

    if run:
        with st.spinner("Computing physics + AI suggestions…"):
            cathode = ElectrodeSpec(material=cathode_material, thickness_um=cathode_thk, porosity=cathode_por, active_frac=0.96)
            anode   = ElectrodeSpec(material=anode_material, thickness_um=anode_thk, porosity=anode_por, active_frac=0.96, silicon_frac=anode_si)
            spec = CellDesignInput(
                cathode=cathode, anode=anode, n_layers=int(n_layers),
                separator_thickness_um=sep_thk, separator_porosity=sep_por, n_p_ratio=float(n_p_ratio),
                cathode_foil_um=foil_al, anode_foil_um=foil_cu, electrolyte=electrolyte, ambient_C=float(ambient_C),
                **dims
            )
            result = design_pouch(spec)

        st.success("Computed successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Capacity (Ah)", f"{result['electrochem']['capacity_Ah']:.2f}")
            st.metric("Nominal Voltage (V)", f"{result['electrochem']['V_nom']:.2f}")
        with col2:
            st.metric("Wh/kg", f"{result['electrochem']['Wh_per_kg']:.0f}")
            st.metric("Wh/L", f"{result['electrochem']['Wh_per_L']:.0f}")
        with col3:
            st.metric("ΔT @1C (°C)", f"{result['thermal']['deltaT_1C_C']:.2f}")
            st.metric("ΔT @3C (°C)", f"{result['thermal']['deltaT_3C_C']:.2f}")

        st.markdown("### Mechanical & Feasibility")
        fz = result["feasibility"]
        cols = st.columns(3)
        cols[0].markdown(f"**Swelling flag:** {fz['swelling_flag']}")
        cols[1].markdown(f"**Thermal @3C:** {fz['thermal_flag_3C']}")
        cols[2].markdown(f"**Swelling % @100% SOC:** {round(result['mechanical']['swelling_pct_100SOC'],2)}%")

        # ---- Temperature advisories & adjusted metrics ----
        st.markdown("### Temperature Advisories")
        tg = result.get("temperature_guidance", {})
        ea = result.get("electrochem_temp_adjusted", {})

        cols_t = st.columns(4)
        cols_t[0].markdown(f"**Ambient:** {tg.get('ambient_C', '—')} °C")
        cols_t[1].markdown(f"**Ideal window:** {tg.get('ideal_low_C','—')}–{tg.get('ideal_high_C','—')} °C")
        try:
            eff_cap = float(ea.get('effective_capacity_Ah_at_ambient', float('nan')))
            cols_t[2].markdown(f"**Effective Capacity @ ambient:** {eff_cap:.2f} Ah")
        except Exception:
            cols_t[2].markdown("**Effective Capacity @ ambient:** —")
        try:
            rel_pow = float(ea.get('relative_power_vs_25C', float('nan')))
            cols_t[3].markdown(f"**Relative Power vs 25 °C:** {rel_pow:.2f}×")
        except Exception:
            cols_t[3].markdown("**Relative Power vs 25 °C:** —")

        risk_msgs = []
        if tg.get("cold_temp_risk", False):
            risk_msgs.append("⚠ Cold-condition risk (≤0 °C): expect higher impedance/lower power; pre-heat or derate C-rate.")
        if tg.get("high_temp_risk", False):
            risk_msgs.append("⚠ High ambient (≥45 °C): accelerated side reactions; consider high-temp electrolyte, charge derating, better cooling.")
        if not risk_msgs:
            risk_msgs.append("✅ Ambient in acceptable range for typical operation.")
        for m in risk_msgs:
            st.write("• " + m)

        st.markdown("### AI Suggestions")
        for s in result["ai_suggestions"]:
            st.write("• " + s)

        st.download_button(
            "Download result JSON",
            data=json.dumps(result, indent=2),
            file_name="BatteryLab_result.json",
            mime="application/json"
        )
    else:
        st.info("Pick a preset for a 1-click demo, or set your recipe parameters, then press **Compute Performance →**")

# =========================
# Utilities for Analytics
# =========================
def _standardize_columns(df: pd.DataFrame):
    cols_l = [c.lower() for c in df.columns]
    vcol = None
    for i, c in enumerate(cols_l):
        if c in ["v", "volt", "voltage", "voltage_v"]:
            vcol = df.columns[i]; break
    qcol = None
    for i, c in enumerate(cols_l):
        if "capacity" in c or c in ["q", "ah", "mah", "capacity_ah", "capacity_mah"]:
            qcol = df.columns[i]; break
    cyc = None
    for i, c in enumerate(cols_l):
        if c in ["cycle", "label", "group"]:
            cyc = df.columns[i]; break
    return vcol, qcol, cyc

def _prep_series(voltage, capacity):
    V = pd.to_numeric(voltage, errors="coerce").to_numpy()
    Qraw = pd.to_numeric(capacity, errors="coerce").to_numpy()
    Q = Qraw / 1000.0 if np.nanmax(Qraw) > 100 else Qraw
    mask = np.isfinite(V) & np.isfinite(Q)
    V = V[mask]; Q = Q[mask]
    order = np.argsort(V)
    return V[order], Q[order]

def _extract_features(V, Q):
    Qs = Q.copy()
    if SCIPY_OK and len(Q) >= 11:
        try:
            Qs = savgol_filter(Q, 11, 3)
        except Exception:
            pass
    dQdV = np.gradient(Qs, V, edge_order=2)
    peak_info = {"n_peaks": 0, "voltages": [], "widths_V": []}
    if SCIPY_OK and np.isfinite(dQdV).any():
        try:
            prom = np.nanmax(np.abs(dQdV))*0.05 if np.nanmax(np.abs(dQdV))>0 else 0.0
            peaks, _ = find_peaks(dQdV, prominence=prom)
            peak_info["n_peaks"] = int(len(peaks))
            peak_info["voltages"] = [float(V[p]) for p in peaks]
            if len(peaks) > 0:
                widths, _, _, _ = peak_widths(dQdV, peaks, rel_height=0.5)
                if len(V) > 1:
                    dv = np.mean(np.diff(V))
                    peak_info["widths_V"] = [float(w*dv) for w in widths]
        except Exception:
            pass
    try:
        dVdQ = np.gradient(V, Qs, edge_order=2)
        dVdQ_med = float(np.nanmedian(np.abs(dVdQ)))
    except Exception:
        dVdQ_med = float("nan")
    return dQdV, peak_info, dVdQ_med

def _compare_two_sets(name_a, feat_a, name_b, feat_b):
    interp = []
    cap_a = feat_a.get("cap_range_Ah", [np.nan, np.nan])[1]
    cap_b = feat_b.get("cap_range_Ah", [np.nan, np.nan])[1]
    if np.isfinite(cap_a) and np.isfinite(cap_b) and cap_a > 0:
        fade_pct = 100.0 * (cap_a - cap_b) / cap_a
        if abs(fade_pct) >= 3:
            interp.append(f"Capacity change from {name_a} to {name_b}: {fade_pct:.1f}% (negative = fade).")
    Va = feat_a.get("ica_peak_voltages_V", [])
    Vb = feat_b.get("ica_peak_voltages_V", [])
    if Va and Vb:
        n = min(len(Va), len(Vb))
        if n >= 1:
            mean_shift_mV = 1000.0 * float(np.nanmean(np.array(Vb[:n]) - np.array(Va[:n])))
            if abs(mean_shift_mV) >= 5:
                direction = "↑" if mean_shift_mV > 0 else "↓"
                interp.append(f"Average ICA peak shift {direction} ~{abs(mean_shift_mV):.0f} mV ({name_b} vs {name_a}) → possible LLI or cathode aging.")
    Wa = feat_a.get("ica_peak_widths_V", [])
    Wb = feat_b.get("ica_peak_widths_V", [])
    if Wa and Wb:
        n = min(len(Wa), len(Wb))
        if n >= 1:
            mean_broad_mV = 1000.0 * float(np.nanmean(np.array(Wb[:n]) - np.array(Wa[:n])))
            if mean_broad_mV > 2:
                interp.append(f"ICA peak broadening ~{mean_broad_mV:.0f} mV ({name_b} vs {name_a}) → rising impedance / polarization.")
    iva = feat_a.get("dVdQ_median_abs", np.nan)
    ivb = feat_b.get("dVdQ_median_abs", np.nan)
    if np.isfinite(iva) and np.isfinite(ivb) and ivb > iva * 1.05:
        interp.append(f"Median |dV/dQ| increased ({name_b} vs {name_a}) → higher polarization/impedance.")
    if not interp:
        interp.append(f"No strong differences detected between {name_a} and {name_b} within prototype sensitivity.")
    return interp

# ---------- FIXED: plot -> PNG bytes for ReportLab ----------
def _plot_to_bytes(df_all, ycol, title):
    """Return PNG bytes for a plot so it can be embedded in a PDF."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
    for c in df_all["Cycle"].unique():
        sub = df_all[df_all["Cycle"] == c]
        ax.plot(sub["Voltage"], sub[ycol], label=str(c))
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generate_pdf_report(features_by_group, richness_notes, suggestions, interps, vc_all, ica_all):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("BatteryLab Analytics Report", styles["Title"]))
    elements.append(Spacer(1, 8))

    # Dataset richness
    elements.append(Paragraph("Dataset Quality & Richness", styles["Heading2"]))
    if richness_notes:
        for r in richness_notes:
            elements.append(Paragraph("• " + r, styles["Normal"]))
    else:
        elements.append(Paragraph("No specific richness notes.", styles["Normal"]))
    elements.append(Spacer(1, 6))

    # Next-step suggestions
    elements.append(Paragraph("Next-Step Suggestions", styles["Heading2"]))
    if suggestions:
        for s in suggestions:
            elements.append(Paragraph("• " + s, styles["Normal"]))
    else:
        elements.append(Paragraph("No suggestions generated.", styles["Normal"]))
    elements.append(Spacer(1, 6))

    # Key features table
    elements.append(Paragraph("Key Features by Curve", styles["Heading2"]))
    table_data = [["Curve", "n_samples", "V Range (V)", "Capacity Range (Ah)",
                   "ICA Peaks", "ICA Voltages (V)", "ICA Widths (V)", "Median |dV/dQ|"]]
    for g, f in features_by_group.items():
        table_data.append([
            str(g),
            f['n_samples'],
            f"{f['voltage_range_V'][0]:.2f}–{f['voltage_range_V'][1]:.2f}",
            f"{f['cap_range_Ah'][0]:.2f}–{f['cap_range_Ah'][1]:.2f}",
            f['ica_peaks_count'],
            ", ".join([f"{v:.3f}" for v in f['ica_peak_voltages_V']]) if f['ica_peak_voltages_V'] else "—",
            ", ".join([f"{w:.3f}" for w in f['ica_peak_widths_V']]) if f['ica_peak_widths_V'] else "—",
            f"{f['dVdQ_median_abs']:.4f}" if np.isfinite(f['dVdQ_median_abs']) else "—"
        ])
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (1,1), (-1,-1), 'CENTER')
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 6))

    # Interpretations
    elements.append(Paragraph("AI-style Interpretations", styles["Heading2"]))
    if interps:
        for i in interps:
            elements.append(Paragraph("• " + i, styles["Normal"]))
    else:
        elements.append(Paragraph("No interpretations generated.", styles["Normal"]))
    elements.append(Spacer(1, 6))

    # Plots (now via PNG bytes in memory)
    elements.append(Paragraph("Visualizations", styles["Heading2"]))
    if vc_all is not None and len(vc_all) > 0:
        elements.append(Paragraph("Voltage vs Capacity", styles["Heading3"]))
        img_bytes = _plot_to_bytes(vc_all, "Capacity_Ah", "Voltage vs Capacity")
        elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))
    if ica_all is not None and len(ica_all) > 0:
        elements.append(Paragraph("ICA: dQ/dV vs Voltage", styles["Heading3"]))
        img_bytes = _plot_to_bytes(ica_all, "dQdV", "ICA: dQ/dV vs Voltage")
        elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# =========================
# TAB 2: Data Analytics (Analyze first → button to visualize → plots → features → interpretations → PDF)
# =========================
with tab2:
    st.subheader("Upload a dataset")
    st.write(
        "Accepted: **.csv** (recommended) or **.mat** (if SciPy is available). "
        "Columns: `Voltage` and `Capacity_Ah` (or `Capacity_mAh`). Optional: `Cycle` (e.g., Fresh/Aged)."
    )

    up = st.file_uploader("Upload CSV or MAT file", type=["csv", "mat"])
    if up is not None:
        with st.spinner("Parsing and extracting features…"):
            df = None
            name = up.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(up)
            elif name.endswith(".mat"):
                if not SCIPY_OK:
                    st.error("SciPy not available. Please upload a CSV for now.")
                else:
                    mat = loadmat(io.BytesIO(up.getvalue()))
                    candidates = {k: np.squeeze(v) for k, v in mat.items()
                                  if isinstance(v, np.ndarray) and v.size > 3}
                    V = None; Q = None
                    for key, arr in candidates.items():
                        lk = key.lower()
                        if V is None and ("volt" in lk or lk == "v"):
                            V = arr
                        if Q is None and ("cap" in lk or lk in ["q","capacity","capacity_ah","capacity_mah"]):
                            Q = arr
                    if V is None or Q is None:
                        st.error("Could not find voltage/capacity arrays in .mat. Use CSV with `Voltage`, `Capacity_Ah`.")
                    else:
                        df = pd.DataFrame({"Voltage": V, "Capacity": Q})

            if df is not None:
                # Standardize
                vcol, qcol, cyc = _standardize_columns(df)
                if vcol is None or qcol is None:
                    st.error("Please include `Voltage` and `Capacity_Ah` (or `Capacity_mAh`).")
                    st.stop()
                if cyc is None:
                    df["_Cycle"] = "Curve1"
                    cyc = "_Cycle"

                # Extract features (no plotting yet)
                groups = list(df[cyc].astype(str).unique())
                features_by_group = {}
                vc_all_rows, ica_all_rows = [], []

                for g in groups:
                    sub = df[df[cyc].astype(str) == g]
                    V, Q = _prep_series(sub[vcol], sub[qcol])
                    if len(V) < 5:
                        continue
                    dQdV, peak_info, dVdQ_med = _extract_features(V, Q)
                    features_by_group[g] = {
                        "n_samples": int(len(V)),
                        "voltage_range_V": [float(np.nanmin(V)), float(np.nanmax(V))],
                        "cap_range_Ah": [float(np.nanmin(Q)), float(np.nanmax(Q))],
                        "ica_peaks_count": int(peak_info.get("n_peaks", 0)),
                        "ica_peak_voltages_V": peak_info.get("voltages", []),
                        "ica_peak_widths_V": peak_info.get("widths_V", []),
                        "dVdQ_median_abs": dVdQ_med,
                    }
                    vc_all_rows.append(pd.DataFrame({"Voltage": V, "Capacity_Ah": Q, "Cycle": g}))
                    ica_all_rows.append(pd.DataFrame({"Voltage": V, "dQdV": dQdV, "Cycle": g}))

                if not features_by_group:
                    st.error("Not enough valid data points to analyze.")
                    st.stop()

               # ---- (1) DATASET QUALITY & RICHNESS ----
st.markdown("### Dataset Quality & Richness")
for g, f in features_by_group.items():
    st.write(
        f"**{g}** — {f['n_samples']} points | "
        f"V range: {f['voltage_range_V'][0]:.2f}–{f['voltage_range_V'][1]:.2f} V | "
        f"Capacity: {f['cap_range_Ah'][0]:.2f}–{f['cap_range_Ah'][1]:.2f} Ah | "
        f"ICA peaks: {f['ica_peaks_count']}"
    )

richness_notes = []
if len(features_by_group) >= 2:
    richness_notes.append("Multiple curves detected -> enables trend comparisons (fade, peak shifts, impedance).")
else:
    richness_notes.append("Single curve detected -> add an aged or baseline curve for richer insights.")

if any(f["n_samples"] < 30 for f in features_by_group.values()):
    richness_notes.append("Some curves have <30 points -> derivatives may be noisy; consider higher-resolution sampling.")

if any(f["ica_peaks_count"] == 0 for f in features_by_group.values()):
    richness_notes.append("ICA shows few/no peaks -> may indicate smooth kinetics or insufficient resolution.")

for rn in richness_notes:
    st.write("• " + rn)

                # ---- (2) NEXT-STEP SUGGESTIONS ----
                st.markdown("### Next-Step Suggestions")
                suggestions = []
                if len(features_by_group) == 1:
                    suggestions = [
                        "Add a comparison curve (e.g., Fresh vs Aged, Cycle 10 vs Cycle 500) to quantify capacity fade and ICA peak shifts.",
                        "Track ICA peak positions/widths across cycles to infer LLI vs LAM vs impedance growth.",
                        "Include IR/temperature columns (if available) to correlate electro-thermal behavior."
                    ]
                else:
                    suggestions = [
                        "Quantify fade: compute % capacity change between earliest and latest curves.",
                        "Track ICA peak shifts (mV) and broadening (mV) → LLI and impedance indicators.",
                        "Build a simple regression using extracted features to predict end-of-life or rate performance.",
                        "If you have cycle index/time, add it to the dataset for richer trend modeling."
                    ]
                for s in suggestions:
                    st.write("• " + s)

                # ---- (3) USER-ACTION: VISUALIZE RECOMMENDED PLOTS ----
                st.divider()
                do_plots = st.button("📈 Visualize recommended plots", type="primary")

                if do_plots:
                    st.markdown("### Visualizations")
                    vc_all = pd.concat(vc_all_rows, ignore_index=True)
                    ica_all = pd.concat(ica_all_rows, ignore_index=True)

                    # On-screen Altair charts
                    vc_chart = (
                        alt.Chart(vc_all)
                        .mark_line()
                        .encode(
                            x=alt.X("Voltage:Q", title="Voltage (V)"),
                            y=alt.Y("Capacity_Ah:Q", title="Capacity (Ah)"),
                            color=alt.Color("Cycle:N", title="Curve")
                        )
                        .properties(title="Voltage vs Capacity")
                    )
                    ica_chart = (
                        alt.Chart(ica_all)
                        .mark_line()
                        .encode(
                            x=alt.X("Voltage:Q", title="Voltage (V)"),
                            y=alt.Y("dQdV:Q", title="dQ/dV (Ah/V)"),
                            color=alt.Color("Cycle:N", title="Curve")
                        )
                        .properties(title="ICA: dQ/dV vs Voltage")
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        st.altair_chart(vc_chart, use_container_width=True)
                    with c2:
                        st.altair_chart(ica_chart, use_container_width=True)

                    # ---- (4) KEY FEATURES ----
                    st.markdown("### Key Features by Curve")
                    st.write(features_by_group)

                    # ---- (5) DYNAMIC INTERPRETATIONS ----
                    st.markdown("### AI-style Interpretations (dynamic)")
                    interps = []
                    if len(features_by_group) == 1:
                        interps = [
                            "Distinct ICA peaks often indicate well-defined phase transitions.",
                            "Broadening of ICA peaks over time is a common sign of rising impedance.",
                            "Compare this curve to an earlier/later cycle to quantify fade and peak shifts.",
                        ]
                    else:
                        gnames = list(features_by_group.keys())[:2]
                        fA, fB = features_by_group[gnames[0]], features_by_group[gnames[1]]
                        featA = {
                            "cap_range_Ah": fA["cap_range_Ah"],
                            "ica_peak_voltages_V": fA["ica_peak_voltages_V"],
                            "ica_peak_widths_V": fA["ica_peak_widths_V"],
                            "dVdQ_median_abs": fA["dVdQ_median_abs"],
                        }
                        featB = {
                            "cap_range_Ah": fB["cap_range_Ah"],
                            "ica_peak_voltages_V": fB["ica_peak_voltages_V"],
                            "ica_peak_widths_V": fB["ica_peak_widths_V"],
                            "dVdQ_median_abs": fB["dVdQ_median_abs"],
                        }
                        interps = _compare_two_sets(gnames[0], featA, gnames[1], featB)

                    for b in interps:
                        st.write("• " + b)

                    # ---- (6) DOWNLOAD FULL REPORT (PDF) ----
                    pdf_bytes = generate_pdf_report(
                        features_by_group=features_by_group,
                        richness_notes=richness_notes,
                        suggestions=suggestions,
                        interps=interps,
                        vc_all=vc_all,
                        ica_all=ica_all
                    )
                    st.download_button(
                        "📑 Download Full Report (PDF)",
                        data=pdf_bytes,
                        file_name="BatteryLab_analytics_report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.info("Click **“Visualize recommended plots”** to render Voltage–Capacity and ICA charts, "
                            "then see Key Features, interpretations, and download the full PDF report.")
    else:
        st.info(
            "Upload a **CSV** with `Voltage`, `Capacity_Ah` (or `Capacity_mAh`). "
            "Optionally include `Cycle` (e.g., Fresh/Aged). MAT is supported if SciPy is available."
        )

