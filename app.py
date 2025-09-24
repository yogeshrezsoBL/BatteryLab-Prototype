import json
import io
import numpy as np
import pandas as pd
import streamlit as st

# Optional (for analytics)
try:
    from scipy.signal import savgol_filter, find_peaks, peak_widths
    from scipy.io import loadmat
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

from batterylab_recipe_engine import ElectrodeSpec, CellDesignInput, design_pouch

st.set_page_config(page_title="BatteryLab Prototype", page_icon="🔋", layout="wide")
st.title("🔋 BatteryLab — Prototype")
st.caption("Core promise: Enter an electrode recipe → get performance + feasibility + AI suggestions. New: upload a dataset → get analytics, plots, and dynamic interpretations. (Prototype, first-order estimates)")

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Recipe → Performance", "Data Analytics (CSV/MAT)"])

# =========================
# TAB 1: Recipe → Performance
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
    # find voltage
    vcol = None
    for i, c in enumerate(cols_l):
        if c in ["v", "volt", "voltage", "voltage_v"]:
            vcol = df.columns[i]; break
    # find capacity
    qcol = None
    for i, c in enumerate(cols_l):
        if "capacity" in c or c in ["q", "ah", "mah", "capacity_ah", "capacity_mah"]:
            qcol = df.columns[i]; break
    # cycle/label column (optional)
    cyc = None
    for i, c in enumerate(cols_l):
        if c in ["cycle", "label", "group"]:
            cyc = df.columns[i]; break
    return vcol, qcol, cyc

def _prep_series(voltage, capacity):
    V = pd.to_numeric(voltage, errors="coerce").to_numpy()
    Qraw = pd.to_numeric(capacity, errors="coerce").to_numpy()
    # convert mAh to Ah if needed (heuristic on column name handled earlier; here we also catch numeric scale)
    if np.nanmax(Qraw) > 100:  # likely in mAh
        Q = Qraw / 1000.0
    else:
        Q = Qraw
    mask = np.isfinite(V) & np.isfinite(Q)
    V = V[mask]; Q = Q[mask]
    order = np.argsort(V)
    return V[order], Q[order]

def _extract_features(V, Q):
    # Smooth for derivative stability
    Qs = Q.copy()
    if SCIPY_OK and len(Q) >= 11:
        try:
            Qs = savgol_filter(Q, 11, 3)
        except Exception:
            pass
    # ICA
    dQdV = np.gradient(Qs, V, edge_order=2)
    # Peaks
    peak_info = {"n_peaks": 0, "voltages": [], "widths_V": []}
    if SCIPY_OK and np.isfinite(dQdV).any():
        try:
            prom = np.nanmax(np.abs(dQdV))*0.05 if np.nanmax(np.abs(dQdV))>0 else 0.0
            peaks, props = find_peaks(dQdV, prominence=prom)
            peak_info["n_peaks"] = int(len(peaks))
            peak_info["voltages"] = [float(V[p]) for p in peaks]
            if len(peaks) > 0:
                widths, h, left, right = peak_widths(dQdV, peaks, rel_height=0.5)
                # convert width from samples to volts (approx via local spacing)
                if len(V) > 1:
                    dv = np.mean(np.diff(V))
                    peak_info["widths_V"] = [float(w*dv) for w in widths]
        except Exception:
            pass
    # dV/dQ for impedance proxy (median abs)
    try:
        dVdQ = np.gradient(V, Qs, edge_order=2)
        dVdQ_med = float(np.nanmedian(np.abs(dVdQ)))
    except Exception:
        dVdQ_med = float("nan")
    return dQdV, peak_info, dVdQ_med

def _compare_two_sets(name_a, feat_a, name_b, feat_b):
    """Return interpretations comparing A vs B using simple heuristics."""
    interp = []
    # Capacity fade (needs capacity ranges)
    cap_a = feat_a.get("cap_range_Ah", [np.nan, np.nan])[1]
    cap_b = feat_b.get("cap_range_Ah", [np.nan, np.nan])[1]
    if np.isfinite(cap_a) and np.isfinite(cap_b) and cap_a > 0:
        fade_pct = 100.0 * (cap_a - cap_b) / cap_a
        if abs(fade_pct) >= 3:
            interp.append(f"Capacity change from {name_a} to {name_b}: {fade_pct:.1f}% (negative = fade).")

    # Peak shifts
    Va = feat_a.get("ica_peak_voltages_V", [])
    Vb = feat_b.get("ica_peak_voltages_V", [])
    if Va and Vb:
        n = min(len(Va), len(Vb))
        if n >= 1:
            mean_shift_mV = 1000.0 * float(np.nanmean(np.array(Vb[:n]) - np.array(Va[:n])))
            if abs(mean_shift_mV) >= 5:
                direction = "↑" if mean_shift_mV > 0 else "↓"
                interp.append(f"Average ICA peak shift {direction} ~{abs(mean_shift_mV):.0f} mV ({name_b} vs {name_a}) → possible LLI or cathode aging.")

    # Peak broadening (widths)
    Wa = feat_a.get("ica_peak_widths_V", [])
    Wb = feat_b.get("ica_peak_widths_V", [])
    if Wa and Wb:
        n = min(len(Wa), len(Wb))
        if n >= 1:
            mean_broad_mV = 1000.0 * float(np.nanmean(np.array(Wb[:n]) - np.array(Wa[:n])))
            if mean_broad_mV > 2:  # >2 mV is small but visible
                interp.append(f"ICA peak broadening ~{mean_broad_mV:.0f} mV ({name_b} vs {name_a}) → rising impedance / polarization.")

    # Impedance proxy via |dV/dQ| median
    iva = feat_a.get("dVdQ_median_abs", np.nan)
    ivb = feat_b.get("dVdQ_median_abs", np.nan)
    if np.isfinite(iva) and np.isfinite(ivb) and ivb > iva*1.05:
        interp.append(f"Median |dV/dQ| increased ({name_b} vs {name_a}) → higher polarization/impedance.")

    if not interp:
        interp.append(f"No strong differences detected between {name_a} and {name_b} within prototype sensitivity.")
    return interp

# =========================
# TAB 2: Data Analytics
# =========================
with tab2:
    st.subheader("Upload a dataset")
    st.write("Accepted: **.csv** (recommended) or **.mat** (if SciPy is available). Columns: `Voltage` and `Capacity_Ah` (or `Capacity_mAh`). Optional: `Cycle` (e.g., Fresh/Aged).")

    up = st.file_uploader("Upload CSV or MAT file", type=["csv", "mat"])
    if up is not None:
        with st.spinner("Parsing and extracting features…"):
            # 1) Load
            df = None
            name = up.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(up)
            elif name.endswith(".mat"):
                if not SCIPY_OK:
                    st.error("SciPy not available. Please upload a CSV for now.")
                else:
                    mat = loadmat(io.BytesIO(up.getvalue()))
                    candidates = {k: np.squeeze(v) for k, v in mat.items() if isinstance(v, np.ndarray) and v.size > 3}
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
                # 2) Standardize
                vcol, qcol, cyc = _standardize_columns(df)
                if vcol is None or qcol is None:
                    st.error("Please include `Voltage` and `Capacity_Ah` (or `Capacity_mAh`).")
                    st.stop()

                # If Cycle missing, treat whole file as one curve
                if cyc is None:
                    df["_Cycle"] = "Curve1"
                    cyc = "_Cycle"

                # 3) Process each group
                groups = list(df[cyc].astype(str).unique())
                features_by_group = {}
                plots_done = False

                st.markdown("### Plots")
                cols = st.columns(2)

                for g in groups:
                    sub = df[df[cyc].astype(str) == g]
                    V, Q = _prep_series(sub[vcol], sub[qcol])
                    if len(V) < 5:
                        continue
                    dQdV, peak_info, dVdQ_med = _extract_features(V, Q)

                    # Save features
                    features_by_group[g] = {
                        "n_samples": int(len(V)),
                        "voltage_range_V": [float(np.nanmin(V)), float(np.nanmax(V))],
                        "cap_range_Ah": [float(np.nanmin(Q)), float(np.nanmax(Q))],
                        "ica_peaks_count": int(peak_info.get("n_peaks", 0)),
                        "ica_peak_voltages_V": peak_info.get("voltages", []),
                        "ica_peak_widths_V": peak_info.get("widths_V", []),
                        "dVdQ_median_abs": dVdQ_med,
                    }

                    # Plots per group (side-by-side)
                    vc_df = pd.DataFrame({"Voltage (V)": V, f"Capacity_Ah [{g}]": Q})
                    ica_df = pd.DataFrame({"Voltage (V)": V, f"dQ/dV (Ah/V) [{g}]": dQdV})
                    with cols[0]:
                        st.line_chart(vc_df.set_index("Voltage (V)"))
                    with cols[1]:
                        st.line_chart(ica_df.set_index("Voltage (V)"))
                    plots_done = True

                if not plots_done:
                    st.error("Not enough valid data points to plot.")
                    st.stop()

                # 4) Show features
                st.markdown("### Key Features by Curve")
                st.write(features_by_group)

                # 5) Dynamic interpretations
                st.markdown("### AI-style Interpretations (dynamic)")
                interps = []
                if len(features_by_group) == 1:
                    g = list(features_by_group.keys())[0]
                    f = features_by_group[g]
                    interps += [
                        "Distinct ICA peaks often indicate well-defined phase transitions.",
                        "Broadening of ICA peaks over time is a common sign of rising impedance.",
                        "Compare this curve to an earlier/later cycle to quantify fade and peak shifts.",
                    ]
                else:
                    # Compare first two groups (e.g., Fresh vs Aged)
                    gnames = list(features_by_group.keys())[:2]
                    fA, fB = features_by_group[gnames[0]], features_by_group[gnames[1]]

                    # build comparison-friendly dicts
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

                # 6) Download features
                st.download_button(
                    "Download extracted features (JSON)",
                    data=json.dumps(features_by_group, indent=2),
                    file_name="BatteryLab_analytics_features.json",
                    mime="application/json"
                )
    else:
        st.info("Upload a **CSV** with `Voltage`, `Capacity_Ah` (or `Capacity_mAh`). Optionally include `Cycle` (e.g., Fresh/Aged). MAT is supported if SciPy is available.")
