import json
import io
import numpy as np
import pandas as pd
import streamlit as st

# optional imports (only used if available)
try:
    from scipy.signal import savgol_filter, find_peaks
    from scipy.io import loadmat
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

from batterylab_recipe_engine import ElectrodeSpec, CellDesignInput, design_pouch

st.set_page_config(page_title="BatteryLab Prototype", page_icon="🔋", layout="wide")
st.title("🔋 BatteryLab — Prototype")
st.caption("Core promise: Enter an electrode recipe → get performance + feasibility + AI suggestions. New: upload a dataset → get analytics, plots, and interpretations. (Prototype, first-order estimates)")

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
# TAB 2: Data Analytics (CSV/MAT)
# =========================
with tab2:
    st.subheader("Upload a dataset")
    st.write("Accepted: **.csv** (recommended) or **.mat** (if SciPy is available). Expect columns like: `Voltage` (V) and `Capacity_Ah` (or `Capacity_mAh`).")

    up = st.file_uploader("Upload CSV or MAT file", type=["csv", "mat"])
    if up is not None:
        with st.spinner("Parsing and extracting features…"):
            # 1) Load data
            df = None
            name = up.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(up)
            elif name.endswith(".mat"):
                if not SCIPY_OK:
                    st.error("SciPy is not available on this runtime. Please upload a CSV for now.")
                else:
                    mat = loadmat(io.BytesIO(up.getvalue()))
                    # Heuristic: find arrays that look like V and Q
                    candidates = {k: np.squeeze(v) for k, v in mat.items() if isinstance(v, np.ndarray) and v.size > 3}
                    # try common names
                    V = None; Q = None
                    for key in candidates:
                        lk = key.lower()
                        if V is None and ("volt" in lk or lk == "v"):
                            V = candidates[key]
                        if Q is None and ("cap" in lk or lk in ["q", "capacity", "capacity_ah", "capacity_mah"]):
                            Q = candidates[key]
                    if V is None or Q is None:
                        st.error("Could not locate voltage/capacity arrays in .mat. Use CSV with columns: Voltage, Capacity_Ah/Capacity_mAh.")
                    else:
                        df = pd.DataFrame({"Voltage": V, "Capacity": Q})

            if df is not None:
                # 2) Standardize column names
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

                if vcol is None or qcol is None:
                    st.error("Please include columns for Voltage and Capacity (Ah or mAh). Example headers: `Voltage`, `Capacity_Ah`.")
                    st.stop()

                V = pd.to_numeric(df[vcol], errors="coerce").to_numpy()
                Qraw = pd.to_numeric(df[qcol], errors="coerce").to_numpy()

                # convert capacity to Ah if needed
                if "mah" in qcol.lower():
                    Q = Qraw / 1000.0
                else:
                    Q = Qraw

                # Remove NaNs and sort by increasing Voltage
                mask = np.isfinite(V) & np.isfinite(Q)
                V = V[mask]; Q = Q[mask]
                order = np.argsort(V)
                V = V[order]; Q = Q[order]

                # 3) Smooth (optional) and derivatives
                Q_s = Q.copy()
                if SCIPY_OK and len(Q) >= 11:
                    try:
                        Q_s = savgol_filter(Q, 11, 3)
                    except Exception:
                        pass

                # dQ/dV (ICA)
                dQdV = np.gradient(Q_s, V, edge_order=2)

                # dV/dQ (differential voltage) - guard for monotonic Q
                try:
                    dVdQ = np.gradient(V, Q_s, edge_order=2)
                except Exception:
                    dVdQ = np.full_like(V, np.nan)

                # 4) Simple peak metrics on ICA
                peak_info = {}
                if SCIPY_OK:
                    try:
                        peaks, props = find_peaks(dQdV, prominence=np.nanmax(np.abs(dQdV))*0.05 if np.nanmax(np.abs(dQdV))>0 else 0.0)
                        peak_info = {
                            "n_peaks": int(len(peaks)),
                            "peak_voltages": [float(V[p]) for p in peaks[:5]],
                        }
                    except Exception:
                        peak_info = {"n_peaks": int(np.nan), "peak_voltages": []}

                # 5) Plots
                st.markdown("### Plots")
                c1, c2 = st.columns(2)
                with c1:
                    st.line_chart(pd.DataFrame({"Voltage (V)": V, "Capacity (Ah)": Q}))
                    st.caption("Voltage vs Capacity")
                with c2:
                    st.line_chart(pd.DataFrame({"Voltage (V)": V, "dQ/dV (Ah/V)": dQdV}))
                    st.caption("Incremental Capacity (ICA): dQ/dV vs Voltage")

                st.markdown("### Key Features (auto-extracted)")
                st.write({
                    "Samples": int(len(V)),
                    "Capacity range (Ah)": [float(np.nanmin(Q)), float(np.nanmax(Q))],
                    "Voltage range (V)": [float(np.nanmin(V)), float(np.nanmax(V))],
                    "ICA peaks (count)": peak_info.get("n_peaks", "n/a"),
                    "ICA peak voltages (V)": peak_info.get("peak_voltages", []),
                })

                # 6) Interpretations (heuristics, safe & useful)
                st.markdown("### AI-style Interpretations")
                bullets = []
                # generic, safe insights:
                bullets.append("Distinct ICA peaks often indicate well-defined phase transitions; broadening over time can suggest rising impedance.")
                bullets.append("Horizontal shifts in Voltage–Capacity curves between cycles typically indicate loss of lithium inventory (LLI).")
                bullets.append("Reduced ICA peak heights at similar state-of-charge can suggest loss of active material (LAM) or increased polarization.")
                if SCIPY_OK and peak_info.get("n_peaks", 0) >= 2:
                    bullets.append("Multiple ICA peaks detected — consider tracking peak positions across cycles to monitor aging modes.")
                bullets.append("Next steps: compare early vs late-cycle ICA to quantify degradation, and correlate with IR/temperature if available.")
                for b in bullets:
                    st.write("• " + b)

                # 7) Download features
                features = {
                    "capacity_range_Ah": [float(np.nanmin(Q)), float(np.nanmax(Q))],
                    "voltage_range_V": [float(np.nanmin(V)), float(np.nanmax(V))],
                    "ica_peaks_count": peak_info.get("n_peaks", None),
                    "ica_peak_voltages_V": peak_info.get("peak_voltages", []),
                }
                st.download_button("Download extracted features (JSON)", data=json.dumps(features, indent=2),
                                   file_name="BatteryLab_analytics_features.json", mime="application/json")
    else:
        st.info("Upload a **CSV** with columns like `Voltage` and `Capacity_Ah` (or `Capacity_mAh`). Optionally, a **MAT** file if SciPy is available.")
