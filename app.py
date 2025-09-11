import json
import streamlit as st
from batterylab_recipe_engine import ElectrodeSpec, CellDesignInput, design_pouch

st.set_page_config(page_title="BatteryLab Prototype", page_icon="🔋", layout="wide")
st.title("🔋 BatteryLab — Recipe → Performance Prototype")
st.caption("Enter an electrode recipe and get performance, thermal & mechanical feasibility plus AI suggestions. Prototype — first-order estimates.")

# ---------- Demo Presets ----------
PRESETS = {
    "LFP 2.5Ah (demo)": {
        "cathode": {"material": "LFP", "thk": 70, "por": 0.35},
        "anode":   {"material": "Graphite", "thk": 85, "por": 0.40, "si": 0.00},
        "geom":    {"mode": "area", "area_cm2": 100.0, "n_layers": 36},
        "sep":     {"thk": 20, "por": 0.45},
        "foil":    {"al": 15, "cu": 8},
        "np": 1.10,
        "elyte": "1M LiPF6 in EC:EMC 3:7",
        "amb": 25
    },
    "NMC811 3Ah (demo)": {
        "cathode": {"material": "NMC811", "thk": 75, "por": 0.33},
        "anode":   {"material": "Graphite", "thk": 90, "por": 0.40, "si": 0.05},
        "geom":    {"mode": "area", "area_cm2": 110.0, "n_layers": 32},
        "sep":     {"thk": 20, "por": 0.45},
        "foil":    {"al": 15, "cu": 8},
        "np": 1.08,
        "elyte": "1M LiPF6 + 2% VC in EC:DEC",
        "amb": 25
    }
}

# ---------- Sidebar Inputs ----------
with st.sidebar:
    st.header("Quick Start")
    preset = st.selectbox("🔖 Preset", ["— none —"] + list(PRESETS.keys()))

    # Defaults (used if no preset)
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
            "area_cm2": p["geom"]["area_cm2"],
            "n_layers": p["geom"]["n_layers"],
            "n_p_ratio": p["np"],
            "electrolyte": p["elyte"],
            "ambient_C": p["amb"],
            "cath_mat": p["cathode"]["material"],
            "cath_thk": p["cathode"]["thk"],
            "cath_por": p["cathode"]["por"],
            "anode_mat": p["anode"]["material"],
            "anode_thk": p["anode"]["thk"],
            "anode_por": p["anode"]["por"],
            "anode_si":  p["anode"]["si"],
            "sep_thk": p["sep"]["thk"],
            "sep_por": p["sep"]["por"],
            "foil_al": p["foil"]["al"],
            "foil_cu": p["foil"]["cu"],
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

# ---------- Light validation (UX polish) ----------
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
            cathode=cathode,
            anode=anode,
            n_layers=int(n_layers),
            separator_thickness_um=sep_thk,
            separator_porosity=sep_por,
            n_p_ratio=float(n_p_ratio),
            cathode_foil_um=foil_al,
            anode_foil_um=foil_cu,
            electrolyte=electrolyte,
            ambient_C=float(ambient_C),
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

    # Download JSON artifact
    st.download_button(
        "Download result JSON",
        data=json.dumps(result, indent=2),
        file_name="BatteryLab_result.json",
        mime="application/json"
    )
else:
    st.info("Pick a preset for a 1-click demo, or set your recipe parameters, then press **Compute Performance →**")
