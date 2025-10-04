# ... (imports and recipe → performance tab remain the same)

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
                        "Track ICA peak shifts (mV) and broadening (mV) -> LLI and impedance indicators.",
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
