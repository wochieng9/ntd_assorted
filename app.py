
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from schisto_tool.config import (
    APP_VERSION,
    CI_INTERVAL_LABEL,
    DEFAULT_N_ITER,
    DEFAULT_SEED,
    OFF_YEAR_FIXED_COST_SHARE,
)
from schisto_tool.utils import _format_ci, _format_ci_or_na, _widget_scope_key, weighted_mean
from schisto_tool.data import load_inputs, load_country_flag
from schisto_tool.parameters import HaematobiumInputs, MansoniInputs
from schisto_tool.caseloads import (
    effective_prevalence,
    estimate_caseloads_haematobium,
    estimate_caseloads_mansoni,
    partitioned_species_defaults,
    threshold_message,
)
from schisto_tool.simulation import run_monte_carlo_haematobium, run_monte_carlo_mansoni
from schisto_tool import elimination as elim
from schisto_tool.economics import (
    add_budget_impact_intervals,
    add_cost_effectiveness_columns,
    add_daly_averted_columns,
    add_economic_impact_intervals,
    adj_daily_wage,
    bladder_cancer_case_summary_table,
    budget_impact_analysis,
    build_combined_daly_df,
    cea_threshold,
    combined_benefit_draws,
    combined_morbidity_budget_draws,
    compute_icer,
    compute_roi,
    cost_effectiveness_summary_table,
    daly_summary_table,
    economic_benefit_cost_ratio_summary,
    economic_impact_analysis,
    health_sector_costs,
    health_sector_result_table,
    morbidity_budget_summary_table,
    productivity_result_table,
    productivity_summary,
    roi_summary_from_draws,
    roi_summary_table,
)
from schisto_tool.health_sector import project_health_sector_costs
from schisto_tool.prevalence import (
    DEFAULT_MANSONI_TRAJECTORY_DECAY_SAC,
    DEFAULT_MANSONI_TRAJECTORY_DECAY_ADULT,
    DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_SAC,
    DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_ADULT,
    annual_equivalent_frequency_factor,
    build_prevalence_scenario_df,
    derive_projection_prevalence_inputs,
    project_caseloads_from_prevalence_scenario,
    project_caseloads_over_time,
)
from schisto_tool.sensitivity import (
    identify_optimal_scenario,
    run_sensitivity_analysis,
    sensitivity_summary_table,
)
from schisto_tool.ui_helpers import download_df, format_daly_averted_calculation, styled_table
from schisto_tool.user_guide import (
    HELP_TEXT,
    render_guided_sidebar,
    render_setup_help,
    render_tab_tip,
    render_user_guide_tab,
)
from schisto_tool.visualization import (
    plot_budget_impact,
    plot_cases_averted_sensitivity,
    plot_ce_plane,
    plot_ceac,
    plot_cost_effectiveness_sensitivity,
    plot_daly_breakdown,
    plot_economic_impact,
    plot_health_sector_cost_trajectory,
    plot_prevalence_projections,
    plot_prevalence_sensitivity,
    plot_prevalence_trajectory,
    plot_elimination_trajectory,
)


st.set_page_config(
    page_title="Schistosomiasis Costing Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(f"Schistosomiasis Endgame Costing Tool v{APP_VERSION}")

try:
    country_inputs, espen_schisto = load_inputs()
except (FileNotFoundError, ImportError) as e:
    st.error(
        "Cannot load required datasets. Please ensure the following files exist in the datasets/ directory:\n"
        "  • consolidated_schisto.csv\n"
        "  • df_gdp.csv\n\n"
        "Also ensure schistosomiasis_data_uploader.py is in the same directory as this script.\n\n"
        f"Error: {e}"
    )
    render_setup_help()
    st.stop()

country_inputs = country_inputs.copy()
country_inputs["Country"] = country_inputs["Country"].astype(str)
country_dict = country_inputs.set_index("Country").T.to_dict()
country_list = sorted(country_inputs["Country"].dropna().astype(str).unique().tolist())

st.sidebar.markdown("## Schistosomiasis Costing Tool")
st.sidebar.markdown("---")
guided_mode = st.sidebar.checkbox(
    "Guided mode",
    value=False,
    help="Show a step-by-step workflow checklist and extra interpretation prompts inside the app.",
)

country = st.sidebar.selectbox("Select country", country_list, help=HELP_TEXT["country"])
espen_country = espen_schisto[espen_schisto["ADMIN0"] == country].copy()

country_flag = load_country_flag(country)
flag1, flag2 = st.columns([1,5], gap="medium")
with flag1:
    if country_flag is not None:
        st.image(country_flag)
    else:
        st.caption("Flag Unavailable")
with flag2:
    st.header(f"Economic Analysis for {country}")

if espen_country.empty:
    st.error(f"No ESPEN schistosomiasis rows were found for {country}.")
    st.stop()

st.sidebar.info("Select disease modules independently below.")

disease_choice = st.sidebar.selectbox(
    "Disease module",
    (
        "S. mansoni (intestinal)",
        "S. haematobium (urogenital)",
        "Both species",
    ),
    help=HELP_TEXT["disease_module"],
)
run_mansoni = "mansoni" in disease_choice.lower() or "both" in disease_choice.lower()
run_haematobium = "haematobium" in disease_choice.lower() or "both" in disease_choice.lower()

default_both = int(np.ceil(espen_country["sm_share_pct"].unique()[0]))
if run_mansoni and run_haematobium:
    both_species_mansoni_share = st.sidebar.slider(
        "Rows marked both species: default population share to mansoni module (%)",
        min_value=0,
        max_value=100,
        value=default_both,
        step=1,
        help=HELP_TEXT["both_species_split"],
    ) / 100.0
else:
    both_species_mansoni_share = 1.0 if run_mansoni else 0.0
both_species_haematobium_share = 1.0 - both_species_mansoni_share if (run_mansoni and run_haematobium) else (1.0 if run_haematobium else 0.0)

st.sidebar.markdown("#### Geographical unit")
country_key = _widget_scope_key(country)
admin1_list = sorted(espen_country["ADMIN1"].dropna().astype(str).unique().tolist())
admin1_list.insert(0, "National level")
admin1 = st.sidebar.selectbox(
    "Administrative Unit 1",
    admin1_list,
    key=f"admin1_{country_key}",
    help=HELP_TEXT["admin1"],
)

admin2 = "National level"
iu_sel = "National level"
if admin1 == "National level":
    espen_unit = espen_country.copy()
    unit_label = country
else:
    espen_adm1 = espen_country[espen_country["ADMIN1"] == admin1]
    admin2_list = sorted(espen_adm1["ADMIN2"].dropna().astype(str).unique().tolist())
    admin2_list.insert(0, f"All of {admin1}")
    admin2 = st.sidebar.selectbox(
        "Administrative Unit 2",
        admin2_list,
        key=f"admin2_{_widget_scope_key(country, admin1)}",
        help=HELP_TEXT["admin2"],
    )

    if admin2.startswith("All of"):
        espen_unit = espen_adm1.copy()
        unit_label = admin1
    else:
        espen_adm2 = espen_adm1[espen_adm1["ADMIN2"] == admin2]
        iu_list = sorted(espen_adm2["IUs_NAME"].dropna().astype(str).unique().tolist())
        iu_list.insert(0, f"All of {admin2}")
        iu_sel = st.sidebar.selectbox(
            "Implementing Unit",
            iu_list,
            key=f"iu_{_widget_scope_key(country, admin1, admin2)}",
            help=HELP_TEXT["iu"],
        )
        espen_unit = (
            espen_adm2.copy()
            if iu_sel.startswith("All of")
            else espen_adm2[espen_adm2["IUs_NAME"] == iu_sel].copy()
        )
        unit_label = admin2 if iu_sel.startswith("All of") else iu_sel

geo_scope_key = _widget_scope_key(country, admin1, admin2, iu_sel)

pop_req_mda = int(pd.to_numeric(espen_unit["PopReq"], errors="coerce").fillna(0.0).sum())
pop_trt_mda = int(pd.to_numeric(espen_unit["PopTreat"], errors="coerce").fillna(0.0).sum())
prev_sac = weighted_mean(espen_unit, "Prev_SAC", "PopReq", 0.0)
prev_adults = weighted_mean(espen_unit, "Prev_Adults", "PopReq", prev_sac)
mda_rounds = int(pd.to_numeric(espen_unit["Sch_MDA_Rounds"], errors="coerce").fillna(0.0).max())
mda_coverage_pct = round((pop_trt_mda / max(pop_req_mda, 1)) * 100.0, 1)

prog_exp = st.sidebar.expander("MDA programme parameters", expanded=True)
with prog_exp:
    default_cov = int(np.clip(round(mda_coverage_pct), 0, 100))
    mda_coverage = st.slider(
        "MDA coverage (%)",
        0,
        100,
        default_cov,
        key=f"mda_coverage_{geo_scope_key}",
        help=HELP_TEXT["mda_coverage"],
    )
    mda_frequency = st.radio(
        "MDA frequency",
        ("Annual", "Biennial"),
        index=0,
        key=f"mda_frequency_{geo_scope_key}",
        help=HELP_TEXT["mda_frequency"],
    )
    mda_target = st.radio(
        "MDA target population",
        ("SAC only", "SAC + at-risk adults"),
        index=1,
        key=f"mda_target_{geo_scope_key}",
        help=HELP_TEXT["mda_target"],
    )
    if mda_target == "SAC + at-risk adults":
        target_multiplier = st.number_input(
            "Target population multiplier vs SAC proxy",
            min_value=1.0,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key=f"target_multiplier_{geo_scope_key}",
            help=HELP_TEXT["target_multiplier"],
        )
    else:
        target_multiplier = 1.0
    disc_costs = st.number_input("Discount rate - costs", value=0.03, step=0.01, min_value=0.0, help=HELP_TEXT["disc_costs"])
    disc_effects = st.number_input("Discount rate - effects", value=0.03, step=0.01, min_value=0.0, help=HELP_TEXT["disc_effects"])
    time_horizon = st.slider("Time horizon (years)", 5, 30, 10, help=HELP_TEXT["time_horizon"])
    bia_horizon = st.slider(
        "Impact-analysis horizon (years)",
        min_value=3,
        max_value=int(time_horizon),
        value=min(5, int(time_horizon)),
        help=HELP_TEXT["bia_horizon"],
    )
    base_year = st.number_input("Impact-analysis start year", min_value=2020, max_value=2050, value=2026, step=1, help=HELP_TEXT["base_year"])
    biennial_effect_factor = st.slider(
            "Biennial annual-equivalent effect vs annual (%)",
            min_value=50,
            max_value=100,
            value=70,
            step=5,
            help=HELP_TEXT["biennial_effect"],
        ) / 100.0

    n_iterations = st.number_input("PSA iterations", min_value=100, max_value=10_000, value=DEFAULT_N_ITER, step=100, help=HELP_TEXT["psa_iterations"])
    seed = st.number_input("Simulation seed", value=DEFAULT_SEED, step=1, help=HELP_TEXT["seed"])
    auto_run_psa = st.checkbox("Auto-run PSA when inputs change", value=False, help=HELP_TEXT["auto_run_psa"])
    run_psa_now = st.button("Run / refresh PSA")

traj_exp = st.sidebar.expander("Prevalence trajectory assumptions", expanded=False)
with traj_exp:
    st.caption("These assumptions affect only the deterministic trajectory and cost-projection preview in the Results tab.")
    trajectory_horizon = st.slider(
        "Trajectory horizon (years)",
        min_value=5,
        max_value=30,
        value=min(10, int(time_horizon)),
        step=1,
        key=f"trajectory_horizon_{geo_scope_key}",
        help=HELP_TEXT["trajectory_horizon"],
    )
    st.markdown("#### Species-specific prevalence decay")
    st.caption("Decay values are annual-equivalent parameters at 75% annual MDA coverage. Defaults are transparent scenario assumptions, not fitted transmission estimates.")
    trajectory_m_decay_sac = DEFAULT_MANSONI_TRAJECTORY_DECAY_SAC
    trajectory_m_decay_adult = DEFAULT_MANSONI_TRAJECTORY_DECAY_ADULT
    trajectory_h_decay_sac = DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_SAC
    trajectory_h_decay_adult = DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_ADULT
    if run_mansoni:
        trajectory_m_decay_sac = st.number_input(
            "S. mansoni annual decay at 75% MDA - SAC",
            min_value=0.0,
            max_value=1.50,
            value=float(DEFAULT_MANSONI_TRAJECTORY_DECAY_SAC),
            step=0.01,
            key=f"trajectory_m_decay_sac_{geo_scope_key}",
            help=HELP_TEXT["trajectory_decay_sac"],
        )
        trajectory_m_decay_adult = st.number_input(
            "S. mansoni annual decay at 75% MDA - adults",
            min_value=0.0,
            max_value=1.50,
            value=float(DEFAULT_MANSONI_TRAJECTORY_DECAY_ADULT),
            step=0.01,
            key=f"trajectory_m_decay_adult_{geo_scope_key}",
            help=HELP_TEXT["trajectory_decay_adult"],
        )
    if run_haematobium:
        trajectory_h_decay_sac = st.number_input(
            "S. haematobium annual decay at 75% MDA - SAC",
            min_value=0.0,
            max_value=1.50,
            value=float(DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_SAC),
            step=0.01,
            key=f"trajectory_h_decay_sac_{geo_scope_key}",
            help=HELP_TEXT["trajectory_decay_sac"],
        )
        trajectory_h_decay_adult = st.number_input(
            "S. haematobium annual decay at 75% MDA - adults",
            min_value=0.0,
            max_value=1.50,
            value=float(DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_ADULT),
            step=0.01,
            key=f"trajectory_h_decay_adult_{geo_scope_key}",
            help=HELP_TEXT["trajectory_decay_adult"],
        )
    trajectory_floor_pct = st.number_input(
        "Residual transmission floor (%)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
        key=f"trajectory_floor_{geo_scope_key}",
        help=HELP_TEXT["trajectory_floor"],
    )
    trajectory_no_mda_change_pct = st.number_input(
        "No-MDA annual prevalence change (%)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        key=f"trajectory_no_mda_change_{geo_scope_key}",
        help=HELP_TEXT["trajectory_no_mda_change"],
    )
    trajectory_include_oscillations = st.checkbox(
        "Show illustrative campaign-cycle oscillations",
        value=False,
        key=f"trajectory_oscillations_{geo_scope_key}",
        help=HELP_TEXT["trajectory_oscillations"],
    )

# Sensitivity analysis controls
sens_exp = st.sidebar.expander("Sensitivity analysis", expanded=False)
with sens_exp:
    st.markdown("#### Coverage & frequency scenarios")
    sens_coverage_min = st.slider("Min coverage (%)", 40, 75, 50, step=5, key="sens_cov_min", help=HELP_TEXT["coverage_min"])
    sens_coverage_max = st.slider("Max coverage (%)", 75, 95, 85, step=5, key="sens_cov_max", help=HELP_TEXT["coverage_max"])
    sens_coverage_step = st.slider("Coverage step (pp)", 5, 20, 10, key="sens_cov_step", help=HELP_TEXT["coverage_step"])
    
    coverage_range_sens = list(range(sens_coverage_min, sens_coverage_max + 1, sens_coverage_step))
    
    st.markdown("#### Frequency scenarios")
    include_annual_sens = st.checkbox("Include annual MDA", value=True, key="sens_annual", help=HELP_TEXT["annual_sens"])
    include_biennial_sens = st.checkbox("Include biennial MDA", value=True, key="sens_biennial", help=HELP_TEXT["biennial_sens"])
    biennial_effect_sens = st.slider(
        "Biennial annual-equivalent effect vs annual (%)",
        50, 100, 70, step=5, key="sens_biennial_eff",
        help=HELP_TEXT["sens_biennial_effect"],
    ) / 100.0
    
    frequency_scenarios_sens = {}
    if include_annual_sens:
        frequency_scenarios_sens["Annual"] = 1.0
    if include_biennial_sens:
        frequency_scenarios_sens["Biennial"] = biennial_effect_sens
    
    run_sens_now = st.button("Run sensitivity analysis", key="run_sens")

cost_exp = st.sidebar.expander("Programme costs")
with cost_exp:
    pzq_unit_cost = st.number_input("PZQ tablet cost (USD)", value=0.08, step=0.01, min_value=0.0, help=HELP_TEXT["cost_pzq"])
    pzq_per_person = st.number_input("Tablets per treatment course", value=6.0, step=1.0, min_value=0.0, help=HELP_TEXT["cost_tablets"])
    delivery_cost = st.number_input("Delivery cost per person treated (USD)", value=0.50, step=0.05, min_value=0.0, help=HELP_TEXT["cost_delivery"])
    mapping_cost = st.number_input("Mapping / M&E annual cost (USD)", value=5_000.0, min_value=0.0, step=500.0, help=HELP_TEXT["cost_mapping"])
    training_cost = st.number_input("Training cost per MDA round (USD)", value=3_000.0, min_value=0.0, step=500.0, help=HELP_TEXT["cost_training"])
    supervision_cost = st.number_input("Supervision cost per MDA round (USD)", value=2_000.0, min_value=0.0, step=500.0, help=HELP_TEXT["cost_supervision"])
    other_prog_cost = st.number_input("Other annual programme costs (USD)", value=500_000.0, min_value=0.0, step=100_000.0, help=HELP_TEXT["cost_other"])

program_target_pop = int(round(pop_req_mda * float(target_multiplier)))
planned_pop_treat = int(round(program_target_pop * (mda_coverage / 100.0)))
pzq_drug_cost = pzq_unit_cost * pzq_per_person * planned_pop_treat
delivery_total = delivery_cost * planned_pop_treat
variable_prog_cost = pzq_drug_cost + delivery_total
fixed_prog_cost_mda_year = mapping_cost + training_cost + supervision_cost + other_prog_cost
programme_cost_mda_year = variable_prog_cost + fixed_prog_cost_mda_year
off_year_fixed_cost = fixed_prog_cost_mda_year * OFF_YEAR_FIXED_COST_SHARE
annualized_prog_cost = (
    programme_cost_mda_year
    if mda_frequency == "Annual"
    else (programme_cost_mda_year + off_year_fixed_cost) / 2.0
)
frequency_effect_factor = annual_equivalent_frequency_factor(mda_frequency, float(biennial_effect_factor))
mda_effect_multiplier = float(np.clip((mda_coverage / 100.0) * frequency_effect_factor, 0.0, 1.0))

country_row = country_dict[country]
econ_exp = st.sidebar.expander("Country economic parameters")
with econ_exp:
    annual_ppp = st.number_input("Per capita GDP PPP (Int$)", value=float(country_row["Annual_PPP(Int$)"]), help=HELP_TEXT["annual_ppp"])
    q1_share = st.number_input(
        "Bottom quintile income share (%)",
        value=float(country_row["inequality_.2_quintile"]),
        help=HELP_TEXT["q1_share"],
    ) / 100.0
    weekly_hrs = st.number_input(
        "Weekly work hours",
        min_value=1.0,
        max_value=100.0,
        value=float(country_row["Weekly_Work_Hours"]),
        help=HELP_TEXT["weekly_hours"],
    )
    life_exp = st.number_input("Life expectancy (years)", value=float(country_row["Life_Expectancy"]), help=HELP_TEXT["life_exp"])
    opd_cost_base = st.number_input("OPD unit cost USD (base year)", value=float(country_row["Primary Hospital_OPD_Costs"]), help=HELP_TEXT["opd_cost"])
    ipd_cost_base = st.number_input("IPD unit cost USD (base year)", value=float(country_row["Primary Hospital_IPD_Costs"]), help=HELP_TEXT["ipd_cost"])

daily_wage_adj = adj_daily_wage(annual_ppp, q1_share, weekly_hrs)
inflation = float(country_row["Inflation rate (consumer prices) (%)"])
med_inflation = (inflation + 3.0) / 100.0
opd_cost_curr = opd_cost_base * (1.0 + med_inflation) ** 10
ipd_cost_curr = ipd_cost_base * (1.0 + med_inflation) ** 10
cet = cea_threshold(annual_ppp)

budget_resource_exp = st.sidebar.expander("Budget impact resource inputs", expanded=False)
with budget_resource_exp:
    st.caption("These inputs drive budget offsets from health states averted. Productivity benefits remain excluded from budget impact.")
    opd_staff_minutes = st.number_input(
        "Staff minutes per OPD visit averted",
        min_value=0.0,
        value=15.0,
        step=5.0,
        key="bia_opd_staff_minutes",
        help=HELP_TEXT["staff_minutes_opd"],
    )
    ipd_staff_minutes = st.number_input(
        "Staff minutes per IPD bed-day averted",
        min_value=0.0,
        value=120.0,
        step=15.0,
        key="bia_ipd_staff_minutes",
        help=HELP_TEXT["staff_minutes_ipd"],
    )
    staff_hourly_cost = st.number_input(
        "Staff cost per hour (USD, optional)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="bia_staff_hourly_cost",
        help=HELP_TEXT["staff_hourly_cost"],
    )
    include_staff_time_value = st.checkbox(
        "Include monetized staff time in net budget impact",
        value=False,
        key="bia_include_staff_value",
        help=HELP_TEXT["include_staff_value"],
    )
    st.markdown("###### Additional direct management cost per case averted")
    st.caption("Use these for costs not captured by the OPD/IPD unit costs, such as diagnostics, medicines, procedures, or oncology care.")
    health_state_costs = {
        "mansoni_anemia": st.number_input("Anemia (USD/case)", min_value=0.0, value=0.0, step=1.0, key="cost_mansoni_anemia"),
        "mansoni_hepatomegaly": st.number_input("Hepatomegaly (USD/case)", min_value=0.0, value=0.0, step=1.0, key="cost_mansoni_hepatomegaly"),
        "mansoni_fibrosis": st.number_input("Periportal fibrosis (USD/case)", min_value=0.0, value=0.0, step=1.0, key="cost_mansoni_fibrosis"),
        "mansoni_portal_htn": st.number_input("Portal hypertension (USD/case)", min_value=0.0, value=0.0, step=1.0, key="cost_mansoni_portal_htn"),
        "mansoni_varices": st.number_input("Esophageal varices (USD/case)", min_value=0.0, value=0.0, step=1.0, key="cost_mansoni_varices"),
        "haematobium_hematuria": st.number_input("Hematuria (USD/case)", min_value=0.0, value=0.0, step=1.0, key="cost_haematobium_hematuria"),
        "haematobium_hydronephrosis": st.number_input("Hydronephrosis (USD/case)", min_value=0.0, value=0.0, step=1.0, key="cost_haematobium_hydronephrosis"),
        "haematobium_fgs": st.number_input("Female genital schistosomiasis (USD/case)", min_value=0.0, value=0.0, step=1.0, key="cost_haematobium_fgs"),
        "haematobium_bladder_cancer": st.number_input("Attributable bladder cancer (USD/case)", min_value=0.0, value=0.0, step=10.0, key="cost_haematobium_bladder_cancer"),
    }

m_defaults = partitioned_species_defaults(espen_unit, "mansoni", both_species_mansoni_share)
h_defaults = partitioned_species_defaults(espen_unit, "haematobium", both_species_mansoni_share)

if run_mansoni and m_defaults["pop_req"] <= 0:
    st.sidebar.warning("No allocated mansoni population was found for this unit. Use manual inputs with care.")
if run_haematobium and h_defaults["pop_req"] <= 0:
    st.sidebar.warning("No allocated haematobium population was found for this unit. Use manual inputs with care.")

m_pop_req_base = float(m_defaults["pop_req"])
h_pop_req_base = float(h_defaults["pop_req"])
m_prev_sac_default = float(m_defaults["prev_sac"])
h_prev_sac_default = float(h_defaults["prev_sac"])
m_prev_adult_default = float(m_defaults["prev_adults"])
h_prev_adult_default = float(h_defaults["prev_adults"])
m_prev_default = effective_prevalence(m_prev_sac_default, m_prev_adult_default, target_multiplier)
h_prev_default = effective_prevalence(h_prev_sac_default, h_prev_adult_default, target_multiplier)
m_pop_default = m_pop_req_base * float(target_multiplier)
h_pop_default = h_pop_req_base * float(target_multiplier)

disease_input_scope_key = _widget_scope_key(
    geo_scope_key,
    disease_choice,
    f"target_{float(target_multiplier):.3f}",
    f"both_mansoni_{float(both_species_mansoni_share):.3f}",
)

render_guided_sidebar(
    enabled=guided_mode,
    country=country,
    unit_label=unit_label,
    disease_choice=disease_choice,
    pop_req_mda=pop_req_mda,
    planned_pop_treat=planned_pop_treat,
    mda_coverage=mda_coverage,
    time_horizon=int(time_horizon),
    bia_horizon=int(bia_horizon),
    n_iterations=int(n_iterations),
)


# =============================================================================
# TABS
# =============================================================================

tabs = st.tabs(
    [
        "About",
        "User Guide",
        "Country inputs",
        "Disease inputs",
        "Results",
        "Economic impact",
        "Budget impact",
        "Elimination Projections",
        "Technical assumptions",
        "Contact",
    ]
)

with tabs[0]:
    st.markdown(
        """
        ### Welcome to the Schistosomiasis Endgame Costing Tool

        This tool supports national NTD programme managers, health economists,
        and policymakers in generating economic and budget evidence for schistosomiasis MDA
        programmes across sub-Saharan Africa.

        #### What the tool does
        - Estimates burden for S. mansoni and S. haematobium using
          ESPEN surveillance data.
        - Runs a Monte Carlo probabilistic sensitivity analysis.
        - Computes DALYs, ICERs, productivity losses, health sector costs, ROI,
          economic impact, and programme budget requirements.
        - Projects time-varying caseloads and costs using transparent prevalence scenarios.
        - Sensitivity analysis showing trade-offs between coverage, frequency, and outcomes.
        """
    )

with tabs[1]:
    render_user_guide_tab()

with tabs[2]:
    if guided_mode:
        render_tab_tip("country")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"Selected unit: {unit_label}")
        st.metric("Population requiring MDA", f"{pop_req_mda:,.0f}")
        st.metric("Target population used for costing", f"{program_target_pop:,.0f}")
        st.metric("Planned population treated", f"{planned_pop_treat:,.0f}")
        st.metric("Last recorded MDA coverage", f"{mda_coverage_pct:.1f}%")
        st.metric("Selected MDA coverage", f"{mda_coverage:.0f}%")
        st.metric("Weighted SAC prevalence", f"{prev_sac:.1f}%")
        st.metric("Weighted adult prevalence", f"{prev_adults:.1f}%")

    with col2:
        st.subheader("Economic parameters")
        econ_display = pd.DataFrame(
            {
                "Parameter": [
                    "Per capita GDP PPP (Int$)",
                    "Bottom quintile income share",
                    "Inequality-adjusted daily wage (USD)",
                    "Weekly work hours",
                    "Life expectancy (years)",
                    "OPD unit cost - current USD",
                    "IPD unit cost - current USD",
                    "CEA threshold - Woods et al. (USD/DALY)",
                ],
                "Value": [
                    f"{annual_ppp:,.0f}",
                    f"{q1_share * 100:.1f}%",
                    f"${daily_wage_adj:.2f}",
                    f"{weekly_hrs:.0f}",
                    f"{life_exp:.0f}",
                    f"${opd_cost_curr:.2f}",
                    f"${ipd_cost_curr:.2f}",
                    f"${cet:,.0f}",
                ],
            }
        )
        st.table(econ_display)

    with st.expander("Click here to view ESPEN unit-level data for 2024"):
        st.caption(
            "All programme denominators and disease-input defaults below are calculated from the rows shown here for the selected administrative unit."
        )
        st.dataframe(espen_unit.reset_index(drop=True), width="stretch")

    with st.expander("Programme cost breakdown"):
        prog_cost_display = pd.DataFrame(
            {
                "Cost category": [
                    "PZQ drug cost",
                    "Delivery cost",
                    "Mapping / M&E",
                    "Training",
                    "Supervision",
                    "Other",
                    "MDA-year total",
                    "Off-year fixed cost (biennial only)",
                    "Annualized programme cost used in ICER/ROI",
                ],
                "USD": [
                    pzq_drug_cost,
                    delivery_total,
                    mapping_cost,
                    training_cost,
                    supervision_cost,
                    other_prog_cost,
                    programme_cost_mda_year,
                    off_year_fixed_cost,
                    annualized_prog_cost,
                ],
            }
        )
        st.table(prog_cost_display.style.format({"USD": "${:,.0f}"}))

with tabs[3]:
    if guided_mode:
        render_tab_tip("disease")
    if run_mansoni:
        st.subheader("S. mansoni - disease inputs")
        col1, col2 = st.columns(2)
        with col1:
            m_prev = st.number_input(
                "S. mansoni effective prevalence (%)",
                value=float(round(m_prev_default, 1)),
                min_value=0.0,
                max_value=100.0,
                key=f"m_prev_{disease_input_scope_key}",
                help=HELP_TEXT["m_prev"],
            )
            m_pop = st.number_input(
                "At-risk population (S. mansoni)",
                value=float(max(m_pop_default, 0.0)),
                min_value=0.0,
                key=f"m_pop_{disease_input_scope_key}",
                help=HELP_TEXT["m_pop"],
            )
            m_heavy_pct = st.slider("% heavy infection (>=400 EPG)", 10, 70, 40, key="m_heavy", help=HELP_TEXT["m_heavy"])
        with col2:
            m_hepatomeg_pct = st.slider("% heavy infection -> hepatomegaly", 5, 50, 22, key="m_hep", help=HELP_TEXT["m_hepatomeg"])
            m_morbidity_red = st.slider("Hepatic morbidity reduction with MDA (%)", 30, 90, 70, key="m_mrh", help=HELP_TEXT["m_morbidity_red"])
            m_cure = st.slider("PZQ cure rate - S. mansoni (%)", 60, 98, 85, key="m_cure", help=HELP_TEXT["m_cure"])

        st.caption(
            f"Default denominator for this selected unit: {m_pop_default:,.0f}. "
            "Changing the administrative unit now resets this field to the new unit-specific default."
        )

        m_params = MansoniInputs(
            at_risk_pop=m_pop,
            pct_heavy=m_heavy_pct / 100.0,
            pct_hepatomegaly=m_hepatomeg_pct / 100.0,
            morbidity_reduction_hepatic=m_morbidity_red / 100.0,
            cure_rate=m_cure / 100.0,
        )
        m_caseloads = estimate_caseloads_mansoni(m_pop, m_prev, m_params)

        with st.expander("Estimated baseline caseloads (S. mansoni)"):
            cl_df = pd.DataFrame(
                [
                    {
                        "Infected": m_caseloads["infected"],
                        "Anemia": m_caseloads["anemia"],
                        "Hepatomegaly": m_caseloads["hepatomegaly"],
                        "Periportal fibrosis": m_caseloads["fibrosis"],
                        "Portal hypertension": m_caseloads["portal_htn"],
                        "Esophageal varices": m_caseloads["varices"],
                    }
                ]
            )
            st.dataframe(cl_df.style.format("{:,.0f}"), width="stretch")
    else:
        m_params = MansoniInputs()
        m_prev = 0.0
        m_pop = 0.0

    if run_haematobium:
        st.markdown("---")
        st.subheader("S. haematobium - disease inputs")
        col1, col2 = st.columns(2)
        with col1:
            h_prev = st.number_input(
                "S. haematobium effective prevalence (%)",
                value=float(round(h_prev_default, 1)),
                min_value=0.0,
                max_value=100.0,
                key=f"h_prev_{disease_input_scope_key}",
                help=HELP_TEXT["h_prev"],
            )
            h_pop = st.number_input(
                "At-risk population (S. haematobium)",
                value=float(max(h_pop_default, 0.0)),
                min_value=0.0,
                key=f"h_pop_{disease_input_scope_key}",
                help=HELP_TEXT["h_pop"],
            )
            h_female_pct = st.slider("% female in at-risk population", 30, 70, 50, key="h_female", help=HELP_TEXT["h_female"])
        with col2:
            h_bg_cancer = st.number_input("Background bladder cancer rate (per 100,000)", value=3.5, min_value=0.0, key="h_bg_ca", help=HELP_TEXT["h_bg_cancer"])
            h_morbidity_red = st.slider("Urinary morbidity reduction with MDA (%)", 30, 90, 75, key="h_mru", help=HELP_TEXT["h_morbidity_red"])
            h_cure = st.slider("PZQ cure rate - S. haematobium (%)", 65, 98, 87, key="h_cure", help=HELP_TEXT["h_cure"])
            h_cancer_red = st.slider("Attributable bladder cancer risk reduction with MDA (%)", 0, 80, 30, key="h_crm", help=HELP_TEXT["h_cancer_red"])

        st.caption(
            f"Default denominator for this selected unit: {h_pop_default:,.0f}. "
            "Changing the administrative unit now resets this field to the new unit-specific default."
        )

        with st.expander("Bladder cancer fatality and survival assumptions"):
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                h_cfr_primary = st.slider("Primary-stage cancer case fatality (%)", 0, 100, 35, key="h_cfr_primary")
                h_surv_primary = st.number_input(
                    "Primary-stage cancer survival duration (years)",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    key="h_surv_primary",
                )
            with ccol2:
                h_cfr_meta = st.slider("Metastatic cancer case fatality (%)", 0, 100, 85, key="h_cfr_meta")
                h_surv_meta = st.number_input(
                    "Metastatic cancer survival duration (years)",
                    min_value=0.0,
                    max_value=20.0,
                    value=1.5,
                    step=0.5,
                    key="h_surv_meta",
                )

        h_params = HaematobiumInputs(
            at_risk_pop=h_pop,
            female_fraction=h_female_pct / 100.0,
            bg_bladder_cancer_rate=h_bg_cancer,
            morbidity_reduction_urinary=h_morbidity_red / 100.0,
            cure_rate=h_cure / 100.0,
            cancer_reduction_mda=h_cancer_red / 100.0,
            cfr_primary=h_cfr_primary / 100.0,
            cfr_meta=h_cfr_meta / 100.0,
            cancer_survival_primary=h_surv_primary,
            cancer_survival_meta=h_surv_meta,
        )
        h_caseloads = estimate_caseloads_haematobium(h_pop, h_prev, h_params, h_female_pct / 100.0)

        with st.expander("Estimated baseline caseloads (S. haematobium)"):
            cl_h_df = pd.DataFrame(
                [
                    {
                        "Infected": h_caseloads["infected"],
                        "Hematuria": h_caseloads["hematuria"],
                        "Hydronephrosis": h_caseloads["hydronephrosis"],
                        "FGS": h_caseloads["fgs"],
                        "Bladder cancer (all-cause observed rate)": h_caseloads["bladder_cancer_total"],
                        "Bladder cancer (non-attributable)": h_caseloads["bladder_cancer_nonattributable"],
                        "Bladder cancer (S. haematobium-attributable)": h_caseloads["bladder_cancer_attributable"],
                        "PAF (%)": h_caseloads["paf"] * 100.0,
                    }
                ]
            )
            st.dataframe(
                cl_h_df.style.format(
                    {
                        "Infected": "{:,.0f}",
                        "Hematuria": "{:,.0f}",
                        "Hydronephrosis": "{:,.0f}",
                        "FGS": "{:,.0f}",
                        "Bladder cancer (all-cause observed rate)": "{:,.2f}",
                        "Bladder cancer (non-attributable)": "{:,.2f}",
                        "Bladder cancer (S. haematobium-attributable)": "{:,.2f}",
                        "PAF (%)": "{:.2f}%",
                    }
                ),
                width="stretch",
            )
    else:
        h_params = HaematobiumInputs()
        h_prev = 0.0
        h_pop = 0.0

# Run PSA
scenario_signature = repr(
    (
        geo_scope_key, disease_input_scope_key, run_mansoni, run_haematobium,
        float(m_pop), float(m_prev), m_params,
        float(h_pop), float(h_prev), h_params, float(life_exp),
        int(n_iterations), float(mda_effect_multiplier), int(seed),
    )
)
scenario_changed = st.session_state.get("scenario_signature") != scenario_signature
should_run_psa = run_psa_now or (auto_run_psa and scenario_changed) or not ("scenario_signature" in st.session_state)

if should_run_psa:
    with st.spinner("Running probabilistic sensitivity analysis..."):
        sim_m_new = None
        sim_h_new = None
        if run_mansoni and m_pop > 0 and m_prev > 0:
            sim_m_new = add_daly_averted_columns(
                run_monte_carlo_mansoni(
                    int(n_iterations), m_pop, m_prev, m_params, mda_effect_multiplier, int(seed)
                )
            )
        if run_haematobium and h_pop > 0 and h_prev > 0:
            sim_h_new = add_daly_averted_columns(
                run_monte_carlo_haematobium(
                    int(n_iterations), h_pop, h_prev, h_female_pct / 100.0, life_exp,
                    h_params, mda_effect_multiplier, int(seed) + 10_007,
                )
            )
        st.session_state["sim_m"] = sim_m_new
        st.session_state["sim_h"] = sim_h_new
        st.session_state["scenario_signature"] = scenario_signature

sim_m = st.session_state.get("sim_m")
sim_h = st.session_state.get("sim_h")

with tabs[4]:
    if guided_mode:
        render_tab_tip("results")
    if sim_m is None and sim_h is None:
        st.info("Set disease prevalence and population in the Disease inputs tab to generate results.")
    else:
        st.subheader("DALY burden analysis")
        st.caption("Ranges shown in the results are 95% PSA intervals from the 2.5th and 97.5th percentiles of simulation draws.")
        
        for label, sim_df, species_tag in [
            ("S. mansoni", sim_m, "mansoni"),
            ("S. haematobium", sim_h, "haematobium"),
        ]:
            if sim_df is None:
                continue
            with st.expander(f"{label} - DALY breakdown"):
                daly_df = daly_summary_table(sim_df, species_tag)
                fmt_cols = {c: "{:,.1f}" for c in daly_df.columns if daly_df[c].dtype == float}
                st.dataframe(daly_df.style.format(fmt_cols), width="stretch")
                st.altair_chart(plot_daly_breakdown(daly_df), width="stretch")

        if sim_h is not None:
            cancer_case_df = bladder_cancer_case_summary_table(sim_h)
            if not cancer_case_df.empty:
                with st.expander("S. haematobium - bladder cancer case impact", expanded=True):
                    st.caption(
                        "Bladder cancer incidence is treated as an observed all-cause rate. "
                        "The Levin PAF estimates the S. haematobium-attributable component; MDA reduces only that attributable component."
                    )
                    st.dataframe(
                        cancer_case_df.style.format(
                            {
                                "No-MDA mean": "{:,.2f}",
                                "No-MDA 95% CI lower": "{:,.2f}",
                                "No-MDA 95% CI upper": "{:,.2f}",
                                "MDA mean": "{:,.2f}",
                                "MDA 95% CI lower": "{:,.2f}",
                                "MDA 95% CI upper": "{:,.2f}",
                                "Cases averted": "{:,.2f}",
                                "Cases averted 95% CI lower": "{:,.2f}",
                                "Cases averted 95% CI upper": "{:,.2f}",
                                "Mean % reduction": "{:.1%}",
                                "Mean % reduction 95% CI lower": "{:.1%}",
                                "Mean % reduction 95% CI upper": "{:.1%}",
                            }
                        ),
                        width="stretch",
                    )

        st.markdown("---")
        st.subheader("Deterministic prevalence trajectory and health-sector cost projection")
        st.caption(
            "This section is a transparent scenario projection for costing. It is not a calibrated transmission model or a direct SCHISTOX simulation. "
            "The no-MDA comparator is constant prevalence by default, unless changed in the sidebar trajectory assumptions."
        )
        if guided_mode:
            st.info(
                "Reviewer-facing interpretation: the chart tests how the selected MDA coverage, frequency, adult targeting, decay assumptions, "
                "and residual transmission floor affect projected prevalence and caseloads. Use the Elimination Projections tab for WHO EPHP target analysis."
            )

        projection_years = np.arange(0, int(trajectory_horizon) + 1, 1)

        def _render_trajectory_block(
            *,
            label: str,
            species_key: str,
            at_risk_pop: float,
            effective_prev: float,
            default_prev_sac: float,
            default_prev_adult: float,
            caseload_fn,
            caseload_params,
            psa_df,
            trajectory_decay_sac_value: float,
            trajectory_decay_adult_value: float,
            expanded: bool = False,
        ) -> None:
            initial_sac, initial_adult = derive_projection_prevalence_inputs(
                effective_prev,
                default_prev_sac,
                default_prev_adult,
                target_multiplier,
            )
            prev_scenarios = build_prevalence_scenario_df(
                years=projection_years,
                initial_prev_sac=initial_sac,
                initial_prev_adult=initial_adult,
                coverage_pct=mda_coverage,
                frequency=mda_frequency,
                target_multiplier=target_multiplier,
                treat_adults=(mda_target != "SAC only"),
                annual_decay_sac_at_reference=trajectory_decay_sac_value,
                annual_decay_adult_at_reference=trajectory_decay_adult_value,
                no_mda_annual_change_pct=trajectory_no_mda_change_pct,
                floor_pct=trajectory_floor_pct,
                include_oscillations=trajectory_include_oscillations,
                species=species_key,
                frequency_effect_factor=frequency_effect_factor,
            )
            cl_with_mda = project_caseloads_from_prevalence_scenario(
                prev_scenarios,
                "MDA scenario",
                at_risk_pop,
                caseload_fn,
                caseload_params,
            )
            cl_no_mda = project_caseloads_from_prevalence_scenario(
                prev_scenarios,
                "No-MDA counterfactual",
                at_risk_pop,
                caseload_fn,
                caseload_params,
            )

            with st.expander(f"{label}: prevalence, caseload, and cost trajectory", expanded=expanded):
                st.markdown("#### Prevalence trajectory")
                st.altair_chart(plot_prevalence_trajectory(prev_scenarios), width="stretch")
                st.caption(
                    f"Starting prevalence split used for the trajectory: SAC {initial_sac:.1f}%, adults {initial_adult:.1f}%. "
                    f"The combined line is weighted using the target-population multiplier ({target_multiplier:.1f}). "
                    f"The frequency effect factor applied to the MDA trajectory is {frequency_effect_factor:.2f}."
                )
                st.caption(
                    "Note: PSA DALYs and ICERs use an annual steady-state effect at the selected coverage/frequency. "
                    "This trajectory is a separate time-path preview for prevalence, caseloads, and health-sector costs, so it should not be used to recalculate the ICER directly."
                )

                if cl_with_mda.empty or cl_no_mda.empty:
                    st.info("No trajectory caseloads were generated for this module.")
                    return

                end_idx = -1
                infected_start = float(cl_with_mda.iloc[0].get("infected", 0.0))
                infected_mda_end = float(cl_with_mda.iloc[end_idx].get("infected", 0.0))
                infected_no_mda_end = float(cl_no_mda.iloc[end_idx].get("infected", 0.0))
                cases_averted_end = infected_no_mda_end - infected_mda_end

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Infected at year 0", f"{infected_start:,.0f}")
                col2.metric(f"MDA infected at year {int(trajectory_horizon)}", f"{infected_mda_end:,.0f}")
                col3.metric(f"No-MDA infected at year {int(trajectory_horizon)}", f"{infected_no_mda_end:,.0f}")
                col4.metric(f"Cases averted by year {int(trajectory_horizon)}", f"{cases_averted_end:,.0f}")

                hs_costs_with_mda = project_health_sector_costs(
                    cl_with_mda, psa_df, opd_cost_curr, ipd_cost_curr,
                    discount_rate=disc_costs, species=species_key,
                )
                hs_costs_no_mda = project_health_sector_costs(
                    cl_no_mda, psa_df, opd_cost_curr, ipd_cost_curr,
                    discount_rate=disc_costs, species=species_key,
                )
                hs_trajectory = pd.DataFrame({
                    "Year": hs_costs_with_mda["Year"],
                    "HS_cost_no_mda": hs_costs_no_mda["HS_cost_discounted_USD"],
                    "HS_cost_with_mda": hs_costs_with_mda["HS_cost_discounted_USD"],
                })

                st.markdown("#### Health-sector cost trajectory")
                st.altair_chart(plot_health_sector_cost_trajectory(hs_trajectory), width="stretch")

                with st.expander("View trajectory data table"):
                    st.dataframe(prev_scenarios.style.format({
                        "Year": "{:.0f}",
                        "Prev_SAC": "{:.2f}",
                        "Prev_Adult": "{:.2f}",
                        "Prev_Combined": "{:.2f}",
                        "MDA_coverage_pct": "{:.0f}",
                        "Adult_coverage_pct": "{:.0f}",
                        "SAC_MDA_intensity": "{:.2f}",
                        "Adult_MDA_intensity": "{:.2f}",
                        "Frequency_effect_factor": "{:.2f}",
                        "SAC_decay_at_reference": "{:.3f}",
                        "Adult_decay_at_reference": "{:.3f}",
                        "No_MDA_annual_change_pct": "{:.1f}",
                        "Residual_floor_pct": "{:.1f}",
                    }), width="stretch")

                safe_species = species_key.lower().replace(" ", "_")
                download_df(prev_scenarios, f"Download {label} prevalence trajectory (CSV)", f"{safe_species}_prevalence_trajectory.csv")
                download_df(cl_with_mda, f"Download {label} MDA caseload trajectory (CSV)", f"{safe_species}_caseloads_with_mda.csv")
                download_df(cl_no_mda, f"Download {label} no-MDA caseload comparator (CSV)", f"{safe_species}_caseloads_no_mda.csv")

        if run_mansoni and m_pop > 0 and m_prev > 0 and sim_m is not None:
            _render_trajectory_block(
                label="S. mansoni",
                species_key="mansoni",
                at_risk_pop=m_pop,
                effective_prev=m_prev,
                default_prev_sac=m_prev_sac_default,
                default_prev_adult=m_prev_adult_default,
                caseload_fn=estimate_caseloads_mansoni,
                caseload_params=m_params,
                psa_df=sim_m,
                trajectory_decay_sac_value=trajectory_m_decay_sac,
                trajectory_decay_adult_value=trajectory_m_decay_adult,
                expanded=True,
            )

        if run_haematobium and h_pop > 0 and h_prev > 0 and sim_h is not None:
            _render_trajectory_block(
                label="S. haematobium",
                species_key="haematobium",
                at_risk_pop=h_pop,
                effective_prev=h_prev,
                default_prev_sac=h_prev_sac_default,
                default_prev_adult=h_prev_adult_default,
                caseload_fn=lambda pop, prev, params: estimate_caseloads_haematobium(pop, prev, params, h_female_pct / 100.0),
                caseload_params=h_params,
                psa_df=sim_h,
                trajectory_decay_sac_value=trajectory_h_decay_sac,
                trajectory_decay_adult_value=trajectory_h_decay_adult,
                expanded=not (run_mansoni and sim_m is not None),
            )

        st.subheader("Productivity losses")
        for label, sim_df in [("S. mansoni", sim_m), ("S. haematobium", sim_h)]:
            if sim_df is None:
                continue
            prod = productivity_summary(sim_df, daily_wage_adj, time_horizon, disc_effects)
            with st.expander(f"{label} - productivity"):
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Workdays lost p.a. (no MDA)",
                    f"{prod['mean_no']:,.0f}",
                    help=f"{CI_INTERVAL_LABEL}: " + _format_ci(prod["mean_no"], prod["lo_no"], prod["hi_no"]),
                )
                col2.metric(
                    "Workdays lost p.a. (MDA)",
                    f"{prod['mean_mda']:,.0f}",
                    help=f"{CI_INTERVAL_LABEL}: " + _format_ci(prod["mean_mda"], prod["lo_mda"], prod["hi_mda"]),
                )
                col3.metric(
                    "Productivity days gained p.a.",
                    f"{prod['days_gained']:,.0f}",
                    help=f"{CI_INTERVAL_LABEL}: " + _format_ci(prod["days_gained"], prod["days_gained_lo"], prod["days_gained_hi"]),
                )
                st.write(
                    f"Estimated annual economic gain from MDA: **\\${prod['econ_gain_pa']:,.0f}** "
                    f"[{prod['econ_gain_pa_lo']:,.0f}, {prod['econ_gain_pa_hi']:,.0f}] p.a.; "
                    f"discounted over {time_horizon} yr: **\\${prod['econ_gain_disc']:,.0f}** "
                    f"[{prod['econ_gain_disc_lo']:,.0f}, {prod['econ_gain_disc_hi']:,.0f}]."
                )
                st.dataframe(
                    productivity_result_table(prod).style.format(
                        {"Mean": "{:,.1f}", "95% CI lower": "{:,.1f}", "95% CI upper": "{:,.1f}"}
                    ),
                    width="stretch",
                )

        st.subheader("Health sector costs")
        total_hs_savings = 0.0
        total_econ_gain = 0.0
        for label, sim_df in [("S. mansoni", sim_m), ("S. haematobium", sim_h)]:
            if sim_df is None:
                continue
            hsc = health_sector_costs(sim_df, opd_cost_curr, ipd_cost_curr, time_horizon, disc_costs)
            prod = productivity_summary(sim_df, daily_wage_adj, time_horizon, disc_effects)
            total_hs_savings += hsc["hs_savings_pa"]
            total_econ_gain += prod["econ_gain_pa"]
            with st.expander(f"{label} - health sector costs"):
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Annual HS cost (no MDA)",
                    f"${hsc['hs_cost_no']:,.0f}",
                    help=f"{CI_INTERVAL_LABEL}: " + _format_ci(hsc["hs_cost_no"], hsc["hs_cost_no_lo"], hsc["hs_cost_no_hi"], fmt=",.0f"),
                )
                c2.metric(
                    "Annual HS cost (MDA)",
                    f"${hsc['hs_cost_mda']:,.0f}",
                    help=f"{CI_INTERVAL_LABEL}: " + _format_ci(hsc["hs_cost_mda"], hsc["hs_cost_mda_lo"], hsc["hs_cost_mda_hi"], fmt=",.0f"),
                )
                c3.metric(
                    "Annual HS savings",
                    f"${hsc['hs_savings_pa']:,.0f}",
                    help=f"{CI_INTERVAL_LABEL}: " + _format_ci(hsc["hs_savings_pa"], hsc["hs_savings_lo"], hsc["hs_savings_hi"], fmt=",.0f"),
                )
                st.caption(
                    f"Discounted savings over {time_horizon} yr: ${hsc['hs_savings_disc']:,.0f} "
                    f"[{hsc['hs_savings_disc_lo']:,.0f}, {hsc['hs_savings_disc_hi']:,.0f}]."
                )
                st.dataframe(
                    health_sector_result_table(hsc).style.format(
                        {"Mean": "${:,.0f}", "95% CI lower": "${:,.0f}", "95% CI upper": "${:,.0f}"}
                    ),
                    width="stretch",
                )

        st.subheader("Cost-effectiveness")
        st.caption(
            "CEA update active: the tables and downloads below include actual draw-level DALYs averted, "
            "draw-level programme cost minus health-sector savings, CEACs, and CE-plane plots."
        )
        st.caption(
            "ICER, CEAC, and CE-plane outputs use programme cost minus draw-level health-sector savings. "
            "The draw-level table still shows annual programme cost and health-sector savings separately."
        )
        combined_df = build_combined_daly_df(sim_m, sim_h)
        if not combined_df.empty:
            combined_ce_df = add_cost_effectiveness_columns(
                combined_df,
                annualized_prog_cost,
                opd_cost_curr,
                ipd_cost_curr,
            )
            combined_icer = compute_icer(
                combined_ce_df,
                annualized_prog_cost,
                annual_ppp,
                cet,
                incremental_cost_draws=combined_ce_df["program_cost_minus_savings"],
            )
            with st.expander("Combined programme ICER", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Computed DALYs averted p.a.",
                    f"{combined_icer['dalys_averted_point']:,.0f}",
                    help=(
                        format_daly_averted_calculation(combined_icer)
                        + " 95% PSA interval for draw-level DALYs averted: "
                        + _format_ci_or_na(
                            combined_icer["dalys_averted_point"],
                            combined_icer["dalys_averted_lo"],
                            combined_icer["dalys_averted_hi"],
                        )
                    ),
                )
                c2.metric(
                    "ICER (USD/DALY)",
                    "NA" if not np.isfinite(combined_icer["icer_mean"]) else f"${combined_icer['icer_mean']:,.0f}",
                    help=_format_ci_or_na(
                        combined_icer["icer_mean"],
                        combined_icer["icer_lo"],
                        combined_icer["icer_hi"],
                        fmt=",.0f",
                        na_text="No positive DALYs averted.",
                    ),
                )
                if combined_icer["positive_daly_draws"] == 0:
                    st.warning("No PSA iterations produced positive DALYs averted, so the ICER is not evaluable. The CEAC/NMB probability remains valid and is shown as zero unless net monetary benefit is positive.")
                elif combined_icer["nonpositive_daly_draws"] > 0:
                    st.caption(
                        f"{combined_icer['nonpositive_daly_draws']:,} of {combined_icer['total_draws']:,} PSA draws had zero or negative DALYs averted; cost-effectiveness probabilities are calculated using NMB across all draws."
                    )
                c3.metric(
                    "% prob. cost-effective (Woods CET)",
                    "NA" if not np.isfinite(combined_icer["pct_cost_effective_cet"]) else f"{combined_icer['pct_cost_effective_cet'] * 100:.0f}%",
                    help=_format_ci_or_na(
                        combined_icer.get("pct_cost_effective_cet", np.nan),
                        combined_icer.get("pct_cost_effective_cet_lo", np.nan),
                        combined_icer.get("pct_cost_effective_cet_hi", np.nan),
                        fmt=".1%",
                        na_text="NA",
                    ),
                )
                st.dataframe(
                    cost_effectiveness_summary_table(combined_icer).style.format(
                        {"Mean": "{:,.1f}", "95% CI lower": "{:,.1f}", "95% CI upper": "{:,.1f}"}
                    ),
                    width="stretch",
                )
                st.caption(format_daly_averted_calculation(combined_icer))
                st.write(
                    f"The Woods et al. CEA threshold for {country} is **\\${cet:,.0f}** per DALY averted. "
                    f"The GDP-based 1x threshold is **\\${annual_ppp:,.0f}**. "
                    f"The combined ICER is **{threshold_message(combined_icer['icer_mean'], cet, annual_ppp)}**."
                )
                st.altair_chart(plot_ceac(combined_icer["ceac_wtp"], combined_icer["ceac_prob"], cet), width="stretch")
                st.altair_chart(
                    plot_ce_plane(
                        combined_icer["draw_level_outputs"],
                        effectiveness_col="dalys_averted",
                        cost_col="incremental_cost_usd",
                        wtp_threshold=cet,
                        additional_thresholds = [annual_ppp],
                        include_ellipse=True,
                    ),
                    width="stretch",
                )
                st.caption(
                    "The cost-effectiveness plane uses actual draw-level DALYs averted and programme cost minus draw-level health-sector savings."
                )
                st.dataframe(
                    combined_icer["draw_level_outputs"].head(25).style.format(
                        {
                            "daly_total_no_mda": "{:,.1f}",
                            "daly_total_mda": "{:,.1f}",
                            "dalys_averted": "{:,.1f}",
                            "annual_program_cost_usd": "${:,.0f}",
                            "health_sector_savings_usd": "${:,.0f}",
                            "program_cost_minus_savings": "${:,.0f}",
                            "incremental_cost_usd": "${:,.0f}",
                            "icer_usd_per_daly": "${:,.0f}",
                            "nmb_at_woods_cet": "${:,.0f}",
                            "nmb_at_gdp_threshold": "${:,.0f}",
                        }
                    ),
                    width="stretch",
                )
                download_df(
                    combined_icer["draw_level_outputs"],
                    "Download combined draw-level CE-plane and ICER data (CSV)",
                    "combined_cea_draw_level_outputs.csv",
                )

        if sim_m is not None and sim_h is not None:
            st.caption(
                "Species-specific ICERs below attribute the full annualized programme cost to each module before subtracting module-specific health-sector savings. "
                "Use the combined ICER above for the programme-level result."
            )

        for label, sim_df in [("S. mansoni", sim_m), ("S. haematobium", sim_h)]:
            if sim_df is None:
                continue
            module_ce_df = add_cost_effectiveness_columns(
                sim_df,
                annualized_prog_cost,
                opd_cost_curr,
                ipd_cost_curr,
            )
            icer_res = compute_icer(
                module_ce_df,
                annualized_prog_cost,
                annual_ppp,
                cet,
                incremental_cost_draws=module_ce_df["program_cost_minus_savings"],
            )
            with st.expander(f"{label} - module ICER and CEAC"):
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Computed DALYs averted p.a.",
                    f"{icer_res['dalys_averted_point']:,.0f}",
                    help=(
                        format_daly_averted_calculation(icer_res)
                        + " 95% PSA interval for draw-level DALYs averted: "
                        + _format_ci_or_na(
                            icer_res["dalys_averted_point"],
                            icer_res["dalys_averted_lo"],
                            icer_res["dalys_averted_hi"],
                        )
                    ),
                )
                c2.metric(
                    "ICER (USD/DALY)",
                    "NA" if not np.isfinite(icer_res["icer_mean"]) else f"${icer_res['icer_mean']:,.0f}",
                    help=_format_ci_or_na(
                        icer_res["icer_mean"],
                        icer_res["icer_lo"],
                        icer_res["icer_hi"],
                        fmt=",.0f",
                        na_text="No positive DALYs averted.",
                    ),
                )
                if icer_res["positive_daly_draws"] == 0:
                    st.warning("No PSA iterations produced positive DALYs averted for this module; the module ICER is not evaluable.")
                c3.metric(
                    "% prob. cost-effective (Woods CET)",
                    "NA" if not np.isfinite(icer_res["pct_cost_effective_cet"]) else f"{icer_res['pct_cost_effective_cet'] * 100:.0f}%",
                    help=_format_ci_or_na(
                        icer_res.get("pct_cost_effective_cet", np.nan),
                        icer_res.get("pct_cost_effective_cet_lo", np.nan),
                        icer_res.get("pct_cost_effective_cet_hi", np.nan),
                        fmt=".1%",
                        na_text="NA",
                    ),
                )
                st.dataframe(
                    cost_effectiveness_summary_table(icer_res).style.format(
                        {"Mean": "{:,.1f}", "95% CI lower": "{:,.1f}", "95% CI upper": "{:,.1f}"}
                    ),
                    width="stretch",
                )
                st.caption(format_daly_averted_calculation(icer_res))
                st.write(f"This module ICER is **{threshold_message(icer_res['icer_mean'], cet, annual_ppp)}**.")
                st.altair_chart(plot_ceac(icer_res["ceac_wtp"], icer_res["ceac_prob"], cet), width="stretch")
                st.altair_chart(
                    plot_ce_plane(
                        icer_res["draw_level_outputs"],
                        effectiveness_col="dalys_averted",
                        cost_col="incremental_cost_usd",
                        wtp_threshold=cet,
                        additional_thresholds = [annual_ppp],
                        include_ellipse=True,
                    ),
                    width="stretch",
                )
                st.caption(
                    "The cost-effectiveness plane uses actual draw-level DALYs averted and programme cost minus draw-level health-sector savings."
                )
                st.dataframe(
                    icer_res["draw_level_outputs"].head(25).style.format(
                        {
                            "daly_total_no_mda": "{:,.1f}",
                            "daly_total_mda": "{:,.1f}",
                            "dalys_averted": "{:,.1f}",
                            "annual_program_cost_usd": "${:,.0f}",
                            "health_sector_savings_usd": "${:,.0f}",
                            "program_cost_minus_savings": "${:,.0f}",
                            "incremental_cost_usd": "${:,.0f}",
                            "icer_usd_per_daly": "${:,.0f}",
                            "nmb_at_woods_cet": "${:,.0f}",
                            "nmb_at_gdp_threshold": "${:,.0f}",
                        }
                    ),
                    width="stretch",
                )
                safe_label = label.lower().replace(" ", "_").replace(".", "").replace("/", "_")
                download_df(
                    icer_res["draw_level_outputs"],
                    f"Download {label} draw-level CE-plane and ICER data (CSV)",
                    f"{safe_label}_cea_draw_level_outputs.csv",
                )

        st.subheader("Return on investment")
        benefit_draws = combined_benefit_draws([sim_m, sim_h], daily_wage_adj, opd_cost_curr, ipd_cost_curr)
        roi_res = roi_summary_from_draws(
            benefit_draws["hs_savings_pa"],
            benefit_draws["econ_gain_pa"],
            annualized_prog_cost,
        )
        roi = roi_res["roi_mean"] if np.isfinite(roi_res["roi_mean"]) else compute_roi(total_hs_savings, total_econ_gain, annualized_prog_cost)
        st.markdown(
            f"For every **\\$1** invested in the schistosomiasis MDA programme at this administrative level, "
            f"the estimated economic return is **\\${roi:.2f}** "
            f"[{roi_res['roi_lo']:.2f}, {roi_res['roi_hi']:.2f}], combining health sector cost savings "
            f"(\\${roi_res['hs_savings_mean']:,.0f} p.a.) and inequality-adjusted productivity gains "
            f"(\\${roi_res['econ_gain_mean']:,.0f} p.a.)."
        )
        st.dataframe(
            roi_summary_table(roi_res).style.format(
                {"Mean": "{:,.2f}", "95% CI lower": "{:,.2f}", "95% CI upper": "{:,.2f}"}
            ),
            width="stretch",
        )

        with st.expander("Download raw Monte Carlo simulation data"):
            st.caption("The PSA CSV downloads include draw-level `dalys_averted` columns computed as no-MDA DALYs minus MDA DALYs.")
            if sim_m is not None:
                download_df(add_daly_averted_columns(sim_m), "Download S. mansoni PSA data (CSV)", "mansoni_psa.csv")
            if sim_h is not None:
                download_df(add_daly_averted_columns(sim_h), "Download S. haematobium PSA data (CSV)", "haematobium_psa.csv")
            if not combined_df.empty:
                download_df(add_daly_averted_columns(combined_df), "Download combined DALY PSA data (CSV)", "combined_daly_psa.csv")

        st.markdown("---")
        st.subheader("Sensitivity analysis: coverage & frequency")
        st.caption(
            "Explore how different MDA coverage (50–85%) and delivery frequencies (annual vs biennial) "
            "affect prevalence, case loads, and cost-effectiveness."
        )
        
        if run_sens_now and run_mansoni and m_pop > 0 and m_prev > 0 and len(frequency_scenarios_sens) > 0:
            with st.spinner("Running sensitivity analysis across coverage & frequency..."):
                sens_df = run_sensitivity_analysis(
                    coverage_range=coverage_range_sens,
                    frequency_scenarios=frequency_scenarios_sens,
                    time_horizon=10,
                    at_risk_pop=m_pop,
                    initial_prev_sac=m_prev,
                    caseload_fn=estimate_caseloads_mansoni,
                    caseload_params=m_params,
                    annual_prog_cost=annualized_prog_cost,
                    psa_df=sim_m,
                    opd_cost=opd_cost_curr,
                    ipd_cost=ipd_cost_curr,
                    discount_rate=disc_costs,
                    p_min_pct=trajectory_floor_pct,
                    species="mansoni",
                    annual_decay_at_reference=trajectory_m_decay_sac,
                    mda_year_prog_cost=programme_cost_mda_year,
                    off_year_fixed_cost=off_year_fixed_cost,
                )
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            optimal_nb = identify_optimal_scenario(sens_df, year=10, criterion="net_benefit")
            optimal_cases = identify_optimal_scenario(sens_df, year=10, criterion="cases_averted")
            
            with col1:
                st.metric("Optimal coverage (max net benefit)", f"{optimal_nb['coverage']:.0f}%", optimal_nb['frequency'])
            with col2:
                st.metric("Optimal for cases averted", f"{optimal_cases['coverage']:.0f}%", optimal_cases['frequency'])
            with col3:
                st.metric("Net benefit at optimum", f"${optimal_nb['net_benefit']:,.0f}")
            with col4:
                st.metric("Prevalence at optimum", f"{optimal_nb['prev_final']:.1f}%")
            
            # Visualizations
            st.markdown("#### Prevalence trajectories by scenario")
            st.altair_chart(
                plot_prevalence_sensitivity(
                    coverage_range_sens,
                    frequency_scenarios_sens,
                    m_prev,
                    time_horizon=10,
                    p_min_pct=trajectory_floor_pct,
                    species="mansoni",
                    annual_decay_at_reference=trajectory_m_decay_sac,
                ),
                width="stretch",
            )
            st.caption("Solid lines = annual, dashed lines = biennial. Darker colors = higher coverage.")
            
            st.markdown("#### Cases averted by year 10")
            st.altair_chart(
                plot_cases_averted_sensitivity(sens_df, year=10),
                width="stretch",
            )
            
            st.markdown("#### Cost-effectiveness landscape (year 10)")
            st.altair_chart(
                plot_cost_effectiveness_sensitivity(sens_df, year=10),
                width="stretch",
            )
            st.caption("Each bubble is a scenario. Higher & further right = more cases averted with higher savings.")
            
            # Summary table
            st.markdown("#### Detailed comparison at year 10")
            summary_table = sensitivity_summary_table(sens_df, year=10)
            fmt_cols = {
                "Frequency effect factor": "{:.2f}",
                "Prevalence (%)": "{:.1f}",
                "Cases averted": "{:,.0f}",
                "HS cost with MDA (USD)": "${:,.0f}",
                "HS savings (USD)": "${:,.0f}",
                "Prog cost (USD)": "${:,.0f}",
                "Net benefit (USD)": "${:,.0f}",
            }
            st.dataframe(
                summary_table.style.format(fmt_cols).highlight_max(
                    subset=["Net benefit (USD)", "Cases averted"], color="lightgreen"
                ),
                width="stretch",
            )
            
            download_df(sens_df, "Download full sensitivity analysis (CSV)", "sensitivity_analysis.csv")
        else:
            if not run_mansoni or m_pop <= 0 or m_prev <= 0:
                st.info("Set disease parameters in the Disease inputs tab first.")
            elif len(frequency_scenarios_sens) == 0:
                st.info("Select at least one frequency scenario in the Sensitivity analysis panel (left sidebar).")
            else:
                st.info("Click **Run sensitivity analysis** in the left sidebar to generate results.")

with tabs[5]:
    if guided_mode:
        render_tab_tip("economic")
    st.subheader("Economic impact analysis")
    st.caption(
        f"Discounted {bia_horizon}-year projection | Discount rate: {disc_costs:.0%} | "
        f"Start year: {int(base_year)}"
    )
    st.info(
        "This section includes morbidity-state health-sector savings and productivity gains, so it is an "
        "economic-impact analysis rather than a budget-impact analysis. Staff time value is included only "
        "when selected in the Budget impact resource inputs panel."
    )

    if sim_m is None and sim_h is None:
        st.info("Complete the Disease inputs tab first to estimate savings and productivity gains.")
    else:
        _benefit_draws = combined_benefit_draws([sim_m, sim_h], daily_wage_adj, opd_cost_curr, ipd_cost_curr)
        _budget_draws_for_econ = combined_morbidity_budget_draws(
            [sim_m, sim_h],
            opd_cost=opd_cost_curr,
            ipd_cost=ipd_cost_curr,
            health_state_costs=health_state_costs,
            opd_staff_minutes=opd_staff_minutes,
            ipd_staff_minutes=ipd_staff_minutes,
            staff_hourly_cost=staff_hourly_cost,
            include_staff_time_value=include_staff_time_value,
        )
        _hs_savings_draws_econ = (
            _budget_draws_for_econ["budget_offset_pa"]
            if _budget_draws_for_econ["budget_offset_pa"].size
            else _benefit_draws["hs_savings_pa"]
        )
        _hs_savings = float(np.nanmean(_hs_savings_draws_econ)) if _hs_savings_draws_econ.size else 0.0
        _econ_gain = float(np.nanmean(_benefit_draws["econ_gain_pa"])) if _benefit_draws["econ_gain_pa"].size else 0.0

        economic_impact_df = economic_impact_analysis(
            programme_cost_mda_year=programme_cost_mda_year,
            off_year_fixed_cost=off_year_fixed_cost,
            hs_savings_pa=_hs_savings,
            econ_gain_pa=_econ_gain,
            horizon=int(bia_horizon),
            disc_rate=disc_costs,
            pzq_cost=pzq_unit_cost,
            pop_treat=planned_pop_treat,
            pzq_per_person=pzq_per_person,
            delivery_c=delivery_cost,
            fixed_costs_mda_year=fixed_prog_cost_mda_year,
            freq=mda_frequency,
            base_year=int(base_year),
        )
        economic_impact_df = add_economic_impact_intervals(
            economic_impact_df,
            _hs_savings_draws_econ,
            _benefit_draws["econ_gain_pa"],
            disc_costs,
        )

        total_programme_cost = float(pd.to_numeric(economic_impact_df["Total_prog_cost_USD"], errors="coerce").fillna(0.0).sum())
        total_hs_savings = float(pd.to_numeric(economic_impact_df["Health_sector_savings_USD"], errors="coerce").fillna(0.0).sum())
        total_productivity_gains = float(pd.to_numeric(economic_impact_df["Economic_gains_USD"], errors="coerce").fillna(0.0).sum())
        net_series = pd.to_numeric(economic_impact_df["Cumulative_net_economic_benefit_USD"], errors="coerce").dropna()
        cumulative_net_benefit = float(net_series.iloc[-1]) if not net_series.empty else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Programme cost", f"${total_programme_cost:,.0f}")
        c2.metric("Health-sector savings", f"${total_hs_savings:,.0f}")
        c3.metric("Productivity gains", f"${total_productivity_gains:,.0f}")
        c4.metric(
            "Cumulative net economic benefit",
            "NA" if not np.isfinite(cumulative_net_benefit) else f"${cumulative_net_benefit:,.0f}",
        )

        st.altair_chart(plot_economic_impact(economic_impact_df), width="stretch")

        st.markdown("---")
        econ_fmt = {
            "Drug_costs_USD": "${:,.0f}",
            "Delivery_costs_USD": "${:,.0f}",
            "Other_prog_USD": "${:,.0f}",
            "Total_prog_cost_USD": "${:,.0f}",
            "Cumulative_prog_cost_USD": "${:,.0f}",
            "Health_sector_savings_USD": "${:,.0f}",
            "Health_sector_savings_USD_95CI_lower": "${:,.0f}",
            "Health_sector_savings_USD_95CI_upper": "${:,.0f}",
            "Economic_gains_USD": "${:,.0f}",
            "Economic_gains_USD_95CI_lower": "${:,.0f}",
            "Economic_gains_USD_95CI_upper": "${:,.0f}",
            "Total_economic_benefits_USD": "${:,.0f}",
            "Total_economic_benefits_USD_95CI_lower": "${:,.0f}",
            "Total_economic_benefits_USD_95CI_upper": "${:,.0f}",
            "Net_economic_benefit_USD": "${:,.0f}",
            "Net_economic_benefit_USD_95CI_lower": "${:,.0f}",
            "Net_economic_benefit_USD_95CI_upper": "${:,.0f}",
            "Cumulative_net_economic_benefit_USD": "${:,.0f}",
            "Cumulative_net_economic_benefit_USD_95CI_lower": "${:,.0f}",
            "Cumulative_net_economic_benefit_USD_95CI_upper": "${:,.0f}",
        }
        styled_table(economic_impact_df, econ_fmt)
        download_df(economic_impact_df, "Download economic impact table (CSV)", "economic_impact.csv")

        bcr_res = economic_benefit_cost_ratio_summary(
            economic_impact_df,
            _hs_savings_draws_econ,
            _benefit_draws["econ_gain_pa"],
            disc_costs,
        )
        bcr = bcr_res["bcr_mean"]
        st.metric(
            f"Economic benefit-cost ratio over {bia_horizon} years",
            "NA" if not np.isfinite(bcr) else f"{bcr:.2f}",
            help=_format_ci_or_na(
                bcr_res.get("bcr_mean", np.nan),
                bcr_res.get("bcr_lo", np.nan),
                bcr_res.get("bcr_hi", np.nan),
                fmt=".2f",
                na_text="NA",
            ),
        )

with tabs[6]:
    if guided_mode:
        render_tab_tip("budget")
    st.subheader("Budget impact analysis")
    st.caption(
        f"Health-system/payer perspective | Discounted {bia_horizon}-year projection | "
        f"Discount rate: {disc_costs:.0%} | Start year: {int(base_year)}"
    )
    st.info(
        "This budget-impact view excludes productivity gains and other societal benefits. "
        "It reports gross programme costs, morbidity-state clinical-management offsets, staff time saved, and net budget impact."
    )

    if sim_m is None and sim_h is None:
        st.info("Complete the Disease inputs tab first to estimate morbidity-state budget offsets.")
    else:
        _budget_draws = combined_morbidity_budget_draws(
            [sim_m, sim_h],
            opd_cost=opd_cost_curr,
            ipd_cost=ipd_cost_curr,
            health_state_costs=health_state_costs,
            opd_staff_minutes=opd_staff_minutes,
            ipd_staff_minutes=ipd_staff_minutes,
            staff_hourly_cost=staff_hourly_cost,
            include_staff_time_value=include_staff_time_value,
        )
        _clinical_offset = (
            float(np.nanmean(_budget_draws["clinical_budget_offset_pa"]))
            if _budget_draws["clinical_budget_offset_pa"].size
            else 0.0
        )
        _staff_value = (
            float(np.nanmean(_budget_draws["staff_time_value_pa"]))
            if _budget_draws["staff_time_value_pa"].size
            else 0.0
        )
        _staff_hours = (
            float(np.nanmean(_budget_draws["staff_hours_saved_pa"]))
            if _budget_draws["staff_hours_saved_pa"].size
            else 0.0
        )

        budget_df = budget_impact_analysis(
            off_year_fixed_cost=off_year_fixed_cost,
            hs_savings_pa=_clinical_offset,
            horizon=int(bia_horizon),
            disc_rate=disc_costs,
            pzq_cost=pzq_unit_cost,
            pop_treat=planned_pop_treat,
            pzq_per_person=pzq_per_person,
            delivery_c=delivery_cost,
            fixed_costs_mda_year=fixed_prog_cost_mda_year,
            freq=mda_frequency,
            base_year=int(base_year),
            staff_time_value_pa=_staff_value,
            staff_hours_saved_pa=_staff_hours,
            include_staff_time_value=include_staff_time_value,
        )
        budget_df = add_budget_impact_intervals(
            budget_df,
            _budget_draws["clinical_budget_offset_pa"],
            disc_costs,
            staff_time_value_draws_arr=_budget_draws["staff_time_value_pa"],
            staff_hours_draws_arr=_budget_draws["staff_hours_saved_pa"],
            include_staff_time_value=include_staff_time_value,
        )

        st.altair_chart(plot_budget_impact(budget_df), width="stretch")

        gross_cost = float(pd.to_numeric(budget_df["Total_prog_cost_USD"], errors="coerce").fillna(0.0).sum())
        clinical_offset = float(pd.to_numeric(budget_df["Clinical_management_budget_offset_USD"], errors="coerce").fillna(0.0).sum())
        staff_hours_total = float(pd.to_numeric(budget_df["Staff_time_hours_saved"], errors="coerce").fillna(0.0).sum())
        hs_offset = float(pd.to_numeric(budget_df["Health_sector_budget_offset_USD"], errors="coerce").fillna(0.0).sum())
        net_series = pd.to_numeric(budget_df["Cumulative_net_budget_impact_USD"], errors="coerce").dropna()
        net_budget_impact = float(net_series.iloc[-1]) if not net_series.empty else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Gross programme budget over {bia_horizon} years", f"${gross_cost:,.0f}")
        c2.metric("Clinical-management offset", f"${clinical_offset:,.0f}")
        c3.metric("Staff time saved", f"{staff_hours_total:,.0f} hours")
        c4.metric(
            "Net budget impact",
            "NA" if not np.isfinite(net_budget_impact) else f"${net_budget_impact:,.0f}",
            help=f"Total budget offset included in the net calculation: ${hs_offset:,.0f}",
        )

        st.markdown("#### Morbidity-state budget offsets")
        state_budget_df = morbidity_budget_summary_table(
            _budget_draws["state_rows"],
            horizon=int(bia_horizon),
            disc_rate=disc_costs,
            include_staff_time_value=include_staff_time_value,
        )
        if state_budget_df.empty:
            st.info("No morbidity-state budget offsets were available for this scenario.")
        else:
            state_fmt = {
                col: "{:,.1f}"
                for col in state_budget_df.columns
                if "Cases" in col or "OPD" in col or "IPD" in col or "hours" in col
            }
            state_fmt.update({col: "${:,.0f}" for col in state_budget_df.columns if "USD" in col})
            styled_table(state_budget_df, state_fmt)
            download_df(state_budget_df, "Download morbidity-state budget offsets (CSV)", "budget_offsets_by_morbidity_state.csv")

        budget_fmt = {
            "Drug_costs_USD": "${:,.0f}",
            "Delivery_costs_USD": "${:,.0f}",
            "Other_prog_USD": "${:,.0f}",
            "Total_prog_cost_USD": "${:,.0f}",
            "Cumulative_prog_cost_USD": "${:,.0f}",
            "Clinical_management_budget_offset_USD": "${:,.0f}",
            "Clinical_management_budget_offset_USD_95CI_lower": "${:,.0f}",
            "Clinical_management_budget_offset_USD_95CI_upper": "${:,.0f}",
            "Staff_time_hours_saved": "{:,.1f}",
            "Staff_time_hours_saved_95CI_lower": "{:,.1f}",
            "Staff_time_hours_saved_95CI_upper": "{:,.1f}",
            "Cumulative_staff_time_hours_saved": "{:,.1f}",
            "Cumulative_staff_time_hours_saved_95CI_lower": "{:,.1f}",
            "Cumulative_staff_time_hours_saved_95CI_upper": "{:,.1f}",
            "Staff_time_value_USD": "${:,.0f}",
            "Staff_time_value_USD_95CI_lower": "${:,.0f}",
            "Staff_time_value_USD_95CI_upper": "${:,.0f}",
            "Staff_time_value_included_USD": "${:,.0f}",
            "Staff_time_value_included_USD_95CI_lower": "${:,.0f}",
            "Staff_time_value_included_USD_95CI_upper": "${:,.0f}",
            "Health_sector_budget_offset_USD": "${:,.0f}",
            "Health_sector_budget_offset_USD_95CI_lower": "${:,.0f}",
            "Health_sector_budget_offset_USD_95CI_upper": "${:,.0f}",
            "Net_budget_impact_USD": "${:,.0f}",
            "Net_budget_impact_USD_95CI_lower": "${:,.0f}",
            "Net_budget_impact_USD_95CI_upper": "${:,.0f}",
            "Cumulative_net_budget_impact_USD": "${:,.0f}",
            "Cumulative_net_budget_impact_USD_95CI_lower": "${:,.0f}",
            "Cumulative_net_budget_impact_USD_95CI_upper": "${:,.0f}",
        }
        styled_table(budget_df, budget_fmt)
        download_df(budget_df, "Download budget impact table (CSV)", "budget_impact.csv")

with tabs[7]:
    if guided_mode:
        render_tab_tip("elimination")
    st.subheader("Elimination endgame projection")
    st.caption("EPHP is defined on heavy-intensity infection prevalence in SAC (<1%).")
    st.caption(f"The elimination scenario uses the same annual-equivalent frequency effect factor as the PSA and prevalence trajectory: {frequency_effect_factor:.2f}.")
    if run_mansoni and m_pop > 0 and m_prev > 0:
        c1, c2 = st.columns(2)
        with c1:
            target_year = st.number_input("WHO target year", 2025, 2050, 2030, 1, help=HELP_TEXT["target_year"])
            reservoir_floor = st.slider("Transmission reservoir floor (% overall)", 0.0, 5.0, 0.5, 0.1, help=HELP_TEXT["reservoir_floor"])
        with c2:
            heavy_accel = st.slider("Heavy-intensity clearance acceleration", 1.0, 2.5, 1.6, 0.1, help=HELP_TEXT["heavy_accel"])
            target_kind = st.radio("Target", ("EPHP (heavy SAC <1%)", "Interruption (overall ~0)"), help=HELP_TEXT["target_kind"])

        target = (elim.ephp_target(int(target_year)) if target_kind.startswith("EPHP")
                  else elim.interruption_target(int(target_year)))
        sac_pop = m_pop / max(float(target_multiplier), 1.0)
        adult_pop = m_pop - sac_pop

        proj = elim.project_elimination(
            species="mansoni",
            prev_sac_pct=m_prev_sac_default, prev_adult_pct=m_prev_adult_default,
            sac_population=sac_pop, adult_population=adult_pop,
            heavy_share=m_heavy_pct / 100.0,
            coverage_pct=mda_coverage, frequency=mda_frequency,
            years=int(target_year) - int(base_year) + 3, base_year=int(base_year),
            p_min_pct=reservoir_floor, heavy_intensity_accel=heavy_accel,
            treat_adults=(mda_target != "SAC only"),
            frequency_effect_factor=frequency_effect_factor,
        )
        ev = elim.evaluate_target(proj, target)
        pr = elim.probability_of_target(
            "mansoni", m_prev_sac_default, m_heavy_pct / 100.0,
            mda_coverage, mda_frequency, target, base_year=int(base_year),
            n_iter=int(n_iterations), seed=int(seed),
            p_min_pct=reservoir_floor, heavy_intensity_accel=heavy_accel,
            frequency_effect_factor=frequency_effect_factor,
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Year target reached", ev["year_reached"] or "Not within horizon")
        k2.metric(f"On track by {target_year}?", "Yes" if ev["reached_by_target_year"] else "No")
        k3.metric(f"P(reach by {target_year})", f"{pr['prob_reached_by_target']:.0%}")

        st.altair_chart(
            plot_elimination_trajectory(proj, target.threshold_pct, target_year, ev["metric_col"]),
            width="stretch",
        )

        scen = elim.compare_elimination_scenarios(
            species="mansoni", prev_sac_pct=m_prev_sac_default,
            heavy_share=m_heavy_pct / 100.0,
            coverage_options=[50, 65, 75, 85], frequency_options=["Annual", "Biennial"],
            target=target, base_year=int(base_year),
            years=int(target_year) - int(base_year) + 3,
            n_iter=int(n_iterations), seed=int(seed),
            p_min_pct=reservoir_floor, heavy_intensity_accel=heavy_accel,
            frequency_effect_factor=float(biennial_effect_factor),
        )
        st.markdown("#### Scenario comparison")
        st.dataframe(scen, width="stretch")
        download_df(scen, "Download endgame scenarios (CSV)", "endgame_scenarios.csv")
    else:
        st.info("Set S. mansoni disease parameters in the Disease inputs tab first.")

with tabs[8]:
    st.markdown(
        """
        ### Technical Assumptions

        **Prevalence trajectory scenario model**:
        - The Results-tab trajectory is a deterministic scenario projection for costing, not a calibrated transmission model and not a direct SCHISTOX simulation.
        - No-MDA comparator: P_noMDA(t) = P0 × (1 + g)^t, where the default annual change g is 0%, so prevalence remains constant unless the analyst changes this assumption.
        - MDA scenario: P_MDA(t) = P_floor + (P0 - P_floor) × exp[-λ_species × I × t].
        - MDA intensity: I = annual-equivalent frequency-effect factor × selected_coverage / 75%. Annual MDA has factor 1.0. Biennial MDA uses the same user-selected effect-vs-annual factor used in the PSA and sensitivity analysis; set it to 0.50 for a literal every-other-year rounds assumption.
        - Species-specific prevalence-response defaults are shown in the sidebar. S. haematobium uses a faster default response than S. mansoni, while all decay values remain editable scenario assumptions rather than fitted transmission parameters.
        - SAC and adult trajectories are projected separately. Adult decay is applied only when the MDA target includes at-risk adults.
        - The residual floor is optional and never forces prevalence upward when baseline prevalence is already below the floor.
        - The trajectory is intended for caseload and health-sector cost projections. The PSA DALY/ICER section uses an annual steady-state effect at the selected coverage and frequency, so trajectory case reductions should not be used to recalculate the ICER directly. The Elimination Projections tab separately evaluates WHO EPHP-style heavy-intensity SAC targets.

        **Caseload Estimation**:
        - Proportional to prevalence at each time point
        - Burden of specific complications (anemia, hepatomegaly, cancer)
          determined by infection prevalence and morbidity probabilities

        **Bladder cancer calculation**:
        - The bladder-cancer rate input is interpreted as an observed all-cause incidence rate per 100,000 population.
        - S. haematobium-attributable cases = observed all-cause cases × Levin PAF, where PAF = P_e(RR-1) / [1 + P_e(RR-1)].
        - MDA reduces only the attributable component: attributable cases with MDA = attributable cases × (1 - effective cancer reduction).
        - Non-attributable bladder-cancer cases are left unchanged; total bladder cancer with MDA = non-attributable cases + residual attributable cases.
        
        **Health Sector Costs**:
        - OPD visits and IPD days scale with infected population size
        - Discounted at specified rate over projection horizon
        
        **Cost-Effectiveness**:
        - ICER = Programme cost / DALYs averted
        - CEA threshold: Woods et al. elasticity method adjusted to local GDP
        - ICER, NMB, DALYs averted, productivity, health-sector cost, ROI, and economic-impact ranges are shown as 95% PSA intervals from the 2.5th to 97.5th percentiles of simulation draws.
        - Cost-effectiveness probability intervals use Wilson binomial intervals across PSA draws.

        **Economic Impact**:
        - Includes programme costs, health-sector savings, and productivity gains.
        - Net economic benefit = health-sector savings + productivity gains - programme costs.
        - The economic benefit-cost ratio uses discounted health-sector savings plus productivity gains as benefits.

        **Budget Impact**:
        - Excludes productivity gains, ROI, benefit-cost ratios, and other societal benefits.
        - Reports gross programme/payer costs, morbidity-state clinical-management offsets, staff time saved, and net budget impact.
        - Clinical-management offsets are estimated separately for anemia, hepatomegaly, periportal fibrosis, portal hypertension, esophageal varices, hematuria, hydronephrosis, FGS, and attributable bladder cancer.
        - Staff time saved is reported in hours from avoided OPD visits and IPD bed-days. Monetized staff time is included in net budget impact only when selected, to avoid double counting if OPD/IPD unit costs already include staff.
        
        **Sensitivity Analysis**:
        - Coverage range: 50–85% (adjustable).
        - Frequency scenarios use the same annual-equivalent frequency-effect factor as the PSA and deterministic trajectory sections.
        - Effect multiplier = (coverage/100) × frequency_effect_factor.
        - Sensitivity analysis is a deterministic scenario screen; ICER results above remain based on the steady-state annual PSA draws.
        """
    )

with tabs[9]:
    st.markdown("**For questions, bug reports, or feature requests**: Contact ocu9@cdc.gov")
    st.markdown("**Key model reference**: Graham et al. SCHISTOX: an individual-based model for schistosomiasis epidemiology and control, Infectious Disease Modelling (2021).")
