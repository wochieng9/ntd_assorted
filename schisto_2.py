"""
Schistosomiasis integrated module.

Architecture
------------
1. SchistoProjectionParams  — validated, documented inputs replacing magic numbers
2. SeverityRatio            — models heavy/total prevalence as a ratio, not absolute
3. schisto_projection()     — fixed trajectory engine (continuous step-down, documented floors)
4. SchistoCaseloads         — translates prevalence at each timestep into clinical cases
5. SchistoPSA               — Monte Carlo engine drawing from the dataclass distributions
6. schisto_run()            — orchestrator: trajectory × PSA → year-by-year DataFrame
7. schisto_summary()        — CEA outputs: DALYs, ICERs, productivity, BIA

Drop-in integration
-------------------
Replace the placeholder in the existing tool:

    if "Schistosomiasis" in ntd_disease:
        from schisto_integrated_module import schisto_ui
        schisto_ui(tabs, country, country_inputs, daily_wage, translate_text)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PROJECTION PARAMETERS
# All formerly magic numbers are now named, documented, and overridable.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchistoProjectionParams:
    """
    Planning-model parameters for the prevalence trajectory.

    Reduction coefficients
    ----------------------
    base_prev_reduction_per_round : float
        Fractional reduction in overall prevalence per MDA round at 100% coverage
        under neutral WASH/snail conditions. Anchored to SCORE trial data
        (Colley et al. 2014; mean ~16% per round at ~75% coverage, back-calculated
        to 100% coverage base). Range in literature: 0.10–0.22.

    base_heavy_reduction_per_round : float
        Analogous coefficient for heavy-intensity prevalence. Heavy infections
        respond more strongly to praziquantel (Zwang & Olliaro 2014 meta-analysis:
        ~33% reduction per round at 100% coverage). Range: 0.22–0.45.

    Aggregation-adjusted rebound (SCHISTOX-derived)
    ------------------------------------------------
    aggregation_alpha : float
        Negative binomial aggregation parameter governing the distribution of
        worm burdens across the population (Graham et al. 2021, SCHISTOX).
        Small α → high aggregation (few heavily-infected individuals carry most
        worms; MDA clears the lightly-infected majority quickly but leaves a
        rebound-prone core). Large α → low aggregation (worms spread evenly;
        MDA effect is more uniform).

        Published range: 0.04–0.24 (Anderson & May 1982; Toor et al. 2019).
        Default 0.07 represents a moderately aggregated community, consistent
        with African S. mansoni settings in SCORE trial data.

        Role in the model: scales the per-round reduction coefficient downward
        at low prevalence via `_aggregation_prev_scalar()`, reflecting that the
        residual population at low prevalence is disproportionately composed of
        heavily-infected, hard-to-clear individuals — the mechanism underlying
        the plateau behaviour observed in SCORE.

    density_dependent_fecundity_z : float
        Exponential decay constant in the density-dependent egg production
        function E ~ λ·W_p·exp(−z·W_F) (Graham et al. 2021; Anderson & May
        1982). As worm burden falls, density-dependent suppression is relieved
        and surviving worms produce proportionally more eggs, creating a
        non-linear rebound. This parameter scales how strongly the rebound
        boost applies when prevalence is low.

        Published values: S. mansoni 0.0007/female worm, S. haematobium
        0.0006/female worm (Anderson & May 1982; Anderson et al. 2016).
        The planning model uses a normalised version of this effect: at baseline
        prevalence the rebound scalar is 1.0; at the floor it approaches
        `1 + density_dependent_fecundity_z * 500` (approximately 1.35 with the
        default), representing ~35% more transmission per surviving worm at very
        low worm burden. This raises the effective floor without changing the
        floor formula directly.

    Reinfection floor
    -----------------
    floor_transmission_intercept : float
        The fraction of baseline prevalence that represents the irreducible
        reinfection floor under no WASH improvement and no snail control.
        Derived from SCORE trial sites where prevalence plateaued:
        ~22% of baseline. Range: 0.15–0.30.

    floor_wash_slope : float
        Reduction in the floor fraction per percentage point of WASH improvement.
        Calibrated so that 100% WASH improvement eliminates ~40% of the floor
        (i.e. slope ≈ 0.004). Biologically: improved water/sanitation reduces
        cercarial exposure and environmental egg contamination (Freeman et al.
        2017; Grimes et al. 2015, cited in Graham et al. 2021).

    floor_snail_reduction : float
        Absolute reduction in floor fraction when snail control is active.
        Based on modelled estimates from multicountry snail control pilots
        (Sokolow et al. 2016): ~4 percentage points.

    floor_minimum_pct : float
        Hard minimum for the reinfection floor (% prevalence). Prevents the
        model projecting to zero in high-transmission settings. Set at 0.1%
        (below EPHP threshold of 1% heavy intensity) so it does not artificially
        block EPHP achievement.

    Step-down policy
    ----------------
    step_down_prev_threshold_low : float
        Prevalence below which treatment frequency shifts to test-and-treat.
        WHO operational threshold: <10% SAC prevalence.

    step_down_prev_threshold_mid : float
        Prevalence below which bi-annual collapses to annual.
        WHO operational threshold: <50% SAC prevalence.

    Test-and-treat adjustment
    -------------------------
    tnt_floor_multiplier : float
        Floor is scaled down in test-and-treat phase because targeted treatment
        reaches only current positives, reducing the pool sustaining transmission.
        Set at 0.6 (i.e. 40% lower floor). Assumption, not empirically derived.

    tnt_efficacy_scalar : float
        Annual reduction in test-and-treat phase is lower than MDA because
        population coverage is effectively the surveillance fraction, not the
        full at-risk population. Set at 0.35, representing ~35% of MDA efficacy.

    Hotspot detection
    -----------------
    hotspot_min_coverage : float
        Coverage threshold below which hotspot flag is not triggered (poor
        coverage is the more parsimonious explanation for slow response).

    hotspot_min_baseline_prev : float
        Only flag hotspots in moderate/high prevalence settings (≥10%).

    hotspot_relative_reduction_threshold : float
        If relative reduction from baseline after `hotspot_review_year` rounds
        is less than this fraction, flag as potential hotspot. Set at 1/3:
        a well-functioning programme should achieve at least 33% relative
        reduction in 2 years at ≥75% coverage.

    hotspot_review_year : int
        Year at which hotspot criterion is evaluated. Set at t=2 (after 2 full
        rounds) to allow one round of operational lag.
    """
    # Reduction coefficients
    base_prev_reduction_per_round: float = 0.16
    base_heavy_reduction_per_round: float = 0.33

    # Reduction caps (biological ceiling — no single round achieves 100%)
    max_prev_reduction_per_round: float = 0.85
    max_heavy_reduction_per_round: float = 0.95

    # SCHISTOX-derived aggregation and density-dependence parameters
    aggregation_alpha: float = 0.07
    density_dependent_fecundity_z: float = 0.0007

    # Reinfection floor construction
    floor_transmission_intercept: float = 0.22
    floor_wash_slope: float = 0.004        # per percentage point WASH gain
    floor_snail_reduction: float = 0.04
    floor_minimum_pct: float = 0.1

    # Step-down thresholds (% prevalence)
    step_down_prev_threshold_low: float = 10.0
    step_down_prev_threshold_mid: float = 50.0

    # Test-and-treat phase
    tnt_floor_multiplier: float = 0.6
    tnt_efficacy_scalar: float = 0.35

    # Hotspot detection
    hotspot_min_coverage: float = 75.0
    hotspot_min_baseline_prev: float = 10.0
    hotspot_relative_reduction_threshold: float = 1 / 3
    hotspot_review_year: int = 2


def _aggregation_prev_scalar(prev_pct: float, baseline_prev_pct: float,
                              alpha: float) -> float:
    """
    Scale the per-round reduction coefficient downward as prevalence falls,
    reflecting that the residual population at low prevalence is increasingly
    composed of heavily-infected, hard-to-clear individuals (high worm burden,
    high predisposition). Derived from the SCHISTOX aggregation parameter alpha
    (Graham et al. 2021; Anderson & May 1982).

    At baseline prevalence the scalar is 1.0 (no adjustment).
    As prevalence approaches zero the scalar approaches alpha / (alpha + 1),
    i.e. the reduction is attenuated proportionally to the degree of aggregation.
    At alpha = 0.07 this floor is ~0.065, meaning at very low prevalence MDA
    achieves only ~6.5% of its baseline per-round effect — consistent with the
    endgame plateau observed in SCORE trial sites.

    Parameters
    ----------
    prev_pct : float
        Current prevalence (%).
    baseline_prev_pct : float
        Baseline (pre-MDA) prevalence (%).
    alpha : float
        Negative binomial aggregation parameter. Range 0.04–0.24.

    Returns
    -------
    float in (0, 1].
    """
    if baseline_prev_pct <= 0:
        return 1.0
    relative_prev = max(0.0, min(1.0, prev_pct / baseline_prev_pct))
    # At relative_prev=1 → scalar=1. At relative_prev=0 → scalar=alpha/(alpha+1).
    floor_scalar = alpha / (alpha + 1.0)
    return floor_scalar + (1.0 - floor_scalar) * relative_prev


def _density_dependent_rebound_scalar(prev_pct: float, baseline_prev_pct: float,
                                       z: float) -> float:
    """
    Apply a transmission rebound boost at low prevalence, reflecting the relief
    of density-dependent fecundity suppression as worm burdens fall
    (Graham et al. 2021; Anderson & May 1982: E ~ lambda * W_p * exp(-z * W_F)).

    At high worm burden (high prevalence), density-dependent suppression
    reduces egg output per worm. As MDA clears worms, surviving worms produce
    proportionally more eggs — a non-linear rebound effect that slows prevalence
    decline in the endgame beyond what a linear model would predict.

    This function scales the effective floor upward at low prevalence without
    modifying the floor formula directly.

    Returns a multiplier >= 1.0, maximum ~1 + z*500 at near-zero prevalence
    (approximately 1.35 with the S. mansoni default z=0.0007).
    """
    if baseline_prev_pct <= 0:
        return 1.0
    relative_prev = max(0.0, min(1.0, prev_pct / baseline_prev_pct))
    # Rebound is strongest at low relative prevalence
    rebound = 1.0 + z * 500.0 * (1.0 - relative_prev)
    return rebound


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AGE STRUCTURE
# Two strata: SAC (5–14 years) and adults (15+).
# Pre-SAC (<5) are excluded from MDA targets per WHO guidelines and omitted.
#
# Contact rates (β_age) from SCHISTOX / Turner et al. (2017):
#   SAC    = 1.00  (reference)
#   Adults = 0.06–0.30 depending on setting (moderate adult burden default 0.06,
#            high adult burden settings e.g. fishing communities up to 0.30)
#
# The environmental transmission pool is shared. Each stratum's force of
# infection scales with its contact rate relative to SAC. MDA strategy
# (SAC-only vs community-wide) determines which strata are treated each round.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgeStructureParams:
    """
    Age-stratified contact rates and demographic fractions.

    contact_rate_sac : float
        Relative cercarial contact rate for SAC (5–14 years). Reference = 1.0.

    contact_rate_adult : float
        Relative cercarial contact rate for adults (15+ years) relative to SAC.
        Published range: 0.06 (moderate adult burden, Turner et al. 2017) to
        0.30 (high adult burden, e.g. fishing communities, Toor et al. 2019).
        Default 0.06 represents a typical school-transmission-dominated setting.

    pop_fraction_sac : float
        Fraction of the total at-risk population aged 5–14 years.
        Approximate sub-Saharan African demographic pyramid value: 0.23
        (consistent with ESPEN target population calculations).

    pop_fraction_adult : float
        Fraction of the total at-risk population aged 15+ years.
        Remainder after SAC fraction: default 0.77.

    mda_strategy : str
        "SAC-only"        — treat SAC only (WHO standard for prevalence <50%)
        "Community-wide"  — treat SAC and adults (WHO recommended for ≥50%
                            prevalence or where adult burden is high;
                            Toor et al. 2019, Journal of Infectious Diseases)
    """
    contact_rate_sac:   float = 1.00
    contact_rate_adult: float = 0.06
    pop_fraction_sac:   float = 0.23
    pop_fraction_adult: float = 0.77
    mda_strategy: str = "SAC-only"

    def __post_init__(self):
        if not (0 < self.pop_fraction_sac < 1):
            raise ValueError("pop_fraction_sac must be in (0, 1)")
        self.pop_fraction_adult = 1.0 - self.pop_fraction_sac
        if self.mda_strategy not in ("SAC-only", "Community-wide"):
            raise ValueError("mda_strategy must be 'SAC-only' or 'Community-wide'")

    @property
    def population_weighted_contact_rate(self) -> float:
        """
        Population-weighted mean contact rate, used to scale the shared
        environmental transmission pool.
        """
        return (self.pop_fraction_sac   * self.contact_rate_sac
                + self.pop_fraction_adult * self.contact_rate_adult)

    def effective_coverage_sac(self, programme_coverage: float) -> float:
        """
        Effective coverage within SAC stratum. Under SAC-only MDA, the
        programme coverage applies only to SAC. Under community-wide MDA,
        coverage applies to both strata but SAC coverage is typically higher
        in practice — we apply the full programme coverage to SAC.
        """
        return min(1.0, programme_coverage)

    def effective_coverage_adult(self, programme_coverage: float) -> float:
        """
        Effective coverage within adult stratum.
        SAC-only: adults receive no MDA (coverage = 0).
        Community-wide: adults receive MDA at the programme coverage rate,
        recognising that adult coverage is often 10–15pp lower than SAC in
        practice (Toor et al. 2019). A 0.85 scalar is applied as a default.
        """
        if self.mda_strategy == "SAC-only":
            return 0.0
        return min(1.0, programme_coverage * 0.85)

    def transmission_rebound_from_untreated_adults(
        self, adult_prev_pct: float, baseline_adult_prev_pct: float
    ) -> float:
        """
        Under SAC-only MDA, untreated adults continue to contaminate the
        environment. This scalar represents the fractional contribution of
        adult transmission to the shared environmental pool, preventing SAC
        prevalence from falling below the level sustained by adult egg output.

        Returns a floor adjustment multiplier (>= 0, <= 1) representing the
        minimum SAC prevalence sustainable from adult transmission alone,
        scaled by the adult-to-SAC contact rate ratio.
        """
        if self.mda_strategy == "Community-wide" or baseline_adult_prev_pct <= 0:
            return 0.0
        adult_relative_prev = min(1.0, adult_prev_pct / baseline_adult_prev_pct)
        return (self.contact_rate_adult / self.contact_rate_sac) * adult_relative_prev


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SEVERITY RATIO MODEL
# Heavy prevalence is modelled as a ratio of total prevalence, not independently.
# This prevents the heavy/total coupling drift identified in the original code.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SeverityRatio:
    """
    Tracks the ratio of heavy-intensity to overall prevalence.

    Epidemiological rationale: the proportion of infections that are heavy
    intensity is not fixed — it tends to decline with MDA because praziquantel
    disproportionately clears high-burden infections (negative binomial
    distribution of worm burden means heavy-tailed, so top infections cleared
    first). We model this as a ratio that itself declines, but more slowly than
    overall prevalence, reflecting residual heavy cases in hard-to-reach groups.
    """
    initial_ratio: float          # heavy_prev / total_prev at baseline
    ratio_decay_per_year: float = 0.05   # ratio declines ~5% p.a. under MDA
                                          # (assumption — calibrate if data available)

    def ratio_at_year(self, t: int, frequency: str) -> float:
        """
        Ratio decays faster under bi-annual MDA, slower under test-and-treat.
        Floors at 10% of initial ratio (some heavy cases are persistent).
        """
        speed = {"Bi-annual": 1.5, "Annual": 1.0, "Test-and-treat": 0.3}.get(frequency, 1.0)
        decayed = self.initial_ratio * (1 - self.ratio_decay_per_year * speed) ** t
        return max(decayed, self.initial_ratio * 0.10)

    def heavy_prev(self, total_prev: float, t: int, frequency: str) -> float:
        return min(total_prev, total_prev * self.ratio_at_year(t, frequency))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIXED TRAJECTORY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _species_factor(species: str) -> float:
    return {"S. haematobium": 1.00, "S. mansoni": 0.95, "Mixed / unknown": 0.97}.get(species, 0.97)


def _rounds_per_year(frequency: str) -> float:
    return {"Test-and-treat": 0.25, "Annual": 1.0, "Bi-annual": 2.0}.get(frequency, 1.0)


def schisto_projection(
    baseline_prev_pct: float,
    baseline_heavy_pct: float,
    start_year: int,
    horizon_years: int,
    coverage_pct: float,
    frequency: str,
    species: str,
    wash_gain_pct: float,
    snail_control: bool,
    auto_step_down: bool = True,
    params: Optional[SchistoProjectionParams] = None,
    age_params: Optional[AgeStructureParams] = None,
    adult_baseline_prev_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Project schistosomiasis prevalence trajectory with age stratification.

    Parameters
    ----------
    baseline_prev_pct : float
        SAC prevalence at baseline (%), sourced from ESPEN Prev_SAC at the
        implementing unit level.
    baseline_heavy_pct : float
        SAC heavy-intensity prevalence at baseline (%).
    adult_baseline_prev_pct : float or None
        Adult prevalence at baseline (%), sourced from ESPEN Prev_Adults where
        available. If None or zero, adult prevalence is derived from the
        contact rate ratio (SCHISTOX equilibrium fallback).
    [all other parameters as before]

    Returns a DataFrame with one row per year containing:
        Year, Frequency, Rounds,
        Prevalence_SAC, HeavyPrev_SAC,
        Prevalence_Adult, HeavyPrev_Adult,
        Prevalence_Overall, HeavyPrev_Overall,
        EPHP_Achieved, MDA_Strategy,
        Adult_Prev_Source  ← "ESPEN" or "Contact-rate fallback"

    DataFrame.attrs keys:
        ephp_year           : int or None  (based on SAC heavy prevalence)
        persistent_hotspot  : bool
        severity_ratio      : SeverityRatio
        age_params          : AgeStructureParams
        adult_prev_source   : str
    """
    if params is None:
        params = SchistoProjectionParams()
    if age_params is None:
        age_params = AgeStructureParams()

    if baseline_prev_pct <= 0:
        raise ValueError("baseline_prev_pct must be > 0")
    if not (0 < baseline_heavy_pct <= baseline_prev_pct):
        raise ValueError("baseline_heavy_pct must be in (0, baseline_prev_pct]")
    if not (0 < coverage_pct <= 100):
        raise ValueError("coverage_pct must be in (0, 100]")

    # ── Derived scalars ───────────────────────────────────────────────────────
    coverage        = coverage_pct / 100.0
    species_mult    = _species_factor(species)
    wash_efficacy_boost  = 1.0 + (wash_gain_pct / 200.0)
    snail_efficacy_boost = 1.08 if snail_control else 1.0

    floor_fraction = max(
        0.02,
        params.floor_transmission_intercept
        - (wash_gain_pct * params.floor_wash_slope)
        - (params.floor_snail_reduction if snail_control else 0.0),
    )

    # ── Age-stratified initial prevalence ─────────────────────────────────────
    # SAC prevalence comes from ESPEN Prev_SAC at the implementing unit level.
    # Adult prevalence uses the following priority order:
    #   1. ESPEN Prev_Adults if available and > 0 for this IU
    #   2. Contact-rate-ratio fallback if ESPEN adult data are missing or zero
    #      (adult_baseline_prev_pct=None signals the fallback path)
    # The contact rate ratio fallback assumes near-equilibrium conditions before
    # MDA — appropriate for units with no prior treatment history but likely to
    # underestimate adult prevalence in settings with high occupational exposure.
    prev_sac = float(baseline_prev_pct)

    if adult_baseline_prev_pct is not None and float(adult_baseline_prev_pct) > 0:
        prev_adult = float(adult_baseline_prev_pct)
    else:
        # Fallback: derive from contact rate ratio (SCHISTOX equilibrium assumption)
        prev_adult = float(baseline_prev_pct
                           * age_params.contact_rate_adult
                           / age_params.contact_rate_sac)

    baseline_sac_prev   = prev_sac
    baseline_adult_prev = prev_adult
    adult_prev_source   = (
        "ESPEN"
        if adult_baseline_prev_pct is not None and float(adult_baseline_prev_pct) > 0
        else "Contact-rate fallback"
    )

    # Overall prevalence = population-weighted mean
    def _overall(ps, pa):
        return (ps * age_params.pop_fraction_sac
                + pa * age_params.pop_fraction_adult)

    baseline_overall = _overall(prev_sac, prev_adult)

    # Floors are set relative to each stratum's baseline
    floor_sac   = max(params.floor_minimum_pct, baseline_sac_prev   * floor_fraction)
    floor_adult = max(params.floor_minimum_pct, baseline_adult_prev * floor_fraction)

    sev = SeverityRatio(initial_ratio=baseline_heavy_pct / baseline_prev_pct)

    # ── State ─────────────────────────────────────────────────────────────────
    current_frequency = frequency
    rounds = _rounds_per_year(current_frequency)
    rows = []
    ephp_year    = None
    hotspot_flag = False
    years = list(range(start_year, start_year + horizon_years + 1))

    for t, year in enumerate(years):
        heavy_sac   = sev.heavy_prev(prev_sac,   t, current_frequency)
        heavy_adult = sev.heavy_prev(prev_adult,  t, current_frequency)
        prev_overall  = _overall(prev_sac, prev_adult)
        heavy_overall = _overall(heavy_sac, heavy_adult)

        if ephp_year is None and heavy_sac < 1.0:
            ephp_year = year

        rows.append({
            "Year":             year,
            "Frequency":        current_frequency,
            "Rounds":           rounds,
            "Prevalence_SAC":   prev_sac,
            "HeavyPrev_SAC":    heavy_sac,
            "Prevalence_Adult": prev_adult,
            "HeavyPrev_Adult":  heavy_adult,
            "Prevalence_Overall": prev_overall,
            "HeavyPrev_Overall":  heavy_overall,
            "EPHP_Achieved":    heavy_sac < 1.0,
            "MDA_Strategy":     age_params.mda_strategy,
            "Adult_Prev_Source": adult_prev_source,
        })

        if t == horizon_years:
            break

        # ── Continuous step-down ──────────────────────────────────────────────
        # Step-down is driven by SAC prevalence, which is the operational
        # measure used by WHO and ESPEN.
        if auto_step_down:
            new_frequency = current_frequency
            if prev_sac < params.step_down_prev_threshold_low:
                new_frequency = "Test-and-treat"
            elif prev_sac < params.step_down_prev_threshold_mid and current_frequency == "Bi-annual":
                new_frequency = "Annual"
            if new_frequency != current_frequency:
                current_frequency = new_frequency
                rounds = _rounds_per_year(current_frequency)

        # ── Coverage per stratum ──────────────────────────────────────────────
        cov_sac   = age_params.effective_coverage_sac(coverage)
        cov_adult = age_params.effective_coverage_adult(coverage)

        # ── SAC reduction ─────────────────────────────────────────────────────
        agg_sac   = _aggregation_prev_scalar(prev_sac, baseline_sac_prev,
                                             params.aggregation_alpha)
        compound_sac = cov_sac * species_mult * wash_efficacy_boost * snail_efficacy_boost * agg_sac
        rr_sac    = min(params.max_prev_reduction_per_round,
                        params.base_prev_reduction_per_round * compound_sac)
        annual_red_sac = 1.0 - (1.0 - rr_sac) ** rounds

        # ── Adult reduction ───────────────────────────────────────────────────
        # Adults have lower contact rates, so their baseline reduction
        # coefficient is scaled by the adult-to-SAC contact rate ratio.
        # Under SAC-only MDA, cov_adult=0 so annual_red_adult=0.
        agg_adult = _aggregation_prev_scalar(prev_adult, baseline_adult_prev,
                                             params.aggregation_alpha)
        adult_contact_scalar = age_params.contact_rate_adult / age_params.contact_rate_sac
        compound_adult = cov_adult * species_mult * wash_efficacy_boost * snail_efficacy_boost * agg_adult * adult_contact_scalar
        rr_adult  = min(params.max_prev_reduction_per_round,
                        params.base_prev_reduction_per_round * compound_adult)
        annual_red_adult = 1.0 - (1.0 - rr_adult) ** rounds

        # ── Density-dependent rebound scalars ─────────────────────────────────
        dd_sac   = _density_dependent_rebound_scalar(
            prev_sac, baseline_sac_prev, params.density_dependent_fecundity_z)
        dd_adult = _density_dependent_rebound_scalar(
            prev_adult, baseline_adult_prev, params.density_dependent_fecundity_z)

        # ── Dynamic floors with adult transmission rebound ────────────────────
        # Under SAC-only MDA, untreated adults sustain environmental
        # contamination, setting a minimum SAC prevalence that cannot be
        # cleared by treating SAC alone (Toor et al. 2019).
        adult_rebound_contrib = age_params.transmission_rebound_from_untreated_adults(
            prev_adult, baseline_adult_prev
        )
        dynamic_floor_sac = max(
            params.floor_minimum_pct,
            min(floor_sac * dd_sac + baseline_sac_prev * adult_rebound_contrib,
                baseline_sac_prev),
        )
        dynamic_floor_adult = max(
            params.floor_minimum_pct,
            min(floor_adult * dd_adult, baseline_adult_prev),
        )

        if current_frequency == "Test-and-treat":
            dynamic_floor_sac   = max(params.floor_minimum_pct,
                                      dynamic_floor_sac   * params.tnt_floor_multiplier)
            dynamic_floor_adult = max(params.floor_minimum_pct,
                                      dynamic_floor_adult * params.tnt_floor_multiplier)
            annual_red_sac   *= params.tnt_efficacy_scalar
            annual_red_adult *= params.tnt_efficacy_scalar

        # ── Update prevalence ─────────────────────────────────────────────────
        prev_sac   = max(dynamic_floor_sac,   prev_sac   * (1.0 - annual_red_sac))
        prev_adult = max(dynamic_floor_adult, prev_adult * (1.0 - annual_red_adult))

        # ── Hotspot detection (SAC-based) ─────────────────────────────────────
        completed_round_years = t + 1
        if (
            not hotspot_flag
            and completed_round_years == params.hotspot_review_year
            and coverage_pct >= params.hotspot_min_coverage
            and baseline_sac_prev >= params.hotspot_min_baseline_prev
        ):
            rel_red = (baseline_sac_prev - prev_sac) / baseline_sac_prev
            if (prev_sac >= params.step_down_prev_threshold_low
                    and rel_red < params.hotspot_relative_reduction_threshold):
                hotspot_flag = True

    projection = pd.DataFrame(rows)
    projection["EPHP_Achieved"] = projection["EPHP_Achieved"].astype(bool)
    projection.attrs["ephp_year"]          = ephp_year
    projection.attrs["persistent_hotspot"] = hotspot_flag
    projection.attrs["severity_ratio"]     = sev
    projection.attrs["age_params"]         = age_params
    projection.attrs["adult_prev_source"]  = adult_prev_source
    return projection


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CASELOAD ESTIMATION
# Translates prevalence (%) + at-risk population → clinical case counts.
# Called at each timestep, so the trajectory drives the disease burden.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchistoClinicalParams:
    """
    Clinical and economic parameters for the PSA.
    All means and SDs are drawn from the literature cited inline.
    Distributions: proportions → Beta; DWs → truncated normal; RRs → Gamma.

    S. mansoni / hepatosplenic cascade
    -----------------------------------
    Sources: van der Werf et al. (2003) Trop Med Int Health;
             King et al. (2005) Lancet; GBD 2019.

    S. haematobium urogenital cascade
    -----------------------------------
    Sources: WHO (2002) Expert Committee; Poggensee & Feldmeier (2001);
             IARC Monograph vol. 100B (2012); GBD 2019.
    """
    n_iterations: int = 1_000

    # ── Infection intensity split ─────────────────────────────────────────────
    # Estimated from SCORE trial and ESPEN survey data. Light = <100 EPG (S. mansoni)
    pct_heavy_of_infected_mean: float = 0.40
    pct_heavy_of_infected_sd:   float = 0.08

    # ── S. mansoni morbidity fractions (% of heavy-infected) ─────────────────
    pct_hepatomegaly_mean: float = 0.22;  pct_hepatomegaly_sd: float = 0.06
    pct_fibrosis_mean:     float = 0.48;  pct_fibrosis_sd:     float = 0.10
    pct_portal_htn_mean:   float = 0.32;  pct_portal_htn_sd:   float = 0.10
    pct_varices_mean:      float = 0.58;  pct_varices_sd:      float = 0.12
    pct_anemia_light_mean: float = 0.18;  pct_anemia_light_sd: float = 0.05
    pct_anemia_heavy_mean: float = 0.45;  pct_anemia_heavy_sd: float = 0.08

    # ── S. haematobium morbidity fractions ───────────────────────────────────
    pct_hematuria_mean:      float = 0.62; pct_hematuria_sd:      float = 0.08
    pct_hydronephrosis_mean: float = 0.15; pct_hydronephrosis_sd: float = 0.05
    pct_fgs_mean:            float = 0.75; pct_fgs_sd:            float = 0.08
    bladder_ca_rr_mean:      float = 4.20; bladder_ca_rr_sd:      float = 1.00
    bg_bladder_ca_per_100k:  float = 3.50; bg_bladder_ca_sd:      float = 0.80

    # ── GBD 2019 disability weights ───────────────────────────────────────────
    dw_anemia_mild_mean:      float = 0.006; dw_anemia_mild_sd:      float = 0.001
    dw_anemia_mod_mean:       float = 0.058; dw_anemia_mod_sd:       float = 0.010
    dw_hepatomegaly_mean:     float = 0.021; dw_hepatomegaly_sd:     float = 0.006
    dw_fibrosis_mean:         float = 0.021; dw_fibrosis_sd:         float = 0.006
    dw_ascites_mean:          float = 0.222; dw_ascites_sd:          float = 0.030
    dw_varices_mean:          float = 0.190; dw_varices_sd:          float = 0.025
    dw_hematuria_mean:        float = 0.020; dw_hematuria_sd:        float = 0.005
    dw_hydronephrosis_mean:   float = 0.149; dw_hydronephrosis_sd:   float = 0.020
    dw_fgs_mean:              float = 0.048; dw_fgs_sd:              float = 0.012
    dw_cancer_primary_mean:   float = 0.288; dw_cancer_primary_sd:   float = 0.035
    dw_cancer_meta_mean:      float = 0.540; dw_cancer_meta_sd:      float = 0.050

    # ── PZQ efficacy (Zwang & Olliaro 2014 meta-analysis) ────────────────────
    cure_rate_mansoni_mean:      float = 0.85; cure_rate_mansoni_sd:      float = 0.07
    cure_rate_haematobium_mean:  float = 0.87; cure_rate_haematobium_sd:  float = 0.06
    morbidity_red_hepatic_mean:  float = 0.70; morbidity_red_hepatic_sd:  float = 0.10
    morbidity_red_urinary_mean:  float = 0.75; morbidity_red_urinary_sd:  float = 0.10
    cancer_red_mda_mean:         float = 0.30; cancer_red_mda_sd:         float = 0.12

    # ── Productivity losses (Audibert 1999; Croce 2010) ───────────────────────
    prod_loss_anemia_mean:      float = 0.12; prod_loss_anemia_sd:      float = 0.04
    prod_loss_hepatomeg_mean:   float = 0.10; prod_loss_hepatomeg_sd:   float = 0.04
    prod_loss_portal_htn_mean:  float = 0.45; prod_loss_portal_htn_sd:  float = 0.10
    prod_loss_varices_mean:     float = 0.65; prod_loss_varices_sd:     float = 0.12
    prod_loss_hematuria_mean:   float = 0.08; prod_loss_hematuria_sd:   float = 0.03
    prod_loss_hydroneph_mean:   float = 0.25; prod_loss_hydroneph_sd:   float = 0.08
    prod_loss_fgs_mean:         float = 0.12; prod_loss_fgs_sd:         float = 0.04
    prod_loss_cancer_mean:      float = 0.75; prod_loss_cancer_sd:      float = 0.10

    working_days_per_year: int = 300


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PSA SAMPLING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _beta(mu: float, sd: float) -> float:
    """Beta distribution parameterised by mean and SD. Clipped to (0,1)."""
    var = sd ** 2
    denom = max(mu * (1 - mu) - var, 1e-9)
    alpha = mu * ((mu * (1 - mu)) / denom - 1)
    beta  = (1 - mu) * ((mu * (1 - mu)) / denom - 1)
    return float(np.clip(np.random.beta(max(alpha, 0.01), max(beta, 0.01)), 1e-6, 1 - 1e-6))


def _tnorm(mu: float, sd: float) -> float:
    """Normal distribution truncated to positive values."""
    v = -1.0
    while v <= 0:
        v = random.gauss(mu, sd)
    return v


def _gamma(mu: float, sd: float) -> float:
    """Gamma distribution parameterised by mean and SD."""
    shape = (mu / sd) ** 2
    scale = sd ** 2 / mu
    return float(np.random.gamma(shape, scale))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SINGLE PSA ITERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _psa_draw(p: SchistoClinicalParams) -> dict:
    """
    Draw one set of stochastic parameters.
    Separated from the calculation so it can be inspected and tested independently.
    """
    return dict(
        # Intensity split
        pct_heavy        = _beta(p.pct_heavy_of_infected_mean, p.pct_heavy_of_infected_sd),
        # Mansoni cascade
        pct_hepatomegaly = _beta(p.pct_hepatomegaly_mean, p.pct_hepatomegaly_sd),
        pct_fibrosis     = _beta(p.pct_fibrosis_mean, p.pct_fibrosis_sd),
        pct_portal_htn   = _beta(p.pct_portal_htn_mean, p.pct_portal_htn_sd),
        pct_varices      = _beta(p.pct_varices_mean, p.pct_varices_sd),
        pct_anemia_light = _beta(p.pct_anemia_light_mean, p.pct_anemia_light_sd),
        pct_anemia_heavy = _beta(p.pct_anemia_heavy_mean, p.pct_anemia_heavy_sd),
        # Haematobium cascade
        pct_hematuria      = _beta(p.pct_hematuria_mean, p.pct_hematuria_sd),
        pct_hydronephrosis = _beta(p.pct_hydronephrosis_mean, p.pct_hydronephrosis_sd),
        pct_fgs            = _beta(p.pct_fgs_mean, p.pct_fgs_sd),
        bladder_ca_rr      = _gamma(p.bladder_ca_rr_mean, p.bladder_ca_rr_sd),
        bg_bladder_ca      = _tnorm(p.bg_bladder_ca_per_100k, p.bg_bladder_ca_sd),
        # Disability weights
        dw_anemia_mild     = _tnorm(p.dw_anemia_mild_mean, p.dw_anemia_mild_sd),
        dw_anemia_mod      = _tnorm(p.dw_anemia_mod_mean, p.dw_anemia_mod_sd),
        dw_hepatomegaly    = _tnorm(p.dw_hepatomegaly_mean, p.dw_hepatomegaly_sd),
        dw_fibrosis        = _tnorm(p.dw_fibrosis_mean, p.dw_fibrosis_sd),
        dw_ascites         = _tnorm(p.dw_ascites_mean, p.dw_ascites_sd),
        dw_varices         = _tnorm(p.dw_varices_mean, p.dw_varices_sd),
        dw_hematuria       = _tnorm(p.dw_hematuria_mean, p.dw_hematuria_sd),
        dw_hydronephrosis  = _tnorm(p.dw_hydronephrosis_mean, p.dw_hydronephrosis_sd),
        dw_fgs             = _tnorm(p.dw_fgs_mean, p.dw_fgs_sd),
        dw_cancer_primary  = _tnorm(p.dw_cancer_primary_mean, p.dw_cancer_primary_sd),
        dw_cancer_meta     = _tnorm(p.dw_cancer_meta_mean, p.dw_cancer_meta_sd),
        # PZQ efficacy
        cure_rate_m        = _beta(p.cure_rate_mansoni_mean, p.cure_rate_mansoni_sd),
        cure_rate_h        = _beta(p.cure_rate_haematobium_mean, p.cure_rate_haematobium_sd),
        morbidity_red_hep  = _beta(p.morbidity_red_hepatic_mean, p.morbidity_red_hepatic_sd),
        morbidity_red_uri  = _beta(p.morbidity_red_urinary_mean, p.morbidity_red_urinary_sd),
        cancer_red         = _beta(p.cancer_red_mda_mean, p.cancer_red_mda_sd),
        # Productivity
        pl_anemia      = _tnorm(p.prod_loss_anemia_mean, p.prod_loss_anemia_sd),
        pl_hepatomeg   = _tnorm(p.prod_loss_hepatomeg_mean, p.prod_loss_hepatomeg_sd),
        pl_portal_htn  = _tnorm(p.prod_loss_portal_htn_mean, p.prod_loss_portal_htn_sd),
        pl_varices     = _tnorm(p.prod_loss_varices_mean, p.prod_loss_varices_sd),
        pl_hematuria   = _tnorm(p.prod_loss_hematuria_mean, p.prod_loss_hematuria_sd),
        pl_hydroneph   = _tnorm(p.prod_loss_hydroneph_mean, p.prod_loss_hydroneph_sd),
        pl_fgs         = _tnorm(p.prod_loss_fgs_mean, p.prod_loss_fgs_sd),
        pl_cancer      = _tnorm(p.prod_loss_cancer_mean, p.prod_loss_cancer_sd),
    )


def _year_burden(
    d: dict,                  # PSA draw
    at_risk: float,
    prev_pct: float,
    heavy_pct: float,
    female_frac: float,
    life_exp: float,
    run_mansoni: bool,
    run_haematobium: bool,
    working_days: int,
) -> dict:
    """
    Calculate disease burden for one PSA draw at one year's prevalence.
    Returns scalars for each outcome in both no-MDA and MDA scenarios.
    The "no-MDA" scenario uses baseline (unadjusted) caseloads.
    The "MDA" scenario applies morbidity reduction factors.
    """
    infected = at_risk * (prev_pct / 100.0)
    heavy_n  = at_risk * (heavy_pct / 100.0)
    light_n  = infected - heavy_n

    out = {}

    # ── S. mansoni cascade ────────────────────────────────────────────────────
    if run_mansoni:
        anemia    = light_n * d["pct_anemia_light"] + heavy_n * d["pct_anemia_heavy"]
        hepatomeg = heavy_n * d["pct_hepatomegaly"]
        fibrosis  = hepatomeg * d["pct_fibrosis"]
        portal    = fibrosis  * d["pct_portal_htn"]
        varices   = portal    * d["pct_varices"]

        anemia_mild = anemia * 0.55
        anemia_mod  = anemia * 0.45

        daly_m = (
            anemia_mild * d["dw_anemia_mild"]
            + anemia_mod  * d["dw_anemia_mod"]
            + hepatomeg   * d["dw_hepatomegaly"]
            + fibrosis    * d["dw_fibrosis"]
            + portal      * d["dw_ascites"]
            + varices     * d["dw_varices"]
        )
        mrh = d["morbidity_red_hep"]
        daly_m_mda = (
            anemia_mild * d["dw_anemia_mild"] * (1 - d["cure_rate_m"])
            + anemia_mod  * d["dw_anemia_mod"]  * (1 - d["cure_rate_m"])
            + hepatomeg   * d["dw_hepatomegaly"] * (1 - mrh)
            + fibrosis    * d["dw_fibrosis"]     * (1 - mrh)
            + portal      * d["dw_ascites"]      * (1 - mrh)
            + varices     * d["dw_varices"]      * (1 - mrh)
        )
        prod_m = (
            anemia    * d["pl_anemia"]
            + hepatomeg * d["pl_hepatomeg"]
            + portal    * d["pl_portal_htn"]
            + varices   * d["pl_varices"]
        ) * working_days
        prod_m_mda = prod_m * (1 - mrh)

        out.update(dict(
            m_infected=infected, m_anemia=anemia,
            m_hepatomeg=hepatomeg, m_fibrosis=fibrosis,
            m_portal=portal, m_varices=varices,
            daly_mansoni=daly_m, daly_mansoni_mda=daly_m_mda,
            prod_days_mansoni=prod_m, prod_days_mansoni_mda=prod_m_mda,
        ))
    else:
        out.update(dict(
            m_infected=0, m_anemia=0, m_hepatomeg=0, m_fibrosis=0,
            m_portal=0, m_varices=0,
            daly_mansoni=0, daly_mansoni_mda=0,
            prod_days_mansoni=0, prod_days_mansoni_mda=0,
        ))

    # ── S. haematobium cascade ────────────────────────────────────────────────
    if run_haematobium:
        hematuria  = infected * d["pct_hematuria"]
        hydroneph  = infected * d["pct_hydronephrosis"]
        fgs        = infected * female_frac * d["pct_fgs"]

        # Bladder cancer PAF: (RR-1)/RR * Pe
        pe  = prev_pct / 100.0
        rr  = d["bladder_ca_rr"]
        paf = (rr - 1) / rr * pe
        bg  = d["bg_bladder_ca"] / 100_000.0
        tot_ca  = at_risk * bg * (1 + paf * (rr - 1))
        ca_prim = tot_ca * 0.72
        ca_meta = tot_ca * 0.28

        mean_age_dx = 55.0
        yll_prim = ca_prim * 0.35 * max(life_exp - mean_age_dx, 0.0)
        yll_meta = ca_meta * 0.85 * max(life_exp - mean_age_dx, 0.0)

        daly_h = (
            hematuria * d["dw_hematuria"]
            + hydroneph * d["dw_hydronephrosis"]
            + fgs       * d["dw_fgs"]
            + ca_prim   * d["dw_cancer_primary"] * 5.0
            + ca_meta   * d["dw_cancer_meta"]    * 1.5
            + yll_prim + yll_meta
        )
        mru = d["morbidity_red_uri"]
        crm = d["cancer_red"]
        daly_h_mda = (
            hematuria * d["dw_hematuria"]        * (1 - mru)
            + hydroneph * d["dw_hydronephrosis"] * (1 - mru)
            + fgs       * d["dw_fgs"]            * (1 - mru)
            + (ca_prim  * d["dw_cancer_primary"] * 5.0
               + ca_meta * d["dw_cancer_meta"]   * 1.5
               + yll_prim + yll_meta)             * (1 - crm)
        )
        prod_h = (
            hematuria * d["pl_hematuria"]
            + hydroneph * d["pl_hydroneph"]
            + fgs       * d["pl_fgs"]
            + tot_ca    * d["pl_cancer"]
        ) * working_days
        prod_h_mda = (
            hematuria * (1 - mru) * d["pl_hematuria"]
            + hydroneph * (1 - mru) * d["pl_hydroneph"]
            + fgs       * (1 - mru) * d["pl_fgs"]
            + tot_ca    * (1 - crm) * d["pl_cancer"]
        ) * working_days

        out.update(dict(
            h_infected=infected, h_hematuria=hematuria,
            h_hydroneph=hydroneph, h_fgs=fgs,
            h_tot_ca=tot_ca, h_paf=paf,
            daly_haematobium=daly_h, daly_haematobium_mda=daly_h_mda,
            prod_days_haematobium=prod_h, prod_days_haematobium_mda=prod_h_mda,
        ))
    else:
        out.update(dict(
            h_infected=0, h_hematuria=0, h_hydroneph=0, h_fgs=0,
            h_tot_ca=0, h_paf=0,
            daly_haematobium=0, daly_haematobium_mda=0,
            prod_days_haematobium=0, prod_days_haematobium_mda=0,
        ))

    out["daly_total"]     = out["daly_mansoni"]     + out["daly_haematobium"]
    out["daly_total_mda"] = out["daly_mansoni_mda"] + out["daly_haematobium_mda"]
    out["prod_days_total"]     = out["prod_days_mansoni"]     + out["prod_days_haematobium"]
    out["prod_days_total_mda"] = out["prod_days_mansoni_mda"] + out["prod_days_haematobium_mda"]

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ORCHESTRATOR
# Runs PSA at every year of the trajectory, not just at baseline.
# This is the key architectural improvement over both Options 1 and 2.
# ═══════════════════════════════════════════════════════════════════════════════

def _annual_program_cost_for_row(
    row: pd.Series,
    at_risk_population: int,
    pzq_cost_per_person_round: float,
    fixed_annual_cost: float,
    test_and_treat_target_share: float = 0.15,
) -> float:
    """Convert treatment frequency into a year-specific programme cost.

    Bi-annual: 2 full rounds
    Annual:    1 full round
    TNT:       targeted treatment only (default 15% of at-risk population)
    """
    freq = str(row["Frequency"])
    rounds = float(row.get("Rounds", 1.0))

    if freq == "Test-and-treat":
        variable_cost = at_risk_population * test_and_treat_target_share * pzq_cost_per_person_round
    else:
        variable_cost = at_risk_population * rounds * pzq_cost_per_person_round

    return float(variable_cost + fixed_annual_cost)


def schisto_run(
    projection: pd.DataFrame,
    at_risk_population: int,
    clinical_params: SchistoClinicalParams,
    daily_wage_usd: float,
    pzq_cost_per_person_round: float,
    fixed_annual_cost: float,
    annual_prog_cost: float,
    disc_rate: float,
    female_fraction: float,
    life_expectancy: float,
    run_mansoni: bool,
    run_haematobium: bool,
) -> pd.DataFrame:
    """
    For each year in the trajectory, run N PSA iterations and store
    the mean and 90% CI for each outcome.

    Returns a year-level summary DataFrame with columns:
        Year, Prevalence, HeavyPrev, Frequency,
        daly_total_mean/lo/hi, daly_total_mda_mean/lo/hi,
        dalys_averted_mean/lo/hi,
        prod_days_total_mean, prod_days_total_mda_mean,
        econ_gain_usd_mean,
        prog_cost_discounted,
        icer_mean/lo/hi
    """
    p = clinical_params
    n = p.n_iterations
    rows = []

    for i, traj_row in projection.iterrows():
        year      = int(traj_row["Year"])
        prev_pct  = float(traj_row["Prevalence_Overall"])
        heavy_pct = float(traj_row["HeavyPrev_Overall"])
        t        = i  # time index

        # Run PSA for this year
        draws = [_psa_draw(p) for _ in range(n)]
        burdens = np.array([
            list(_year_burden(
                d, at_risk_population, prev_pct, heavy_pct,
                female_fraction, life_expectancy,
                run_mansoni, run_haematobium,
                p.working_days_per_year,
            ).values())
            for d in draws
        ])
        col_names = list(_year_burden(
            draws[0], at_risk_population, prev_pct, heavy_pct,
            female_fraction, life_expectancy,
            run_mansoni, run_haematobium,
            p.working_days_per_year,
        ).keys())
        sim_df = pd.DataFrame(burdens, columns=col_names)

        dalys_averted = sim_df["daly_total"] - sim_df["daly_total_mda"]
        econ_gain     = (sim_df["prod_days_total"] - sim_df["prod_days_total_mda"]) * daily_wage_usd
        annual_prog_cost = _annual_program_cost_for_row(
            row=traj_row,
            at_risk_population=at_risk_population,
            pzq_cost_per_person_round=pzq_cost_per_person_round,
            fixed_annual_cost=fixed_annual_cost,
        )
        disc          = (1 + disc_rate) ** (t + 1)
        prog_cost_disc= annual_prog_cost / disc

        with np.errstate(invalid="ignore", divide="ignore"):
            icer_vec = np.where(dalys_averted > 0, annual_prog_cost / dalys_averted, np.nan)

        def _ci(col):
            return col.mean(), col.quantile(0.05), col.quantile(0.95)

        dm, dl, dh   = _ci(sim_df["daly_total"])
        dmm, dlm, dhm= _ci(sim_df["daly_total_mda"])
        am, al, ah   = _ci(dalys_averted)
        im, il, ih   = (np.nanmean(icer_vec),
                        np.nanpercentile(icer_vec, 5),
                        np.nanpercentile(icer_vec, 95))

        rows.append({
            "Year":                    year,
            "Prevalence":              prev_pct,
            "HeavyPrev":               heavy_pct,
            "Frequency":               traj_row["Frequency"],
            "EPHP_Achieved":           bool(traj_row["EPHP_Achieved"]),
            # DALYs
            "daly_total_mean":         dm,  "daly_total_lo": dl,  "daly_total_hi": dh,
            "daly_mda_mean":           dmm, "daly_mda_lo":   dlm, "daly_mda_hi":   dhm,
            "dalys_averted_mean":      am,  "dalys_averted_lo": al, "dalys_averted_hi": ah,
            # Productivity
            "prod_days_mean":          sim_df["prod_days_total"].mean(),
            "prod_days_mda_mean":      sim_df["prod_days_total_mda"].mean(),
            "econ_gain_usd_mean":      econ_gain.mean(),
            # Costs
            "prog_cost_discounted":    prog_cost_disc,
            # ICER
            "icer_mean":               im, "icer_lo": il, "icer_hi": ih,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def schisto_cea_summary(
    results: pd.DataFrame,
    cet_usd: float,
    gdp_ppp: float,
) -> pd.DataFrame:
    """
    Produce a one-row-per-indicator summary table for the Results tab.
    """
    total_dalys_averted = results["dalys_averted_mean"].sum()
    total_prog_cost     = results["prog_cost_discounted"].sum()
    total_econ_gain     = results["econ_gain_usd_mean"].sum()
    icer_end            = results["icer_mean"].iloc[-1]

    rows = [
        ["Baseline prevalence (%)",           f"{results['Prevalence'].iloc[0]:.1f}%"],
        ["End-horizon prevalence (%)",         f"{results['Prevalence'].iloc[-1]:.2f}%"],
        ["End-horizon heavy-intensity (%)",    f"{results['HeavyPrev'].iloc[-1]:.2f}%"],
        ["EPHP achieved",                      "Yes" if results["EPHP_Achieved"].any() else "No"],
        ["First year EPHP achieved",           str(results.loc[results["EPHP_Achieved"], "Year"].min())
                                               if results["EPHP_Achieved"].any() else "Not in horizon"],
        ["Cumulative DALYs averted",           f"{total_dalys_averted:,.0f}"],
        ["Discounted cumulative programme cost (USD)", f"${total_prog_cost:,.0f}"],
        ["Cumulative productivity gains (USD)",f"${total_econ_gain:,.0f}"],
        ["End-year ICER (USD/DALY)",           f"${icer_end:,.0f}"],
        ["Cost-effective vs Woods CET",        "Yes" if icer_end < cet_usd else "No"],
        ["Cost-effective vs GDP threshold",    "Yes" if icer_end < gdp_ppp else "No"],
    ]
    return pd.DataFrame(rows, columns=["Indicator", "Value"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def chart_trajectory(results: pd.DataFrame) -> alt.Chart:
    value_vars = ["Prevalence_SAC", "Prevalence_Adult",
                  "HeavyPrev_SAC",  "HeavyPrev_Adult"]
    # Keep only columns that exist (guard against partial DataFrames)
    value_vars = [c for c in value_vars if c in results.columns]
    df = results[["Year"] + value_vars].melt(
        "Year", var_name="Series", value_name="Percent"
    )
    label_map = {
        "Prevalence_SAC":   "SAC overall prevalence",
        "Prevalence_Adult": "Adult overall prevalence",
        "HeavyPrev_SAC":    "SAC heavy-intensity prevalence",
        "HeavyPrev_Adult":  "Adult heavy-intensity prevalence",
    }
    df["Series"] = df["Series"].map(label_map)
    # Dashed lines for heavy-intensity, solid for overall
    df["Dash"] = df["Series"].str.contains("heavy").map({True: "heavy", False: "overall"})

    ephp_line = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(color="#E74C3C", strokeDash=[6, 3])
        .encode(y="y:Q")
    )
    traj = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:O"),
            y=alt.Y("Percent:Q", title="Prevalence (%)"),
            color=alt.Color("Series:N"),
            strokeDash=alt.StrokeDash("Dash:N",
                scale=alt.Scale(domain=["overall", "heavy"],
                                range=[[1, 0], [4, 2]])),
            tooltip=["Year:O", "Series:N", alt.Tooltip("Percent:Q", format=".2f")],
        )
        .properties(title="Prevalence trajectory by age group — dashed: heavy intensity, red line: EPHP threshold (1%)")
    )
    return (traj + ephp_line).interactive()


def chart_dalys(results: pd.DataFrame) -> alt.Chart:
    df = results[["Year", "daly_total_mean", "daly_mda_mean"]].melt(
        "Year", var_name="Scenario", value_name="DALYs"
    )
    df["Scenario"] = df["Scenario"].map({
        "daly_total_mean": "No MDA", "daly_mda_mean": "With MDA"
    })
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:O"),
            y=alt.Y("DALYs:Q", axis=alt.Axis(format=",.0f")),
            color=alt.Color("Scenario:N",
                            scale=alt.Scale(domain=["No MDA", "With MDA"],
                                            range=["#C0392B", "#1A7851"])),
            tooltip=["Year:O", "Scenario:N", alt.Tooltip("DALYs:Q", format=",.0f")],
        )
        .properties(title="Annual DALYs: no-MDA vs MDA scenario")
        .interactive()
    )


def chart_icer(results: pd.DataFrame, cet: float, gdp: float) -> alt.Chart:
    base = (
        alt.Chart(results)
        .mark_line(point=True, color="#1A5276")
        .encode(
            x=alt.X("Year:O"),
            y=alt.Y("icer_mean:Q", title="ICER (USD/DALY averted)",
                    axis=alt.Axis(format="$,.0f")),
            tooltip=["Year:O",
                     alt.Tooltip("icer_mean:Q", format="$,.0f"),
                     alt.Tooltip("icer_lo:Q",   format="$,.0f"),
                     alt.Tooltip("icer_hi:Q",   format="$,.0f")],
        )
    )
    cet_line = (
        alt.Chart(pd.DataFrame({"ICER": [cet]}))
        .mark_rule(color="#E74C3C", strokeDash=[6, 3])
        .encode(y="ICER:Q")
    )
    gdp_line = (
        alt.Chart(pd.DataFrame({"ICER": [gdp]}))
        .mark_rule(color="#F39C12", strokeDash=[4, 4])
        .encode(y="ICER:Q")
    )
    return (base + cet_line + gdp_line).properties(
        title="ICER over time — red: Woods CET, orange: GDP threshold"
    ).interactive()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — STREAMLIT UI ENTRY POINT
# Call this from the main tool: schisto_ui(tabs, country, country_inputs, ...)
# ═══════════════════════════════════════════════════════════════════════════════

def schisto_ui(tabs, country: str, country_inputs: pd.DataFrame,
               daily_wage_fn, translate_fn, cea_threshold_fn, **kwargs):
    """
    Drop-in Streamlit UI for the schistosomiasis module.

    Parameters
    ----------
    tabs            : list of st.tab objects from the parent tool
    country         : selected country string
    country_inputs  : the df_gdp DataFrame already loaded in the parent tool
    daily_wage_fn   : the daily_wage() function from the parent tool
    translate_fn    : the translate_text() function from the parent tool
    cea_threshold_fn: the cea_threshold() function from the parent tool
    """
    T = translate_fn  # alias

    espen_full = kwargs.get("espen_unit", pd.DataFrame())

    # ── Admin hierarchy selectors ─────────────────────────────────────────────
    # ESPEN uses: admin0, admin1, admin2, iusName as the hierarchy columns.
    # Filter to selected country (admin0) and most recent year available.
    if not espen_full.empty:
        espen_country = espen_full[espen_full["admin0"] == country].copy()
        latest_year   = int(espen_country["year"].max()) if "year" in espen_country.columns else None
        if latest_year:
            espen_country = espen_country[espen_country["year"] == latest_year]
    else:
        espen_country = pd.DataFrame()

    # Admin 1
    if not espen_country.empty and "admin1" in espen_country.columns:
        admin1_opts = sorted(espen_country["admin1"].dropna().unique().tolist())
        admin1_opts.insert(0, T("National level"))
        admin1_sel = st.sidebar.selectbox(T("Administrative Unit 1"), admin1_opts)
    else:
        admin1_sel = T("National level")

    if admin1_sel == T("National level"):
        espen_unit = espen_country.copy()
        unit_label = country
    else:
        espen_adm1 = espen_country[espen_country["admin1"] == admin1_sel]
        if "admin2" in espen_adm1.columns:
            admin2_opts = sorted(espen_adm1["admin2"].dropna().unique().tolist())
            admin2_opts.insert(0, T(f"All of {admin1_sel}"))
            admin2_sel = st.sidebar.selectbox(T("Administrative Unit 2"), admin2_opts)
        else:
            admin2_sel = T(f"All of {admin1_sel}")

        if admin2_sel.startswith(T("All of")):
            espen_unit = espen_adm1.copy()
            unit_label = admin1_sel
        else:
            espen_adm2 = espen_adm1[espen_adm1["admin2"] == admin2_sel]
            if "iusName" in espen_adm2.columns:
                iu_opts = sorted(espen_adm2["iusName"].dropna().unique().tolist())
                iu_opts.insert(0, T(f"All of {admin2_sel}"))
                iu_sel = st.sidebar.selectbox(T("Implementing Unit"), iu_opts)
            else:
                iu_sel = T(f"All of {admin2_sel}")

            if iu_sel.startswith(T("All of")):
                espen_unit = espen_adm2.copy()
                unit_label = admin2_sel
            else:
                espen_unit = espen_adm2[espen_adm2["iusName"] == iu_sel].copy()
                unit_label = iu_sel

    # ── Extract prevalence and population from ESPEN ──────────────────────────
    sac_prev_espen   = _espen_prevalence(espen_unit, "SAC")   if not espen_unit.empty else None
    adult_prev_espen = _espen_prevalence(espen_unit, "Adults") if not espen_unit.empty else None
    pop_req_sac      = _espen_pop_req(espen_unit, "SAC")       if not espen_unit.empty else 0
    pop_req_adults   = _espen_pop_req(espen_unit, "Adults")    if not espen_unit.empty else 0
    pop_req_total    = _espen_pop_req(espen_unit, "Total")     if not espen_unit.empty else 0
    who_band         = _espen_who_band(espen_unit)             if not espen_unit.empty else "Unknown"
    default_freq     = _who_band_to_frequency(who_band)

    # SAC fraction of total at-risk population (from ESPEN target populations)
    pop_frac_sac = (pop_req_sac / pop_req_total) if pop_req_total > 0 else 0.23

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    prog_exp = st.sidebar.expander(T("Schistosomiasis programme details"), expanded=True)
    with prog_exp:
        st.caption(T(f"Unit: **{unit_label}** | WHO band: **{who_band}**"))

        species = st.radio(T("Species"), ["S. mansoni", "S. haematobium", "Both species"])
        run_m = "mansoni" in species.lower() or "both" in species.lower()
        run_h = "haematobium" in species.lower() or "both" in species.lower()

        baseline_prev = st.slider(
            T("Baseline SAC prevalence (%)"),
            1, 100,
            int(round(sac_prev_espen)) if sac_prev_espen else 25,
            help=T("Pre-populated from ESPEN pc (SAC) for this unit. Adjust if using a more recent survey."),
        )
        baseline_heavy = st.slider(
            T("Baseline heavy-intensity prevalence (%)"),
            1, baseline_prev,
            max(1, int(baseline_prev * 0.15)),
        )

        # Show ESPEN adult prevalence as read-only information
        if adult_prev_espen is not None:
            st.info(T(f"ESPEN adult prevalence (pc): **{adult_prev_espen:.1f}%** — used as adult baseline."))
        else:
            st.warning(T("No ESPEN adult prevalence data for this unit — contact-rate fallback will be used."))

        coverage = st.slider(T("MDA coverage (%)"), 10, 100, 75)

        freq_opts = ["Bi-annual", "Annual", "Test-and-treat"]
        frequency = st.radio(
            T("Treatment frequency"),
            freq_opts,
            index=freq_opts.index(default_freq),
            help=T("Default set from WHO endemicity band for this unit."),
        )
        wash_gain  = st.slider(T("Incremental WASH improvement (%)"), 0, 100, 20)
        snail_ctrl = st.radio(T("Snail control"), ["No", "Yes"]) == "Yes"
        horizon    = st.slider(T("Time horizon (years)"), 5, 25, 10)
        female_frac= st.slider(T("% female in at-risk population"), 30, 70, 50) / 100.0

        st.markdown("---")
        st.markdown(T("**Age structure and MDA strategy**"))
        mda_strategy = st.radio(
            T("MDA target population"),
            ["SAC-only", "Community-wide"],
            index=0 if baseline_prev < 50 else 1,
            help=T("WHO recommends community-wide MDA when SAC prevalence ≥ 50% "
                   "or where adult burden is high (Toor et al. 2019)."),
        )
        contact_rate_adult = st.slider(
            T("Adult contact rate relative to SAC"),
            min_value=0.01, max_value=0.50,
            value=0.06, step=0.01,
            help=T("0.06 = moderate adult burden. Up to 0.30 for fishing communities "
                   "(Turner et al. 2017; Toor et al. 2019)."),
        )
        pop_fraction_sac = st.slider(
            T("SAC fraction of at-risk population (%)"),
            min_value=10, max_value=40,
            value=int(round(pop_frac_sac * 100)) if pop_frac_sac else 23,
            help=T("Pre-populated from ESPEN SAC/Total popReq ratio for this unit."),
        ) / 100.0

    cost_exp = st.sidebar.expander(T("Costing inputs"))
    with cost_exp:
        # Default at-risk population from ESPEN Total popReq for the unit
        pop_default = pop_req_total if pop_req_total > 0 else int(
            country_inputs["Officialfigure"][country_inputs["Country"] == country].values[0] * 0.25
        )
        at_risk_pop = st.number_input(T("Population requiring treatment"), value=pop_default, step=1000)
        pzq_cost    = st.number_input(T("PZQ cost per person per round (USD)"), value=0.45, step=0.05)
        fixed_cost  = st.number_input(T("Fixed annual programme costs (USD)"), value=25_000.0, step=1000.0)
        disc_rate   = st.number_input(T("Discount rate"), value=0.03, step=0.01)
        n_iter      = st.select_slider(T("PSA iterations"), [100, 500, 1000], value=1000)

    # ── Derived economic inputs from parent tool ──────────────────────────────
    annual_ppp   = float(country_inputs["Annual_PPP(Int$)"][country_inputs["Country"] == country].values[0])
    life_exp     = float(country_inputs["Life_Expectancy"][country_inputs["Country"] == country].values[0])
    daily_wage   = float(daily_wage_fn()[0].iloc[0] if hasattr(daily_wage_fn()[0], "iloc") else daily_wage_fn()[0])
    cet          = cea_threshold_fn()
    annual_prog_cost = pzq_cost * at_risk_pop + fixed_cost

    # Adult baseline prevalence: ESPEN first, contact-rate fallback if absent
    adult_baseline_prev = adult_prev_espen  # None triggers fallback in schisto_projection

    # ── Run model ─────────────────────────────────────────────────────────────
    proj_params     = SchistoProjectionParams()
    clinical_params = SchistoClinicalParams(n_iterations=n_iter)
    age_params      = AgeStructureParams(
        contact_rate_adult=contact_rate_adult,
        pop_fraction_sac=pop_fraction_sac,
        mda_strategy=mda_strategy,
    )

    projection = schisto_projection(
        baseline_prev_pct=float(baseline_prev),
        baseline_heavy_pct=float(baseline_heavy),
        start_year=2025,
        horizon_years=horizon,
        coverage_pct=float(coverage),
        frequency=frequency,
        species=species if species != "Both species" else "Mixed / unknown",
        wash_gain_pct=float(wash_gain),
        snail_control=snail_ctrl,
        params=proj_params,
        age_params=age_params,
        adult_baseline_prev_pct=adult_baseline_prev,
    )

    with st.spinner(T("Running probabilistic sensitivity analysis...")):
        results = schisto_run(
            projection=projection,
            at_risk_population=int(at_risk_pop),
            clinical_params=clinical_params,
            daily_wage_usd=daily_wage,
            annual_prog_cost=annual_prog_cost,
            disc_rate=disc_rate,
            female_fraction=female_frac,
            life_expectancy=life_exp,
            run_mansoni=run_m,
            run_haematobium=run_h,
        )

    summary = schisto_cea_summary(results, cet_usd=cet, gdp_ppp=annual_ppp)
    ephp_year   = projection.attrs.get("ephp_year")
    hotspot     = projection.attrs.get("persistent_hotspot", False)

    # ── Tab content ───────────────────────────────────────────────────────────
    with tabs[2]:  # Disease inputs tab
        with st.expander(T("Prevalence trajectory"), expanded=True):
            st.altair_chart(chart_trajectory(results), use_container_width=True)
            st.dataframe(projection[["Year", "Frequency", "Prevalence", "HeavyPrev", "EPHP_Achieved"]]
                         .style.format({"Prevalence": "{:.2f}", "HeavyPrev": "{:.2f}"}))

    with tabs[3]:  # Results tab
        if ephp_year:
            st.success(T(f"Elimination as a public health problem projected in {ephp_year}."))
        else:
            st.warning(T("EPHP threshold not reached within the selected time horizon."))
        if hotspot:
            st.warning(T("Persistent hotspot indicator triggered. Review coverage quality and local transmission drivers."))

        with st.expander(T("Summary of results"), expanded=True):
            st.dataframe(summary)

        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(chart_dalys(results), use_container_width=True)
        with col2:
            st.altair_chart(chart_icer(results, cet, annual_ppp), use_container_width=True)

        with st.expander(T("Year-by-year results table")):
            fmt = {c: "${:,.0f}" if "usd" in c or "cost" in c
                   else ("{:.2f}" if c in ("Prevalence", "HeavyPrev") else "{:,.0f}")
                   for c in results.columns if results[c].dtype == float}
            st.dataframe(results.style.format(fmt))
            st.download_button(T("Download results CSV"), results.to_csv(index=False).encode(), "schisto_results.csv")

    with tabs[4]:  # Technical assumptions tab
        with st.expander(T("Projection model parameters")):
            proj_tech = pd.DataFrame([
                ["Base prev. reduction/round", proj_params.base_prev_reduction_per_round, "SCORE trial (Colley 2014)"],
                ["Base heavy reduction/round", proj_params.base_heavy_reduction_per_round, "Zwang & Olliaro 2014 meta-analysis"],
                ["Floor transmission intercept", proj_params.floor_transmission_intercept, "SCORE plateau sites"],
                ["Floor WASH slope (per pp)", proj_params.floor_wash_slope, "Model assumption — calibrate locally"],
                ["Floor snail reduction", proj_params.floor_snail_reduction, "Sokolow et al. 2016"],
                ["Step-down threshold (low)", proj_params.step_down_prev_threshold_low, "WHO 2022 guidelines"],
                ["Step-down threshold (mid)", proj_params.step_down_prev_threshold_mid, "WHO 2022 guidelines"],
                ["TNT efficacy scalar", proj_params.tnt_efficacy_scalar, "Model assumption"],
                ["Hotspot review year", proj_params.hotspot_review_year, "Operational convention"],
            ], columns=["Parameter", "Value", "Source"])
            st.table(proj_tech)

        with st.expander(T("Clinical and economic parameters")):
            clin_tech = pd.DataFrame([
                ["PZQ cure rate (S. mansoni)", "85% (SD 7%)", "Zwang & Olliaro 2014"],
                ["PZQ cure rate (S. haematobium)", "87% (SD 6%)", "Zwang & Olliaro 2014"],
                ["Hepatic morbidity reduction", "70% (SD 10%)", "Garba et al. 2013"],
                ["Urinary morbidity reduction", "75% (SD 10%)", "Garba et al. 2013"],
                ["Bladder cancer RR", "4.2 (SD 1.0)", "IARC Monograph vol. 100B (2012)"],
                ["DW anemia mild", "0.006", "GBD 2019"],
                ["DW ascites/portal HTN", "0.222", "GBD 2019"],
                ["DW FGS", "0.048", "GBD 2019"],
                ["DW bladder cancer (primary)", "0.288", "GBD 2019"],
                ["Prod. loss — varices", "65% (SD 12%)", "Croce et al. 2010"],
                ["Prod. loss — anemia", "12% (SD 4%)", "Audibert et al. 1999"],
                ["CEA threshold", f"${cet:,.0f} / DALY", "Woods et al. 2016"],
            ], columns=["Parameter", "Value", "Source"])
            st.table(clin_tech)
