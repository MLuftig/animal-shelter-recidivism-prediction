import streamlit as st
import numpy as np
import requests
from scipy import stats

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Shelter Overflow Risk Forecaster", page_icon="🐾", layout="wide")

# ============================================================
# City parameter sets, derived from real shelter data
# (see src/ notebooks for full derivation of both)
# ============================================================
CITY_PARAMS = {
    "Austin, TX": {
        "p_long_stay": 0.0319,
        "shape_short": 0.504, "scale_short": 25.278,
        "shape_long": 0.736, "scale_long": 126.786, "threshold": 100,
        "n_nb": 7.462, "p_nb": 0.141,
        "mean_intake": 45.42,
        "dispersion_ratio": 7.09,
        "pct_change_per_degF": 0.007642,
        "historical_avg_temp": 69.25,
        "default_capacity": 900,
        "capacity_range": (700, 1200),
    },
    "Bloomington, IN": {
        "p_long_stay": 0.0711,
        "shape_short": 0.516, "scale_short": 42.578,
        "shape_long": 0.703, "scale_long": 228.157, "threshold": 100,
        "n_nb": 1.7141, "p_nb": 0.1274,
        "mean_intake": 11.7417,
        "dispersion_ratio": 7.8499,
        "pct_change_per_degF": 0.016290,     # steeper per-degree effect than Austin
        "historical_avg_temp": 55.98,        # converted from 13.32 degC; raw source data is Celsius, unlike Austin's Fahrenheit
        "default_capacity": 450,
        "capacity_range": (350, 600),
    },
}


# ============================================================
# Core model functions
# ============================================================
def adjusted_intake_mean(forecast_temp, params):
    temp_diff = forecast_temp - params["historical_avg_temp"]
    multiplier = (1 + params["pct_change_per_degF"]) ** temp_diff
    return params["mean_intake"] * multiplier


def nb_params_from_mean(mean, dispersion_ratio):
    var = mean * dispersion_ratio
    p = mean / var
    n = mean * p / (1 - p)
    return n, p


def generate_steady_state_population(params, n_days=365, seed=42):
    rng = np.random.default_rng(seed)
    current_animals = []
    for day in range(n_days):
        current_animals = [los - 1 for los in current_animals if los - 1 > 0]
        new_arrivals = stats.nbinom.rvs(params["n_nb"], params["p_nb"], random_state=rng)
        for _ in range(new_arrivals):
            if rng.random() < params["p_long_stay"]:
                los = stats.gamma.rvs(params["shape_long"], loc=params["threshold"], scale=params["scale_long"], random_state=rng)
            else:
                los = stats.gamma.rvs(params["shape_short"], loc=0, scale=params["scale_short"], random_state=rng)
            current_animals.append(los)
    return current_animals


def simulate_forecast_period(temps, capacity, starting_population, params, rng):
    current_animals = starting_population.copy()
    days_over_capacity = 0
    daily_population = []

    for day_temp in temps:
        current_animals = [los - 1 for los in current_animals if los - 1 > 0]
        adj_mean = adjusted_intake_mean(day_temp, params)
        n_day, p_day = nb_params_from_mean(adj_mean, params["dispersion_ratio"])
        new_arrivals = stats.nbinom.rvs(n_day, p_day, random_state=rng)

        for _ in range(new_arrivals):
            if rng.random() < params["p_long_stay"]:
                los = stats.gamma.rvs(params["shape_long"], loc=params["threshold"], scale=params["scale_long"], random_state=rng)
            else:
                los = stats.gamma.rvs(params["shape_short"], loc=0, scale=params["scale_short"], random_state=rng)
            current_animals.append(los)

        pop_today = len(current_animals)
        daily_population.append(pop_today)
        if pop_today > capacity:
            days_over_capacity += 1

    return days_over_capacity, daily_population


# ============================================================
# ZIP -> coordinates -> live 7-day forecast
# ============================================================
def fetch_location_from_zip(zip_code):
    """ZIP -> lat/lon/place name via Zippopotam.us (free, no API key)."""
    response = requests.get(f"http://api.zippopotam.us/us/{zip_code}", timeout=10)
    response.raise_for_status()
    data = response.json()
    place = data["places"][0]
    return {
        "lat": float(place["latitude"]),
        "lon": float(place["longitude"]),
        "name": f"{place['place name']}, {place['state abbreviation']}",
    }


def fetch_7day_forecast(lat, lon):
    """Live 7-day daily-mean-temperature forecast via Open-Meteo (free, no API key)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_mean&temperature_unit=fahrenheit"
        f"&forecast_days=7&timezone=auto"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data["daily"]["time"], data["daily"]["temperature_2m_mean"]


# ============================================================
# Streamlit UI
# ============================================================
st.title("🐾 Shelter Overflow Risk Forecaster")
st.write(
    "Estimate the probability of exceeding shelter capacity this week, "
    "comparing two independently-trained weather-intake models side by side "
    "-- one fit on real Austin Animal Center data, one on real Bloomington "
    "Animal Care & Control data. Both models found 'feels-like'/average "
    "temperature to be a significant, positive predictor of daily intake."
)
st.info(
    "⚠️ **Note on scope:** both models were trained and validated only on their "
    "own home shelter's historical data. Applying either model to a third, "
    "unrelated ZIP code's weather is an extrapolation beyond what's been "
    "tested -- this tool is meant to compare how two real, differently-trained "
    "models respond to the same local conditions, not as a validated "
    "prediction for your specific shelter."
)

st.subheader("7-Day Temperature Forecast")
st.caption(
    "Enter a ZIP code to pull a live 7-day forecast for both cities, or "
    "adjust each city's daily average temperatures (°F) manually below. "
    "Before fetching, each city defaults to its own historical average, "
    "so the starting comparison isn't biased toward either city. Warmer "
    "conditions are associated with higher shelter intake in both cities' data."
)

col_zip, col_button = st.columns([2, 1])
with col_zip:
    zip_code = st.text_input("ZIP code", value="", max_chars=5, placeholder="e.g. 78701")
with col_button:
    st.write("")
    st.write("")
    fetch_clicked = st.button("Fetch live forecast", type="secondary")

if fetch_clicked and zip_code:
    try:
        location = fetch_location_from_zip(zip_code)
        dates, temps_fetched = fetch_7day_forecast(location["lat"], location["lon"])
        for i, t in enumerate(temps_fetched):
            st.session_state[f"austin_temp_{i}"] = round(t)
            st.session_state[f"bloom_temp_{i}"] = round(t)
        st.session_state["fetched_location"] = location["name"]
        st.session_state["fetched_dates"] = dates
        st.success(f"Loaded live 7-day forecast for {location['name']} -- applied to both cities' models.")
    except Exception as e:
        st.error(f"Couldn't fetch forecast for that ZIP code: {e}. Enter temperatures manually below.")

if "fetched_location" in st.session_state:
    st.caption(f"📍 Both cities' models are being tested against real weather for **{st.session_state['fetched_location']}**")

city_temps = {}
for city_name, key_prefix, default_temp in [
    ("Austin, TX", "austin_temp", round(CITY_PARAMS["Austin, TX"]["historical_avg_temp"])),
    ("Bloomington, IN", "bloom_temp", round(CITY_PARAMS["Bloomington, IN"]["historical_avg_temp"])),
]:
    st.markdown(f"**{city_name}**" + (" *(defaults to its own historical average)*" if "fetched_location" not in st.session_state else ""))
    cols = st.columns(7)
    temps = []
    for i, col in enumerate(cols):
        with col:
            label = f"Day {i+1}"
            if "fetched_dates" in st.session_state:
                label = st.session_state["fetched_dates"][i]
            t = st.number_input(
                label, min_value=-20, max_value=120, step=1,
                key=f"{key_prefix}_{i}", value=st.session_state.get(f"{key_prefix}_{i}", default_temp)
            )
            temps.append(t)
    city_temps[city_name] = temps

st.subheader("Kennel Capacity")
st.caption("Each city's shelter operates at a different scale -- set capacity separately for each.")
cap_col1, cap_col2 = st.columns(2)
capacities = {}
with cap_col1:
    lo, hi = CITY_PARAMS["Austin, TX"]["capacity_range"]
    capacities["Austin, TX"] = st.slider("Austin, TX capacity (max animals)", lo, hi, CITY_PARAMS["Austin, TX"]["default_capacity"], step=10)
with cap_col2:
    lo, hi = CITY_PARAMS["Bloomington, IN"]["capacity_range"]
    capacities["Bloomington, IN"] = st.slider("Bloomington, IN capacity (max animals)", lo, hi, CITY_PARAMS["Bloomington, IN"]["default_capacity"], step=10)

n_simulations = st.slider("Number of simulations", 100, 5000, 1000, step=100)

if st.button("Run Simulation", type="primary"):
    city_columns = st.columns(2)

    for city_col, (city_name, params) in zip(city_columns, CITY_PARAMS.items()):
        with city_col:
            st.markdown(f"### {city_name}")
            with st.spinner(f"Running Monte Carlo simulation ({city_name})..."):
                starting_population = generate_steady_state_population(params)
                rng = np.random.default_rng()
                capacity = capacities[city_name]

                results = [
                    simulate_forecast_period(city_temps[city_name], capacity, starting_population, params, rng)
                    for _ in range(n_simulations)
                ]
                days_over = [r[0] for r in results]
                overflow_pct = (np.array(days_over) > 0).mean() * 100
                avg_days_over = np.mean(days_over)

            m1, m2 = st.columns(2)
            m1.metric("Probability of overflow this week", f"{overflow_pct:.1f}%")
            m2.metric("Average days over capacity", f"{avg_days_over:.2f}")

            all_pops = np.array([r[1] for r in results])
            mean_pop = all_pops.mean(axis=0)
            st.line_chart(mean_pop)
            st.caption(f"Average simulated shelter population across the forecast week (capacity: {capacity})")

st.divider()
with st.expander("About this model"):
    st.write(
        """
        This tool compares two independently-trained models, each combining
        two pieces of analysis derived from real shelter data:

        1. **Length-of-stay mixture model** — a two-component gamma
           distribution fit to each city's own historical intake data,
           separating typical short stays from a smaller population of
           long-term residents. Bloomington's long-stay rate (7.11% of
           animals staying over 100 days) is more than double Austin's
           (3.19%) — a real, notable difference between the two shelters.
        2. **Weather-adjusted intake model** — a Negative Binomial regression
           on each city's own daily intake and weather records. **Austin:**
           each 1°F rise in "feels-like" temperature is associated with a
           ~0.76% increase in expected daily intake (p < 0.001); barometric
           pressure is not significant once temperature is controlled for
           (p = 0.311). **Bloomington:** each 1°F rise in average temperature
           is associated with a ~1.63% increase in expected daily intake
           (p < 0.001) — more than double Austin's per-degree sensitivity;
           pressure retains a small, borderline-significant residual effect
           here (p = 0.046), unlike Austin's cleaner result, though
           temperature remains the clearly dominant driver in both cities.

        **Note on an earlier version:** this app originally used barometric
        pressure as the sole weather driver, based on an initial univariate
        finding. A follow-up multivariate test found pressure's apparent
        effect on intake disappears (or, for Bloomington, becomes marginal)
        once temperature is controlled for — temperature is the real,
        dominant driver in both cities.

        The simulator runs thousands of randomized 7-day scenarios per city
        using these fitted distributions to estimate the probability of
        exceeding each city's own kennel capacity.

        **Note:** temperature explains only a small share of day-to-day
        intake variation in both cities (many other factors matter too), so
        this should be read as a modest risk adjustment on top of baseline
        volume, not a precise forecast. Both models were trained and
        validated only on their own home shelter's data — applying either to
        a third location's weather (via the ZIP lookup above) is an
        extrapolation, useful for comparison but not a validated prediction
        for that specific shelter.
        """
    )
