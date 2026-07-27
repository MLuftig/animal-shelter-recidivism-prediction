# Shelter Overflow Risk Forecaster

A Monte Carlo simulation tool that estimates the probability of a shelter exceeding
kennel capacity in the coming week, adjusted for forecasted temperature, comparing
two independently-trained weather-intake models side by side — one fit on real
Austin Animal Center data, one on real Bloomington Animal Care & Control data.

**Live app:** [shelter-overflow-forecaster.streamlit.app](https://shelter-overflow-forecaster.streamlit.app/)

---

## Model Correction Note

An earlier version of this app used **barometric pressure** as the weather driver
of daily intake, based on the univariate finding in the
[moon-phase & weather analysis](https://github.com/MLuftig/moon-phase-weather-shelter-analysis)
project (p = 2.58e-23). A follow-up multivariate test — refitting the same
Negative Binomial regression with both pressure and "feels-like" temperature
included together — found that pressure's apparent effect on Austin's intake
**disappears** once temperature is controlled for (p = 0.311), while temperature
remains a genuine, independent driver (p < 0.001). Extending the same test to a
second, independent shelter (Bloomington, IN) mostly replicated this: temperature
is again the clearly dominant driver (p < 0.001), though pressure retains a small,
borderline-significant residual effect there (p = 0.046) — an honest nuance worth
noting rather than a clean repeat of Austin's result.

The model has been rebuilt around temperature for both cities. All results below
reflect the corrected version.

## Problem

Shelter intake volume is inherently random, and prior work in this portfolio
identified that "feels-like"/average temperature is a statistically significant
driver of intake surges. But knowing an effect is *significant* doesn't tell you
how much it actually matters operationally — shelter directors need an actual risk
estimate, not a p-value. And a model built at one shelter isn't guaranteed to
reflect another — this tool tests that directly by comparing two independently-fit
models rather than assuming one generalizes.

## Solution

This project quantifies the weather effect for two shelters and builds each into
its own stochastic capacity-planning model:

1. **Length-of-stay mixture model** — historical intake data at both shelters
   shows a clearly bimodal length-of-stay distribution: most animals turn over
   quickly, but a smaller subpopulation stays for months. Modeling this as a
   single distribution understated both groups, so two gamma distributions were
   fit separately per city and combined into a weighted mixture. Bloomington's
   long-stay rate (7.11% of animals staying over 100 days) is more than double
   Austin's (3.19%) — a real, notable difference between the two shelters.
2. **Weather-adjusted intake model** — a Negative Binomial regression per city
   (with an explicitly estimated dispersion parameter, since the default
   assumption understated real overdispersion in both datasets) found that each
   1°F rise in temperature is associated with a ~0.76% increase in expected daily
   intake at Austin, and a steeper ~1.63% increase at Bloomington — more than
   double Austin's per-degree sensitivity.
3. **Monte Carlo simulation** — thousands of randomized 7-day scenarios are run
   per city using each fitted distribution, tracking daily shelter population
   against a city-specific capacity to estimate overflow probability. Bloomington's
   shelter operates at a much smaller scale (steady-state population ~388 vs.
   Austin's ~880), so capacity is set independently for each.
4. **Live forecast lookup** — enter a ZIP code and the app fetches a real 7-day
   temperature forecast (via Open-Meteo) for that location, then runs both cities'
   models against those same conditions, so you can see how two differently-trained
   models respond to identical real-world weather.

## Analysis Notebooks

The parameters used in `app.py` were derived from real shelter intake/outcome
data for both cities, not assumed or hardcoded arbitrarily. The full derivation
is reproducible from these notebooks:

- [`01-load-and-summarize-inputs.ipynb`](src/01-load-and-summarize-inputs.ipynb) —
  fits Austin's two-component length-of-stay gamma mixture and baseline daily
  arrival Negative Binomial distribution.
- [`02-weather-intake-regression.ipynb`](src/02-weather-intake-regression.ipynb) —
  fits Austin's corrected Negative Binomial regression, testing temperature and
  pressure together to confirm pressure's effect is not independent, then
  isolating the temperature-only coefficient used downstream.
- [`03-build-simulation.ipynb`](src/03-build-simulation.ipynb) — builds the
  forecast-period simulation function using Austin's corrected temperature-driven
  arrival rate, and validates the headline impact numbers below against real
  Monte Carlo output.
- `04-bloomington-intake-regression.ipynb` *(to be added)* — mirrors `01` and `02`
  for Bloomington: length-of-stay gamma mixture, daily intake NB regression
  (temperature + pressure tested together), and the resulting corrected
  parameters used in `app.py`.

## Impact

At each city's own capacity, a forecasted heat wave more than **doubled** the
probability of at least one overflow day compared to an average week — Austin:
47.4% → 85.6% (capacity 900); Bloomington: 2.8% → 37.8% (capacity 450) — a
meaningful, quantified operational risk that a shelter director could act on
before it happens, not after, at either shelter.

The analysis also showed this effect is **not constant**: it's most pronounced
when capacity is already tight relative to a shelter's typical population, and
largely disappears when there's enough buffer above average. That nuance —
*weather risk depends on how much slack you already have* — held for both
shelters despite their very different scale and per-degree sensitivity.

## How It Works

- Enter a ZIP code to fetch a live 7-day temperature forecast, applied to both
  cities' models at once, or adjust each city's daily temperatures (°F)
  manually -- each defaults to that city's own historical average before any
  ZIP is entered, so the starting comparison isn't biased toward either city
- Set kennel capacity separately for each city, since Austin and Bloomington
  operate at very different scales
- The app simulates thousands of possible weeks per city using each city's own
  fitted arrival and length-of-stay distributions, adjusting daily intake based
  on that day's forecasted temperature
- Output: probability of exceeding capacity at least once, average days over
  capacity, and the average simulated population trajectory across the week,
  shown side by side for direct comparison

## Limitations

- Temperature explains only a small share of day-to-day intake variance at
  either shelter (other factors — season, day-of-week, local events — matter
  more). This tool should be read as a **modest risk adjustment on top of
  baseline volume**, not a precise intake forecast.
- Bloomington's pressure coefficient did not cleanly disappear the way Austin's
  did (p = 0.046 vs. p = 0.311) — a small, borderline-significant residual
  effect remains unexplained by temperature alone there.
- **Neither model has been validated on any location other than its own home
  shelter.** Applying either model to an arbitrary ZIP code via the live
  forecast lookup is an extrapolation — useful for comparing how two real,
  differently-trained models respond to given conditions, not a validated
  prediction for that specific location's actual shelter.
- Bloomington's underlying dataset is smaller and covers a shorter reliable
  window (2017–2019, since earlier years have near-zero shelter records) than
  Austin's, so its estimates should be read as somewhat more approximate.

## Tech Stack

`Python`, `NumPy`, `SciPy` (distribution fitting, negative binomial regression),
`Requests` (ZIP geocoding and live forecast retrieval via Zippopotam.us and
Open-Meteo), `Streamlit` (deployment)

## Related Projects

- [Moon Phase & Weather Analysis](https://github.com/MLuftig/moon-phase-weather-shelter-analysis) — original discovery of the weather-mortality relationship and the pressure/temperature confounding finding this tool is built on
- [Shelter Medical Supply Forecaster](https://github.com/MLuftig/shelter-supply-forecaster) — companion tool independently replicating the same temperature-driven relationship, for mortality rather than intake, across the same two shelters
- [Animal Shelter Recidivism Prediction](https://github.com/MLuftig/animal-shelter-recidivism-prediction) — companion project testing individual-animal return risk across the same two shelters
- [Shelter Return Risk Predictor](https://github.com/MLuftig/shelter-risk-predictor) — deployed app for the recidivism model
