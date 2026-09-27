SimuData/
├── README.md
├── pyproject.toml
│
├── data/
│   └── simulated/
│
├── notebooks/
│   ├── 01_temperature.ipynb
│   ├── 02_humidity.ipynb
│   └── 03_pollution.ipynb
│
├── src/
│   └── simudata/
│       ├── distributions.py
│       ├── seasonality.py
│       ├── temperature.py
│       ├── humidity.py
│       ├── pollution.py
│       └── simulator.py
│
└── tests/


SimuData/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── temperature.yaml
│   ├── humidity.yaml
│   ├── pollution.yaml
│   └── simulation.yaml
│
├── data/
│   ├── raw/
│   ├── simulated/
│   └── processed/
│
├── notebooks/
│   ├── 01_temperature_simulation.ipynb
│   ├── 02_humidity_simulation.ipynb
│   ├── 03_pollution_simulation.ipynb
│   ├── 04_multivariate_simulation.ipynb
│   └── 05_simulation_diagnostics.ipynb
│
├── src/
│   └── simudata/
│       ├── __init__.py
│       │
│       ├── distributions/
│       │   ├── __init__.py
│       │   ├── normal.py
│       │   ├── lognormal.py
│       │   └── gamma.py
│       │
│       ├── temporal/
│       │   ├── __init__.py
│       │   ├── seasonality.py
│       │   ├── trend.py
│       │   └── noise.py
│       │
│       ├── environmental/
│       │   ├── __init__.py
│       │   ├── temperature.py
│       │   ├── humidity.py
│       │   └── pollution.py
│       │
│       ├── multivariate/
│       │   ├── __init__.py
│       │   └── dependencies.py
│       │
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── simulator.py
│       │   └── scenarios.py
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── distributions.py
│           ├── correlations.py
│           └── visualization.py
│
├── scripts/
│   ├── simulate_temperature.py
│   ├── simulate_environment.py
│   └── run_simulation.py
│
└── tests/
    ├── test_distributions.py
    ├── test_temperature.py
    ├── test_humidity.py
    └── test_pollution.py




| Pollutant | Typical ambient concentration | Possible peaks | Unit |
|-----------|-------------------------------|----------------|------|
| PM2.5     | 5–20                          | >50, sometimes >100 | µg/m³ |
| NO₂       | 5–30                          | >50–100        | µg/m³ |
| O₃        | 40–100                        | >100–150       | µg/m³ |
| CO        | 0.2–2                         | Several mg/m³  | mg/m³ |





Voici une version complète du README.md, alignée sur l’architecture actuelle de SimuData, les mécanismes retenus et les paramètres par défaut. J’ai aussi distingué clairement les concentrations physiques des paramètres sur l’échelle logarithmique.

# SimuData
**SimuData** is a Python package for generating synthetic environmental
time-series data designed for experiments in **deep learning, statistical
modeling, and longevity/mortality research**.
The package generates coherent multivariate environmental scenarios by
simulating:
- temperature;
- relative humidity;
- air pollutants;
- temporal variability, seasonality, and stochastic noise;
- dependencies between environmental variables.
The main objective is to provide controlled synthetic datasets for studying
how environmental exposures can be incorporated into statistical and
deep-learning models for longevity, mortality, and health-risk analysis.
---
## Project structure
```text
SimuData/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── temperature.yaml
│   ├── humidity.yaml
│   ├── pollution.yaml
│   └── simulation.yaml
│
├── data/
│   ├── raw/
│   ├── simulated/
│   └── processed/
│
├── notebooks/
│   ├── 01_temperature_simulation.ipynb
│   ├── 02_humidity_simulation.ipynb
│   ├── 03_pollution_simulation.ipynb
│   ├── 04_multivariate_simulation.ipynb
│   └── 05_simulation_diagnostics.ipynb
│
├── src/
│   └── simudata/
│       ├── __init__.py
│       │
│       ├── distributions/
│       │   ├── __init__.py
│       │   ├── normal.py
│       │   ├── lognormal.py
│       │   └── gamma.py
│       │
│       ├── temporal/
│       │   ├── __init__.py
│       │   ├── seasonality.py
│       │   ├── trend.py
│       │   └── noise.py
│       │
│       ├── environmental/
│       │   ├── __init__.py
│       │   ├── temperature.py
│       │   ├── humidity.py
│       │   └── pollution.py
│       │
│       ├── multivariate/
│       │   ├── __init__.py
│       │   └── dependencies.py
│       │
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── simulator.py
│       │   └── scenarios.py
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── distributions.py
│           ├── correlations.py
│           └── visualization.py
│
├── scripts/
│   ├── simulate_temperature.py
│   ├── simulate_environment.py
│   └── run_simulation.py
│
└── tests/
    ├── test_distributions.py
    ├── test_temperature.py
    ├── test_humidity.py
    └── test_pollution.py

⸻

1. Environmental simulation framework

SimuData follows a hierarchical simulation strategy.

First, meteorological variables are generated:

[
\text{Temperature}
\rightarrow
\text{Humidity}
]

Then atmospheric pollutants are generated conditionally on the
meteorological variables:

[
(\text{Temperature},\text{Humidity})
\rightarrow
\text{Pollutants}.
]

This creates dependencies between environmental variables instead of
simulating each pollutant independently.

The simulated environmental process can therefore be represented as

[
\mathbf{X}t =
\left(
T_t,
H_t,
PM{2.5,t},
NO_{2,t},
O_{3,t},
CO_t
\right).
]

⸻

2. Temperature

Temperature is simulated as a seasonal time series with optional trend and
stochastic variability.

The basic seasonal component is

[
T_t =
\mu
+
A
\sin\left(
\frac{2\pi t}{P}+\phi
\right)
+
\delta_t
+
\epsilon_t,
]

where:

* (\mu) is the mean temperature;
* (A) is the seasonal amplitude;
* (P) is the seasonal period;
* (\phi) is the phase;
* (\delta_t) represents an optional long-term trend;
* (\epsilon_t) represents stochastic variability.

For daily environmental simulations, the default seasonal period is

[
P=365.
]

⸻

3. Humidity

Relative humidity is simulated as a positive bounded environmental variable
depending on the meteorological state.

Humidity can incorporate:

* seasonal variability;
* temperature dependence;
* stochastic noise;
* optional temporal structure.

The simulated humidity is constrained to physically meaningful values:

[
0 \leq H_t \leq 100.
]

The humidity process is therefore generated separately from temperature but can
retain dependence on the temperature trajectory.

⸻

4. Air pollution

Pollutants are generated on a logarithmic scale:

[
Y_t = \exp(\eta_t),
]

which guarantees positive concentrations and naturally allows occasional
large values.

The pollutant models deliberately use different functional forms. This
prevents the synthetic dataset from containing several pollutants that are
essentially different parameterizations of the same model.

The four pollutants currently simulated are:

* PM2.5;
* NO₂;
* O₃;
* CO.

⸻

4.1 PM2.5

PM2.5 is modeled using linear effects of standardized temperature and
humidity:

[
\log(PM_{2.5,t})

\beta_0
+
\beta_T T_t^*
+
\beta_H H_t^*
+
\epsilon_t,
]

where

[
T_t^*

\frac{T_t-\bar T}{s_T},
\qquad
H_t^*

\frac{H_t-\bar H}{s_H}.
]

Default parameters

pm25:
  intercept: 2.0
  temperature_slope: 0.15
  humidity_slope: 0.10
  noise_mean: 0.0
  noise_std: 0.30

The intercept corresponds approximately to

[
\exp(2.0)\approx7.4\ \mu g/m^3
]

before accounting for meteorological effects and stochastic variability.

⸻

4.2 NO₂

NO₂ uses a different temperature-humidity relationship:

[
\log(NO_{2,t})

\beta_0

\beta_T T_t^*
+
\beta_H H_t^*
+
\epsilon_t.
]

The negative temperature effect and positive humidity effect create a
different dependence structure from PM2.5.

Default parameters

no2:
  intercept: 2.0
  temperature_slope: 0.08
  humidity_slope: 0.15
  noise_mean: 0.0
  noise_std: 0.25

The baseline concentration is approximately

[
\exp(2.0)\approx7.4\ \mu g/m^3.
]

⸻

4.3 O₃

Ozone uses a nonlinear model because its relationship with meteorological
conditions is not adequately represented by two simple linear effects.

The model is

[
\log(O_{3,t})

\beta_0
+
\beta_T T_t^*
+
\beta_H H_t^*
+
\beta_{T^2}(T_t^*)^2
+
\beta_{TH}(T_t^H_t^)
+
\epsilon_t.
]

The quadratic temperature term allows the concentration to respond
nonlinearly to temperature, while the interaction allows the temperature
effect to depend on humidity.

Default parameters

o3:
  intercept: 4.0
  temperature_slope: 0.15
  humidity_slope: -0.08
  temperature_squared_slope: 0.05
  temperature_humidity_slope: -0.04
  noise_mean: 0.0
  noise_std: 0.25

The baseline concentration is approximately

[
\exp(4.0)\approx54.6\ \mu g/m^3.
]

This is intentionally higher than the baseline levels used for PM2.5 and
NO₂.

⸻

4.4 CO

CO uses a different nonlinear structure based on a logarithmic humidity
effect.

The model is

[
\log(CO_t)

\beta_0
+
\beta_T T_t^*
+
\beta_H H_t^*
+
\beta_{\log}
\log(1+H_t^{+})
+
\beta_{TH}(T_t^H_t^)
+
\epsilon_t,
]

where (H_t^{+}) is a positive rescaling of the standardized humidity.

The logarithmic term provides a nonlinear but gradually saturating humidity
effect.

Default parameters

co:
  intercept: -0.70
  temperature_slope: -0.10
  humidity_slope: 0.06
  humidity_log_slope: 0.08
  temperature_humidity_slope: 0.03
  noise_mean: 0.0
  noise_std: 0.25

The baseline concentration is approximately

[
\exp(-0.70)\approx0.50\ mg/m^3.
]

⸻

5. Pollution concentration ranges

The following ranges are used as approximate physical reference points for
calibrating the synthetic simulations. They are not intended to represent
universal thresholds or fixed values for every location.

Pollutant	Typical ambient concentration	Possible peaks	Unit
PM2.5	5–20	>50, sometimes >100	µg/m³
NO₂	5–30	>50–100	µg/m³
O₃	40–100	>100–150	µg/m³
CO	0.2–2	Several mg/m³	mg/m³

The baseline parameters are defined on the logarithmic scale. Therefore, the
intercept should not be interpreted directly as a concentration.

For a model

[
Y=\exp(\eta),
]

the baseline corresponding to the intercept is approximately

[
Y_0=\exp(\beta_0).
]

⸻

6. Extreme pollution events

The simulation deliberately allows occasional extreme observations.

The pollutant models use Gaussian noise on the logarithmic scale:

[
\epsilon_t\sim\mathcal N(0,\sigma^2).
]

Because the final concentration is obtained through an exponential
transformation,

[
Y_t=\exp(\eta_t),
]

moderately large positive noise realizations can generate relatively large
pollution concentrations.

For example, with

[
\sigma=0.25,
]

a realization at (+3\sigma) increases the multiplicative component by

[
\exp(3\times0.25)\approx2.12.
]

Thus, extreme events can occur occasionally without making the entire
simulation dominated by extreme values.

⸻

7. Multivariate environmental scenarios

The final objective is not only to simulate individual variables but to
generate coherent multivariate environmental scenarios.

A scenario can therefore be represented as

[
\mathbf{X}

{T_t,H_t,PM_{2.5,t},NO_{2,t},O_{3,t},CO_t}_{t=1}^{N}.
]

The simulation pipeline is:

Temperature
     │
     ▼
  Humidity
     │
     ├──────────────┐
     ▼              ▼
 PM2.5            NO₂
     │              │
     └──────┬───────┘
            ▼
       O₃ and CO

The exact dependency structure can be extended through
src/simudata/multivariate/dependencies.py.

⸻

8. Simulation scenarios

Different scenarios can be used to generate heterogeneous environmental
conditions, for example:

* normal environmental conditions;
* hot conditions;
* cold conditions;
* humid conditions;
* dry conditions;
* high-pollution episodes;
* combined meteorological extremes.

Scenario definitions are stored in:

configs/
└── simulation.yaml

This allows the same simulation engine to generate multiple environments
without modifying the underlying Python functions.

⸻

9. Configuration

Simulation parameters are separated from the Python implementation.

Example:

pm25:
  intercept: 2.0
  temperature_slope: 0.15
  humidity_slope: 0.10
  noise_mean: 0.0
  noise_std: 0.30
no2:
  intercept: 2.0
  temperature_slope: 0.08
  humidity_slope: 0.15
  noise_mean: 0.0
  noise_std: 0.25
o3:
  intercept: 4.0
  temperature_slope: 0.15
  humidity_slope: -0.08
  temperature_squared_slope: 0.05
  temperature_humidity_slope: -0.04
  noise_mean: 0.0
  noise_std: 0.25
co:
  intercept: -0.70
  temperature_slope: -0.10
  humidity_slope: 0.06
  humidity_log_slope: 0.08
  temperature_humidity_slope: 0.03
  noise_mean: 0.0
  noise_std: 0.25

Keeping the parameters in YAML files makes the simulations reproducible and
facilitates systematic sensitivity analyses.

⸻

10. Reproducibility

Random components are generated using PyTorch.

A random seed can be specified to reproduce a simulation:

import torch
torch.manual_seed(42)

For experiments involving multiple scenarios, each scenario should preferably
use a documented seed.

⸻

11. Diagnostics

The evaluation/ module provides tools for checking whether simulated data
behave as expected.

Diagnostics include:

* marginal distributions;
* temporal patterns;
* seasonal structure;
* summary statistics;
* correlations;
* cross-variable dependencies;
* extreme-event frequency;
* comparison between simulation scenarios.

The main diagnostic notebook is:

notebooks/05_simulation_diagnostics.ipynb

⸻

12. Intended applications

SimuData is primarily designed to support research involving environmental
exposures and health outcomes.

Potential applications include:

* deep-learning experiments;
* mortality forecasting;
* longevity modeling;
* environmental mortality studies;
* exposure-response modeling;
* synthetic cohort generation;
* uncertainty analysis;
* robustness testing;
* transfer learning across environmental scenarios;
* benchmarking statistical and machine-learning models.

The synthetic data are intended for methodological experimentation and do not
replace real environmental measurements.

⸻

13. Installation

Clone the repository and install the package in editable mode:

git clone <repository-url>
cd SimuData
pip install -e .

For development dependencies:

pip install -e ".[dev]"

⸻

14. Running a simulation

A complete environmental simulation can be launched with:

python scripts/run_simulation.py

Individual components can also be simulated separately:

python scripts/simulate_temperature.py
python scripts/simulate_environment.py

⸻

15. Testing

Run the test suite with:

pytest

The tests cover:

* probability distributions;
* temperature generation;
* humidity generation;
* pollution generation;
* input validation;
* output shapes;
* positivity constraints.

⸻

16. Research perspective

SimuData is designed as a modular foundation for generating synthetic
environmental exposure data that can subsequently be linked to health and
longevity models.

A future extension is to introduce more complex environmental-health
relationships, such as

[
\text{Environment}_t
\rightarrow
\text{Exposure}_t
\rightarrow
\text{Vulnerability}_t
\rightarrow
\text{Mortality}_t.
]

This could include socioeconomic characteristics, spatial heterogeneity,
lagged exposure effects, nonlinear dose-response relationships, and
population-level differences.

The ultimate objective is to provide controlled synthetic environments in
which statistical and deep-learning models can be evaluated under known data
generating processes.

**Petite correction importante par rapport à tes versions précédentes :** j’ai mis `O₃` à `intercept: 4.0` et `CO` à `-0.70`, car ce sont les deux cas où l'interprétation de `exp(intercept)` changeait fortement l’échelle physique. Les quatre modèles ont maintenant des structures réellement différentes, et pas seulement des coefficients différents.