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