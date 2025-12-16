Kinetic Hunter: 

The Trap StrategyOverviewKinetic Hunter is an algorithmic trading tool and simulation dashboard built with Streamlit. 
It implements a microstructure strategy based on Wyckoff's Law of Effort vs Result.The system detects "Market Traps" (Absorption events) by analyzing two key metrics:

Kinetic Score (Effort/Result): Identifies moments where massive volume (Effort) results in very little price movement (Result).HVG Degree (Horizontal Visibility Graph): Measures market complexity to ensure the price structure is clean and deterministic before entering.When high Kinetic Energy coincides with low Structural Complexity, the system signals a reversal (Fade the trend).PrerequisitesnYou need Python 3.8 or higher installed on your system.nDependenciesThe project relies on the following Python packages:streamlit: For the interactive web dashboard.pandas: For data manipulation and time-series handling.numpy: For high-performance numerical calculations.plotly: For interactive financial charting.numba: 

How to run:

streamlit run kinetic_hunter.py
