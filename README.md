# EV Route Planner App

# Installation

This project is written in Python and hence requires an installation of Python3.
The easiest way to get started is to create a virtual environment, install the dependencies, and then run the app:

```
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then, the app should be running at `http://localhost:8501` in your web browser (browser will also be opened).

# Usage

The app allows you to input a start and destination location, as well as vehicle parameters such as manufacturer rated range and current state of charge. It will then calculate the optimal route including necessary charging stops and display it on an interactive map.

# Notes

- Make sure to have an active internet connection as the app uses external APIs for routing and map tiles.
- If you encounter any issues with missing dependencies, ensure that all packages in `requirements.txt` are installed in your virtual environment.
- The waiting time is predicted by a machine learning model trained on a synthetical data set. The chosen model is a Random Forest Regressor. The training of the model can be found in `train_model.py`.
