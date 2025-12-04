import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Chargement des artefacts du modèle (déjà présent)
artefacts = joblib.load('./modele_housing.pkl')
theta = artefacts.get("modele_poids", artefacts.get("theta", None))
mean_x = artefacts.get("x_mean", None)
std_x = artefacts.get("x_std", None)
scaler_y = artefacts.get("y_scaler", None)
