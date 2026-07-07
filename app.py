from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

from Home import Home
from ModelComparison import ModelComparison
from FeatureSelection import FeatureSelection
from Statistics import Statistics
from AboutMe import AboutMe

ROOT = Path(__file__).resolve().parent
STYLE_PATH = ROOT / "styles.css"
LOGO_PATH = ROOT / "logo.png"

st.set_page_config(page_title="Credit Card Default Prediction", page_icon=":credit_card:", layout="wide")


# Load style css (neu co)
def load_custom_css():
    try:
        with open(STYLE_PATH, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_custom_css()


def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Model Comparison", "Feature Selection", "Statistics", "About Us"],
            icons=["house", "bar-chart", "funnel", "eye", "people"],
            menu_icon="cast",
            default_index=0,
            styles={
                "menu-title": {"font-size": "18px", "white-space": "nowrap"},
            },
        )
    if selected == "Home":
        Home()
    elif selected == "Model Comparison":
        ModelComparison()
    elif selected == "Feature Selection":
        FeatureSelection()
    elif selected == "Statistics":
        Statistics()
    elif selected == "About Us":
        AboutMe()


try:
    st.sidebar.image(str(LOGO_PATH), caption="")
except Exception:
    pass

sideBar()
