import streamlit as st
import pickle
import streamlit.components.v1 as components

# Incrustar HTML completo desde archivo
with open("index.html", "r", encoding="utf-8") as f:
    html_string = f.read()


