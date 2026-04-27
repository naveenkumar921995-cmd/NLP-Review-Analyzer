import spacy
import streamlit as st

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()
