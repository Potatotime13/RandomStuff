from pages import pages
import streamlit as st
from streamlit_multipage import MultiPage

app = MultiPage()
app.st = st

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

app.start_button = "Go to the main page"
app.navbar_name = "Mikroökonomik 1"
app.next_page_button = "Next Chapter"
app.previous_page_button = "Previous Chapter"
app.reset_button = "Delete Cache"
app.navbar_style = "VerticalButton"

app.hide_menu = True
app.hide_navigation = True

st.set_page_config(layout="wide")

m = st.markdown("""
<style>
div.stButton > button:first-child {
    height: 3em;
    width: 12em;
    margin: auto;
    display: block;
}

div.stButton > button:active {
    position:relative;
    top:3px;
}

</style>""", unsafe_allow_html=True)

for app_name, app_function in pages.items():
    app.add_app(app_name, app_function)

app.run()