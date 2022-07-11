import streamlit as st
from pages.resources.db_func import *

st.set_page_config(layout="wide")
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
conn = sqlite3.connect('./pages/resources/data.db')
c = conn.cursor()
col1, col2, col3 = st.columns(3)   
with col2: 
    st.subheader('Login')
    username = st.text_input("User Name")
    password = st.text_input("Password",type='password')

    if st.button("Login", key=1):
        create_usertable(c)
        hashed_pswd = make_hashes(password)
        result = login_user(c,username,check_hashes(password,hashed_pswd))
        if result:
            st.success("Logged In as {}".format(username))
            st.session_state.logged_in = True

        else:
            st.warning("Incorrect Username/Password")