from .db_func import *

def sign_page(st, **state):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    col1, col2, col3 = st.columns(3)   
    with col2:     
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')
        if st.button("Signup"):
            create_usertable(c)
            add_userdata(c,conn,new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")