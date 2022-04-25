import streamlit as st
import pandas as pd
import hashlib
import sqlite3

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def Content_task():
	task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
	if task == "Add Post":
		st.subheader("Add Your Post")

	elif task == "Analytics":
		st.subheader("Analytics")
	elif task == "Profiles":
		st.subheader("User Profiles")
		user_result = view_all_users()
		clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
		st.dataframe(clean_db)

st.title('Übungsklausuren Mikro')

page = st.sidebar.selectbox('Seitenauswahl', ('Home','Tasks', 'Sign Up'))

def Home():
	"""Simple Login App"""
	if 'logged_in' not in st.session_state:
		st.session_state.logged_in = False
	st.title("Simple Login App")
	st.subheader("Home")

def Tasks():
	if st.session_state.logged_in:
		Content_task()
	else:
		st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):

			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
				st.success("Logged In as {}".format(username))
				st.session_state.logged_in = True
				Content_task()
			else:
				st.warning("Incorrect Username/Password")
	
def SignUp():
	st.subheader("Create New Account")
	new_user = st.text_input("Username")
	new_password = st.text_input("Password",type='password')

	if st.button("Signup"):
		create_usertable()
		add_userdata(new_user,make_hashes(new_password))
		st.success("You have successfully created a valid Account")
		st.info("Go to Login Menu to login")

if page == 'Home':
    Home()
elif page == 'Tasks':
    Tasks()
elif page == 'Sign Up':
    SignUp()