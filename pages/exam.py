from PIL import Image

def set_type(st, str_type):
    if 'page' not in st.session_state:
        st.session_state['page'] = 'default'
    st.session_state['page'] = str_type

def random_exam(st, **state):
    st.subheader('Random Exam')
    st.button('back', key='b3', on_click=set_type, args=(st, 'default'))


def full_exam(st, **state):
    st.subheader('Full Exam')
    st.selectbox('Klausurauswahl',['2015','2017'])
    st.button('back', key='b2', on_click=set_type, args=(st, 'default'))


def topics(st, **state):
    st.subheader('Topics')
    st.selectbox('Themenwahl',['Thema A','Thema B'])
    st.button('back', key='b1', on_click=set_type, args=(st, 'default'))


def exam_page(st, **state):
    image = Image.open('klausur.png')
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'page' not in st.session_state:
        st.session_state['page'] = 'default'

    st.title("Exam prep")
    if st.session_state.logged_in:
        if st.session_state['page'] == 'FullExam':
            full_exam(st, **state)
        if st.session_state['page'] == 'RandomGenerated':
            topics(st, **state)
        if st.session_state['page'] == 'Topics':
            random_exam(st, **state)
        if st.session_state['page'] == 'default':
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container():
                    st.subheader("Full Exam")
                    st.image(image,use_column_width=True)             
                    st.button('Select', key='s1', on_click=set_type, args=(st,'FullExam'))
            with col2:
                with st.container():
                    st.subheader("Random Generated")
                    st.image(image,use_column_width=True)
                    st.button('Select', key='s2', on_click=set_type, args=(st,'RandomGenerated'))
            with col3:
                with st.container():
                    st.subheader("Topics")
                    st.image(image,use_column_width=True)
                    st.button('Select', key='s3', on_click=set_type, args=(st,'Topics'))

    else:
        st.warning('pls log in')