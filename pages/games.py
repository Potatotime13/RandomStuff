from streamlit_multipage import MultiPage
from PIL import Image

def set_type(st, str_type):
    if 'page_game' not in st.session_state:
        st.session_state['page_game'] = 'default'
    st.session_state['page_game'] = str_type

def game1(st, **state):
    st.subheader('Game1')
    st.button('back', key='b3', on_click=set_type, args=(st, 'default'))


def game2(st, **state):
    st.subheader('Game2')
    st.button('back', key='b2', on_click=set_type, args=(st, 'default'))


def game3(st, **state):
    st.subheader('Game3')
    st.button('back', key='b1', on_click=set_type, args=(st, 'default'))

def game_page(st, **state):

    st.title("Single Player Games")
    image = Image.open('dummy.png')
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if 'page_game' not in st.session_state:
        st.session_state['page_game'] = 'default'

    if st.session_state.logged_in:
        if st.session_state['page_game'] == 'game1':
            game1(st, **state)
        if st.session_state['page_game'] == 'game2':
            game2(st, **state)
        if st.session_state['page_game'] == 'game3':
            game3(st, **state)
        if st.session_state['page_game'] == 'default':
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container():
                    st.subheader("Game1")
                    st.image(image,use_column_width=True)                
                    st.button('play', key='p1', on_click=set_type, args=(st,'game1'))
            with col2:
                with st.container():
                    st.subheader("Game2")
                    st.image(image,use_column_width=True)
                    st.button('play', key='p2', on_click=set_type, args=(st,'game2'))
            with col3:
                with st.container():
                    st.subheader("Game3")
                    st.image(image,use_column_width=True)
                    st.button('play', key='p3', on_click=set_type, args=(st,'game3'))

    else:
        st.warning('pls log in')