from streamlit_multipage import MultiPage
import streamlit.components.v1 as components
from PIL import Image
import random as rn 

def set_type(st, str_type):
    if 'page_game' not in st.session_state:
        st.session_state['page_game'] = 'default'
    st.session_state['page_game'] = str_type
    # gamestate := round:0 , points:0 , history_b:[]...
    st.session_state['game_state'] = {'round':0, 'points':0, 'history_b':[], 'history_p':[], 'final':rn.randint(3,7)}

def set_game_state(st, dict):
    # gestehen := 1
    # lügen := 0
    if dict['history_b'][-1] == dict['history_p'][-1]:
        if dict['history_b'][-1] == 'gestehen':
            dict['points'] = 2
        else:
            dict['points'] = 1
    elif dict['history_b'] == 'lügen':
        dict['points'] = 0
    else:
        dict['points'] = 3

    for stat in dict:
        st.session_state['game_state'][stat] += dict[stat]

def bot(game_state):
    return rn.choice(['gestehen','lügen'])

def game1(st, **state):
    def get_html(score_p, score_b):
        html_string ="""
                    <canvas id="Canvas1" width="200" height="70" style="border:2px solid #000000;">
                    </canvas>
                    <script>
                    var c = document.getElementById("Canvas1");
                    var ctx = c.getContext("2d");
                    ctx.moveTo(0, 70);
                    ctx.lineTo(200, 0);
                    ctx.stroke();
                    ctx.font = "30px Arial";
                    ctx.fillText(" """+str(score_b)+""" ", 20, 30);
                    ctx.fillText(" """+str(score_p)+""" ", 160, 50);
                    </script>
                """

    st.subheader('Wiederholtes Gefangenendilemma')
    st.write(st.session_state['game_state']['round'])
    st.write(st.session_state['game_state']['final'])
    best=3
    worst=0
    eq_h=2
    eq_l=1
    if st.session_state['game_state']['round']<st.session_state['game_state']['final']:
        if st.session_state['game_state']['round']>0:
            st.write('dein Gegner hat letzte Runde',st.session_state['game_state']['history_b'])
            st.write('du hast letzte Runde',st.session_state['game_state']['history_p'])
        components.html(
        """
            <div style="text-align: center"></div>
        """,
        height=100,)    
        c1, c2, c3, c4 = st.columns((1,1,1,1))
        with c1:
            components.html(
                """
                    <div style="text-align: center"> gestehen </div>
                """,
                height=100,
            )
            components.html(
                """
                    <div style="text-align: center"> lügen </div>
                """,
                height=100,
            )       
        with c2:
            components.html(
                    get_html(eq_h,eq_h)
                ,
                height=100,
            )
            components.html(
                """
                    <canvas id="Canvas1" width="200" height="70" style="border:2px solid #000000;">
                    </canvas>
                    <script>
                    var c = document.getElementById("Canvas1");
                    var ctx = c.getContext("2d");
                    ctx.moveTo(0, 70);
                    ctx.lineTo(200, 0);
                    ctx.stroke();
                    ctx.font = "30px Arial";
                    ctx.fillText(" """+str(best)+""" ", 20, 30);
                    ctx.fillText(" """+str(worst)+""" ", 160, 50);
                    </script>
                """,
                height=100,
            )
        c2.button('gestehen', key='coopb', on_click=set_game_state, args=(st, {'round':1, 'points':0, 'history_b':[bot(st.session_state['game_state'])], 'history_p':['gestehen']}))
        with c3:
            components.html(
                """
                    <canvas id="Canvas1" width="200" height="70" style="border:2px solid #000000;">
                    </canvas>
                    <script>
                    var c = document.getElementById("Canvas1");
                    var ctx = c.getContext("2d");
                    ctx.moveTo(0, 70);
                    ctx.lineTo(200, 0);
                    ctx.stroke();
                    ctx.font = "30px Arial";
                    ctx.fillText(" """+str(worst)+""" ", 20, 30);
                    ctx.fillText(" """+str(best)+""" ", 160, 50);
                    </script>
                """,
                height=100,
            )
            components.html(
                """
                    <canvas id="Canvas1" width="200" height="70" style="border:2px solid #000000;">
                    </canvas>
                    <script>
                    var c = document.getElementById("Canvas1");
                    var ctx = c.getContext("2d");
                    ctx.moveTo(0, 70);
                    ctx.lineTo(200, 0);
                    ctx.stroke();
                    ctx.font = "30px Arial";
                    ctx.fillText(" """+str(eq_l)+""" ", 20, 30);
                    ctx.fillText(" """+str(eq_l)+""" ", 160, 50);
                    </script>
                """,
                height=100,
            )
        c3.button('lügen', key='defb', on_click=set_game_state, args=(st, {'round':1, 'points':0, 'history_b':[bot(st.session_state['game_state'])], 'history_p':['lügen']}))
        components.html(
        """
            <div style="text-align: center"></div>
        """,
        height=100,)
    else:
        st.write('Du hast',st.session_state['game_state']['points'],'punkte erzielt')
        st.button('rerun', key='r1', on_click=set_type, args=(st,'game1'))
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
                    st.subheader("Wiederholtes Gefangenendilemma")
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