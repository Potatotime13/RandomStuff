import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import random as rn 
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0069d4;
        color:#ffffff;
        font-size: 20px;
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
    }
    div.stButton > button:hover {
        background-color: #004993;
        color:#ffffff;
        }
    </style>""", unsafe_allow_html=True)


if st.session_state.logged_in:
    if 'game_state_p' not in st.session_state:
        st.session_state.game_state_p = {'round':0, 'points':[], 'history_b':[], 'history_p':[],'history_eq':['e/e'], 'final':rn.randint(3,7)}

    ############################### HELPER METHODS ####################################
    def set_game_state(dummy, dict):
        if dict['history_b'][-1] == dict['history_p'][-1]:
            if dict['history_b'][-1] == 'gestehen':
                dict['points'] = [2]
                dict['history_eq'] = ['g/g']
            else:
                dict['points'] = [1]
                dict['history_eq'] = ['l/l']
        elif dict['history_b'][-1] == 'lügen':
            dict['points'] = [0]
            dict['history_eq'] = ['l/g']
        else:
            dict['points'] = [3]
            dict['history_eq'] = ['g/l']

        for stat in dict:
            st.session_state.game_state_p[stat] += dict[stat]

    def reset_game_state():
        st.session_state.game_state_p = {'round':0, 'points':[], 'history_b':[], 'history_p':[],'history_eq':['e/e'], 'final':rn.randint(3,7)}

    def bot(game_state):
        return rn.choice(['gestehen','lügen'])

    def get_html(score_p, score_b, eq_type):
        if eq_type == st.session_state.game_state_p['history_eq'][-1]:
            border_style = "2px solid #3944dd"
        else:
            border_style = "1px solid #000000"
        html_string ="""
                    <div id="d1" style="position:absolute; top:0px; left:10px">
                        <canvas id="Canvas1" style="border:""" +border_style+ """; border-radius: 4px 4px 4px 4px;"></canvas>
                    </div>
                    
                    <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                    <script>
                        const draw = (canvas) => {
                            const ctx = canvas.getContext("2d");
                            width = canvas.width
                            height = canvas.height
                            ctx.moveTo(0, height);
                            ctx.lineTo(width, 0);
                            ctx.lineTo(0,0);
                            ctx.closePath();
                            ctx.stroke();
                            ctx.fillStyle = "#d4d4d4";
                            ctx.fill();
                            ctx.fillStyle = "#969696";
                            ctx.font = "25px Sans-Serif";
                            ctx.fillText(" """+str(score_b)+""" ", parseInt(0.2*width), parseInt(0.45*height));
                            ctx.fillStyle = "#292929";                    
                            ctx.fillText(" """+str(score_p)+""" ", parseInt(0.7*width), parseInt(0.75*height));
                        }

                        const canvas = document.getElementById("Canvas1");
                        canvas.width = parseInt(window.innerWidth*0.9);
                        canvas.height = parseInt(window.innerHeight*0.9);
                        draw(canvas)

                        window.addEventListener('resize', () => {
                            canvas.width = parseInt(window.innerWidth*0.9);
                            canvas.height = parseInt(window.innerHeight*0.9);
                            draw(canvas)
                        })
                    </script>
                """
        return html_string

    ###################################################################################
    ########################### GAME CODE #############################################
    ###################################################################################
    st.subheader('Wiederholtes Gefangenendilemma')
    best=3
    worst=0
    eq_h=2
    eq_l=1
    with st.sidebar:
        genre = st.radio(
            "Spielmodus",
            ('typ1', 'typ2', 'typ3'))

    if st.session_state.game_state_p['round']<st.session_state.game_state_p['final']:
        if st.session_state.game_state_p['round']>0:
            txt_gegner = 'Dein Gegner hat letzte Runde '+ str(st.session_state.game_state_p['history_b'][-1])+' gespielt'
            txt_punkte = 'Du erhälst dadurch '+ str(st.session_state.game_state_p['points'][-1])+ ' Punkte'
        else:
            txt_gegner = ''
            txt_punkte = ''
        components.html(
        """
            <div>
                <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                <p style="font-family:IBM Plex Sans">""" +txt_gegner+ """</p>
                <p style="font-family:IBM Plex Sans">""" +txt_punkte+ """</p>
            </div>
        """,
        height=100,)    
        c1, c2, c3, c4 = st.columns((1,1,1,1))
        HEIGHT = 75
        with c1:
            components.html(
                """
                    <div style="text-align:center">
                        <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                        <p style="border:1px; border-style:solid; border-color:#474747; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">gestehen</p>
                    </div>
                """,
                height=HEIGHT)
            components.html(
                """
                    <div style="text-align:center">
                        <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                        <p style="border:1px; border-style:solid; border-color:#474747; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">lügen</p>
                    </div>
                """,
                height=HEIGHT)       
        with c2:
            components.html(
                get_html(eq_h,eq_h,'g/g'),
                height=HEIGHT)
            components.html(
                get_html(worst,best,'l/g'),
                height=HEIGHT)
        c2.button('gestehen', key='coopb', on_click=set_game_state, 
                args=(5, {'round':1, 'points':[], 'history_b':[bot(st.session_state.game_state_p)], 'history_p':['gestehen'], 'history_eq':[]})
            )
        
        with c3:
            components.html(
                get_html(best,worst,'g/l'),
                height=HEIGHT)
            components.html(
                get_html(eq_l,eq_l,'l/l'),
                height=HEIGHT)

        c3.button('lügen', key='defb', on_click=set_game_state, 
                args=(5, {'round':1, 'points':[], 'history_b':[bot(st.session_state.game_state_p)], 'history_p':['lügen'], 'history_eq':[]})
            )

        components.html(
        """
            <div></div>
        """,
        height=100)
    else:
        st.write('Du hast',sum(st.session_state.game_state_p['points']),'punkte erzielt')
        st.button('Neue Runde', on_click=reset_game_state)
else:
    st.write('bitte einloggen')