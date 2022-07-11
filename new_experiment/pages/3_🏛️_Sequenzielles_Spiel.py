import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import random as rn 
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    if 'game_state_s' not in st.session_state:
        st.session_state.game_state_s = {'round':0, 'points':[], 'history_b':[], 'history_p':[],'history_eq':['e/e'], 'final':rn.randint(3,7)}

    ############################### HELPER METHODS ####################################
    def set_game_state(dict, player_first, point_structure):
        point_index = 0 if player_first else 1
        if dict['history_b'][-1] == dict['history_p'][-1]:
            if dict['history_b'][-1] == 'up':
                dict['history_eq'] = ['u/u']
            else:
                dict['history_eq'] = ['d/d']
        elif player_first:
            if dict['history_p'][-1] == 'down':
                dict['history_eq'] = ['d/u']
            else:
                dict['history_eq'] = ['u/d']
        elif dict['history_b'][-1] == 'down':
            dict['history_eq'] = ['d/u']
        else:
            dict['history_eq'] = ['u/d']

        dict['points'] = [point_structure[dict['history_eq'][-1]][point_index]]
        for stat in dict:
            st.session_state.game_state_s[stat] += dict[stat]

    def reset_game_state():
        st.session_state.game_state_s = {'round':0, 'points':[], 'history_b':[], 'history_p':[],'history_eq':['e/e'], 'final':rn.randint(3,7)}


    def bot(game_state):
        return rn.choice(['up','down'])

    def get_html(score, eq_type, direction, last_col=None, bot_move=None):
        if direction == 'up':
            direction_int = 1
        else:
            direction_int = 0

        if not last_col is None:
            if last_col == 'player':
                c_first = 'red'
                c_sec = 'green'
            else:
                c_first = 'green'
                c_sec = 'red'

            txt1 = 'ctx.font = "20px Sans-Serif"; ctx.fillStyle = "'+c_first+'"; ctx.fillText("'+str(score[0])+'", parseInt(0.82*width), parseInt(height*'+str(1-direction_int)+'+15*'+str(direction_int)+'));'
            txt2 = 'ctx.font = "20px Sans-Serif"; ctx.fillStyle = "black"; ctx.fillText("|", parseInt(0.89*width), parseInt(height*'+str(1-direction_int)+'+15*'+str(direction_int)+'));'
            txt3 = 'ctx.font = "20px Sans-Serif"; ctx.fillStyle = "'+c_sec+'"; ctx.fillText("'+str(score[1])+'", parseInt(0.93*width), parseInt(height*'+str(1-direction_int)+'+15*'+str(direction_int)+'));'
            txt = txt1+txt2+txt3
            fct = str(0.8)
        else:
            txt = ''
            fct = str(1)
            grad = ''

        if bot_move == direction:
            draw_style = "#e27777"
        else:
            draw_style = "#000000"

        if st.session_state.game_state_s['history_eq'][-1] in eq_type:
            draw_eq = 'ctx.setLineDash([5, 10]); ctx.beginPath(); ctx.moveTo(0, height*'+str(direction_int)+'); ctx.lineTo(width*'+fct+', height*'+str(1-direction_int)+'); ctx.strokeStyle = "#ffff52"; ctx.stroke();'
        else:
            draw_eq = ''

        draw = 'ctx.moveTo(0, height*'+str(direction_int)+'); ctx.lineTo(width*'+fct+', height*'+str(1-direction_int)+');'

        

        html_string ="""
                    <div id="d1" style="position:absolute; top:0px; left:15px">
                        <canvas id="Canvas1"></canvas>
                    </div>

                    <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                    <script>
                        const draw = (canvas) => {
                            const ctx = canvas.getContext("2d");
                            width = canvas.width
                            height = canvas.height
                            ctx.beginPath();
                            """+draw+"""
                            ctx.lineWidth = 2;
                            ctx.strokeStyle = " """ +draw_style+ """ ";
                            ctx.stroke();
                            """+txt+"""
                            """+draw_eq+"""
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
    bot_decission = bot(st.session_state.game_state_s)
    point_structure = {'u/u':[1,2],'d/d':[0,0],'d/u':[2,1],'u/d':[0,2]}
    st.subheader('Wiederholtes sequenzielles Spiel')
    HEIGHT = 60
    if st.session_state.game_state_s['round']<st.session_state.game_state_s['final']:
        if st.session_state.game_state_s['round']>0:
            txt_gegner = 'Dein Gegner hat letzte Runde '+ str(st.session_state.game_state_s['history_b'][-1])+' gespielt'
            txt_punkte = 'Du erhälst dadurch '+ str(st.session_state.game_state_s['points'][-1])+ ' Punkte'
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
        height=70,) 
        player_role = rn.choice(['first','second'])
        c1, c2, c3, c4 = st.columns((1,4,4,1))
        with c2:
            if player_role=='first':
                components.html(
                    """
                        <div style="text-align:center">
                            <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                            <p style="border:1px; border-style:solid; border-color:#9be277; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">Dein Zug</p>
                        </div>
                    """,
                    height=70)
                bot_move = None
            else:
                components.html(
                    """
                        <div style="text-align:center">
                            <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                            <p style="border:1px; border-style:solid; border-color:#e27777; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">Gegner Zug</p>
                        </div>
                    """,
                    height=70)    
                bot_move = bot_decission
            components.html(
                """
                    <div></div>
                """,
                height=HEIGHT-15)        
            components.html(
                    get_html([],['u/d','u/u'],'up', bot_move=bot_move),
                    height=HEIGHT+30)
            components.html(
                    get_html([],['d/u','d/d'],'down', bot_move=bot_move),
                    height=HEIGHT)
            components.html(
                """
                    <div></div>
                """,
                height=HEIGHT-15)

            if player_role=='first':
                player_first = True
                st.button('up', key='u1',on_click=set_game_state, args=({'round':1, 'points':[], 'history_b':[bot_decission], 'history_p':['up'], 'history_eq':[]}, player_first, point_structure))
                st.button('down', key='d1',on_click=set_game_state, args=({'round':1, 'points':[], 'history_b':[bot_decission], 'history_p':['down'], 'history_eq':[]}, player_first, point_structure))
        with c3:
            if player_role=='second':
                components.html(
                    """
                        <div style="text-align:center">
                            <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                            <p style="border:1px; border-style:solid; border-color:#9be277; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">Dein Zug</p>
                        </div>
                    """,
                    height=70)
                last_col = 'player'
            else:
                components.html(
                    """
                        <div style="text-align:center">
                            <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                            <p style="border:1px; border-style:solid; border-color:#e27777; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">Gegner Zug</p>
                        </div>
                    """,
                    height=70)
                last_col = 'bot'
            components.html(
                    get_html(point_structure['u/u'],['u/u'],'up', last_col=last_col),
                    height=HEIGHT)
            components.html(
                    get_html(point_structure['u/d'],['u/d'],'down', last_col=last_col),
                    height=HEIGHT)
            components.html(
                    get_html(point_structure['d/u'],['d/u'],'up', last_col=last_col),
                    height=HEIGHT)
            components.html(
                    get_html(point_structure['d/d'],['d/d'],'down', last_col=last_col),
                    height=HEIGHT)

            if player_role=='second':
                player_first = False
                st.button('up', key='u2',on_click=set_game_state, args=({'round':1, 'points':[], 'history_b':[bot_decission], 'history_p':['up'], 'history_eq':[]}, player_first, point_structure))
                st.button('down', key='d2',on_click=set_game_state, args=({'round':1, 'points':[], 'history_b':[bot_decission], 'history_p':['down'], 'history_eq':[]}, player_first, point_structure))
        components.html(
            """
                <div></div>
            """,
            height=100)
    else:
        st.write('Du hast',sum(st.session_state.game_state_s['points']),'punkte erzielt')
        st.button('Neue Runde', on_click=reset_game_state)
else:
    st.write('bitte einloggen')