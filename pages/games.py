from streamlit_multipage import MultiPage
import streamlit.components.v1 as components
from PIL import Image
import random as rn 

def set_type(st, str_type):
    if 'page_game' not in st.session_state:
        st.session_state['page_game'] = 'default'
    st.session_state['page_game'] = str_type
    st.session_state['game_state'] = {'round':0, 'points':[], 'history_b':[], 'history_p':[],'history_eq':['e/e'], 'final':rn.randint(3,7)}

'''
#######################################################################################
'''

def game1(st, **state):
    ############################### HELPER METHODS ####################################
    def set_game_state(st, dict):
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
            st.session_state['game_state'][stat] += dict[stat]
    def bot(game_state):
        return rn.choice(['gestehen','lügen'])    
    def get_html(score_p, score_b, eq_type):
        if eq_type == st.session_state['game_state']['history_eq'][-1]:
            border_style = "2px solid #3944dd"
        else:
            border_style = "1px solid #000000"
        html_string ="""
                    <div id="d1" style="position:absolute; top:0px; left:15px">
                        <canvas id="Canvas1" width="200" height="70" style="border:""" +border_style+ """; border-radius: 4px 4px 4px 4px;">
                        </canvas>
                    </div>
                    <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                    <script>
                    var c = document.getElementById("Canvas1");
                    width = parseInt(window.innerWidth*0.9);
                    height = parseInt(window.innerHeight*0.9);
                    c.width = width;
                    c.height = height;
                    var ctx = c.getContext("2d");
                    ctx.moveTo(0, height);
                    ctx.lineTo(width, 0);
                    ctx.stroke();
                    ctx.font = "25px Sans-Serif";
                    ctx.fillText(" """+str(score_b)+""" ", parseInt(0.2*width), parseInt(0.45*height));
                    ctx.fillText(" """+str(score_p)+""" ", parseInt(0.7*width), parseInt(0.75*height));
                    </script>
                """
        return html_string

    st.subheader('Wiederholtes Gefangenendilemma')
    best=3
    worst=0
    eq_h=2
    eq_l=1
    ###################################################################################
    ########################### GAME CODE #############################################
    ###################################################################################

    if st.session_state['game_state']['round']<st.session_state['game_state']['final']:
        if st.session_state['game_state']['round']>0:
            txt_gegner = 'Dein Gegner hat letzte Runde '+ str(st.session_state['game_state']['history_b'][-1])+' gespielt'
            txt_punkte = 'Du erhälst dadurch '+ str(st.session_state['game_state']['points'][-1])+ ' Punkte'
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
                        <p style="border:1px; border-style:solid; border-color:#9be277; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">gestehen</p>
                    </div>
                """,
                height=HEIGHT)
            components.html(
                """
                    <div style="text-align:center">
                        <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                        <p style="border:1px; border-style:solid; border-color:#e27777; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">lügen</p>
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
        c2.button('gestehen', key='coopb', on_click=set_game_state, args=(st, {'round':1, 'points':[], 'history_b':[bot(st.session_state['game_state'])], 'history_p':['gestehen'], 'history_eq':[]}))
        
        with c3:
            components.html(
                get_html(best,worst,'g/l'),
                height=HEIGHT)
            components.html(
                get_html(eq_l,eq_l,'l/l'),
                height=HEIGHT)
        c3.button('lügen', key='defb', on_click=set_game_state, args=(st, {'round':1, 'points':[], 'history_b':[bot(st.session_state['game_state'])], 'history_p':['lügen'], 'history_eq':[]}))
        
        components.html(
        """
            <div></div>
        """,
        height=100)
    else:
        st.write('Du hast',sum(st.session_state['game_state']['points']),'punkte erzielt')
        st.button('rerun', key='r1', on_click=set_type, args=(st,'game1'))
    st.button('back', key='b3', on_click=set_type, args=(st, 'default'))


def game2(st, **state):
    ############################### HELPER METHODS ####################################
    def set_game_state(st, dict, player_first, point_structure):
        point_index = 0 if player_first else 1
        if dict['history_b'][-1] == dict['history_p'][-1]:
            if dict['history_b'][-1] == 'up':
                dict['history_eq'] = ['u/u']
            else:
                dict['history_eq'] = ['d/d']
        elif dict['history_b'][-1] == 'down':
            dict['history_eq'] = ['d/u']
        else:
            dict['history_eq'] = ['u/d']
        dict['points'] = [point_structure[dict['history_eq'][-1]][point_index]]
        for stat in dict:
            st.session_state['game_state'][stat] += dict[stat]
    def bot(game_state):
        return rn.choice(['up','down'])    
    def get_html(score, eq_type, direction, last_col):
        if direction == 'up':
            direction_int = 1
        else:
            direction_int = 0
        if eq_type == st.session_state['game_state']['history_eq'][-1]:
            border_style = "2px solid #3944dd"
        else:
            border_style = "1px solid #000000"
        if last_col:
            txt = 'ctx.font = "20px Sans-Serif"; ctx.fillText("'+str(score[0])+' | '+str(score[1])+'", parseInt(0.82*width), parseInt(height*'+str(1-direction_int)+'+15*'+str(direction_int)+'));'
            fct = str(0.8)
        else:
            txt = ''
            fct = str(1)

        draw = 'ctx.moveTo(0, height*'+str(direction_int)+'); ctx.lineTo(width*'+fct+', height*'+str(1-direction_int)+');'

        html_string ="""
                    <div id="d1" style="position:absolute; top:0px; left:15px">
                        <canvas id="Canvas1" width="200" height="70">
                        </canvas>
                    </div>
                    <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                    <script>
                    var c = document.getElementById("Canvas1");
                    width = parseInt(window.innerWidth*0.9);
                    height = parseInt(window.innerHeight*0.9);
                    c.width = width;
                    c.height = height;
                    var ctx = c.getContext("2d");
                    """+draw+"""
                    """+txt+"""
                    ctx.stroke();
                    </script>
                """
        return html_string
    ###################################################################################
    ########################### GAME CODE #############################################
    ###################################################################################

    point_structure = {'u/u':[1,2],'d/d':[0,0],'d/u':[2,1],'u/d':[0,2]}
    st.subheader('Wiederholtes sequenzielles Spiel')
    HEIGHT = 60
    if st.session_state['game_state']['round']<st.session_state['game_state']['final']:
        if st.session_state['game_state']['round']>0:
            txt_gegner = 'Dein Gegner hat letzte Runde '+ str(st.session_state['game_state']['history_b'][-1])+' gespielt'
            txt_punkte = 'Du erhälst dadurch '+ str(st.session_state['game_state']['points'][-1])+ ' Punkte'
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
        c1, c2, c3, c4 = st.columns((1,1,1,1))
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
            else:
                components.html(
                    """
                        <div style="text-align:center">
                            <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                            <p style="border:1px; border-style:solid; border-color:#e27777; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">Gegner Zug</p>
                        </div>
                    """,
                    height=70)    
            components.html(
                """
                    <div></div>
                """,
                height=HEIGHT-15)        
            components.html(
                    get_html([],'g/l','up', False),
                    height=HEIGHT+30)
            components.html(
                    get_html([],'g/l','down', False),
                    height=HEIGHT)
            components.html(
                """
                    <div></div>
                """,
                height=HEIGHT-15)

            if player_role=='first':
                player_first = True
                st.button('up', key='u1',on_click=set_game_state, args=(st, {'round':1, 'points':[], 'history_b':[bot(st.session_state['game_state'])], 'history_p':['up'], 'history_eq':[]}, player_first, point_structure))
                st.button('down', key='d1',on_click=set_game_state, args=(st, {'round':1, 'points':[], 'history_b':[bot(st.session_state['game_state'])], 'history_p':['down'], 'history_eq':[]}, player_first, point_structure))
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
            else:
                components.html(
                    """
                        <div style="text-align:center">
                            <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                            <p style="border:1px; border-style:solid; border-color:#e27777; border-radius: 4px 4px 4px 4px; font-family:IBM Plex Sans; padding: 1em;">Gegner Zug</p>
                        </div>
                    """,
                    height=70)   
            components.html(
                    get_html(point_structure['u/u'],'g/l','up', True),
                    height=HEIGHT)
            components.html(
                    get_html(point_structure['u/d'],'g/l','down', True),
                    height=HEIGHT)
            components.html(
                    get_html(point_structure['d/u'],'g/l','up', True),
                    height=HEIGHT)
            components.html(
                    get_html(point_structure['d/d'],'g/l','down', True),
                    height=HEIGHT)

            if player_role=='second':
                player_first = False
                st.button('up', key='u2',on_click=set_game_state, args=(st, {'round':1, 'points':[], 'history_b':[bot(st.session_state['game_state'])], 'history_p':['up'], 'history_eq':[]}, player_first, point_structure))
                st.button('down', key='d2',on_click=set_game_state, args=(st, {'round':1, 'points':[], 'history_b':[bot(st.session_state['game_state'])], 'history_p':['down'], 'history_eq':[]}, player_first, point_structure))
        components.html(
            """
                <div></div>
            """,
            height=100)
    else:
        st.write('Du hast',sum(st.session_state['game_state']['points']),'punkte erzielt')
        st.button('rerun', key='r1', on_click=set_type, args=(st,'game1'))
    st.button('back', key='b2', on_click=set_type, args=(st, 'default'))


def game3(st, **state):
    st.subheader('Game3')
    st.button('back', key='b1', on_click=set_type, args=(st, 'default'))

'''
#######################################################################################
'''

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
                    components.html(
                        """
                            <div>
                                <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                                <p style="font-family:IBM Plex Sans; font-size: 26px"><b>Wiederholtes Gefangenendilemma</b></p>
                            </div>
                        """,
                        height=100,)
                    st.image(image,use_column_width=True)                
                    st.button('play', key='p1', on_click=set_type, args=(st,'game1'))
            with col2:
                with st.container():
                    components.html(
                        """
                            <div>
                                <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                                <p style="font-family:IBM Plex Sans; font-size: 26px"><b>Sequenzielles Spiel</b></p>
                            </div>
                        """,
                        height=100,)
                    st.image(image,use_column_width=True)
                    st.button('play', key='p2', on_click=set_type, args=(st,'game2'))
            with col3:
                with st.container():
                    components.html(
                        """
                            <div>
                                <link href='https://fonts.googleapis.com/css?family=IBM Plex Sans' rel='stylesheet'>
                                <p style="font-family:IBM Plex Sans; font-size: 26px"><b>Game3</b></p>
                            </div>
                        """,
                        height=100,)
                    st.image(image,use_column_width=True)
                    st.button('play', key='p3', on_click=set_type, args=(st,'game3'))

    else:
        st.warning('pls log in')