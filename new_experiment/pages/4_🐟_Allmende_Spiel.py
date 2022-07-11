import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import random as rn 
import plotly.express as px
import pandas as pd
import numpy as np
import pydeck as pdk

st.set_page_config(layout="wide")
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    if 'game_state_a' not in st.session_state:
        st.session_state.game_state_a = {'round':0, 'points':[], 'history_b':[], 'history_p':[],'history_eq':['e/e'], 'final':rn.randint(3,7)}

    ############################### HELPER METHODS ####################################
    def set_game_state(dummy, dict):
        resource_sum = dict['history_p'][-1] + dict['history_b'][-1]
        population = st.session_state.game_state_a['history_eq'][-1]
        dict['history_p'][-1]= round(min(dict['history_p'][-1],population*(dict['history_p'][-1]/resource_sum)))
        dict['history_b'][-1] = round(min(dict['history_b'][-1],population*(dict['history_b'][-1]/resource_sum)))
        dict['points'] = [dict['history_p'][-1]]
        dict['history_eq'] = [population - dict['history_p'][-1] - dict['history_b'][-1]]
        dict['history_eq'][-1] += int(dict['history_eq'][-1]*(dict['history_eq'][-1]/100)**2*0.5)
        dict['history_eq'][-1] = max(0, dict['history_eq'][-1])
        for stat in dict:
            st.session_state.game_state_a[stat] += dict[stat]

    def reset_game_state():
        st.session_state.game_state_a = {'round':0, 'points':[], 'history_b':[], 'history_p':[],'history_eq':['e/e'], 'final':rn.randint(3,7)}

    def bot(game_state):
        return int(rn.random() * 30)

    def get_html():
        pass

    ###################################################################################
    ########################### GAME CODE #############################################
    ###################################################################################

    st.subheader('Allmende Spiel')
    if st.session_state.game_state_a['history_eq'][-1] == 'e/e':
        st.session_state.game_state_a['history_eq'][-1] = 100

    HEIGHT = 60
    if st.session_state.game_state_a['round']<st.session_state.game_state_a['final'] and st.session_state.game_state_a['history_eq'][-1]>0:
        c1, c2, = st.columns((1,2))
        with c2:
            diameter = {
                    'a':[50.0097544467139, 8.619416562978586],
                    'b':[50.0187584253731, 8.619416562978586],
                    }
            coords = {
                    1:[50.009459236206304, 8.613143826021536],
                    2:[50.01122432518706, 8.613165283958244],
                    3:[50.01122432518706, 8.615461254702526],
                    4:[50.01308587235735, 8.615461254702526],
                    5:[50.01510932696573, 8.615461254702526],
                    6:[50.01510932696573, 8.610483074879927],
                    7:[50.0171477639355, 8.610483074879927],
                    8:[50.0171477639355, 8.615082712374176],
                    9:[50.01824267077451, 8.615082712374176]
                    }
            coords = {
                    1:[50.009459236206304, 8.618127664732916],
                    2:[50.0109209551541, 8.618127664732916],
                    3:[50.01121053579544, 8.624387646079429],
                    4:[50.01715344666876, 8.62376537365095],
                    5:[50.01715344666876, 8.618127664732916],
                    6:[50.01887689122687, 8.620527664732916],
                    7:[50.0141209551541, 8.618127664732916],
                    8:[50.0153209551541, 8.618127664732916],
                    }
            rechteck_rechts = {
                    2:[50.0109209551541, 8.618127664732916],
                    3:[50.01121053579544, 8.624387646079429],
                    4:[50.01715344666876, 8.62376537365095],
                    6:[50.01887689122687, 8.618127664732916],
            }
            gerade_mitte = {
                    1:[50.0096, 8.6181],
                    12:[50.0102, 8.6181],
                    2:[50.0109, 8.6181],
                    3:[50.0171, 8.6181],
                    4:[50.0188, 8.6181],
                    5:[50.0141, 8.6181],
                    6:[50.0153, 8.6181],
            }
            gerade_mitte2 = [
                    [50.0096, 8.6181],
                    [50.01052, 8.6181],
                    [50.01144, 8.6181],
                    [50.01236, 8.6181],
                    [50.01328, 8.6181],
                    [50.0142, 8.6181],
                    [50.01512, 8.6181],
                    [50.01604, 8.6181],
                    [50.01696, 8.6181],
                    [50.01788, 8.6181],
                    [50.0188, 8.6181]
            ]
            li_re = {
                1:[[50.009459236206304, 8.613127664732916],[50.009459236206304, 8.618127664732916]],
                2:[[50.009459236206304, 8.613127664732916],[50.01122432518706, 8.615461254702526]],
                3:[[50.0141209551541, 8.615927664732916],[50.0141209551541, 8.615461254702526]],
                4:[[50.0153209551541, 8.610127664732916],[50.0153209551541, 8.615461254702526]],
                5:[[50.009459236206304, 8.611127664732916],[50.01122432518706, 8.615461254702526]],
                6:[[50.01887689122687, 8.618127664732916],[50.01887689122687,8.620527664732916]],
            }
            li_re2 = [
                [[50.0096, 8.6131],[50.0096, 8.6181]],
                [[50.01052, 8.6131],[50.01052, 8.6181]],
                [[50.01144, 8.6131],[50.01144, 8.6244]],
                [[50.01236, 8.6159],[50.01236, 8.6244]],
                [[50.01328, 8.6159],[50.01328, 8.6244]],
                [[50.0142, 8.6159],[50.0142, 8.6244]],
                [[50.01512, 8.6159],[50.01512, 8.6244]],
                [[50.01604, 8.6101],[50.01604, 8.6244]],
                [[50.01696, 8.6101],[50.01696, 8.6244]],
                [[50.01788, 8.6151],[50.01788, 8.6205]],
                [[50.0188, 8.6181],[50.0188, 8.6205]],
            ]
            soos = np.array(li_re2)
            rand_pos = np.random.rand(30)
            for i in range(len(rand_pos)):
                ind = int(rand_pos[i]*110//10)
                upper = soos[ind,0,0]

            coord_df = pd.DataFrame(
                gerade_mitte,
                index=['lat', 'lon'])
            coord_df = coord_df.T

            def find_closest(df, lat):
                dist = (df['lat'] - lat).abs()
                return df.loc[dist.idxmin()]

            diff = coord_df['lat'].max()-coord_df['lat'].min()
            samples = np.random.rand(100)
            for s in samples:
                lat = coord_df['lat'].min()+ diff*s
                n1 = find_closest(coord_df, lat)
                n2 = find_closest(coord_df.drop(n1.name), lat)

            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=50.0129,
                    longitude=8.6186,
                    zoom=14.5,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=coord_df,
                        get_position='[lon, lat]',
                        get_color='[200, 30, 0, 160]',
                        get_radius=10,
                    ),
                ],
            ))

            with st.form('slider_form'):
                resources = st.slider('Fischfang')
                submit = st.form_submit_button('submit')
                if submit:
                    set_game_state(5, {'round':1, 'points':[], 'history_b':[bot(st.session_state.game_state_a)], 'history_p':[resources], 'history_eq':[]})
        with c1:
            if len(st.session_state.game_state_a['history_p']) < 1:
                fish_p = 0
                fish_b = 0
            else:
                fish_p = st.session_state.game_state_a['history_p'][-1]
                fish_b = st.session_state.game_state_a['history_b'][-1]
            population = st.session_state.game_state_a['history_eq'][-1]
            stats = pd.DataFrame({'':['Fisch','zuletzt gefangen', 'zuletzt gefangen'], 'Player':['Population', 'Gegner', 'Du'], 'Menge':[population,fish_b,fish_p]})
            fig = px.bar(stats, x='', y='Menge', color="Player", title="Zustand")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False,})
            
            hide_fs = '''
                <style>
                button[title="View fullscreen"]{
                    visibility: hidden;}
                </style>
                '''
            st.markdown(hide_fs, unsafe_allow_html=True)
        components.html(
            """
                <div></div>
            """,
            height=100)
    else:
        st.write('Du hast',sum(st.session_state.game_state_a['points']),'punkte erzielt')
        st.button('Neue Runde', on_click=reset_game_state)
else:
    st.write('bitte einloggen')