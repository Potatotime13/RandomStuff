import numpy as np
import pandas as pd
import re
import streamlit as st
import random as rn

st.title('Klausur Generator CBEN')

page = st.sidebar.selectbox('Seitenauswahl', ('Klausur','Aufgabe Hinzufügen', 'Developer Kram'))

def klausur():

    txt = open('Questions.txt', 'r',encoding='utf8').read()

    q_a = re.split('#*#', txt)

    q_a_table = []
    for num in q_a:
        if len(num)>10:
            q_a_table.append(re.split('--*--', num))

    q_a_df = pd.DataFrame(q_a_table)

    KLAUSUR_LEN = st.sidebar.slider('Anzahlfragen', 10,q_a_df.shape[0],10)

    def clear_text():
        for i in range(KLAUSUR_LEN):
            st.session_state['A'+str(i)] = ''

    if st.sidebar.button('Neue Klausur', on_click=clear_text):
        st.session_state.ind = np.random.permutation(q_a_df.shape[0])[0:KLAUSUR_LEN]
        st.session_state.ans = ['' for _ in range(KLAUSUR_LEN)]

    def exercises():
        st.header('Praktische Aufgaben')
        # params PA problem
        c_a = 100 + rn.randint(1,10) * 10
        c_b = 100
        o_a = 1000
        o_b = 500
        q = rn.randint(2,6) * 0.1
        p = q + rn.randint(1,3) * 0.1
        w0 = rn.randint(0,20) * 10

        st.write('### Gegeben sei folgendes PA Problem:')
        st.write('Der effort ist nur indirekt durch das outbut beobachtbar')
        st.write('effort:'+'  \n'
                    +'hoher effort mit kosten c_a='+str(c_a)+'  \n'+'niedriger effort mit kosten c_b='+str(c_b))
        st.write('output:'+'  \n'
                    + 'output für hohen effort o_a = p * '+str(o_a)+' + (1-p) *'+str(o_b)+' mit p='+str(p)+'  \n'
                    +'output für niedrigen effort o_b = q * '+str(o_a)+' + (1-q) *'+str(o_b)+' mit q='+str(q))
        st.write('A hat ein Grundeinkommen von w0='+str(w0))
        st.write('### Löse die Folgenden Aufgaben:')

        st.write('a) Unter welchen Bedingungen ist der Aufwand a effizient')
        with st.expander('Lösung'):
            c_diff = p * o_a + (1-p) * o_b - (q * o_a + (1-q) * o_b)
            st.write('c_a - c_b <= E[o|a] - E[o|b] -> c_a - c_b<='+str(c_diff))
        st.write('b) Unter beobachtbarem effort, welchen Vertrag würde P A anbieten?')
        with st.expander('Lösung'):
            w_p = c_a + w0
            st.write('c_a + w0 = w -> w='+str(w_p))
        st.write('c) Unter unbeobachtbarem effort, welchen vertrag bietet P A an, so dass dieser den effort a einsetzt')
        with st.expander('Lösung'):
            st.write('p*w_a+(1-p)*w_b-c_a=q*w_a+(1-q)*w_b-c_a')        
        st.write('d) Gegeben c) unter welcher Bedingung würde A arbeiten')
        with st.expander('Lösung'):
            st.write('p*w_a+(1-p)*w_b>=w0+c_a')
        st.write('e) welche sind unter den gegebenen Parametern die Gehälter die sich aus den Bedingungen ergeben?')
        with st.expander('Lösung'):
            w_a = ((w_p)/(1-p)+(c_a-c_b)/(p-q))*(1-p)
            w_b = w_a - (c_a-c_b)/(p-q)
            st.write('w_a='+str(w_a)+'  \n'+'w_b='+str(w_b))
        
        w02 = rn.randint(5,15) * 0.1
        exp_c = rn.randint(1,3)
        if exp_c == 1:
            exp_e = rn.randint(2,3)
        else:
            exp_e = rn.randint(1,3)
        st.write('### Gegeben sei folgendes PA Problem:')
        st.write('Der Aufwand e ist beobachtbar'+'  \n'+'Das Marktgehalt ist w0='+str(w02)+'  \n'+
                    'Die Kostenfunktion des Agenten ist c(e)=e^'+str(exp_c)+'  \n'+
                    'Die Ertragsfunktion lautet p(e)=e^(1/'+str(exp_e)+')')
        st.write('a) Stelle das Participation Constraint auf')
        with st.expander('Lösung'):
            st.write('c(e) + w0 <= w -> w = c(e) + '+str(w02))
        st.write('b) Welchen Aufwand würde P A abverlangen, so dass er seinen Ertrag maximiert?')
        with st.expander('Lösung'):
            e_pa = (exp_c * exp_e)**(exp_e/(1-(exp_c * exp_e)))
            st.write('e*='+str(e_pa))
        st.write('c) Gegeben der Lösung aus b) welches Gehalt würde P anbieten?')
        with st.expander('Lösung'):
            w_pa = w02+e_pa**exp_c
            st.write('w*='+str(w_pa))

    if 'ind' not in st.session_state:
        st.session_state.ind = np.random.permutation(q_a_df.shape[0])[0:KLAUSUR_LEN]

    if 'ans' not in st.session_state:
        st.session_state.ans = ['' for _ in range(KLAUSUR_LEN)]

    st.header('Vorlesungs Fragen')

    exp_list = [st.empty() for _ in range(KLAUSUR_LEN)]
    for num, i in enumerate(st.session_state.ind):
        st.write(q_a_df.iloc[i,0])
        exp_list[num] = st.expander(label='Lösung', expanded=False)
        exp_list[num].write(q_a_df.iloc[i,1])
        st.session_state.ans[num] = st.text_area('Antwort'+str(num), height=100, key='A'+str(num))
    exercises()

def add_task():
    st.write('Fragestellung mit Nummerierung im Format (Foliensatznummer).(Folie):')
    fragestellung = st.text_area('Fragestellung', height=100, key='F')
    antwort = st.text_area('Antwort', height=100, key='A')

    def write_qa():
        txt = open('Questions.txt', 'a',encoding='utf8')
        txt.write('\n'+fragestellung)
        txt.write('\n'+'-----')
        txt.write('\n'+antwort)
        txt.write('\n'+'###')
        txt.close()
        st.session_state['F'] = ''
        st.session_state['A'] = ''
        st.success('Hinzufügen erfolgreich')

    st.button('Hinzufügen', on_click=write_qa)

def save_updates():
    with open('Questions.txt') as f:
        st.download_button('Download QA', f)

if page == 'Klausur':
    klausur()
elif page == 'Aufgabe Hinzufügen':
    add_task()
elif page == 'Developer Kram':
    save_updates()
