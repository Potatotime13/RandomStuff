import numpy as np
import pandas as pd
import re
import streamlit as st

txt = open('Questions.txt', 'r',encoding='utf8').read()

q_a = re.split('#*#', txt)

q_a_table = []
for num in q_a:
    if len(num)>10:
        q_a_table.append(re.split('--*--', num))

q_a_df = pd.DataFrame(q_a_table)
KLAUSUR_LEN = 10

def clear_text():
    for i in range(KLAUSUR_LEN):
        st.session_state['A'+str(i)] = ''

if st.sidebar.button('Neue Klausur', on_click=clear_text):
    st.session_state.ind = np.random.permutation(q_a_df.shape[0])[0:KLAUSUR_LEN]
    st.session_state.ans = ['' for _ in range(KLAUSUR_LEN)]

def exercises(sol=False):
    st.header('Praktische Aufgaben')
    # params PA problem
    c_a = 150
    c_b = 100
    o_a = 1000
    o_b = 500
    p = 0.5
    q = 0.3
    w0 = 100

    st.write('### Gegeben sei folgendes PA Problem:')
    st.write('effort:'+'  \n'
                +'hoher effort mit kosten c_a='+str(c_a)+'  \n'+'niedriger effort mit kosten c_b='+str(c_b))
    st.write('output:'+'  \n'
                + 'output für hohen effort o_a = p * '+str(o_a)+' + (1-p) *'+str(o_b)+' mit p='+str(p)+'  \n'
                +'output für niedrigen effort o_b = q * '+str(o_a)+' + (1-q) *'+str(o_b)+' mit q='+str(q))
    st.write('A hat ein Grundeinkommen von w0='+str(w0))
    st.write('### Löse die Folgenden Aufgaben:')
    if not sol:
        st.write('a) Unter welchen Bedingungen ist der Aufwand a effizient')
        st.write('b) Unter beobachtbarem effort, welchen Vertrag würde P A anbieten?')
        st.write('c) Unter unbeobachtbarem effort, welchen vertrag bietet P A an, so dass dieser den effort a einsetzt')
        st.write('d) Gegeben c) unter welcher Bedingung würde A arbeiten')
        st.write('e) welche sind unter den gegebenen Parametern die Gehälter die sich aus den Bedingungen ergeben?')

st.title('Klausur Generator CBEN')

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

