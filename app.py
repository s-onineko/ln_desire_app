import streamlit as st
from PIL import Image
import subprocess
import pandas as pd
import os
import datetime
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import base64
import os
import json
import pickle
import uuid
import re

#desktop_path = os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop\\"

#============================================================================
#                                タイトル領域
#============================================================================
image_1 = Image.open("materials/head.png")
st.image(image_1, use_column_width=True)
'''
下記の各設問について個人の回答を入力すると、欲求のタイプを判定することができます。
'''
#============================================================================
#                           欲求フラグ判定結果
#============================================================================

def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=True)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


#STEP-2

mode = st.sidebar.radio("グラフ化",["設問入力モード","欲求フラグ判定結果"])
if mode == "欲求フラグ判定結果":
    st.subheader('2. 欲求フラグ判定結果')
   
    answer = st.file_uploader("回答結果のCSVファイルの読み込み",type = "csv")
    df_answer = pd.read_csv(answer)
    df_answer = df_answer['answer']
    df_answer_b = df_answer
    df_answer = df_answer.replace(1, 33)
    df_answer = df_answer.replace(3, 1)
    df_answer = df_answer.replace(33, 3)
    df_answer = df_answer.replace(4, 0)
    df_answer = df_answer.replace(5, -1)
    df_answer = df_answer.replace(6, -2)
    df_answer = df_answer.replace(7, -3)
    df_answer_b = df_answer_b.replace(1, -3)
    df_answer_b = df_answer_b.replace(2, -2)
    df_answer_b = df_answer_b.replace(3, -1)
    df_answer_b = df_answer_b.replace(4, 0)
    df_answer_b = df_answer_b.replace(5, 1)
    df_answer_b = df_answer_b.replace(6, 2)
    df_answer_b = df_answer_b.replace(7, 3)

    df_answer_ab = df_answer.append(df_answer_b, ignore_index=True)
    df_answer_ab.index = ['flg1', 'flg2','flg3','flg4','flg5','flg6','flg7','flg8','flg9','flg10',\
        'flg11','flg12','flg13','flg14','flg15','flg16','flg17','flg18','flg19','flg20',\
        'flg21','flg22','flg23','flg24','flg25','flg26','flg27','flg28','flg29','flg30',\
        'flg31','flg32','flg33','flg34','flg35','flg36','flg37','flg38','flg39','flg40',\
        'flg41','flg42','flg43','flg44','flg45','flg46']
    df_clst = pd.merge(df_answer_ab,df_answer_ab,how = "left",left_index = True, right_index = True).T
    #df_clst.to_csv("clst_out.csv")

    val2 = st.selectbox("比較対象 ( 比較対象の調査を実施した際のn数 )",["クリックして比較対象を選択ください","全体 ( n = 37,830 )",\
        "男性全体 ( n = 15,480 )","女性全体 ( n = 22,350 )",\
        "男性10代 ( n = 248 )","男性20代 ( n = 1,356 )","男性30代 ( n = 2,851 )","男性40代 ( n = 3,712 )","男性50代 ( n = 3,224 )","男性60代 ( n = 2,711 )","男性70代 ( n = 1,378 )",\
        "女性10代 ( n = 430 )","女性20代 ( n = 3,267 )","女性30代 ( n = 5,015 )","女性40代 ( n = 5,628 )","女性50代 ( n = 4,097 )","女性60代 ( n = 2,487 )","女性70代 ( n = 1,426 )"])

    if val2 == "男性全体 ( n = 15,480 )":
        values2 = [-0.06,0.55,-0.12,0.44,-0.15,0.14,0.25,0.06,-0.41,-0.11,0.24,0.21,0.08,0.29,0.06,0.05,0.12,0.06,-0.2,-0.41,-0.27,0.26,-0.05,0.19,-0.55,0.15,-0.25,0.12,-0.14,-0.44,0.41,0.11,-0.06,-0.24,-0.21,-0.29,-0.05,-0.12,0.2,0.41,0.27,-0.26,0.05,-0.19,-0.06,-0.08]
    elif val2 == "女性全体 ( n = 22,350 )":
        values2 = [0.2,0.95,0.13,0.41,0.06,0.2,0.56,0.16,-0.48,-0.07,0.47,0.1,-0.29,0.06,-0.25,-0.34,-0.06,-0.2,-0.59,-0.79,-0.74,0.11,-0.12,0.08,-0.95,-0.06,-0.56,-0.13,-0.2,-0.41,0.48,0.07,-0.16,-0.47,-0.1,-0.06,0.34,0.06,0.59,0.79,0.74,-0.11,0.12,-0.08,0.25,0.29]
    elif val2 == "男性10代 ( n = 248 )":
        values2 = [-0.34,0.21,-0.5,0.17,-0.44,-0.3,-0.16,-0.52,-0.73,-0.64,-0.27,-0.13,0.42,0.38,0.19,0.15,0.33,0.34,0.19,-0.41,0.21,0.59,0.4,0.79,-0.21,0.44,0.16,0.5,0.3,-0.17,0.73,0.64,0.52,0.27,0.13,-0.38,-0.15,-0.33,-0.19,0.41,-0.21,-0.59,-0.4,-0.79,-0.19,-0.42]
    elif val2 == "男性20代 ( n = 1,356 )":
        values2 = [-0.15,0.26,-0.3,0.32,-0.39,0.01,-0.02,-0.25,-0.71,-0.35,-0.08,-0.05,0.15,0.27,0.14,0.06,0.24,0.15,-0.05,-0.38,-0.15,0.37,0.22,0.55,-0.26,0.39,0.02,0.3,-0.01,-0.32,0.71,0.35,0.25,0.08,0.05,-0.27,-0.06,-0.24,0.05,0.38,0.15,-0.37,-0.22,-0.55,-0.14,-0.15]
    elif val2 == "男性30代 ( n = 2,851 )":
        values2 = [-0.1,0.45,-0.29,0.3,-0.27,0.04,0.09,-0.2,-0.66,-0.38,0.02,-0.01,0.1,0.23,0.1,0.08,0.28,0.1,-0.17,-0.38,-0.19,0.29,0.14,0.32,-0.45,0.27,-0.09,0.29,-0.04,-0.3,0.66,0.38,0.2,-0.02,0.01,-0.23,-0.08,-0.28,0.17,0.38,0.19,-0.29,-0.14,-0.32,-0.1,-0.1]
    elif val2 == "男性40代 ( n = 3,712 )":
        values2 = [-0.07,0.55,-0.22,0.41,-0.15,0.14,0.27,-0.04,-0.52,-0.22,0.2,0.14,0.06,0.25,0.04,0.03,0.21,0.07,-0.2,-0.42,-0.21,0.26,-0.07,0.24,-0.55,0.15,-0.27,0.22,-0.14,-0.41,0.52,0.22,0.04,-0.2,-0.14,-0.25,-0.03,-0.21,0.2,0.42,0.21,-0.26,0.07,-0.24,-0.04,-0.06]
    elif val2 == "男性50代 ( n = 3,224 )":
        values2 = [-0.11,0.66,-0.08,0.48,-0.07,0.18,0.34,0.14,-0.32,-0.07,0.33,0.24,0.05,0.3,0.02,0.05,0.15,0.11,-0.18,-0.4,-0.28,0.25,-0.15,0.12,-0.66,0.07,-0.34,0.08,-0.18,-0.48,0.32,0.07,-0.14,-0.33,-0.24,-0.3,-0.05,-0.15,0.18,0.4,0.28,-0.25,0.15,-0.12,-0.02,-0.05]
    elif val2 == "男性60代 ( n = 2,711 )":
        values2 = [0,0.69,0.09,0.54,-0.05,0.21,0.39,0.36,-0.15,0.19,0.49,0.45,0.06,0.35,0.06,0.08,-0.05,0,-0.26,-0.43,-0.41,0.24,-0.18,0.01,-0.69,0.05,-0.39,-0.09,-0.21,-0.54,0.15,-0.19,-0.36,-0.49,-0.45,-0.35,-0.08,0.05,0.26,0.43,0.41,-0.24,0.18,-0.01,-0.06,-0.06]
    elif val2 == "男性70代 ( n = 1,378 )":
        values2 = [0.24,0.61,0.28,0.65,0.01,0.31,0.41,0.52,0.02,0.37,0.52,0.58,0.07,0.38,0.03,0,-0.31,-0.24,-0.36,-0.48,-0.52,0.16,-0.25,-0.18,-0.61,-0.01,-0.41,-0.28,-0.31,-0.65,-0.02,-0.37,-0.52,-0.52,-0.58,-0.38,0,0.31,0.36,0.48,0.52,-0.16,0.25,0.18,-0.03,-0.07]
    elif val2 == "女性10代 ( n = 430 )":
        values2 = [-0.36,0.27,-0.63,-0.07,-0.4,-0.43,-0.1,-0.29,-0.93,-0.42,-0.06,-0.49,0.14,0.13,-0.01,-0.05,0.44,0.36,0,-0.5,-0.23,0.43,0.33,0.78,-0.27,0.4,0.1,0.63,0.43,0.07,0.93,0.42,0.29,0.06,0.49,-0.13,0.05,-0.44,0,0.5,0.23,-0.43,-0.33,-0.78,0.01,-0.14]
    elif val2 == "女性20代 ( n = 3,267 )":
        values2 = [-0.04,0.95,-0.2,0.33,-0.09,0.08,0.48,-0.03,-0.83,-0.22,0.35,-0.23,-0.22,-0.01,-0.15,-0.42,0.06,0.04,-0.47,-0.82,-0.69,0.21,0.06,0.42,-0.95,0.09,-0.48,0.2,-0.08,-0.33,0.83,0.22,0.03,-0.35,0.23,0.01,0.42,-0.06,0.47,0.82,0.69,-0.21,-0.06,-0.42,0.15,0.22]
    elif val2 == "女性30代 ( n = 5,015 )":
        values2 = [0.22,1.02,-0.01,0.38,0.01,0.15,0.58,0.04,-0.67,-0.17,0.41,-0.04,-0.37,-0.03,-0.25,-0.44,-0.05,-0.22,-0.62,-0.9,-0.79,0.08,-0.09,0.17,-1.02,-0.01,-0.58,0.01,-0.15,-0.38,0.67,0.17,-0.04,-0.41,0.04,0.03,0.44,0.05,0.62,0.9,0.79,-0.08,0.09,-0.17,0.25,0.37]
    elif val2 == "女性40代 ( n = 5,628 )":
        values2 = [0.22,0.95,0.13,0.43,0.08,0.22,0.57,0.11,-0.46,-0.13,0.43,0.14,-0.34,0.04,-0.28,-0.31,-0.04,-0.22,-0.66,-0.78,-0.71,0.09,-0.14,0.05,-0.95,-0.08,-0.57,-0.13,-0.22,-0.43,0.46,0.13,-0.11,-0.43,-0.14,-0.04,0.31,0.04,0.66,0.78,0.71,-0.09,0.14,-0.05,0.28,0.34]
    elif val2 == "女性50代 ( n = 4,097 )":
        values2 = [0.2,0.94,0.27,0.44,0.13,0.28,0.57,0.24,-0.31,-0.03,0.56,0.24,-0.32,0.08,-0.28,-0.29,-0.06,-0.2,-0.61,-0.73,-0.75,0.08,-0.17,-0.07,-0.94,-0.13,-0.57,-0.27,-0.28,-0.44,0.31,0.03,-0.24,-0.56,-0.24,-0.08,0.29,0.06,0.61,0.73,0.75,-0.08,0.17,0.07,0.28,0.32]
    elif val2 == "女性60代 ( n = 2,487 )":
        values2 = [0.37,0.97,0.45,0.49,0.18,0.32,0.62,0.49,-0.16,0.23,0.7,0.39,-0.24,0.19,-0.3,-0.29,-0.18,-0.37,-0.64,-0.74,-0.82,0.09,-0.24,-0.18,-0.97,-0.18,-0.62,-0.45,-0.32,-0.49,0.16,-0.23,-0.49,-0.7,-0.39,-0.19,0.29,0.18,0.64,0.74,0.82,-0.09,0.24,0.18,0.3,0.24]
    elif val2 == "女性70代 ( n = 1,426 )":
        values2 = [0.48,0.92,0.59,0.5,0.17,0.34,0.68,0.59,0,0.4,0.67,0.45,-0.21,0.29,-0.26,-0.27,-0.35,-0.48,-0.55,-0.73,-0.76,0.11,-0.27,-0.26,-0.92,-0.17,-0.68,-0.59,-0.34,-0.5,0,-0.4,-0.59,-0.67,-0.45,-0.29,0.27,0.35,0.55,0.73,0.76,-0.11,0.27,0.26,0.26,0.21]
    else:
        values2 = [0.09,0.79,0.03,0.42,-0.03,0.18,0.43,0.12,-0.45,-0.09,0.38,0.14,-0.14,0.15,-0.12,-0.18,0.02,-0.09,-0.43,-0.63,-0.55,0.17,-0.09,0.12,-0.79,0.03,-0.43,-0.03,-0.18,-0.42,0.45,0.09,-0.12,-0.38,-0.14,-0.15,0.18,-0.02,0.43,0.63,0.55,-0.17,0.09,-0.12,0.12,0.14]


    import numpy as np
    def plot_polar(labels, values, values2):
        from PIL import Image
        angles = np.linspace(0, 2 * np.pi, 47, endpoint=True)
        angles2 = np.linspace(0, 2 * np.pi, 47, endpoint=True)
        values = np.concatenate((values, [values[0]]))  # 閉じた多角形にする
        values2 = np.concatenate((values2, [values2[0]]))  # 閉じた多角形にする
        fig = plt.figure()
        #画像の読み込み
        im = Image.open("materials/radar.png")
        #画像をarrayに変換
        im_list = np.asarray(im)
        #貼り付け
        plt.axis("off")
        plt.imshow(im_list)

        ax = fig.add_subplot(111, polar=True, alpha=0.05)
        ax.patch.set_alpha(0.001)
        marker_props = {
            "markersize": 2,
        }
        ax.plot(angles, values, 'o-',**marker_props, alpha=1, color="#006e54")  # 外枠
        ax.plot(angles2, values2, '-',**marker_props, alpha=0.25, color="blue")  # 外枠
        ax.fill(angles, values, alpha=0.25, color= '#006e54' )  # 塗りつぶし
        ax.set_rlim(-5 ,10)
        ax.axis("off")
        dt_now = datetime.datetime.now()
        time = dt_now.strftime('%Y%m%d %H%M')        
        plt.savefig(str(time) + str('-rader')  + str('.jpeg'))

    labels = ['1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',\
        '21','22','23','24','25','26','27','28','29','30','31','32','33','34',\
            '35','36','37','38','39','40','41','42','43','44','45','46']
    values = [df_clst['flg29'][0], df_clst['flg36'][0],df_clst['flg38'][0], df_clst['flg40'][0],\
        df_clst['flg37'][0], df_clst['flg39'][0],df_clst['flg41'][0], df_clst['flg44'][0],\
        df_clst['flg42'][0], df_clst['flg43'][0],df_clst['flg45'][0], df_clst['flg46'][0],\
        df_clst['flg1'][0], df_clst['flg2'][0],df_clst['flg3'][0], df_clst['flg4'][0],\
        df_clst['flg5'][0], df_clst['flg6'][0],df_clst['flg7'][0], df_clst['flg8'][0],\
        df_clst['flg9'][0], df_clst['flg10'][0],df_clst['flg11'][0], df_clst['flg12'][0],\
        df_clst['flg13'][0], df_clst['flg14'][0],df_clst['flg18'][0], df_clst['flg15'][0],\
        df_clst['flg16'][0], df_clst['flg17'][0],df_clst['flg19'][0], df_clst['flg20'][0],\
        df_clst['flg21'][0], df_clst['flg22'][0],df_clst['flg23'][0], df_clst['flg25'][0],\
        df_clst['flg27'][0], df_clst['flg28'][0],df_clst['flg30'][0], df_clst['flg31'][0],\
        df_clst['flg32'][0], df_clst['flg33'][0],df_clst['flg34'][0], df_clst['flg35'][0],\
        df_clst['flg26'][0], df_clst['flg24'][0]]
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot_polar(labels, values, values2))
    df_rader = pd.DataFrame(values, columns = ['個人スコア'])
    df_mean = pd.DataFrame(values2, columns = ['平均スコア'])
    df_table = pd.merge(df_rader,df_mean,how = "left", left_index=True,right_index=True)

    sort_label = [35,36,37,38,39,40,41,42,43,44,45,46,\
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,\
            21,22,23,24,25,26,27,28,29,30,31,32,33,34]   
    df_sort_label = pd.DataFrame(sort_label, columns = ['Number'])
    df_table = pd.merge(df_sort_label,df_table,how = "left", left_index=True,right_index=True)
    df_table = df_table.sort_values(by='Number')
    df_table = df_table.reset_index().drop('index', axis=1)
    df_table['差分'] = (df_table['個人スコア'] - df_table['平均スコア'])
    df_question = pd.read_csv("materials/question.csv", encoding = "cp932")
    df_table_view = pd.merge(df_question,df_table,how = "left", left_index=True,right_index=True)
    df_table_view = df_table_view.set_index('Number')
    image_0 = Image.open("materials/rader_list.PNG")
    st.image(image_0, use_column_width=True)
    st.table(df_table_view)


#============================================================================
#                       1. 欲求判定設問（全23問）
#============================================================================

else:
    #STEP-1
    st.subheader('1. 欲求判定設問（全23問）')
    '''
    あなたは、以下のAとBの項目について、どちらを重視しますか。
    より近いと思う方をお知らせください。
    '''
    st.markdown('''
    ##### 1：Ａに非常に近い\n
    ##### 2：Ａにかなり近い\n
    ##### 3：Ａにやや近い\n
    ##### 4：どちらともいえない\n
    ##### 5：Ｂにやや近い\n
    ##### 6：Ｂにかなり近い\n
    ##### 7：Ｂに非常に近い\n
    ***
    ''')
    image_1 = Image.open("materials/q1.png")
    st.image(image_1, use_column_width=True)
    q1 = st.selectbox("Q1：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q1 == "1：Ａに非常に近い":
        q1 = int(1)
    elif q1 == "2：Ａにかなり近い":
        q1 = int(2)
    elif q1 == "3：Ａにやや近い":
        q1 = int(3)
    elif q1 == "4：どちらともいえない":
        q1 = int(4)
    elif q1 == "5：Ｂにやや近い":
        q1 = int(5)
    elif q1 == "6：Ｂにかなり近い":
        q1 = int(6)
    else:
        q1 = int(7)

    image_2 = Image.open("materials/q2.png")
    st.image(image_2, use_column_width=True)
    q2 = st.selectbox("Q2：1～7から1つお選びください ",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q2 == "1：Ａに非常に近い":
        q2 = int(1)
    elif q2 == "2：Ａにかなり近い":
        q2 = int(2)
    elif q2 == "3：Ａにやや近い":
        q2 = int(3)
    elif q2 == "4：どちらともいえない":
        q2 = int(4)
    elif q2 == "5：Ｂにやや近い":
        q2 = int(5)
    elif q2 == "6：Ｂにかなり近い":
        q2 = int(6)
    else:
        q2 = int(7)


    image_3 = Image.open("materials/q3.png")
    st.image(image_3, use_column_width=True)
    q3 = st.selectbox("Q3：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q3 == "1：Ａに非常に近い":
        q3 = int(1)
    elif q3 == "2：Ａにかなり近い":
        q3 = int(2)
    elif q3 == "3：Ａにやや近い":
        q3 = int(3)
    elif q3 == "4：どちらともいえない":
        q3 = int(4)
    elif q3 == "5：Ｂにやや近い":
        q3 = int(5)
    elif q3 == "6：Ｂにかなり近い":
        q3 = int(6)
    else:
        q3 = int(7)


    image_4 = Image.open("materials/q4.png")
    st.image(image_4, use_column_width=True)
    q4 = st.selectbox("Q4：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q4 == "1：Ａに非常に近い":
        q4 = int(1)
    elif q4 == "2：Ａにかなり近い":
        q4 = int(2)
    elif q4 == "3：Ａにやや近い":
        q4 = int(3)
    elif q4 == "4：どちらともいえない":
        q4 = int(4)
    elif q4 == "5：Ｂにやや近い":
        q4 = int(5)
    elif q4 == "6：Ｂにかなり近い":
        q4 = int(6)
    else:
        q4 = int(7)


    image_5 = Image.open("materials/q5.png")
    st.image(image_5, use_column_width=True)
    q5 = st.selectbox("Q5：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q5 == "1：Ａに非常に近い":
        q5 = int(1)
    elif q5 == "2：Ａにかなり近い":
        q5 = int(2)
    elif q5 == "3：Ａにやや近い":
        q5 = int(3)
    elif q5 == "4：どちらともいえない":
        q5 = int(4)
    elif q5 == "5：Ｂにやや近い":
        q5 = int(5)
    elif q5 == "6：Ｂにかなり近い":
        q5 = int(6)
    else:
        q5 = int(7)


    image_6 = Image.open("materials/q6.png")
    st.image(image_6, use_column_width=True)
    q6 = st.selectbox("Q6：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q6 == "1：Ａに非常に近い":
        q6 = int(1)
    elif q6 == "2：Ａにかなり近い":
        q6 = int(2)
    elif q6 == "3：Ａにやや近い":
        q6 = int(3)
    elif q6 == "4：どちらともいえない":
        q6 = int(4)
    elif q6 == "5：Ｂにやや近い":
        q6 = int(5)
    elif q6 == "6：Ｂにかなり近い":
        q6 = int(6)
    else:
        q6 = int(7)


    image_7 = Image.open("materials/q7.png")
    st.image(image_7, use_column_width=True)
    q7 = st.selectbox("Q7：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q7 == "1：Ａに非常に近い":
        q7 = int(1)
    elif q7 == "2：Ａにかなり近い":
        q7 = int(2)
    elif q7 == "3：Ａにやや近い":
        q7 = int(3)
    elif q7 == "4：どちらともいえない":
        q7 = int(4)
    elif q7 == "5：Ｂにやや近い":
        q7 = int(5)
    elif q7 == "6：Ｂにかなり近い":
        q7 = int(6)
    else:
        q7 = int(7)


    image_8 = Image.open("materials/q8.png")
    st.image(image_8, use_column_width=True)
    q8 = st.selectbox("Q8：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q8 == "1：Ａに非常に近い":
        q8 = int(1)
    elif q8 == "2：Ａにかなり近い":
        q8 = int(2)
    elif q8 == "3：Ａにやや近い":
        q8 = int(3)
    elif q8 == "4：どちらともいえない":
        q8 = int(4)
    elif q8 == "5：Ｂにやや近い":
        q8 = int(5)
    elif q8 == "6：Ｂにかなり近い":
        q8 = int(6)
    else:
        q8 = int(7)


    image_9 = Image.open("materials/q9.png")
    st.image(image_9, use_column_width=True)
    q9 = st.selectbox("Q9：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q9 == "1：Ａに非常に近い":
        q9 = int(1)
    elif q9 == "2：Ａにかなり近い":
        q9 = int(2)
    elif q9 == "3：Ａにやや近い":
        q9 = int(3)
    elif q9 == "4：どちらともいえない":
        q9 = int(4)
    elif q9 == "5：Ｂにやや近い":
        q9 = int(5)
    elif q9 == "6：Ｂにかなり近い":
        q9 = int(6)
    else:
        q9 = int(7)

    image_10 = Image.open("materials/q10.png")
    st.image(image_10, use_column_width=True)
    q10 = st.selectbox("Q10：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q10 == "1：Ａに非常に近い":
        q10 = int(1)
    elif q10 == "2：Ａにかなり近い":
        q10 = int(2)
    elif q10 == "3：Ａにやや近い":
        q10 = int(3)
    elif q10 == "4：どちらともいえない":
        q10 = int(4)
    elif q10 == "5：Ｂにやや近い":
        q10 = int(5)
    elif q10 == "6：Ｂにかなり近い":
        q10 = int(6)
    else:
        q10 = int(7)


    image_11 = Image.open("materials/q11.png")
    st.image(image_11, use_column_width=True)
    q11 = st.selectbox("Q11：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q11 == "1：Ａに非常に近い":
        q11 = int(1)
    elif q11 == "2：Ａにかなり近い":
        q11 = int(2)
    elif q11 == "3：Ａにやや近い":
        q11 = int(3)
    elif q11 == "4：どちらともいえない":
        q11 = int(4)
    elif q11 == "5：Ｂにやや近い":
        q11 = int(5)
    elif q11 == "6：Ｂにかなり近い":
        q11 = int(6)
    else:
        q11 = int(7)

    image_12 = Image.open("materials/q12.png")
    st.image(image_12, use_column_width=True)
    q12 = st.selectbox("Q12：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q12 == "1：Ａに非常に近い":
        q12 = int(1)
    elif q12 == "2：Ａにかなり近い":
        q12 = int(2)
    elif q12 == "3：Ａにやや近い":
        q12 = int(3)
    elif q12 == "4：どちらともいえない":
        q12 = int(4)
    elif q12 == "5：Ｂにやや近い":
        q12 = int(5)
    elif q12 == "6：Ｂにかなり近い":
        q12 = int(6)
    else:
        q12 = int(7)


    image_13 = Image.open("materials/q13.png")
    st.image(image_13, use_column_width=True)
    q13 = st.selectbox("Q13：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q13 == "1：Ａに非常に近い":
        q13 = int(1)
    elif q13 == "2：Ａにかなり近い":
        q13 = int(2)
    elif q13 == "3：Ａにやや近い":
        q13 = int(3)
    elif q13 == "4：どちらともいえない":
        q13 = int(4)
    elif q13 == "5：Ｂにやや近い":
        q13 = int(5)
    elif q13 == "6：Ｂにかなり近い":
        q13 = int(6)
    else:
        q13 = int(7)


    image_14 = Image.open("materials/q14.png")
    st.image(image_14, use_column_width=True)
    q14 = st.selectbox("Q14：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q14 == "1：Ａに非常に近い":
        q14 = int(1)
    elif q14 == "2：Ａにかなり近い":
        q14 = int(2)
    elif q14 == "3：Ａにやや近い":
        q14 = int(3)
    elif q14 == "4：どちらともいえない":
        q14 = int(4)
    elif q14 == "5：Ｂにやや近い":
        q14 = int(5)
    elif q14 == "6：Ｂにかなり近い":
        q14 = int(6)
    else:
        q14 = int(7)


    image_15 = Image.open("materials/q15.png")
    st.image(image_15, use_column_width=True)
    q15 = st.selectbox("Q15：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q15 == "1：Ａに非常に近い":
        q15 = int(1)
    elif q15 == "2：Ａにかなり近い":
        q15 = int(2)
    elif q15 == "3：Ａにやや近い":
        q15 = int(3)
    elif q15 == "4：どちらともいえない":
        q15 = int(4)
    elif q15 == "5：Ｂにやや近い":
        q15 = int(5)
    elif q15 == "6：Ｂにかなり近い":
        q15 = int(6)
    else:
        q15 = int(7)


    image_16 = Image.open("materials/q16.png")
    st.image(image_16, use_column_width=True)
    q16 = st.selectbox("Q16：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q16 == "1：Ａに非常に近い":
        q16 = int(1)
    elif q16 == "2：Ａにかなり近い":
        q16 = int(2)
    elif q16 == "3：Ａにやや近い":
        q16 = int(3)
    elif q16 == "4：どちらともいえない":
        q16 = int(4)
    elif q16 == "5：Ｂにやや近い":
        q16 = int(5)
    elif q16 == "6：Ｂにかなり近い":
        q16 = int(6)
    else:
        q16 = int(7)


    image_17 = Image.open("materials/q17.png")
    st.image(image_17, use_column_width=True)
    q17 = st.selectbox("Q17：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q17 == "1：Ａに非常に近い":
        q17 = int(1)
    elif q17 == "2：Ａにかなり近い":
        q17 = int(2)
    elif q17 == "3：Ａにやや近い":
        q17 = int(3)
    elif q17 == "4：どちらともいえない":
        q17 = int(4)
    elif q17 == "5：Ｂにやや近い":
        q17 = int(5)
    elif q17 == "6：Ｂにかなり近い":
        q17 = int(6)
    else:
        q17 = int(7)


    image_18 = Image.open("materials/q18.png")
    st.image(image_18, use_column_width=True)
    q18 = st.selectbox("Q18：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q18 == "1：Ａに非常に近い":
        q18 = int(1)
    elif q18 == "2：Ａにかなり近い":
        q18 = int(2)
    elif q18 == "3：Ａにやや近い":
        q18 = int(3)
    elif q18 == "4：どちらともいえない":
        q18 = int(4)
    elif q18 == "5：Ｂにやや近い":
        q18 = int(5)
    elif q18 == "6：Ｂにかなり近い":
        q18 = int(6)
    else:
        q18 = int(7)

    image_19 = Image.open("materials/q19.png")
    st.image(image_19, use_column_width=True)
    q19 = st.selectbox("Q19：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q19 == "1：Ａに非常に近い":
        q19 = int(1)
    elif q19 == "2：Ａにかなり近い":
        q19 = int(2)
    elif q19 == "3：Ａにやや近い":
        q19 = int(3)
    elif q19 == "4：どちらともいえない":
        q19 = int(4)
    elif q19 == "5：Ｂにやや近い":
        q19 = int(5)
    elif q19 == "6：Ｂにかなり近い":
        q19 = int(6)
    else:
        q19 = int(7)


    image_20 = Image.open("materials/q20.png")
    st.image(image_20, use_column_width=True)
    q20 = st.selectbox("Q20：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q20 == "1：Ａに非常に近い":
        q20 = int(1)
    elif q20 == "2：Ａにかなり近い":
        q20 = int(2)
    elif q20 == "3：Ａにやや近い":
        q20 = int(3)
    elif q20 == "4：どちらともいえない":
        q20 = int(4)
    elif q20 == "5：Ｂにやや近い":
        q20 = int(5)
    elif q20 == "6：Ｂにかなり近い":
        q20 = int(6)
    else:
        q20 = int(7)

 
    image_21 = Image.open("materials/q21.png")
    st.image(image_21, use_column_width=True)
    q21 = st.selectbox("Q21：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q21 == "1：Ａに非常に近い":
        q21 = int(1)
    elif q21 == "2：Ａにかなり近い":
        q21 = int(2)
    elif q21 == "3：Ａにやや近い":
        q21 = int(3)
    elif q21 == "4：どちらともいえない":
        q21 = int(4)
    elif q21 == "5：Ｂにやや近い":
        q21 = int(5)
    elif q21 == "6：Ｂにかなり近い":
        q21 = int(6)
    else:
        q21 = int(7)


    image_22 = Image.open("materials/q22.png")
    st.image(image_22, use_column_width=True)
    q22 = st.selectbox("Q22：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q22 == "1：Ａに非常に近い":
        q22 = int(1)
    elif q22 == "2：Ａにかなり近い":
        q22 = int(2)
    elif q22 == "3：Ａにやや近い":
        q22 = int(3)
    elif q22 == "4：どちらともいえない":
        q22 = int(4)
    elif q22 == "5：Ｂにやや近い":
        q22 = int(5)
    elif q22 == "6：Ｂにかなり近い":
        q22 = int(6)
    else:
        q22 = int(7)


    image_23 = Image.open("materials/q23.png")
    st.image(image_23, use_column_width=True)
    q23 = st.selectbox("Q23：1～7から1つお選びください",["クリックして回答を選択ください","1：Ａに非常に近い","2：Ａにかなり近い","3：Ａにやや近い","4：どちらともいえない","5：Ｂにやや近い","6：Ｂにかなり近い","7：Ｂに非常に近い"])
    if q23 == "1：Ａに非常に近い":
        q23 = int(1)
    elif q23 == "2：Ａにかなり近い":
        q23 = int(2)
    elif q23 == "3：Ａにやや近い":
        q23 = int(3)
    elif q23 == "4：どちらともいえない":
        q23 = int(4)
    elif q23 == "5：Ｂにやや近い":
        q23 = int(5)
    elif q23 == "6：Ｂにかなり近い":
        q23 = int(6)
    else:
        q23 = int(7)

    df_clst = pd.DataFrame({'q_1' : int(q1),
                            'q_2' : int(q2),
                            'q_3' : int(q3),
                            'q_4' : int(q4),
                            'q_5' : int(q5),
                            'q_6' : int(q6),
                            'q_7' : int(q7),
                            'q_8' : int(q8),
                            'q_9' : int(q9),
                            'q_10' : int(q10),
                            'q_11' : int(q11),
                            'q_12' : int(q12),
                            'q_13' : int(q13),
                            'q_14' : int(q14),
                            'q_15' : int(q15),
                            'q_16' : int(q16),
                            'q_17' : int(q17),
                            'q_18' : int(q18),
                            'q_19' : int(q19),
                            'q_20' : int(q20),
                            'q_21' : int(q21),
                            'q_22' : int(q22),
                            'q_23' : int(q23)},index=['answer',])
    dt_now = datetime.datetime.now()
    time = dt_now.strftime('%Y%m%d %H%M')        
    #df_clst.T.to_csv(str(time) + str('-figure')  + str('-result.csv'))

    '''
    ***
    '''
    if st.checkbox('回答結果をダウンロードするにはチェックを入れてください'):
        st.write('ファイル名を入力後、ダウンロードボタンを押してください。ダウンロードしたファイルは「欲求フラグ判定結果」モードでレーダーチャートとして可視化できます。')

        # Enter text for testing
        s = 'pd.DataFrame'

        filename = st.text_input('Enter output filename and ext (e.g. my-question.csv, )', 'my-question.csv')
        #pickle_it = st.checkbox('Save as pickle file')
        sample_dtypes = {'list': [1,'a', [2, 'c'], {'b': 2}],
                         'str': 'Hello Streamlit!',
                         'int': 17,
                         'float': 17.0,
                         'dict': {1: 'a', 'x': [2, 'c'], 2: {'b': 2}},
                         'bool': True,
                         'pd.DataFrame': df_clst}
        sample_dtypes = sample_dtypes

        # Download sample
        download_button_str = download_button(sample_dtypes[s].T, filename, f'Click here to download {filename}', pickle_it=False)
        st.markdown(download_button_str, unsafe_allow_html=True)
