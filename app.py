import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import jieba
import os

# --- 頁面設定 ---
st.set_page_config(
    page_title="DTDA 第8堂社課回饋儀表板",
    page_icon="📊",
    layout="wide"
)



# 載入字體屬性，這將在所有 matplotlib 圖表中使用
FONT_PATH = 'NotoSansTC-Regular.ttf'
font_prop = None
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
else:
    st.error(f"字體檔案未找到: {FONT_PATH}。請確保 NotoSansTC-Regular.ttf 已經上傳到 GitHub。")

# --- 載入資料 ---
@st.cache_data
def load_data():
    df = pd.read_csv("feedback_data.csv")
    new_columns = [
        'role', 's_content', 's_lecturer', 's_structure', 's_practicality',
        's_knowledge', 's_interaction', 's_time', 's_overall', 'f_hypothesis',
        'f_p_value', 'f_error_type', 'f_ab_flow', 'f_ab_code', 'useful_content',
        'attractive_part', 'suggestions', 'feedback_to_lecturer', 'additional_comments'
    ]
    if len(df.columns) == len(new_columns):
        df.columns = new_columns
    else:
        raise ValueError(f"CSV欄位數量({len(df.columns)})與預期({len(new_columns)})不符！")
    return df

df = load_data()

# --- 主畫面 ---
st.title("📊 DTDA 下學期第8堂社課回饋儀表板")
if not font_prop:
    st.warning("警告：中文字體載入失敗，圖表可能無法正常顯示中文。")

# --- 側邊欄篩選器 ---
st.sidebar.title("篩選器")
st.sidebar.markdown("---")
roles = df['role'].unique()
selected_roles = st.sidebar.multiselect("選擇身份：", options=roles, default=roles)
filtered_df = df[df['role'].isin(selected_roles)] if selected_roles else df

# --- 關鍵指標 (KPIs) ---
total_responses = len(filtered_df)
avg_satisfaction = filtered_df['s_overall'].mean()
col1, col2 = st.columns(2)
col1.metric("總回饋數", f"{total_responses} 份")
col2.metric("平均整體滿意度", f"{avg_satisfaction:.2f} / 5.0")
st.markdown("---")

# --- 儀表板內容 (使用分頁) ---
tab1, tab2, tab3 = st.tabs(["📈 滿意度與熟悉度分析", "☁️ Word Cloud", "📄 完整回饋留言"])

# 只有在字體成功載入時才繪製圖表
if font_prop:
    with tab1:
        st.header("各項滿意度與主題熟悉度分佈")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("課程滿意度(1-5分)")
            satisfaction_cols_map = {
                's_content': '課程內容', 's_lecturer': '講師技巧', 's_structure': '課程結構',
                's_practicality': '課程實用性', 's_knowledge': '新知學習', 's_interaction': '互動參與',
                's_time': '時間合理性', 's_overall': '整體滿意度'
            }
            satisfaction_data = filtered_df[satisfaction_cols_map.keys()].mean().rename(satisfaction_cols_map)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            satisfaction_data.sort_values().plot(kind='barh', ax=ax, color='skyblue')
            
            # 【明確指定字體】
            ax.set_title('各項滿意度平均分數', fontproperties=font_prop, fontsize=16)
            ax.set_xlabel('平均分數', fontproperties=font_prop, fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
            ax.set_ylabel('') # 清除 Y 軸標籤
            
            # 對Y軸刻度標籤單獨設置字體
            ax.set_yticklabels(satisfaction_data.sort_values().index, fontproperties=font_prop, fontsize=12)
            
            ax.set_xlim(0, 5.5)
            for index, value in enumerate(satisfaction_data.sort_values()):
                ax.text(value + 0.05, index, f'{value:.2f}')
            st.pyplot(fig)

        with col2:
            st.subheader("課後主題熟悉度(1-5分)")
            familiarity_cols_map = {
                'f_hypothesis': '假設檢定', 'f_p_value': 'P值概念', 'f_error_type': '型一/二錯誤',
                'f_ab_flow': 'A/B Test流程', 'f_ab_code': 'A/B Test程式碼'
            }
            familiarity_data = filtered_df[familiarity_cols_map.keys()].mean().rename(familiarity_cols_map)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            familiarity_data.sort_values().plot(kind='barh', ax=ax, color='lightgreen')

            # 【明確指定字體】
            ax.set_title('課後主題熟悉度平均分數', fontproperties=font_prop, fontsize=16)
            ax.set_xlabel('平均分數', fontproperties=font_prop, fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
            ax.set_ylabel('') # 清除 Y 軸標籤

            # 對Y軸刻度標籤單獨設置字體
            ax.set_yticklabels(familiarity_data.sort_values().index, fontproperties=font_prop, fontsize=12)

            ax.set_xlim(0, 5.5)
            for index, value in enumerate(familiarity_data.sort_values()):
                ax.text(value + 0.05, index, f'{value:.2f}')
            st.pyplot(fig)

    with tab2:
        st.header("Word Cloud")
        text_cols = ['attractive_part', 'suggestions', 'feedback_to_lecturer', 'additional_comments']
        text_data = filtered_df[text_cols].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)
        full_text = ' '.join(text_data)
        if full_text.strip():
            seg_list = jieba.cut(full_text)
            stopwords = set(['的', '是', '在', '我', '有', '也', '了', '都', '個', '很', '可以', '老師', '講師',
                             ' ', '無', '沒有', '希望', '覺得', '課程', '部分', '什麼', '地方', '一些', '這個',
                             'the', 'to', 'and', 'a', 'test', 'ab', 'testing'])
            words = [word for word in seg_list if word not in stopwords and len(word) > 1]
            if words:
                try:
                    # 【明確指定字體路徑】
                    wordcloud = WordCloud(
                        font_path=FONT_PATH, 
                        width=800, height=400, background_color='white', collocations=False
                    ).generate(" ".join(words))
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"產生文字雲時發生錯誤: {e}")
            else:
                st.info("在目前的篩選條件下，沒有足夠的文字回饋來產生文字雲。")
        else:
            st.info("在目前的篩選條件下，沒有足夠的文字回饋來產生文字雲。")

with tab3:
    st.header("完整回饋留言")
    st.markdown("以下是篩選後，每份回饋的完整文字內容。")
    for index, row in filtered_df.reset_index(drop=True).iterrows():
        with st.expander(f"💬 回饋 #{index + 1} ({row['role']})"):
            st.markdown(f"**【最吸引我的部分】**\n> {row['attractive_part']}")
            st.markdown(f"**【期待與建議】**\n> {row['suggestions']}")
            st.markdown(f"**【給講師的回饋】**\n> {row['feedback_to_lecturer']}")
            if pd.notna(row['additional_comments']) and row['additional_comments'].strip():
                st.markdown(f"**【補充事項】**\n> {row['additional_comments']}")
