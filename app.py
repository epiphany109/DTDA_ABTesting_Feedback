import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba

# --- 頁面設定 ---
st.set_page_config(
    page_title="DTDA 第8堂社課回饋儀表板",
    page_icon="📊",
    layout="wide"
)

# --- 中文字體設定 for Matplotlib ---
plt.rcParams['font.sans-serif'] = ['Noto Sans TC Regular']
plt.rcParams['axes.unicode_minus'] = False 

# --- 載入資料 (使用快取避免重複載入) ---
@st.cache_data
def load_data():
    df = pd.read_csv("feedback_data.csv")
    # 簡化欄位名稱
    df.columns = [col.strip().replace('\n', '') for col in df.columns]
    short_cols = {
        '專案生/研習生/幹部': 'role',
        '對於這次社課的滿意程度！（1表示非常不滿意，5為非常滿意） [課程內容滿意度]': 's_content',
        '對於這次社課的滿意程度！（1表示非常不滿意，5為非常滿意） [講師的授課技巧]': 's_lecturer',
        '對於這次社課的滿意程度！（1表示非常不滿意，5為非常滿意） [課程結構與安排]': 's_structure',
        '對於這次社課的滿意程度！（1表示非常不滿意，5為非常滿意） [課程實用性]': 's_practicality',
        '對於這次社課的滿意程度！（1表示非常不滿意，5為非常滿意） [學習到的新知識或技能]': 's_knowledge',
        '對於這次社課的滿意程度！（1表示非常不滿意，5為非常滿意） [互動性與參與度]': 's_interaction',
        '對於這次社課的滿意程度！（1表示非常不滿意，5為非常滿意） [課程時間的合理性]': 's_time',
        '對於這次社課的滿意程度！（1表示非常不滿意，5為非常滿意） [整體滿意度]': 's_overall',
        '課後主題熟悉度 [假設檢定基礎概念]': 'f_hypothesis',
        '課後主題熟悉度 [P 值基礎概念]': 'f_p_value',
        '課後主題熟悉度 [型一/型二錯誤]': 'f_error_type',
        '課後主題熟悉度 [A/B Testing 標準流程]': 'f_ab_flow',
        '課後主題熟悉度 [A/B Testing Python Code]': 'f_ab_code',
        '根據課程架構，這次講座讓你有收穫的內容是？（可複選）': 'useful_content',
        '這次社課最吸引你的部分是什麼？': 'attractive_part',
        '如果有機會再次參加類似課程，你會期待哪些不同之處？或是有無需要調整的地方？': 'suggestions',
        '有沒有想要跟講師回饋或分享的呢？': 'feedback_to_lecturer',
        '最後還有沒有想要補充什麼呢～': 'additional_comments'
    }
    df.rename(columns=short_cols, inplace=True)
    return df

df = load_data()

# --- 側邊欄篩選器 ---
st.sidebar.title("篩選器")
st.sidebar.markdown("---")
# 身份篩選
roles = df['role'].unique()
selected_roles = st.sidebar.multiselect(
    "選擇身份查看",
    options=roles,
    default=roles
)

# 根據篩選結果過濾 DataFrame
if selected_roles:
    filtered_df = df[df['role'].isin(selected_roles)]
else:
    filtered_df = df

# --- 主畫面 ---
st.title("📊 DTDA 下學期第8堂社課回饋儀表板")
st.markdown("這份儀表板整理了【A/B Testing】社課的學員回饋，可透過左側篩選器查看不同身份成員的意見。")

# --- 關鍵指標 (KPIs) ---
total_responses = len(filtered_df)
avg_satisfaction = filtered_df['s_overall'].mean()

col1, col2 = st.columns(2)
col1.metric("總回饋數", f"{total_responses} 份")
col2.metric("平均整體滿意度", f"{avg_satisfaction:.2f} / 5.0")

st.markdown("---")

# --- 儀表板內容 (使用分頁) ---
tab1, tab2, tab3 = st.tabs(["📈 滿意度與熟悉度分析", "☁️ 文字回饋與詞雲", "📄 完整回饋留言"])

with tab1:
    st.header("各項滿意度與主題熟悉度分佈")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("課程滿意度 (1-5分)")
        satisfaction_cols = {
            's_content': '課程內容', 's_lecturer': '講師技巧', 's_structure': '課程結構',
            's_practicality': '課程實用性', 's_knowledge': '新知學習', 's_interaction': '互動參與',
            's_time': '時間合理性', 's_overall': '整體滿意度'
        }
        satisfaction_data = filtered_df[satisfaction_cols.keys()].mean().rename(satisfaction_cols)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        satisfaction_data.sort_values().plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title('各項滿意度平均分數')
        ax.set_xlabel('平均分數')
        ax.set_xlim(0, 5)
        for index, value in enumerate(satisfaction_data.sort_values()):
            ax.text(value + 0.05, index, f'{value:.2f}')
        st.pyplot(fig)

    with col2:
        st.subheader("課後主題熟悉度 (1-5分)")
        familiarity_cols = {
            'f_hypothesis': '假設檢定', 'f_p_value': 'P值概念', 'f_error_type': '型一/二錯誤',
            'f_ab_flow': 'A/B Test流程', 'f_ab_code': 'A/B Test程式碼'
        }
        familiarity_data = filtered_df[familiarity_cols.keys()].mean().rename(familiarity_cols)

        fig, ax = plt.subplots(figsize=(10, 8))
        familiarity_data.sort_values().plot(kind='barh', ax=ax, color='lightgreen')
        ax.set_title('課後主題熟悉度平均分數')
        ax.set_xlabel('平均分數')
        ax.set_xlim(0, 5)
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
        
        try:
            wordcloud = WordCloud(
                font_path='NotoSansTC-Regular.ttf', 
                width=800, height=400, background_color='white', collocations=False
            ).generate(" ".join(words))
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"產生詞雲時發生錯誤: {e}")
    else:
        st.info("在目前的篩選條件下，沒有足夠的文字回饋來產生詞雲。")

with tab3:
    st.header("完整回饋留言")
    st.markdown("以下是篩選後，每份回饋的完整文字內容。")

    # 使用 reset_index() 來產生一個匿名的序號
    for index, row in filtered_df.reset_index(drop=True).iterrows():
        with st.expander(f"💬 回饋 #{index + 1} ({row['role']})"):
            st.markdown(f"**【最吸引我的部分】**\n> {row['attractive_part']}")
            st.markdown(f"**【期待與建議】**\n> {row['suggestions']}")
            st.markdown(f"**【給講師的回饋】**\n> {row['feedback_to_lecturer']}")
            if pd.notna(row['additional_comments']) and row['additional_comments'].strip():
                st.markdown(f"**【補充事項】**\n> {row['additional_comments']}")
