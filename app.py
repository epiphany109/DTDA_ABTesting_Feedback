import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 【核心修正】導入字體管理器
from wordcloud import WordCloud
import jieba
import os # 導入 os 模組來檢查檔案是否存在

# --- 頁面設定 ---
st.set_page_config(
    page_title="DTDA 第8堂社課回饋儀表板",
    page_icon="📊",
    layout="wide"
)

# --- 【核心修正】手動註冊中文字體 ---
# 字體檔案的路徑 (因為字體檔跟 app.py 在同一個目錄，所以直接寫檔名即可)
FONT_PATH = 'NotoSansTC-Regular.ttf'

# 檢查字體檔案是否存在
if os.path.exists(FONT_PATH):
    # 將字體註冊到 Matplotlib 的字體管理器
    fm.fontManager.addfont(FONT_PATH)
    
    # 設置 Matplotlib 的默認字體
    # 'Noto Sans TC' 是這個字體檔案內部定義的名稱
    plt.rc('font', family='Noto Sans TC') 
    plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
else:
    # 如果在 Streamlit Cloud 上找不到字體檔，顯示錯誤訊息
    # 這有助於部署時的偵錯
    st.error(f"字體檔案未找到: {FONT_PATH}。請確保 NotoSansTC-Regular.ttf 已經上傳到 GitHub 儲存庫的根目錄。")


# --- 載入資料 (使用快取避免重複載入) ---
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

# --- 側邊欄篩選器 ---
st.sidebar.title("篩選器")
st.sidebar.markdown("---")
roles = df['role'].unique()
selected_roles = st.sidebar.multiselect(
    "選擇身份：",
    options=roles,
    default=roles
)

if selected_roles:
    filtered_df = df[df['role'].isin(selected_roles)]
else:
    filtered_df = df

# --- 主畫面 ---
st.title("📊 DTDA 下學期第8堂社課回饋儀表板")
st.markdown("我們整理了A/B Testing 社課的綜合回饋，可透過左側篩選器查看不同身份成員的意見。")

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
        st.subheader("課程滿意度(1-5分)")
        satisfaction_cols_map = {
            's_content': '課程內容', 's_lecturer': '講師技巧', 's_structure': '課程結構',
            's_practicality': '課程實用性', 's_knowledge': '新知學習', 's_interaction': '互動參與',
            's_time': '時間合理性', 's_overall': '整體滿意度'
        }
        satisfaction_data = filtered_df[satisfaction_cols_map.keys()].mean().rename(satisfaction_cols_map)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        satisfaction_data.sort_values().plot(kind='barh', ax=ax, color='skyblue')
        ax.set_title('各項滿意度平均分數')
        ax.set_xlabel('平均分數')
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
        ax.set_title('課後主題熟悉度平均分數')
        ax.set_xlabel('平均分數')
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
                wordcloud = WordCloud(
                    font_path=FONT_PATH, # 直接使用我們定義好的字體路徑變數
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
