import os
import platform
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np

# 1) 페이지 설정 및 사이드바 크기 조정
st.set_page_config(
    page_title="전북 지역의 고령화와 노인 의료 ",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { width: 100px; min-width: 100px; }
    /* 사이드바 라디오, 셀렉트, 멀티셀렉트 요소 크기 조정 */
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stMultiSelect {
        font-size: 16px;
        line-height: 2;
        padding: 8px 0;
    }
    /* 멀티셀렉트 옵션 높이 조정 */
    [data-testid="stSidebar"] div[role="listbox"] { max-height: 250px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 초고령화사회 속 의료 불균형, 데이터로 드러나다")

# 2) 한글 폰트 설정
def ensure_nanum():
    font_dir = os.path.join(os.getcwd(), "fonts")
    os.makedirs(font_dir, exist_ok=True)
    path = os.path.join(font_dir, "NanumGothic.ttf")
    if not os.path.isfile(path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        resp = requests.get(url); resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    fm.fontManager.addfont(path)
    return fm.FontProperties(fname=path).get_name()

_sys = platform.system()
if _sys == "Windows":
    font_name = "Malgun Gothic"
elif _sys == "Darwin":
    font_name = "AppleGothic"
else:
    font_name = ensure_nanum()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# 3) 데이터 로드 유틸
@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path, encoding='cp949')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='utf-8-sig')

# 연도별 파일 및 병원 파일 지정
years = [2020, 2021, 2022, 2023]
filepaths = {yr: f"{yr}disease.csv" for yr in years}
hosp_path = "hospital_count.csv"

# 4) 2023년도 데이터 로드 및 TOP10 추출
df2023 = load_data(filepaths[2023])
df2023.columns = df2023.columns.str.strip()
numeric_cols = ['진료실인원','내원일수','급여일수','진료비','급여비']
for c in numeric_cols:
    if c in df2023.columns:
        df2023[c] = pd.to_numeric(df2023[c].astype(str).str.replace(',',''), errors='coerce')
top10 = df2023.nlargest(10, '진료실인원')['상병명'].tolist()
disease_options = [d for d in top10 if 'U07' not in d]

# 색상 매핑: Top10 질환 고정 색상
palette = sns.color_palette('tab10', n_colors=len(disease_options))
color_map = {d: palette[i] for i, d in enumerate(disease_options)}

# 5) 사이드바 옵션
if 'direction' not in st.session_state:
    st.session_state.direction = '상위'
with st.sidebar:
    st.markdown("#### 🔍 노인 주요 질환의 연도에 따른 추세 시각화 옵션")
    vis_mode = st.selectbox("", ["라인 차트📈","스캐터 플롯📌","스텝 차트👟","히트맵🧱"], label_visibility="collapsed")
    st.markdown("#### 📝 노인 주요 질환 종류")
    selected_diseases = st.multiselect("", options=disease_options, default=disease_options, label_visibility="collapsed")


# 7) SI 포맷터
def si_format(x, pos):
    if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if x >= 1_000: return f"{x/1_000:.0f}K"
    return f"{int(x)}"

# 8) Top/Bottom Bar Chart
# 👉 라디오 버튼을 타이틀 옆에 배치 (columns 사용)
col_title, col_radio = st.columns([4, 1])
direction = st.radio(
        "",  # 라벨 없음
        ["상위 10위", "하위 10위"],
        horizontal=True,
        label_visibility="collapsed",
        key="direction_radio"
    )

direction_str = "상위" if direction.startswith("상위") else "하위"

# 선택에 따라 disp_df 추출
disp_df = (
    df2023.nlargest(10, '진료실인원')
    if direction_str == '상위'
    else df2023.nsmallest(10, '진료실인원')
)

# 타이틀도 바꿔서 보여주기
st.markdown(
    f"""
    <div style="text-align:center; font-size:20px; font-weight:600;">
      🏥 노인 주요 질환별 환자 수 현황 <span style="color:#4682B4">{direction_str} Top 10</span>
    </div>
    """,
    unsafe_allow_html=True
)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=disp_df, y='상병명', x='진료실인원', palette='Set2', ax=ax1)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(si_format))
maxv = disp_df['진료실인원'].max()
for p in ax1.patches:
    w = p.get_width()
    ax1.text(
        w + maxv * 0.005,
        p.get_y() + p.get_height() / 2,
        si_format(w, None),
        va='center',
        clip_on=False,
    )
ax1.set_xlabel("환자수")
ax1.set_ylabel("질병명")
plt.tight_layout()
st.pyplot(fig1)


#구분선
st.markdown("---")

# 9) 연도별 질환 증가 추세 (유형별)
st.subheader(f"📽노인 주요 질환의 연도에 따른 추세")
trends = {d: [] for d in disease_options}
for yr in years:
    dfy = load_data(filepaths[yr])
    dfy.columns = dfy.columns.str.strip()
    dfy['진료실인원'] = pd.to_numeric(dfy['진료실인원'].astype(str).str.replace(',',''), errors='coerce')
    for d in disease_options:
        vals = dfy.loc[dfy['상병명']==d, '진료실인원']
        trends[d].append(vals.iloc[0] if not vals.empty else np.nan)
allv = [v for arr in trends.values() for v in arr if not np.isnan(v)]
ymin, ymax = min(allv), max(allv)

if vis_mode == "라인 차트📈":
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for d in selected_diseases:
        ax2.plot(years, trends[d], marker='o', label=d, color=color_map[d])
elif vis_mode == "스캐터 플롯📌":
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for d in selected_diseases:
        ax2.scatter(years, trends[d], label=d, color=color_map[d])
elif vis_mode == "스텝 차트👟":
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for d in selected_diseases:
        ax2.step(years, trends[d], where='mid', marker='o', label=d, color=color_map[d])
elif vis_mode == "히트맵🧱":
    dfhm = pd.DataFrame(trends, index=years)
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(dfhm[selected_diseases].T, cmap='Blues', ax=ax2, cbar_kws={'format': ticker.FuncFormatter(si_format)})

if vis_mode != "히트맵🧱":
    ax2.set_xticks(years)
    ax2.set_ylim(ymin*0.95, ymax*1.05)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(si_format))
    ax2.set_xlabel("연도"); ax2.set_ylabel("진료실인원")
    ax2.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout(); st.pyplot(fig2)

st.subheader("⏳전북의 급속한 고령화... 충북보다 심각하다.")
youtube_url = "https://www.youtube.com/embed/EY9p79gmtGw?start=17"  # 17초부터 시작
st.markdown(
    f"""
    <iframe width="780" height="460"
    src="{youtube_url}"
    frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>
    """,
    unsafe_allow_html=True
)


#구분선
st.markdown("---")
#df_old define
df_old = load_data("old_count_new.csv")
df_old.columns = df_old.columns.str.strip()
df_old['년도'] = df_old['년도'].astype(str).str.strip()
df_old['년도'] = pd.to_numeric(df_old['년도'], errors='coerce')

## 년도별로 선택할 수 있음  <-- 이 부분 '복사'
st.write("**연도별 보기 옵션**")
cols = st.columns([0.1, 0.1, 0.1])
show_2015 = cols[0].checkbox("2015", value=True)
show_2021 = cols[1].checkbox("2019", value=True)
show_2023 = cols[2].checkbox("2023", value=True)

years_to_show = []
if show_2015: years_to_show.append(2015)
if show_2021: years_to_show.append(2019)
if show_2023: years_to_show.append(2023)

if years_to_show:
    df_selected = df_old[df_old['년도'].isin(years_to_show)].copy()
    pivot = df_selected.pivot_table(
        index='지역', columns='년도', values='65세 이상합계', aggfunc='sum'
    ).fillna(0)
    pivot = pivot.reindex(columns=years_to_show, fill_value=0)
    # 맨 마지막 선택된 연도를 기준으로 오름차순 정렬
    pivot = pivot.sort_values(by=years_to_show[-1], ascending=False)

    st.subheader(f"📊 전북특별자치도 내 지역별 65세 이상 인구수 변화 추이 ({', '.join(map(str, years_to_show))}년)")
    fig, ax = plt.subplots(figsize=(10,6))
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel("65세 이상 인구수")
    ax.set_xlabel("지역")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("최소 한 개 이상의 연도를 선택해야 그래프가 표시됩니다.")

#구분선
st.markdown("---")

# 14) old_count_new.csv 활용: '노인 1000명당 …' 지표 생성 및 특정 연도만 Bar Chart
# 14-1) 데이터 로드 및 컬럼명 정리
df_old = load_data("old_count_new.csv")
df_old.columns = df_old.columns.str.strip()

# 14-2) '년도' 숫자 변환 후 2019~2023 필터링
df_old['년도'] = df_old['년도'].astype(str).str.strip()
df_old['년도'] = pd.to_numeric(df_old['년도'], errors='coerce')
df_bar = df_old[df_old['년도'].between(2015, 2023)].copy()
df_bar['년도'] = df_bar['년도'].astype(int)

# 14-3) '노인 1000명당 …' 칼럼을 바로 숫자형으로 변환
df_bar['노인 1000명당 병원수'] = pd.to_numeric(
    df_bar['노인 1000명당 병원수'].astype(str).str.replace(',', ''),
    errors='coerce'
)
df_bar['노인 1000명당 치과수'] = pd.to_numeric(
    df_bar['노인 1000명당 치과수'].astype(str).str.replace(',', ''),
    errors='coerce'
)

# 14-4) Bar Chart 생성: 2019, 2021, 2023년만 & 옆으로 나란히 배치
selected_years = [2015, 2019, 2023]

# 2개의 컬럼 생성
col1, col2 = st.columns(2)


# 왼쪽: 치과수
with col1:
    metric = '노인 1000명당 치과수'
    st.subheader(
        f"📈 지역별 · 년도별 {metric} 비교 ({', '.join(map(str, selected_years))}년)"
    )
    show_avg = st.checkbox("평균선 표시", value=True, key=f"avg_{metric}")

    pivot = (
        df_bar.pivot_table(
            index='지역', columns='년도', values=metric, aggfunc='mean'
        )
        .reindex(columns=selected_years, fill_value=0)
        .sort_values(by=2023, ascending=True)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette('tab10', n_colors=len(selected_years))
    # 범례 완전 제거
    pivot.plot(kind='bar', ax=ax, color=colors, legend=False)

    ax.set_ylim(top=6.0)

    if show_avg:
        means = pivot.mean(axis=0)
        for i, yr in enumerate(selected_years):
            ax.axhline(
                means[yr],
                color=colors[i],
                linestyle='--',
                linewidth=1.2
            )

    ax.set_xlabel("")
    ax.set_ylabel(metric)
    # legend 호출 부분 삭제

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    unsafe_allow_html=True
st.info("🔎 노인 치과 질환 수요가 가장 높음에도 불구하고, 병원 대비 치과 인프라는 2.5배 부족한 현상을 그래프를 통해 확인 가능.")

# 오른쪽: 병원수
with col2:
    metric = '노인 1000명당 병원수'
    st.subheader(f"📈 지역별 · 년도별 {metric} 비교 ({', '.join(map(str, selected_years))}년)")
    show_avg = st.checkbox("평균선 표시", value=True, key=f"avg_{metric}")

    pivot = df_bar.pivot_table(
        index='지역', columns='년도', values=metric, aggfunc='mean'
    ).reindex(columns=selected_years, fill_value=0)
    pivot = pivot.sort_values(by=2023, ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette('tab10', n_colors=len(selected_years))
    pivot.plot(kind='bar', ax=ax, color=colors)

    # y축 Major tick 을 0.5 단위로 설정
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    # y축 Max를 6.0 단위로 설정
    ax.set_ylim(top=6.0)

    if show_avg:
        means = pivot.mean(axis=0)
        for i, yr in enumerate(selected_years):
            ax.axhline(means[yr], color=colors[i], linestyle='--', linewidth=1.2, label=f"{yr}년 평균")
    ax.set_xlabel(""); ax.set_ylabel(metric)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), bbox_to_anchor=(1,1))
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    st.pyplot(fig)

#구분선
st.markdown("---")
st.markdown('<h3 style="margin-top:0.3em; margin-bottom:0em;">📑 요점 · 요약 </h2>', unsafe_allow_html=True)
st.markdown('<h6 style="margin-top:0.07em; margin-bottom:0em;">· 초고령화 사회 속, 노인들이 가장 많이 겪는 주요 질환은 치은염 및 치주질환 🦷</h2>', unsafe_allow_html=True)
st.markdown('<h6 style="margin-top:0.07em; margin-bottom:0em;">· 치은염 및 치주질환은 해가 갈 수록 진료 수요가 빠르게 증가하고 있음 🔼</h2>', unsafe_allow_html=True)
st.markdown('<h6 style="margin-top:0.07em; margin-bottom:0em;">· 65세 이상 인구는 전국적으로 증가하고 있으며, 전북은 그 중에서도 평균을 웃도는 빠른 고령화 속도를 보이고 있음 ⏫</h2>', unsafe_allow_html=True)
st.markdown('<h6 style="margin-top:0.07em; margin-bottom:0em;">· 늘어난 치주질환의 수요에 대비 노인 1000명당 병원수에 비해 치과 의료기관 수는 턱없이 부족해 명확한 공급 불균형(비대칭)을 보이고 있음 📌</h2>', unsafe_allow_html=True)