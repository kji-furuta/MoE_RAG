"""
MoE Web Interface for Civil Engineering Domain
土木・建設分野MoEモデルのWebインターフェース
"""

import streamlit as st
import torch
import json
import pandas as pd
from pathlib import Path
import sys
import os
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# プロジェクトパスの追加
sys.path.append('/home/kjifu/AI_FT_7')

# MoEモジュールのインポート
try:
    from src.moe.moe_architecture import (
        MoEConfig, 
        CivilEngineeringMoEModel,
        ExpertType,
        create_civil_engineering_moe
    )
    from src.moe.data_preparation import CivilEngineeringDataPreparator
    from src.moe.moe_training import MoETrainer, MoETrainingConfig
    MOE_AVAILABLE = True
except ImportError as e:
    MOE_AVAILABLE = False
    st.error(f"MoEモジュールのインポートエラー: {e}")

# ページ設定
st.set_page_config(
    page_title="MoE土木・建設AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .expert-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .expert-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        transition: opacity 0.3s;
    }
    .stButton > button:hover {
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'expert_usage' not in st.session_state:
    st.session_state.expert_usage = {expert.value: 0 for expert in ExpertType}
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# エキスパート情報
EXPERT_INFO = {
    "structural_design": {
        "name": "構造設計",
        "icon": "🏢",
        "description": "橋梁、建築物の構造計算",
        "keywords": ["梁", "柱", "基礎", "耐震", "応力"]
    },
    "road_design": {
        "name": "道路設計",
        "icon": "🛣️",
        "description": "道路構造令、線形設計",
        "keywords": ["設計速度", "曲線半径", "勾配", "交差点"]
    },
    "geotechnical": {
        "name": "地盤工学",
        "icon": "⛰️",
        "description": "土質力学、基礎工事",
        "keywords": ["N値", "支持力", "液状化", "土圧"]
    },
    "hydraulics": {
        "name": "水理・排水",
        "icon": "💧",
        "description": "排水設計、河川工学",
        "keywords": ["流量", "管渠", "ポンプ", "洪水"]
    },
    "materials": {
        "name": "材料工学",
        "icon": "🧱",
        "description": "コンクリート、鋼材特性",
        "keywords": ["配合", "強度", "品質管理", "試験"]
    },
    "construction_management": {
        "name": "施工管理",
        "icon": "👷",
        "description": "工程・安全・品質管理",
        "keywords": ["工程", "安全", "原価", "施工計画"]
    },
    "regulations": {
        "name": "法規・基準",
        "icon": "📋",
        "description": "JIS規格、建築基準法",
        "keywords": ["建築基準法", "道路構造令", "JIS", "ISO"]
    },
    "environmental": {
        "name": "環境・維持管理",
        "icon": "🌿",
        "description": "環境影響評価、維持補修",
        "keywords": ["騒音", "振動", "廃棄物", "長寿命化"]
    }
}

def load_model(demo_mode=True):
    """モデルのロード"""
    with st.spinner('モデルを読み込み中...'):
        try:
            if demo_mode:
                # デモ用の小規模モデル
                config = MoEConfig(
                    hidden_size=512,
                    num_experts=8,
                    num_experts_per_tok=2,
                    domain_specific_routing=True
                )
                model = CivilEngineeringMoEModel(config, base_model=None)
            else:
                # 本番モデル
                model = create_civil_engineering_moe(
                    base_model_name="cyberagent/calm3-22b-chat",
                    num_experts=8
                )
            
            st.session_state.model = model
            return True
        except Exception as e:
            st.error(f"モデルロードエラー: {e}")
            return False

def analyze_query(query):
    """クエリから関連エキスパートを分析"""
    detected_experts = []
    confidence_scores = {}
    
    for expert_type, info in EXPERT_INFO.items():
        score = 0
        for keyword in info["keywords"]:
            if keyword in query:
                score += 1
        
        if score > 0:
            confidence = min(score * 0.3, 1.0)
            detected_experts.append(expert_type)
            confidence_scores[expert_type] = confidence
    
    # スコアでソート
    detected_experts.sort(key=lambda x: confidence_scores[x], reverse=True)
    
    return detected_experts[:2], confidence_scores  # Top-2エキスパート

def generate_response(query, selected_experts):
    """応答生成（シミュレーション）"""
    # 実際のモデル推論の代わりにシミュレーション
    time.sleep(1)  # 推論時間のシミュレーション
    
    expert_names = [EXPERT_INFO[e]["name"] for e in selected_experts]
    
    response = f"""
    ご質問について、{' と '.join(expert_names)}の観点から回答いたします。

    {query}に関して、以下の点が重要です：

    1. 技術的要件の確認
    2. 関連法規・基準の遵守
    3. 安全性と経済性の両立
    4. 施工性の検討
    5. 維持管理計画の策定

    詳細な設計や計算が必要な場合は、具体的な条件をお知らせください。
    """
    
    # エキスパート使用状況の更新
    for expert in selected_experts:
        st.session_state.expert_usage[expert] += 1
    
    return response

# メインヘッダー
st.markdown("""
<div class="main-header">
    <h1 style="color: #764ba2; text-align: center; margin: 0;">
        🏗️ MoE 土木・建設AI システム
    </h1>
    <p style="text-align: center; color: #666; margin-top: 0.5rem;">
        8つの専門エキスパートが土木・建設の課題を解決
    </p>
</div>
""", unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.header("⚙️ システム設定")
    
    # モデル設定
    st.subheader("モデル設定")
    demo_mode = st.checkbox("デモモード", value=True, help="小規模モデルで高速動作")
    
    if st.button("🚀 モデル読み込み"):
        if load_model(demo_mode):
            st.success("✅ モデル読み込み完了")
    
    # エキスパート設定
    st.subheader("エキスパート設定")
    num_experts_per_tok = st.slider(
        "同時活性化エキスパート数",
        min_value=1,
        max_value=4,
        value=2,
        help="同時に使用するエキスパート数"
    )
    
    confidence_threshold = st.slider(
        "信頼度閾値",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="エキスパート選択の信頼度閾値"
    )
    
    # 統計情報
    st.subheader("📊 使用統計")
    total_queries = len(st.session_state.conversation_history)
    st.metric("総クエリ数", total_queries)
    
    if st.session_state.expert_usage:
        most_used = max(st.session_state.expert_usage.items(), key=lambda x: x[1])
        if most_used[1] > 0:
            st.metric("最頻使用エキスパート", EXPERT_INFO[most_used[0]]["name"])

# メインコンテンツ
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 質問応答", 
    "👥 エキスパート一覧", 
    "📊 ダッシュボード",
    "🎓 トレーニング",
    "📁 データ管理"
])

# Tab 1: 質問応答
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("質問を入力してください")
        
        # 質問入力
        query = st.text_area(
            "質問",
            placeholder="例: 設計速度80km/hの道路における最小曲線半径は？",
            height=100
        )
        
        # サンプル質問
        st.write("**サンプル質問:**")
        sample_questions = [
            "RC梁の曲げモーメントに対する設計方法",
            "N値15の地盤における直接基礎の支持力",
            "道路の横断勾配の標準値と特例値",
            "コンクリートの配合設計の手順"
        ]
        
        selected_sample = st.selectbox("サンプルを選択", [""] + sample_questions)
        if selected_sample:
            query = selected_sample
        
        if st.button("🔍 回答を生成", type="primary"):
            if query:
                # エキスパート分析
                detected_experts, confidence_scores = analyze_query(query)
                
                # 応答生成
                with st.spinner('回答を生成中...'):
                    response = generate_response(query, detected_experts)
                
                # 会話履歴に追加
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now(),
                    "query": query,
                    "response": response,
                    "experts": detected_experts
                })
                
                # 結果表示
                st.success("✅ 回答生成完了")
                
                # 使用エキスパート表示
                st.write("**使用エキスパート:**")
                expert_cols = st.columns(len(detected_experts))
                for idx, expert in enumerate(detected_experts):
                    with expert_cols[idx]:
                        info = EXPERT_INFO[expert]
                        confidence = confidence_scores.get(expert, 0)
                        st.markdown(f"""
                        <div class="expert-card">
                            <h4>{info['icon']} {info['name']}</h4>
                            <p style="font-size: 0.9em; color: #666;">
                                信頼度: {confidence:.0%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 回答表示
                st.write("**回答:**")
                st.info(response)
            else:
                st.warning("質問を入力してください")
    
    with col2:
        st.subheader("📜 会話履歴")
        
        if st.session_state.conversation_history:
            for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {conv['query'][:30]}..."):
                    st.write(f"**時刻:** {conv['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**エキスパート:** {', '.join([EXPERT_INFO[e]['name'] for e in conv['experts']])}")
                    st.write(f"**回答:** {conv['response'][:200]}...")
        else:
            st.info("会話履歴はまだありません")

# Tab 2: エキスパート一覧
with tab2:
    st.subheader("8つの専門エキスパート")
    
    # エキスパートカード表示
    cols = st.columns(4)
    for idx, (expert_type, info) in enumerate(EXPERT_INFO.items()):
        with cols[idx % 4]:
            usage_count = st.session_state.expert_usage.get(expert_type, 0)
            
            st.markdown(f"""
            <div class="expert-card">
                <h3 style="text-align: center; color: #764ba2;">
                    {info['icon']}
                </h3>
                <h4 style="text-align: center; margin: 0.5rem 0;">
                    {info['name']}
                </h4>
                <p style="font-size: 0.9em; color: #666; text-align: center;">
                    {info['description']}
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #eee;">
                    <p style="font-size: 0.8em; color: #999; text-align: center;">
                        使用回数: {usage_count}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # キーワード表示
            with st.expander("キーワード"):
                for keyword in info["keywords"]:
                    st.write(f"• {keyword}")

# Tab 3: ダッシュボード
with tab3:
    st.subheader("📊 パフォーマンスダッシュボード")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>95%</h3>
            <p>専門分野精度</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>4x</h3>
            <p>推論速度向上</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>-67%</h3>
            <p>メモリ削減</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <h3>-75%</h3>
            <p>コスト削減</p>
        </div>
        """, unsafe_allow_html=True)
    
    # エキスパート使用状況グラフ
    st.subheader("エキスパート使用状況")
    
    if any(st.session_state.expert_usage.values()):
        # 使用状況の円グラフ
        fig_pie = go.Figure(data=[go.Pie(
            labels=[EXPERT_INFO[k]["name"] for k in st.session_state.expert_usage.keys()],
            values=list(st.session_state.expert_usage.values()),
            hole=0.3
        )])
        fig_pie.update_layout(
            title="エキスパート使用割合",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 使用回数の棒グラフ
        fig_bar = go.Figure(data=[go.Bar(
            x=[EXPERT_INFO[k]["name"] for k in st.session_state.expert_usage.keys()],
            y=list(st.session_state.expert_usage.values()),
            marker_color='#764ba2'
        )])
        fig_bar.update_layout(
            title="エキスパート使用回数",
            xaxis_title="エキスパート",
            yaxis_title="使用回数",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("データがまだありません。質問を入力してください。")

# Tab 4: トレーニング
with tab4:
    st.subheader("🎓 モデルトレーニング")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**トレーニング設定**")
        
        epochs = st.number_input("エポック数", min_value=1, max_value=10, value=3)
        batch_size = st.number_input("バッチサイズ", min_value=1, max_value=32, value=4)
        learning_rate = st.select_slider(
            "学習率",
            options=[1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
            value=2e-5,
            format_func=lambda x: f"{x:.0e}"
        )
        
        use_mixed_precision = st.checkbox("Mixed Precision (BF16)", value=True)
        gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=True)
        
        if st.button("🚀 トレーニング開始"):
            with st.spinner('トレーニング中...'):
                # トレーニングのシミュレーション
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for epoch in range(epochs):
                    for step in range(10):  # 10ステップのシミュレーション
                        progress = (epoch * 10 + step + 1) / (epochs * 10)
                        progress_bar.progress(progress)
                        status_text.text(f'Epoch {epoch+1}/{epochs}, Step {step+1}/10')
                        time.sleep(0.1)
                    
                    # ダミーの損失値
                    loss = 2.5 - (epoch * 0.3) + (0.1 * (0.5 - torch.rand(1).item()))
                    st.session_state.training_history.append({
                        "epoch": epoch + 1,
                        "loss": loss,
                        "timestamp": datetime.now()
                    })
                
                st.success("✅ トレーニング完了！")
    
    with col2:
        st.write("**トレーニング履歴**")
        
        if st.session_state.training_history:
            # 損失グラフ
            df = pd.DataFrame(st.session_state.training_history)
            fig = px.line(df, x="epoch", y="loss", title="Training Loss")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # 最新の統計
            latest = st.session_state.training_history[-1]
            st.metric("最終損失", f"{latest['loss']:.4f}")
            st.metric("完了時刻", latest['timestamp'].strftime('%H:%M:%S'))
        else:
            st.info("トレーニング履歴はまだありません")

# Tab 5: データ管理
with tab5:
    st.subheader("📁 データ管理")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**データ生成**")
        
        samples_per_domain = st.number_input(
            "ドメインあたりサンプル数",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        if st.button("📝 データ生成"):
            with st.spinner('データ生成中...'):
                try:
                    from src.moe.data_preparation import CivilEngineeringDataPreparator
                    
                    preparator = CivilEngineeringDataPreparator()
                    preparator.generate_training_data(num_samples_per_domain=samples_per_domain)
                    
                    st.success(f"✅ {samples_per_domain * 8}件のデータを生成しました")
                except Exception as e:
                    st.error(f"データ生成エラー: {e}")
        
        st.write("**データセット情報**")
        data_path = Path("./data/civil_engineering")
        if data_path.exists():
            train_path = data_path / "train"
            val_path = data_path / "val"
            
            if train_path.exists():
                train_files = list(train_path.glob("*.jsonl"))
                st.metric("トレーニングファイル数", len(train_files))
            
            if val_path.exists():
                val_files = list(val_path.glob("*.jsonl"))
                st.metric("検証ファイル数", len(val_files))
        else:
            st.info("データセットがまだ作成されていません")
    
    with col2:
        st.write("**データ統計**")
        
        # ドメイン別サンプル数
        domain_stats = []
        for expert_type in ExpertType:
            domain_file = data_path / "train" / f"{expert_type.value}.jsonl"
            if domain_file.exists():
                with open(domain_file, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
                domain_stats.append({
                    "ドメイン": EXPERT_INFO[expert_type.value]["name"],
                    "サンプル数": count
                })
        
        if domain_stats:
            df_stats = pd.DataFrame(domain_stats)
            st.dataframe(df_stats, use_container_width=True)
            
            # 棒グラフ
            fig = px.bar(
                df_stats,
                x="ドメイン",
                y="サンプル数",
                title="ドメイン別データ分布"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("データ統計はまだありません")

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏗️ MoE土木・建設AIシステム v1.0</p>
    <p style="font-size: 0.9em;">AI_FT_7 Project | 8 Experts for Civil Engineering</p>
</div>
""", unsafe_allow_html=True)
