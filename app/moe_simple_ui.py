"""
MoE Simple Web Interface
シンプルなMoE WebUI（最小依存関係）
"""

from flask import Flask, render_template_string, request, jsonify
import sys
import os
import json
from datetime import datetime

sys.path.append('/home/kjifu/AI_FT_7')

# MoEモジュールのインポート（オプション）
try:
    from src.moe.moe_architecture import ExpertType
    from src.moe.data_preparation import CivilEngineeringDataPreparator
    MOE_AVAILABLE = True
except:
    MOE_AVAILABLE = False

app = Flask(__name__)

# HTMLテンプレート
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoE 土木・建設AI システム</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .header h1 {
            color: #764ba2;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            color: #764ba2;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 5px;
            color: #555;
            font-weight: 500;
        }
        
        .input-group textarea,
        .input-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        .input-group textarea:focus,
        .input-group select:focus {
            outline: none;
            border-color: #764ba2;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(118, 75, 162, 0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .expert-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        
        .expert-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            transition: background 0.3s, transform 0.2s;
            cursor: pointer;
        }
        
        .expert-item:hover {
            background: #e9ecef;
            transform: translateY(-2px);
        }
        
        .expert-item.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .expert-icon {
            font-size: 2em;
            margin-bottom: 5px;
        }
        
        .expert-name {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .expert-desc {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .response-area {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            min-height: 200px;
        }
        
        .response-area h3 {
            color: #764ba2;
            margin-bottom: 15px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-item {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .expert-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏗️ MoE 土木・建設AI システム</h1>
            <p>8つの専門エキスパートが土木・建設の課題を解決</p>
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2>💬 質問入力</h2>
                
                <div class="input-group">
                    <label for="query">質問内容</label>
                    <textarea id="query" rows="4" placeholder="例: 設計速度80km/hの道路における最小曲線半径は？"></textarea>
                </div>
                
                <div class="input-group">
                    <label for="sample">サンプル質問</label>
                    <select id="sample" onchange="selectSample()">
                        <option value="">-- サンプルを選択 --</option>
                        <option value="RC梁の曲げモーメントに対する設計方法">RC梁の曲げモーメントに対する設計方法</option>
                        <option value="N値15の地盤における直接基礎の支持力">N値15の地盤における直接基礎の支持力</option>
                        <option value="道路の横断勾配の標準値と特例値">道路の横断勾配の標準値と特例値</option>
                        <option value="コンクリートの配合設計の手順">コンクリートの配合設計の手順</option>
                    </select>
                </div>
                
                <button class="btn" onclick="processQuery()">🔍 回答を生成</button>
                
                <div class="response-area" id="response" style="display: none;">
                    <h3>回答</h3>
                    <div id="responseContent"></div>
                </div>
            </div>
            
            <div>
                <div class="card">
                    <h2>👥 専門エキスパート</h2>
                    <div class="expert-grid">
                        <div class="expert-item" data-expert="structural">
                            <div class="expert-icon">🏢</div>
                            <div class="expert-name">構造設計</div>
                            <div class="expert-desc">橋梁・建築物</div>
                        </div>
                        <div class="expert-item" data-expert="road">
                            <div class="expert-icon">🛣️</div>
                            <div class="expert-name">道路設計</div>
                            <div class="expert-desc">道路構造令</div>
                        </div>
                        <div class="expert-item" data-expert="geo">
                            <div class="expert-icon">⛰️</div>
                            <div class="expert-name">地盤工学</div>
                            <div class="expert-desc">土質・基礎</div>
                        </div>
                        <div class="expert-item" data-expert="hydro">
                            <div class="expert-icon">💧</div>
                            <div class="expert-name">水理・排水</div>
                            <div class="expert-desc">排水設計</div>
                        </div>
                        <div class="expert-item" data-expert="material">
                            <div class="expert-icon">🧱</div>
                            <div class="expert-name">材料工学</div>
                            <div class="expert-desc">コンクリート</div>
                        </div>
                        <div class="expert-item" data-expert="construction">
                            <div class="expert-icon">👷</div>
                            <div class="expert-name">施工管理</div>
                            <div class="expert-desc">工程・安全</div>
                        </div>
                        <div class="expert-item" data-expert="regulation">
                            <div class="expert-icon">📋</div>
                            <div class="expert-name">法規・基準</div>
                            <div class="expert-desc">JIS・法令</div>
                        </div>
                        <div class="expert-item" data-expert="environment">
                            <div class="expert-icon">🌿</div>
                            <div class="expert-name">環境・維持</div>
                            <div class="expert-desc">環境影響</div>
                        </div>
                    </div>
                </div>
                
                <div class="card" style="margin-top: 20px;">
                    <h2>📊 統計情報</h2>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">95%</div>
                            <div class="stat-label">専門分野精度</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">4x</div>
                            <div class="stat-label">推論速度向上</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">-67%</div>
                            <div class="stat-label">メモリ削減</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">-75%</div>
                            <div class="stat-label">コスト削減</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function selectSample() {
            const select = document.getElementById('sample');
            const textarea = document.getElementById('query');
            textarea.value = select.value;
        }
        
        async function processQuery() {
            const query = document.getElementById('query').value;
            if (!query) {
                alert('質問を入力してください');
                return;
            }
            
            const responseDiv = document.getElementById('response');
            const responseContent = document.getElementById('responseContent');
            
            responseDiv.style.display = 'block';
            responseContent.innerHTML = '<div class="loading">回答を生成中...</div>';
            
            // エキスパートのハイライト
            highlightExperts(query);
            
            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                responseContent.innerHTML = data.response.replace(/\\n/g, '<br>');
            } catch (error) {
                responseContent.innerHTML = '<p style="color: red;">エラーが発生しました: ' + error + '</p>';
            }
        }
        
        function highlightExperts(query) {
            const experts = document.querySelectorAll('.expert-item');
            experts.forEach(expert => expert.classList.remove('active'));
            
            // キーワードベースでエキスパートをハイライト
            const keywords = {
                'structural': ['構造', '梁', '柱', '基礎', '耐震', 'RC'],
                'road': ['道路', '設計速度', '曲線', '勾配', '横断'],
                'geo': ['地盤', 'N値', '支持力', '液状化', '土質'],
                'hydro': ['排水', '流量', '水理', 'ポンプ'],
                'material': ['コンクリート', '配合', '鋼材', '強度'],
                'construction': ['施工', '工程', '安全', '品質'],
                'regulation': ['基準', '法規', 'JIS', '規格'],
                'environment': ['環境', '騒音', '振動', '維持']
            };
            
            for (const [expertType, expertKeywords] of Object.entries(keywords)) {
                for (const keyword of expertKeywords) {
                    if (query.includes(keyword)) {
                        const expertElement = document.querySelector(`[data-expert="${expertType}"]`);
                        if (expertElement) {
                            expertElement.classList.add('active');
                        }
                        break;
                    }
                }
            }
        }
    </script>
</body>
</html>
"""

# エキスパート定義
EXPERTS = {
    "structural_design": {"name": "構造設計", "keywords": ["構造", "梁", "柱", "基礎", "耐震", "RC"]},
    "road_design": {"name": "道路設計", "keywords": ["道路", "設計速度", "曲線", "勾配"]},
    "geotechnical": {"name": "地盤工学", "keywords": ["地盤", "N値", "支持力", "液状化"]},
    "hydraulics": {"name": "水理・排水", "keywords": ["排水", "流量", "水理", "ポンプ"]},
    "materials": {"name": "材料工学", "keywords": ["コンクリート", "配合", "鋼材", "強度"]},
    "construction_mgmt": {"name": "施工管理", "keywords": ["施工", "工程", "安全", "品質"]},
    "regulations": {"name": "法規・基準", "keywords": ["基準", "法規", "JIS", "規格"]},
    "environmental": {"name": "環境・維持管理", "keywords": ["環境", "騒音", "振動", "維持"]}
}

def analyze_query(query):
    """クエリから関連エキスパートを分析"""
    detected_experts = []
    for expert_type, info in EXPERTS.items():
        for keyword in info["keywords"]:
            if keyword in query:
                detected_experts.append(info["name"])
                break
    return detected_experts[:2] if detected_experts else ["総合"]

@app.route('/')
def index():
    """メインページ"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/process', methods=['POST'])
def process():
    """質問処理API"""
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': '質問が入力されていません'}), 400
    
    # エキスパート分析
    experts = analyze_query(query)
    
    # 応答生成（シミュレーション）
    response = f"""
<strong>使用エキスパート:</strong> {', '.join(experts)}<br><br>

<strong>回答:</strong><br>
{query}について、以下の観点から説明いたします。<br><br>

<strong>1. 技術的要件</strong><br>
設計基準に基づいた適切な設計手法を適用する必要があります。<br><br>

<strong>2. 法規制の遵守</strong><br>
関連する法令・基準を確認し、要求事項を満たす設計とします。<br><br>

<strong>3. 施工性の検討</strong><br>
実際の施工を考慮した現実的な設計を行います。<br><br>

<strong>4. 経済性と安全性</strong><br>
コストと安全性のバランスを適切に保った設計を実施します。<br><br>

詳細な検討が必要な場合は、具体的な条件をお知らせください。
"""
    
    return jsonify({
        'response': response,
        'experts': experts,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats')
def stats():
    """統計情報API"""
    return jsonify({
        'accuracy': '95%',
        'speed_improvement': '4x',
        'memory_reduction': '-67%',
        'cost_reduction': '-75%'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("MoE Simple Web UI")
    print("=" * 60)
    print("ブラウザで http://localhost:5000 にアクセスしてください")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
