# **32B日本語LLMのローカルDPOファインチューニング実践ガイド：cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japaneseの実装**

## **序論**

### **目的と範囲**

本レポートは、cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japaneseモデルに対し、ローカル環境で直接的選好最適化（Direct Preference Optimization, DPO）を適用するための、専門家レベルの包括的なガイドを提供する。本稿では、この先進的なファインチューニング技術の実現可能性、ハードウェアおよびソフトウェアの前提条件、データ準備戦略、そしてエンドツーエンドの実装プロセスについて詳述する。

### **選好チューニングの台頭**

大規模言語モデル（LLM）のカスタマイズは、教師ありファインチューニング（Supervised Fine-Tuning, SFT）から、人間のフィードバックを用いた強化学習（Reinforcement Learning from Human Feedback, RLHF）のような、より高度なアラインメント技術へと進化してきた。しかし、RLHFは実装が複雑で不安定な場合がある。その効率的な後継技術として登場したのがDPOである 1。DPOは、独立した報酬モデルを必要とせず、人間の選好に対して直接最適化を行うため、より安定性が高く、計算負荷も軽いという利点を持つ 4。

### **ローカルハードウェアにおける大規模モデルの課題**

本レポートが取り組む課題は、クラウド環境外で強力な大規模パラメータモデルをカスタマイズしたいという高まる需要に応えることである。特に320億パラメータを持つモデルは特有の課題を提示するが、この課題を解決する鍵となる技術がQLoRA（Quantized Low-Rank Adaptation）である 6。

### **対象モデル：cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese**

本稿で対象とするモデルは、知識蒸留によって論理的推論、コーディング、数学的能力に秀でた性能を発揮する 9。Qwen2アーキテクチャと広大なコンテキストウィンドウを特徴とし、非常に魅力的であると同時に、要求スペックの高いベースモデルである 10。

---

## **第1章：実現可能性の分析と32Bモデル学習のためのハードウェアアーキテクチャ**

ユーザーの「可能か？」という根源的な問いに対する答えは、「特定のハードウェアとソフトウェア構成を前提とすれば可能である」となる。本章では、VRAMのボトルネックを分析し、ローカル環境でのDPO学習に不可欠なベースラインを確立する。

### **1.1 VRAMの制約：なぜ24GBがエントリーポイントなのか**

#### **フル精度での実行不可能性**

32Bモデルをフル精度（FP16/BF16）で実行または学習させることは、コンシューマ向けハードウェアでは不可能である。モデルの重みだけで64GB以上のVRAMを必要とし、学習時のオーバーヘッドを含めると総需要は約82GBに達する 10。これはNVIDIA A100/H100のようなデータセンター級のGPUを必要とすることを意味する 11。

#### **量子化の力**

この問題を解決するのが4ビット量子化である。モデルを量子化することで、重みのメモリフットプリントが約64GBから16GB～21GB程度まで劇的に削減される 12。これが、本タスクが実現可能となる根本的な理由である。

#### **DPOにおけるVRAM要件の計算**

DPO学習中のVRAM使用量は、以下の要素で構成される。

1. **量子化されたベースモデルの重み**：最大の構成要素であり、4ビット32Bモデルで約16GB～21GBを占める 12。  
2. **参照モデル**：DPOは凍結された2つ目の参照モデルを必要とする。QLoRAでは効率的に扱われるが、それでもオーバーヘッドは生じる 1。実装によっては、2つのモデルを同時にロードするのを避けるためにログ確率を事前計算する手法があり、これは重要な最適化となる 14。  
3. **LoRAアダプタ**：学習可能なパラメータ（LoRA行列）は小さいが、より高い精度で保持される 6。  
4. **勾配とオプティマイザの状態**：これらはLoRAパラメータに関連付けられるため、フルファインチューニングと比較してフットプリントが大幅に削減される 17。  
5. **活性化（Activations）**：順伝播の中間結果であり、かなりのメモリを消費する可能性がある。勾配チェックポインティングは、活性化を保存する代わりに再計算することで、計算時間と引き換えにメモリを節約する重要な技術である 14。

#### **24GBというコンセンサス**

複数の情報源 7 を総合すると、量子化された32Bモデルのファインチューニングには、24GBのVRAMを搭載したGPU（NVIDIA RTX 3090, RTX 4090など）が標準的かつ推奨される最小要件として確立されている。実際のテストでは、4ビット32Bモデルが約21GBを使用することが示されており、24GBは安全なマージンと言える 12。

### **1.2 VRAMを超えて：バランスの取れたシステムの構築**

#### **システムRAM**

GPU VRAMが逼迫している場合、システムRAMの不足は深刻なボトルネックとなり、モデルの一部レイヤーがオフロードされる原因となる 11。最低でも32GB、本格的な作業には64GBがはるかに安全で堅牢な構成として推奨される 13。

#### **CPU性能**

低速なCPUはGPUを待機させることになる。データのロード、前処理、タスクのオーケストレーションはCPUに依存するため、最新のマルチコアCPU（例：近年のIntel i7/i9やAMD Ryzen 7/9）がパイプラインを円滑に維持するために不可欠である 11。

#### **ストレージ速度**

32Bモデル（量子化後も）のロード時間は非常に長い。モデル、データセット、環境をNVMe SSDに保存し、I/O待機時間を最小限に抑えることは、譲れない要件である 11。

#### **電源と冷却**

RTX 4090のようなハイエンドGPUは消費電力が大きく、大量の熱を発生させる。長時間の学習セッション中のサーマルスロットリングやシステムの不安定化を防ぐため、最低でも850W～1000Wの高品質な電源ユニット（PSU）と、優れた冷却性能を持つ通気性の良いケースが必須である 13。

### **1.3 考察と示唆**

DeepSeek-R1-32Bのような強力なオープンソースモデルとTRL/QLoRAのようなフレームワークの存在は、AI開発の民主化というトレンドを示している。しかし、同時に、その実践には24GB VRAMという、依然として「プロシューマー」またはハイエンド愛好家カテゴリに属するハードウェアの壁が存在する 13。ここには一つの緊張関係が生まれる。理論的にはアクセスは開かれているが、より高性能な大規模モデルのファインチューニングを実践できるのは、相応のハードウェア投資が可能な一部のユーザーに限られる。これは完全な民主化ではなく、参入障壁が「個人には不可能」から「十分な機材を持つ個人には可能」へと移行した段階にあることを示唆している。

また、VRAMだけに注目するのは一般的な誤りである。CPU、システムRAM、ストレージに関する分析 11 は、このような大規模タスクにおいて、ローカルマシンが単なる部品の集合体ではなく、一つの統合された処理ユニットとして機能することを示している。いずれか一つの要素（例えば低速なSSD）がボトルネックになると、その影響はシステム全体に波及し、高性能GPUの利点を無効化しかねない。その因果連鎖は「低速なストレージ → データロードの遅延 → CPUの待機 → GPUの非効率な利用 → 学習時間の大幅な増加」となる。したがって、予算と計画は、最高級のGPUだけでなく、システム全体のバランスを考慮に入れる必要がある。

#### **表1：32B DPOファインチューニング向けハードウェア仕様ティア**

| ティア | GPU | VRAM | システムRAM | CPU | ストレージ | PSU | 備考 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **最低限** | 中古 NVIDIA RTX 3090 / Tesla P40 | 24 GB | 32 GB | 最新8コアCPU | 1 TB NVMe SSD | 850W Gold | 学習は可能だが、バッチサイズやコンテキスト長に制約が生じやすい。 |
| **推奨** | NVIDIA RTX 4090 | 24 GB | 64 GB | 最新12-16コアCPU | 2 TB+ NVMe SSD | 1000W+ Platinum | 安定したパフォーマンスと柔軟な設定が可能。最もバランスの取れた構成。 |
| **高性能** | NVIDIA H100 (参考) | 80 GB | 128 GB+ | ハイエンドワークステーションCPU | 高速RAID NVMeアレイ | 1200W+ Platinum | フル精度に近い実験や、より大規模なバッチサイズでの高速な学習が可能。 |

---

## **第2章：高性能DPOのためのソフトウェアエコシステム**

本章では、必要となる特定のソフトウェアコンポーネントを詳述し、それぞれの役割と相互依存関係を解説する。これにより、堅牢で正確な開発環境を構築するための明確な道筋を示す。

### **2.1 コアライブラリスタック：Hugging Faceエコシステム**

* **transformers**：DeepSeek-R1-Distill-Qwen-32B-Japaneseモデルとそのトークナイザをダウンロード、ロード、操作するための基盤ライブラリ 11。  
* **trl (Transformer Reinforcement Learning)**：DPO学習ループを統括する中心的なコンポーネントであるDPOTrainerを提供する高レベルライブラリ 2。  
  DPOTrainerはtransformers.Trainerの特殊なサブクラスである。  
* **peft (Parameter-Efficient Fine-Tuning)**：LoRAやその他のPEFT手法を実装するライブラリ。LoraConfigを作成し、ベースモデルにアダプタを適用することで、学習可能なパラメータ数を劇的に削減する 16。  
* **datasets**：選好データセットをメモリ効率良くロードし、処理するためのライブラリ 23。

### **2.2 最適化・高速化ライブラリ**

* **bitsandbytes**：4ビット量子化（QLoRA）に不可欠なライブラリ。BitsAndBytesConfigを通じて、量子化された重みでモデルをロードする役割（load\_in\_4bit=True）を担う 23。CUDAへの依存とプラットフォームの制約（Linux/Windows、macOS非対応）に注意が必要である 26。  
* **accelerate**：ハードウェアの仕様を抽象化するライブラリ。デバイス配置（device\_map="auto"）を管理し、単一GPUやマルチGPU環境で分散学習を正しく初期化するためのaccelerate launchコマンドを提供する 23。  
* **（上級者向け）unsloth**：オプションだが強く推奨される最適化レイヤー。独自のTritonカーネルと手動の逆伝播エンジンを提供し、精度を犠牲にすることなく学習速度を大幅に（最大2倍）向上させ、VRAM使用量をさらに削減できる可能性がある 18。

### **2.3 環境構築：ステップバイステップガイド**

1. **NVIDIAドライバとCUDAツールキット**：PyTorchとbitsandbytesが要求する最新のNVIDIAドライバと互換性のあるCUDAツールキットのバージョンをインストールすることが不可欠である 11。  
2. **Python環境**：Python 3.9以上をサポートする仮想環境（例：condaやvenv）の使用を推奨する 11。  
3. **PyTorchのインストール**：公式サイトから、CUDAサポート付きの正しいPyTorchバージョンをインストールする。  
4. **コアライブラリのインストール**：transformers, trl, peft, datasets, bitsandbytes, accelerate、そしてオプションでunslothのpip installコマンドを提供する 23。必要に応じて、最新機能を利用するためにソースからインストールすることの重要性にも言及する 24。  
5. **accelerateの設定**：accelerate configを実行し、使用するハードウェアに合わせたデフォルトの学習環境を設定する手順を案内する 30。

### **2.4 考察と示唆**

このソフトウェアスタックは単なるツールのリストではなく、複雑に相互接続されたシステムである。pytorch, cuda, bitsandbytes, transformers間のバージョン不整合は、不可解なコンパイルエラーや、気づかぬうちにパフォーマンスが低下する原因となり得る。インストールに関する問題 29 は、この脆弱性を浮き彫りにしている。ここから導かれるのは、Dockerコンテナや詳細な

requirements.txtファイルを用いた厳密な環境管理が、単なる良い習慣ではなく、成功のための絶対的な必須要件であるということだ。プロジェクトの再現性は、この複雑な依存関係の連鎖を固定化できるかどうかにかかっている。

一方で、accelerateやtrlのようなライブラリは、MLエンジニアリングにおける抽象化という強力なトレンドを体現している。accelerateは分散コンピューティングの複雑さを隠蔽し 30、

DPOTrainerはDPOアルゴリズム全体をカプセル化する 4。この抽象化こそが、このような複雑なタスクを個人が実行可能にする要因である。開発者はデバイス配置の定型コードを書いたり、DPOの損失関数をゼロから実装したりする必要がない。これは、ツールが複雑さを管理するように進化し、開発者が低レベルの実装詳細ではなく、データ品質やモデルの振る舞いといったより高レベルのタスクに集中できるようになった、この分野の成熟を示している。

---

## **第3章：日本語選好データセットの調達と合成**

本章では、最も大きな実践的ハードルである、DPOに必要な形式の高品質な日本語データセットの入手方法に取り組む。必要なデータ構造を詳述し、合成データセットを生成するための堅牢な方法論を提示する。

### **3.1 DPOデータセットの形式：(prompt, chosen, rejected)**

DPOには3つのカラムを持つデータ構造が必須である 1。

* **prompt**：モデルへの入力。  
* **chosen**：選好される「良い」応答。  
* **rejected**：選好されない「悪い」応答。

DPOアルゴリズムは、与えられたpromptに対して、chosen応答の確率を最大化し、rejected応答の確率を最小化するように学習する 1。

### **3.2 日本語選好データの希少性**

英語と比較して、日本語には大規模で汎用的なオープンソースの選好データセットが不足している現状がある 41。

llm-japanese-dataset-vanillaのような指示データセットは存在するものの 41、DPOに必要な選好形式ではない。このため、合成データセットを生成するアプローチが必要となる。

### **3.3 合成データセット生成のための実践的パイプライン**

Self-InstructやUltraFeedbackのようなフレームワークに着想を得て、distilabelのようなツールで編成される多段階のプロセスを概説する 43。

1. **ステージ1：シード指示の生成（Self-Instruct）**  
   * 少数の高品質な人間が書いた日本語の指示をシードとして開始する。  
   * 強力なLLM（「ジェネレータ」）を使い、これらのシードに基づいて、より大規模で多様な新しい指示を生成する。これがSelf-Instructの中核的なアイデアである 50。  
   * llm-jp/text2datasetのようなツールを使い、既存の英語の指示データセットを日本語に翻訳してシードプールを作成することも可能である 54。  
2. **ステージ2：応答の生成**  
   * 生成された各指示（プロンプト）に対して、1つまたは複数のLLMを使い、複数の候補応答を生成する。例えば、ベースとなるDeepSeek-R1-32Bモデルから1つの応答を、別の高品質モデルからもう1つの応答を生成する。  
3. **ステージ3：選好のラベリング（LLM-as-a-Judge）**  
   * 強力で独立したLLM（「ジャッジ」、例：GPT-4, Claude 3）を使い、各プロンプトに対する候補応答を評価する。  
   * ジャッジに対して、評価基準（例：有用性、正確性、指示への準拠、安全性）と望ましい出力形式（例：どちらの応答が優れているかとその根拠）を定義した詳細なプロンプトを作成する 55。  
   * ジャッジの出力を用いてchosenとrejectedカラムを埋め、最終的な選好データセットを作成する。  
4. **ステージ4：（任意だが推奨）人間参加型キュレーション**  
   * Argillaのようなツール（distilabelと統合されていることが多い）を使い、合成された選好データの一部をレビューする。これにより、LLMジャッジによる体系的なバイアスを品質管理し、修正することが可能になる 43。

### **3.4 考察と示唆**

合成データパイプライン全体が、メタレベルのAIタスクである。我々は、あるLLMを改善するための学習データを作成するために、別の高度なLLM（ジェネレータとジャッジ）を使用している。これは強力でスケーラブルなパラダイムであるが、同時に深く再帰的でもある。最終的なファインチューニング済みモデルの品質は、そのデータ作成に使用されたジェネレータモデルとジャッジモデルの能力および内在するバイアスに因果的に依存する。これにより、ある世代のモデルのバイアスが次の世代に伝播し、増幅される可能性のある閉ループが形成される。

さらに、研究は「**選好の漏洩（Preference Leakage）**」という、微妙だが重大な汚染問題を指摘している 60。もし「ジャッジ」LLMが応答を生成したモデルと同じファミリーに属している場合、客観的に優れているからではなく、文体の類似性やその他の共通の学習アーティファクトのために、そのモデルの出力を好むようにバイアスがかかる可能性がある。これは三次的な効果である。ジャッジモデルの

*選択*が、選好データの*妥当性*に直接影響を与え、それがDPOファインチューニングの*成功*を決定する。実践的な示唆として、選好シグナルの完全性を確保するためには、応答生成モデルとは可能な限り無関係なジャッジモデルを戦略的に選択する必要がある。

LLM-as-a-Judgeの出力品質は、そのプロンプトの品質に完全に依存する 55。役割、基準、評価尺度、例を定義するこのジャッジプロンプトの作成は、高度な「データセットエンジニアリング」の一形態である。不十分に設計されたジャッジプロンプトは、ノイズが多く、バイアスのかかった、あるいは低品質な選好データセットを生み出し、それはハードウェアや学習パラメータに関係なくDPOプロセスを台無しにする。したがって、プロンプトエンジニアリングのスキルは、単一の出力を生成することから、データセット全体の生成を設計することへと進化したと言える。

#### **表2：日本語選好データセットの構造と内容のサンプル**

| prompt | chosen | rejected |
| :---- | :---- | :---- |
| 東京についての短い俳句を詠んでください。 | 古池や 蛙飛び込む 水の音 | 東京はとても大きな都市で、たくさんの人が住んでいます。 |
| 日本の首都はどこですか？ | 東京です。 | 大阪です。 |
| フィボナッチ数を計算するPython関数を書いてください。 | def fibonacci(n): a, b \= 0, 1; for \_ in range(n): a, b \= b, a \+ b; return a | def fibonacci(n): if n \<= 1: return n; else: return fibonacci(n-1) \+ fibonacci(n-2) |

---

## **第4章：QLoRAを用いたDPOのエンドツーエンド実装**

本章はレポートの実践的な核となる部分であり、これまでの章で述べたすべての概念を統合した、コメント付きの完全なPythonスクリプトを提示する。

### **4.1 スクリプトの概要とロジック**

スクリプトのワークフローの概要は以下の通りである。

1. 必要なライブラリをインポートする。  
2. モデル、データセット、設定パラメータを定義する。  
3. 4ビット量子化設定（BitsAndBytesConfig）をセットアップする。  
4. 量子化設定を用いてベースモデルとトークナイザをロードする。  
5. QLoRAのためのPEFT設定（LoraConfig）をセットアップする。  
6. 選好データセットをロードし、フォーマットする。  
7. 学習引数（TrainingArguments / DPOConfig）を定義する。  
8. DPOTrainerをインスタンス化する。  
9. 学習プロセスを開始する。  
10. 結果として得られたLoRAアダプタを保存する。

### **4.2 詳細なコードウォークスルー（コメント付き）**

以下に、DPOファインチューニングを実行するための完全なスクリプト例を示す。

Python

import torch  
from datasets import load\_dataset  
from peft import LoraConfig, get\_peft\_model, prepare\_model\_for\_kbit\_training  
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments  
from trl import DPOTrainer

\# 1\. 設定と初期化  
model\_id \= "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"  
\# データセットは第3章で作成したものを想定  
dataset\_path \= "path/to/your/japanese\_preference\_dataset"

\# 4ビット量子化設定 (BitsAndBytesConfig)  
\# bnb\_4bit\_compute\_dtypeは、Ampere世代以降のGPU（RTX 3090/4090, A100など）で  
\# パフォーマンスを向上させるためにtorch.bfloat16を指定する \[26, 28\]  
quantization\_config \= BitsAndBytesConfig(  
    load\_in\_4bit=True,  
    bnb\_4bit\_quant\_type="nf4",  
    bnb\_4bit\_use\_double\_quant=True,  
    bnb\_4bit\_compute\_dtype=torch.bfloat16,  
)

\# モデルとトークナイザのロード  
\# device\_map="auto"は、accelerateライブラリが自動的にデバイス配置を処理する \[28\]  
model \= AutoModelForCausalLM.from\_pretrained(  
    model\_id,  
    quantization\_config=quantization\_config,  
    device\_map="auto",  
    trust\_remote\_code=True,  
)  
model.config.use\_cache \= False

tokenizer \= AutoTokenizer.from\_pretrained(model\_id, trust\_remote\_code=True)  
tokenizer.pad\_token \= tokenizer.eos\_token

\# 2\. PEFT (QLoRA) の設定  
\# LoraConfigの主要パラメータ \[25, 61, 62\]:  
\# r: LoRA行列のランク。小さいほど学習パラメータが少なくなる。  
\# lora\_alpha: LoRAのスケーリング係数。  
\# target\_modules: LoRAを適用する層。モデルのアーキテクチャに依存する。  
\# 一般的には'q\_proj', 'k\_proj', 'v\_proj', 'o\_proj'などが対象となる。  
peft\_config \= LoraConfig(  
    r=64,  
    lora\_alpha=128,  
    lora\_dropout=0.05,  
    bias="none",  
    task\_type="CAUSAL\_LM",  
    target\_modules=\[  
        "q\_proj",  
        "k\_proj",  
        "v\_proj",  
        "o\_proj",  
        "gate\_proj",  
        "up\_proj",  
        "down\_proj",  
    \],  
)

\# 3\. データのロードと処理  
\# 第3章で作成したデータセットをロード  
train\_dataset \= load\_dataset(dataset\_path, split="train")

\# 4\. DPOTrainerの設定  
training\_args \= TrainingArguments(  
    per\_device\_train\_batch\_size=1,  \# VRAM制約のためバッチサイズは1に設定 \[63, 64\]  
    gradient\_accumulation\_steps=8, \# 実効バッチサイズを8にする \[1, 34\]  
    gradient\_checkpointing=True,   \# 活性化のメモリを節約するために必須 \[65\]  
    learning\_rate=5e-6,  
    lr\_scheduler\_type="cosine",  
    num\_train\_epochs=1,  
    logging\_steps=10,  
    save\_steps=100,  
    output\_dir="./dpo\_output",  
    optim="paged\_adamw\_8bit",       \# メモリをさらに節約する8ビットオプティマイザ \[26, 63\]  
    bf16=True,                     \# bfloat16混合精度学習を有効化 (Ampere以降のGPU) \[61\]  
    remove\_unused\_columns=False,  
)

\# DPOTrainerのインスタンス化  
\# ref\_model=Noneとすると、trainerが自動的に参照モデルのコピーを作成する \[15\]  
dpo\_trainer \= DPOTrainer(  
    model,  
    ref\_model=None,  
    args=training\_args,  
    train\_dataset=train\_dataset,  
    tokenizer=tokenizer,  
    peft\_config=peft\_config,  
    beta=0.1,  \# DPOのbetaハイパーパラメータ (後述)  
    max\_prompt\_length=1024,  
    max\_length=2048,  
)

\# 5\. 学習の開始  
dpo\_trainer.train()

\# 6\. アダプタの保存  
dpo\_trainer.save\_model("./final\_adapter")

#### **4.5 betaハイパーパラメータの解説**

DPOTrainerにおけるbetaパラメータ（デフォルト値0.1）は、DPOアルゴリズムの挙動を制御する重要な要素である。このパラメータは、選好データへのアラインメントと、参照モデルからの逸脱を防ぐ正則化との間のトレードオフを調整する 1。

* **低いbeta値**：選好データへのより積極的なアラインメントを促すが、ベースモデルの知識を忘れる「破滅的忘却」のリスクが高まる。  
* **高いbeta値**：より保守的な更新を行い、ベースモデルに近い状態を維持するが、選好の学習が限定的になる可能性がある。

近年の研究では、データ品質に応じてbetaを動的に調整する手法が提案されているが 66、最初の実装としては、0.1から0.5の間の静的な値を用いるのが標準的である 40。

### **4.6 学習の実行**

このスクリプトは、python your\_script.pyではなく、以下のコマンドで実行する。accelerate launchは、accelerateライブラリが環境を正しく初期化し、リソースを管理するために必要である 30。

Bash

accelerate launch your\_dpo\_script.py

### **4.7 考察と示唆**

学習引数は、独立して調整できるノブの集まりではない。それらは主に速度、メモリ使用量、モデル性能の間で複雑なトレードオフのシステムを形成している。例えば、per\_device\_train\_batch\_sizeとgradient\_accumulation\_stepsは、実効バッチサイズを一定に保つために逆相関の関係にある 1。

gradient\_checkpointingは計算コストの増加と引き換えにメモリを節約する 18。

learning\_rateとbetaは相互に作用し、アラインメントプロセスの安定性と積極性を決定する 34。一つのパラメータを単独で最適化することはできず、24GBのGPUという制約された環境を乗り切るためには、これらの相互依存関係を理解する必要がある。

また、dpo\_trainer.train()という一行のコードの単純さ 38 は、その背後にある複雑な処理を覆い隠している。DPOは強化学習問題に対する解析的な解法である 1。

DPOTrainerは、ポリシーモデルと参照モデルからの対数確率に基づいて暗黙的に報酬を計算し、単純な分類損失を介してポリシーの更新を実行している。ユーザーは、使いやすいAPIにカプセル化された強力な理論的成果（DPO論文）を活用しているのである。これは、*どのように*行うかは単純だが、効果的なデバッグやハイパーパラメータチューニングのためには、*何が*行われているか（根底にある数学と理論）を理解することが重要であることを示唆している。

---

## **第5章：学習後のワークフロー：マージ、推論、評価**

本最終章では、学習スクリプトが完了した後の作業、すなわち学習済みアダプタを実用的なモデルに変換し、その性能を評価する方法について案内する。

### **5.1 LoRAアダプタのマージ**

学習の出力は、完全な32Bモデルではなく、小さなLoRAアダプタの重みである 34。PEFTライブラリの

merge\_and\_unload()メソッドを使い、元の4ビット量子化ベースモデルをロードし、学習済みのLoRAの重みをマージするスクリプトを提供する。マージ後のモデルはsave\_pretrainedを用いて保存し、推論用に簡単にロードできる新しいスタンドアロンのモデルディレクトリを作成する。

Python

from peft import PeftModel  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch

base\_model\_id \= "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"  
adapter\_path \= "./final\_adapter"  
merged\_model\_path \= "./dpo\_merged\_model"

\# ベースモデルを4-bitでロード  
quantization\_config \= BitsAndBytesConfig(  
    load\_in\_4bit=True,  
    bnb\_4bit\_compute\_dtype=torch.bfloat16,  
)

base\_model \= AutoModelForCausalLM.from\_pretrained(  
    base\_model\_id,  
    quantization\_config=quantization\_config,  
    device\_map="auto",  
    trust\_remote\_code=True,  
)

tokenizer \= AutoTokenizer.from\_pretrained(base\_model\_id, trust\_remote\_code=True)

\# LoRAアダプタをロードしてマージ  
model \= PeftModel.from\_pretrained(base\_model, adapter\_path)  
model \= model.merge\_and\_unload()

\# マージしたモデルを保存  
model.save\_pretrained(merged\_model\_path)  
tokenizer.save\_pretrained(merged\_model\_path)

### **5.2 DPOチューニング済みモデルでの推論**

新たにマージされたモデルとトークナイザをロードし、日本語のプロンプトでテキストを生成する簡単な推論スクリプトを提供する。これは、モデルの振る舞いが望ましい方向に変化したかを定性的に評価するための「感覚的なチェック」として機能する。

### **5.3 評価戦略**

#### **定性的評価**

対象ドメインに関連する多様なプロンプトを用いて、広範な対話的テストを実施することを推奨する。

#### **定量的評価**

自動評価の難しさを認識した上で、より厳密なアプローチとして標準的なベンチマークを紹介する。

* MT-BenchとFastChatを使い、GPT-4のような強力なジャッジモデルに対して対話能力を構造的に評価する方法がある 23。  
* 合成データセットからホールドアウト（評価用）の選好ペアを作成し、モデルがchosen応答により高い確率を割り当てるかどうかの精度を測定する。これは学習目標に対するアラインメントを直接的に評価する手法である 64。

### **5.4 考察と示唆**

学習後のワークフローは、根本的なトレードオフを明らかにする。ベースモデルとLoRAアダプタを分離しておくことは**モジュール性**を提供する。単一のベースモデルに対して、異なるタスク用に異なるアダプタを交換できる。一方、それらをマージすると、デプロイや共有が容易な単一のポータブルなアーティファクトが作成されるが、そのモジュール性は失われる。どちらを選択するかは、デプロイ戦略に依存する。研究環境では実験のためにモジュール性が好まれるかもしれないが、本番環境では単純さと堅牢性のためにマージされたモデルが好まれるだろう。

また、本レポートは*学習*のための明確な道筋を提供したが、評価に関する記述 23 は、その学習の成功を

*評価する*ことがはるかに困難であることを示している。「感覚的なチェック」は主観的であり、MT-Benchのような自動ベンチマークは別のLLMに依存するため、再び「LLM-as-a-Judge」パラダイムとそのバイアスの問題に直面する。これは、我々がモデルを選好*データセット*にアラインさせる強力なツールを持っている一方で、そのアラインメントが人間の真の*意図*に沿った、一般化可能な改善につながるかを堅牢に検証することは、依然として未解決の困難な研究課題であることを意味する。DPOの旅はtrain()が終了したときに終わるのではなく、検証という、より曖昧な新しい段階に入るのである。

## **結論と戦略的提言**

### **総括**

cyberagent/DeepSeek-R1-Distill-Qwen-32B-JapaneseモデルのローカルDPOファインチューニングは、特定のハイエンドハードウェア構成（24GB以上のVRAMを持つGPUとバランスの取れたシステム）、最新のソフトウェアスタック（TRL, PEFT, bitsandbytes）、そして堅牢なデータ生成パイプラインを前提とすれば、実現可能である。

### **戦略的提言**

1. **小さく始める**：本格的な32Bモデルに取り組む前に、より小さな7Bクラスのモデルでパイプライン全体（データ生成、学習スクリプト、マージ）を検証する。これにより、迅速かつ低コストでデバッグが可能になる。  
2. **データ品質に投資する**：DPOの成功は、選好データセットの品質に大きく依存する。データ生成とジャッジプロンプトの作成に十分な時間とリソースを割り当てる。合成データの一部を人間がレビューし、体系的なエラーを検出することが重要である。  
3. **計画的に反復する**：プロセスを科学的実験として扱う。一度に一つのハイパーパラメータ（例：learning\_rate, beta, LoRAのr）を変更し、その影響を評価する。実験の詳細なログを記録する。  
4. **高度な最適化を検討する**：ベースラインが機能したら、Unslothのようなツールを検討し、学習速度を倍増させたり、同じVRAMにより大きなバッチを収めたりすることで、高価なハードウェアの効率を最大化することを推奨する 18。

#### **引用文献**

1. Platform For AI:Guide to fine-tuning LLMs \- Alibaba Cloud, 10月 3, 2025にアクセス、 [https://www.alibabacloud.com/help/doc-detail/2863790.html](https://www.alibabacloud.com/help/doc-detail/2863790.html)  
2. LLM fine-tuning with Direct Preference Optimization (DPO) with code | by Ufuk Birbiri, 10月 3, 2025にアクセス、 [https://medium.com/@ufuk.birbiri/llm-fine-tuning-with-direct-preference-optimization-dpo-with-code-12ed92259215](https://medium.com/@ufuk.birbiri/llm-fine-tuning-with-direct-preference-optimization-dpo-with-code-12ed92259215)  
3. DPO In Kaggle \- Working.... \!\!\!, 10月 3, 2025にアクセス、 [https://www.kaggle.com/code/eugeniokukes/dpo-in-kaggle-working](https://www.kaggle.com/code/eugeniokukes/dpo-in-kaggle-working)  
4. DPO Trainer \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/trl/dpo\_trainer](https://huggingface.co/docs/trl/dpo_trainer)  
5. DPO Trainer \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/trl/main/dpo\_trainer](https://huggingface.co/docs/trl/main/dpo_trainer)  
6. How can I fine-tune large language models on a budget using LoRA and QLoRA on cloud GPUs? \- Runpod, 10月 3, 2025にアクセス、 [https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget](https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget)  
7. What are VRAM requirements for QLoRA Finetuning? : r/LocalLLaMA \- Reddit, 10月 3, 2025にアクセス、 [https://www.reddit.com/r/LocalLLaMA/comments/159g3hy/what\_are\_vram\_requirements\_for\_qlora\_finetuning/](https://www.reddit.com/r/LocalLLaMA/comments/159g3hy/what_are_vram_requirements_for_qlora_finetuning/)  
8. artidoro/qlora \- Efficient Finetuning of Quantized LLMs \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/artidoro/qlora](https://github.com/artidoro/qlora)  
9. Deepseek R1 Distill Qwen 32B: Memory Requirements Guide \- BytePlus, 10月 3, 2025にアクセス、 [https://www.byteplus.com/en/topic/397894](https://www.byteplus.com/en/topic/397894)  
10. DeepSeek-R1-Distill-Qwen-Japanese-32B Model | MAX Builds \- Code with Modular, 10月 3, 2025にアクセス、 [https://builds.modular.com/models/DeepSeek-R1-Distill-Qwen-Japanese/32B](https://builds.modular.com/models/DeepSeek-R1-Distill-Qwen-Japanese/32B)  
11. Deepseek R1 32B Requirements: Specs & System Needs 2025 \- BytePlus, 10月 3, 2025にアクセス、 [https://www.byteplus.com/en/topic/415701](https://www.byteplus.com/en/topic/415701)  
12. GPU System Requirements for Running DeepSeek-R1 \- ApX Machine Learning, 10月 3, 2025にアクセス、 [https://apxml.com/posts/gpu-requirements-deepseek-r1](https://apxml.com/posts/gpu-requirements-deepseek-r1)  
13. Deepseek R1 32b VRAM Requirements Explained \- BytePlus, 10月 3, 2025にアクセス、 [https://www.byteplus.com/en/topic/553063](https://www.byteplus.com/en/topic/553063)  
14. michaelnny/DPO-LLaMA: A clean implementation of direct preference optimization (DPO) to train the LLaMA 2 model to align with human preferences. \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/michaelnny/DPO-LLaMA](https://github.com/michaelnny/DPO-LLaMA)  
15. Fine-tune Llama 2 with DPO \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/blog/dpo-trl](https://huggingface.co/blog/dpo-trl)  
16. In-depth guide to fine-tuning LLMs with LoRA and QLoRA \- Mercity AI, 10月 3, 2025にアクセス、 [https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora](https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora)  
17. How much VRAM do I need for LLM model fine-tuning? | Modal Blog, 10月 3, 2025にアクセス、 [https://modal.com/blog/how-much-vram-need-fine-tuning](https://modal.com/blog/how-much-vram-need-fine-tuning)  
18. \[P\] Train your own Reasoning model \- GRPO works on just 5GB VRAM : r/MachineLearning, 10月 3, 2025にアクセス、 [https://www.reddit.com/r/MachineLearning/comments/1iyv12c/p\_train\_your\_own\_reasoning\_model\_grpo\_works\_on/](https://www.reddit.com/r/MachineLearning/comments/1iyv12c/p_train_your_own_reasoning_model_grpo_works_on/)  
19. How Much VRAM Do you need to run a 32B with 32k context? : r/LocalLLaMA \- Reddit, 10月 3, 2025にアクセス、 [https://www.reddit.com/r/LocalLLaMA/comments/1j5kdcm/how\_much\_vram\_do\_you\_need\_to\_run\_a\_32b\_with\_32k/](https://www.reddit.com/r/LocalLLaMA/comments/1j5kdcm/how_much_vram_do_you_need_to_run_a_32b_with_32k/)  
20. DeepSeek R1: Architecture, Training, Local Deployment, and Hardware Requirements, 10月 3, 2025にアクセス、 [https://dev.to/askyt/deepseek-r1-architecture-training-local-deployment-and-hardware-requirements-3mf8](https://dev.to/askyt/deepseek-r1-architecture-training-local-deployment-and-hardware-requirements-3mf8)  
21. DeepSeek R1 Hardware Requirements Explained \- YouTube, 10月 3, 2025にアクセス、 [https://www.youtube.com/watch?v=5RhPZgDoglE](https://www.youtube.com/watch?v=5RhPZgDoglE)  
22. DeepSeek-R1 671B: Complete Hardware Requirements \- DEV Community, 10月 3, 2025にアクセス、 [https://dev.to/askyt/deepseek-r1-671b-complete-hardware-requirements-optimal-deployment-setup-2e48](https://dev.to/askyt/deepseek-r1-671b-complete-hardware-requirements-optimal-deployment-setup-2e48)  
23. RLHF in 2024 with DPO & Hugging Face \- Philschmid, 10月 3, 2025にアクセス、 [https://www.philschmid.de/dpo-align-llms-in-2024-with-trl](https://www.philschmid.de/dpo-align-llms-in-2024-with-trl)  
24. huggingface/trl: Train transformer language models with reinforcement learning. \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/huggingface/trl](https://github.com/huggingface/trl)  
25. Fine-Tune Gemma using Hugging Face Transformers and QloRA | Google AI for Developers, 10月 3, 2025にアクセス、 [https://ai.google.dev/gemma/docs/core/huggingface\_text\_finetune\_qlora](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)  
26. Bitsandbytes \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/transformers/quantization/bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)  
27. bitsandbytes-foundation/bitsandbytes: Accessible large language models via k-bit quantization for PyTorch. \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)  
28. How to Quantize LLMs Using BitsandBytes \- ApX Machine Learning, 10月 3, 2025にアクセス、 [https://apxml.com/posts/efficient-llm-quantization-bitsandbytes](https://apxml.com/posts/efficient-llm-quantization-bitsandbytes)  
29. Using \`bitsandbytes\` 4-bit quantization requires the latest version of bitsandbytes: \`pip install \-U bitsandbytes\` \- Stack Overflow, 10月 3, 2025にアクセス、 [https://stackoverflow.com/questions/79344565/using-bitsandbytes-4-bit-quantization-requires-the-latest-version-of-bitsandby](https://stackoverflow.com/questions/79344565/using-bitsandbytes-4-bit-quantization-requires-the-latest-version-of-bitsandby)  
30. Accelerate \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/transformers/accelerate](https://huggingface.co/docs/transformers/accelerate)  
31. Accelerate library Lightning AI \- Docs, 10月 3, 2025にアクセス、 [https://lightning.ai/docs/overview/pretrain-models/accelerate-library](https://lightning.ai/docs/overview/pretrain-models/accelerate-library)  
32. Speeding Up AI Workflows: How Hugging Face Uses the Accelerate Library \- Aditya Mangal, 10月 3, 2025にアクセス、 [https://adityamangal98.medium.com/speeding-up-ai-workflows-how-hugging-face-uses-the-accelerate-library-16da00a7ba5a](https://adityamangal98.medium.com/speeding-up-ai-workflows-how-hugging-face-uses-the-accelerate-library-16da00a7ba5a)  
33. Accelerate \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/accelerate/index](https://huggingface.co/docs/accelerate/index)  
34. Fine-tuning LLMs Guide | Unsloth Documentation, 10月 3, 2025にアクセス、 [https://docs.unsloth.ai/get-started/fine-tuning-llms-guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)  
35. unslothai/unsloth: Fine-tuning & Reinforcement Learning for LLMs. Train OpenAI gpt-oss, DeepSeek-R1, Qwen3, Gemma 3, TTS 2x faster with 70% less VRAM. \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)  
36. 4-bit quantization requires the latest version of bitsandbytes on Google Co-lab \- Reddit, 10月 3, 2025にアクセス、 [https://www.reddit.com/r/LocalLLaMA/comments/1j1mq6y/4bit\_quantization\_requires\_the\_latest\_version\_of/](https://www.reddit.com/r/LocalLLaMA/comments/1j1mq6y/4bit_quantization_requires_the_latest_version_of/)  
37. huggingface/accelerate: A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed support \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/huggingface/accelerate](https://github.com/huggingface/accelerate)  
38. DPO Trainer \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/trl/v0.9.6/dpo\_trainer](https://huggingface.co/docs/trl/v0.9.6/dpo_trainer)  
39. DPO Trainer \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/trl/v0.11.2/dpo\_trainer](https://huggingface.co/docs/trl/v0.11.2/dpo_trainer)  
40. DPO Post-Training on a Budget: A Practical Guide | by Ramesh Subrahmanyam | Medium, 10月 3, 2025にアクセス、 [https://medium.com/@rameshsubrahmanyam/dpo-post-training-on-a-budget-a-practical-guide-2251b2ec68e2](https://medium.com/@rameshsubrahmanyam/dpo-post-training-on-a-budget-a-practical-guide-2251b2ec68e2)  
41. From Base to Conversational: Japanese Instruction Dataset and Tuning Large Language Models \- ResearchGate, 10月 3, 2025にアクセス、 [https://www.researchgate.net/publication/376613886\_From\_Base\_to\_Conversational\_Japanese\_Instruction\_Dataset\_and\_Tuning\_Large\_Language\_Models](https://www.researchgate.net/publication/376613886_From_Base_to_Conversational_Japanese_Instruction_Dataset_and_Tuning_Large_Language_Models)  
42. \[2305.12720\] llm-japanese-dataset v0: Construction of Japanese Chat Dataset for Large Language Models and its Methodology \- arXiv, 10月 3, 2025にアクセス、 [https://arxiv.org/abs/2305.12720](https://arxiv.org/abs/2305.12720)  
43. Create a legal preference dataset \- distilabel, 10月 3, 2025にアクセス、 [https://distilabel.argilla.io/0.6.0/tutorials/pipeline-notus-instructions-preferences-legal/](https://distilabel.argilla.io/0.6.0/tutorials/pipeline-notus-instructions-preferences-legal/)  
44. Generate a Preference Dataset with distilabel \- Hugging Face Open-Source AI Cookbook, 10月 3, 2025にアクセス、 [https://huggingface.co/learn/cookbook/generate\_preference\_dataset\_distilabel](https://huggingface.co/learn/cookbook/generate_preference_dataset_distilabel)  
45. Generate a preference dataset \- Distilabel Docs, 10月 3, 2025にアクセス、 [https://distilabel.argilla.io/dev/sections/pipeline\_samples/tutorials/generate\_preference\_dataset/](https://distilabel.argilla.io/dev/sections/pipeline_samples/tutorials/generate_preference_dataset/)  
46. Distilabel \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/hub/datasets-distilabel](https://huggingface.co/docs/hub/datasets-distilabel)  
47. Distilabel is a framework for synthetic data and AI feedback for engineers who need fast, reliable and scalable pipelines based on verified research papers. \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/argilla-io/distilabel](https://github.com/argilla-io/distilabel)  
48. Create a legal preference dataset \- Hugging Face Open-Source AI Cookbook, 10月 3, 2025にアクセス、 [https://huggingface.co/learn/cookbook/pipeline\_notus\_instructions\_preferences\_legal](https://huggingface.co/learn/cookbook/pipeline_notus_instructions_preferences_legal)  
49. Tutorials \- Distilabel Docs, 10月 3, 2025にアクセス、 [http://distilabel.argilla.io/dev/sections/pipeline\_samples/](http://distilabel.argilla.io/dev/sections/pipeline_samples/)  
50. yizhongw/self-instruct: Aligning pretrained language models with instruction data generated by themselves. \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)  
51. Self-Instruct Framework, Explained \- Towards Data Science, 10月 3, 2025にアクセス、 [https://towardsdatascience.com/self-instruct-framework-explained-16bce90f4683/](https://towardsdatascience.com/self-instruct-framework-explained-16bce90f4683/)  
52. Aligning Language Model with Self Generated Instructions \- YouTube, 10月 3, 2025にアクセス、 [https://www.youtube.com/watch?v=FefyD2Vk-Wg](https://www.youtube.com/watch?v=FefyD2Vk-Wg)  
53. Synthetic dataset generation techniques: Self-Instruct \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/blog/davanstrien/self-instruct](https://huggingface.co/blog/davanstrien/self-instruct)  
54. llm-jp/text2dataset: Easily turn large English text datasets into Japanese text datasets using open LLMs. \- GitHub, 10月 3, 2025にアクセス、 [https://github.com/llm-jp/text2dataset](https://github.com/llm-jp/text2dataset)  
55. LLM-as-a-judge: a complete guide to using LLMs for evaluations \- Evidently AI, 10月 3, 2025にアクセス、 [https://www.evidentlyai.com/llm-guide/llm-as-a-judge](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)  
56. How to Create an LLM Judge That Aligns with Human Labels | Towards Data Science, 10月 3, 2025にアクセス、 [https://towardsdatascience.com/how-to-create-an-llm-judge-that-aligns-with-human-labels/](https://towardsdatascience.com/how-to-create-an-llm-judge-that-aligns-with-human-labels/)  
57. Evidence-Based Prompting Strategies for LLM-as-a-Judge: Explanations and Chain-of-Thought \- Arize AI, 10月 3, 2025にアクセス、 [https://arize.com/blog/evidence-based-prompting-strategies-for-llm-as-a-judge-explanations-and-chain-of-thought/](https://arize.com/blog/evidence-based-prompting-strategies-for-llm-as-a-judge-explanations-and-chain-of-thought/)  
58. LLM-as-a-Judge: A Practical Guide | Towards Data Science, 10月 3, 2025にアクセス、 [https://towardsdatascience.com/llm-as-a-judge-a-practical-guide/](https://towardsdatascience.com/llm-as-a-judge-a-practical-guide/)  
59. Clean an Existing Preference Dataset with LLMs as Judges \- Colab, 10月 3, 2025にアクセス、 [https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/clean\_dataset\_judges\_distilabel.ipynb](https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/clean_dataset_judges_distilabel.ipynb)  
60. Preference Leakage: A Contamination Problem in LLM-as-a-judge, 10月 3, 2025にアクセス、 [https://howiehwong.github.io/preference\_leakage.pdf](https://howiehwong.github.io/preference_leakage.pdf)  
61. Preference Optimization for Vision Language Models with TRL \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/blog/dpo\_vlm](https://huggingface.co/blog/dpo_vlm)  
62. $β$-DPO: Direct Preference Optimization with Dynamic $β \- arXiv, 10月 3, 2025にアクセス、 [https://arxiv.org/pdf/2407.08639](https://arxiv.org/pdf/2407.08639)  
63. DPO: Direct Preference Optimization with Dynamic $\\beta \- OpenReview, 10月 3, 2025にアクセス、 [https://openreview.net/forum?id=ZfBuhzE556¬eId=jIWyqRg4rw](https://openreview.net/forum?id=ZfBuhzE556&noteId=jIWyqRg4rw)  
64. Quickstart \- Hugging Face, 10月 3, 2025にアクセス、 [https://huggingface.co/docs/trl/quickstart](https://huggingface.co/docs/trl/quickstart)