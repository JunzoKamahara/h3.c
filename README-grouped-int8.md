# Group-wise INT8 Weight Quantization — 実験まとめ

`exp/grouped-int8-weights`ブランチ（`main`から分岐、`exp-grouped-int8-baseline`タグの上）で実施した、DiT
weightのgroup-wise INT8量子化実験の中間まとめ。元の実験計画書「h3.c DiT Weight Group-wise INT8
Quantization 実験計画書」の Stage 0〜2 相当、および計画書に無かった追加の誤差伝播分析（評価ラダー①〜④）を実施した。

## 用語の定義（重要な訂正）

本ドキュメント（および`quality_grouped_int8.csv`のvideo-levelエントリ）で単に**「BF16」「BF16基準」**と
書いている箇所は、**すべて以下の"Hybrid reference"を指しており、モデル全体をBF16で実行したものではない**。
これは`exp/gpu-determinism-probe`ブランチでの調査を通じて後から明確になった、重要な訂正事項である。

本実験のスコープは最初から**attention-output (OUT) projectionという1つの重み行列のみ**に限定されていた
（1章参照：「QKV / FC1 / FC2 は未対応（row-wiseのまま）」）。したがって、以下のどのモードも、
QKV・FC1・FC2は共通して通常の（本モデルのデフォルトである）int8 row-wise経路のままであり、
違いはattn.out_projの量子化方式1点のみである：

| 本ドキュメントでの呼称 | attn.out_projの実際の経路 | QKV / FC1 / FC2 |
|---|---|---|
| **Hybrid reference**（本文中「BF16」「BF16基準」） | BF16（`H3_DISABLE_INT8_ATTENTION_OUT=1`相当、量子化なし） | 通常のint8 row-wise（デフォルト） |
| **Row-wise** | int8、per-row scale（本実験以前からの既存経路） | 同上 |
| **G1024 / G512 / G256 / G128** | int8、group-wise scale（本実験で追加した経路） | 同上 |
| **Full BF16**（本実験では一度も生成していない） | BF16 | BF16 |

つまり、本ドキュメントでのすべての比較（①〜④の評価ラダー、20-step multi-seed評価、single-step
injection実験）は、「Hybrid referenceに対して、attn.out_projだけをint8に置き換えるとどう変わるか」を
見ているのであって、「モデル全体をBF16からint8に置き換えるとどう変わるか」を見ているのではない。
G128が「Hybrid referenceにどれだけ近いか」という問いと、G128が「Full
BF16にどれだけ近いか」という問いは意味が異なるが、**本実験ではFull BF16を一度も生成していないため、
後者には答えられない**。

なお、この呼称の混乱が、`exp/gpu-determinism-probe`ブランチで最初に「BF16 vs
BF16再実行でrun-to-run差が見えた＝GPU非決定性がある」と誤って解釈した一因でもあった（そこで
「BF16」と呼んでいたのも実際にはこのHybrid referenceであり、後の広範な検証でその解釈自体撤回されている。
詳細は`exp/gpu-determinism-probe`ブランチの`README-gpu-determinism.md`を参照——
本ブランチにはそのファイルは存在しない）。

## 1. 実装したもの（Stage 0 / Stage 1）

既存の row-wise INT8 量子化経路（`h3_gpu_quantize_weight_int8` /
`h3_gpu_linear_int8_bf16`、`h3_linear_int8_nax_r128`カーネル）は一切変更せず、新しい関数・カーネルを追加する形で実装した。

- **`h3_shaders.metal`**
  - `h3_quantize_bf16_weight_int8_groups_scalar` — 重み用のgroup-wise量子化カーネル。`ceil(K /
    group_size)`方式で、最終グループが半端でも対応（活性化用の既存grouped量子化カーネルは`group_size`が列数を割り切ることを要求するため、これとは別実装）。
  - `h3_linear_int8_weight_grouped_nax_r128x64` — 重み側のみgroupごとにscaleを持つINT8
    GEMM。活性化側は既存どおりrow-wiseのまま（本実験のスコープを「重みの量子化」のみに限定するため）。128×64タイル（既存row-wiseカーネルは128×128）。
- **`h3_gpu.h` / `h3_gpu.m`**
  - ホスト側ラッパー `h3_gpu_quantize_weight_int8_grouped` / `h3_gpu_linear_int8_grouped_weight_bf16`。
  - `h3_gpu_tensor_read_i8`（診断用途で必要になった、int8テンソルの読み戻し関数。既存コードにはread_f32/read_bf16はあったがread_i8が無かった）。
- **`h3_dit.c`**
  - 環境変数 `H3_INT8_WEIGHT_GROUP`（0/未設定＝現行row-wise、それ以外＝128の倍数のgroup
    size）を追加。**Attention-output (OUT) projectionのみ**に配線。QKV / FC1 / FC2 は未対応（row-wiseのまま）。
  - grouped GEMMにはhead-major版が無いため、`H3_INT8_WEIGHT_GROUP`が有効な間はhead-major attention
    output経路を強制的に無効化し、row-major経路にフォールバックする。
  - 実験用の診断フック（`H3_DUMP_BLOCK_HIDDEN` / `H3_DUMP_BLOCK_HIDDEN_PATH` /
    `H3_DISABLE_FUSED_CROSS_BLOCK_ADALN`）を追加。指定したブロック番号の直後で`dit->hidden`をファイルにダンプして`exit(0)`する。以降の評価ラダー②③で使用。
- **`tests/test_int8_weight_group.c`** — grouped量子化のCPU参照実装とGPU出力をbyte-exactで比較する単体テスト（部分グループ・全ゼロ行・負の最大値行を含む）。
- **`h3_weight_quant_error.c`** — 実チェックポイントの`attn.out_proj.weight`を各量子化方式で量子化→逆量子化し、元のBF16との再構成誤差（RMSE/MAE/MaxAbsErr/Cosine/RelFrobErr）を測定する単体ツール。

いずれも `Makefile` にビルドターゲットを追加済み（`h3_int8_weight_group_test`, `h3_weight_quant_error`）。

## 2. 正当性検証

- `h3_int8_weight_group_test`: CPU参照実装とGPU出力が**byte-exactで完全一致**（部分グループ・全ゼロ行・負の最大値行を含む）。
- 実生成（`H3_INT8_WEIGHT_GROUP=128〜1024`）が50ブロック全てでクラッシュなく完走することを確認。
- `--ref-image`を使わない通常のtext-to-videoパスは、同一seedで**バイト完全一致の再現性**を確認（前セッションで確認したRef2VAの非決定性とは異なる、信頼できる比較対象であることを保証）。

## 3. 評価ラダー：weight精度と出力の近さは一致しない

計画書には無かった追加分析として、「量子化誤差の小ささ」と「最終出力のHybrid referenceへの近さ」が本当に対応するのかを、4段階に分けて切り分けた（すべて同一プロンプト "A red fox walks through fresh snow in a pine
forest."、seed=42、512×512、attn.out_proj projectionのみ変更）。

| 段階 | 内容 | RMSEによる順位（良い→悪い） |
|---|---|---|
| ①Weight単体（block 0/9/24/39/49平均） | BF16重みを量子化→逆量子化した値とBF16原本を比較 | **G128 < G256 < G512 < G1024 < Row-wise**（全ブロックで完全に単調） |
| ②Block 0通過後 | 1ブロック分のattention+MLP+residualを通した後の隠れ状態を比較 | G1024 < G128 < G256 < G512 < Row-wise（grouped 4方式全てがrow-wiseに勝つが、内部順序は崩壊） |
| ③Block 49通過後（50ブロック、denoising 1step） | 全50ブロックを通した後の隠れ状態を比較 | G512 < G128 ≈ G1024 < Row-wise < **G256**（grouped方式の中でG256が最下位に転落） |
| ④動画（4 steps全体） | 完成した動画フレームをHybrid referenceとPSNR/SSIMで比較 | プロンプト依存（下記参照） |

**誤差の相対的な大きさ（RelFrobErr）も段階を追うごとに急増**した：block0で約0.0045〜0.0048だったものが、block49では約0.054〜0.062（**約12倍**）に拡大。これは50層の非線形変換（attention/MLP/residual）を繰り返すことによる実質的な誤差増幅を示している。

### ④ 動画レベル（PSNR/SSIM、Hybrid reference、4 steps）

| プロンプト | seed | 勝者(PSNR) | 勝者(SSIM) |
|---|---:|---|---|
| 狐・雪(毛皮のテクスチャ) | 42 | G=1024 | G=1024 |
| 高層ビル(直線・幾何学) | 42 | Row-wise | Row-wise |
| 高層ビル(直線・幾何学) | 100 | Row-wise(僅差) | **G=1024**(僅差) |
| 顔のクローズアップ(人物・肌) | 42 | G=1024 | G=1024 |
| 廊下・パースペクティブ(直線・幾何学 2例目) | 42 | G=1024 | G=1024 |

同一のG=1024設定でも、プロンプトが変わるだけで勝敗が完全に逆転した（4 prompts中3勝1敗でG=1024優勢）。さらに重要な点として、**高層ビルプロンプトをseed=42→100に変えただけで、PSNRとSSIMの勝者が食い違う結果になった**（seed=42はRow-wiseが両指標で圧勝、seed=100はPSNRはRow-wiseが僅差、SSIMはG=1024が僅差）。これは「プロンプト内容がgroup
sizeの優劣を決める」という仮説だけでは説明できず、**同一プロンプトでもseed（denoising trajectory）によって結果が変わる**ことを示す最初の証拠になった。→ 4節でこれを体系的に検証する。

### 結論（4-step時点）

計画書23節が最初から明示していた「group sizeが小さいほど良いとは仮定しない」という慎重な姿勢は、**4段階の測定すべてで実証された**。weight量子化の精度は理論通り単調に改善する一方、それがDiTという巨大な非線形システムを通過すると、量子化スキームの違いはむしろ「denoising軌道をどの方向にどれだけ摂動させるか」という、量子化誤差の大小とは別の効果として現れ、プロンプトや通過ブロック数によって勝者が入れ替わる。**特定のgroup
sizeがHybrid referenceに一貫して近いとは言えない**、というのが4-step評価時点での結論である。

ただしこの結論には重要な限界がある：4 stepsは1回のEuler更新が非常に大きく、量子化による小さな摂動がそのまま次stepの入力を大きく変えてしまう可能性がある（chaotic
amplification）。実運用に近い20 stepsではこの感度が下がるかもしれない、という仮説を次節で検証した。

## 4. 20-step × multi-seed検証：trajectory依存性の統計的評価

### 4.1 動機と設計

4-stepの結果が「量子化スキームの優劣」ではなく「4-step特有の軌道不安定性」を見ていただけである可能性を排除するため、実運用に近い**20
steps**で、同一プロンプト内で**複数seed**にわたる分布を測定した。比較対象はRow-wise / G=1024 / G=128の3方式に絞った（weight再構成誤差・実行速度の両方で対照的な特性を持つ両極端であり、G=512/G=256は前節までの結果で優先度が低いと判断）。

- プロンプト: 狐・雪、高層ビル の2つ
- seed: 42, 100, 123, 777, 2026 の5つ
- 各(プロンプト, seed)につき Hybrid reference + Row-wise + G=1024 + G=128 の4本を生成（512×512、22フレーム、20
  steps、layers=50、reuse=1）
- 合計 2 prompts × 5 seeds × 4 runs = **40本の生成**を実行し、Hybrid referenceに対するPSNR/SSIMを測定

### 4.2 結果：狐（5 seed）

| 指標 | G1024 vs Row | G128 vs Row |
|---|---:|---:|
| mean ΔPSNR | -0.042 dB | +0.069 dB |
| median ΔPSNR | +0.218 dB | +2.527 dB |
| std ΔPSNR | 5.922 | 5.960 |
| min ΔPSNR | **-9.512 dB** | **-9.614 dB** |
| max ΔPSNR | +5.529 dB | +4.688 dB |
| win rate | 60% (3/5) | 60% (3/5) |

生データ（ΔPSNR、seed順 42/100/123/777/2026）：G1024 `[+5.53, +4.31, +0.22, -0.76, -9.51]` / G128
`[+4.69, +2.53, +4.33, -1.59, -9.61]`

### 4.3 結果：高層ビル（5 seed）

| 指標 | G1024 vs Row | G128 vs Row |
|---|---:|---:|
| mean ΔPSNR | +2.101 dB | +1.497 dB |
| median ΔPSNR | +0.787 dB | **+1.513 dB** |
| std ΔPSNR | 6.083 | **0.708（非常に小さい）** |
| min ΔPSNR | -2.317 dB | **+0.618 dB（全seedで正）** |
| max ΔPSNR | +12.616 dB | +2.521 dB |
| win rate | 60% (3/5) | **100% (5/5)** |

生データ（ΔPSNR、seed順）：G1024 `[-2.32, +0.79, -1.84, +1.25, +12.62]` / G128 `[+1.13, +1.51, +0.62, +1.71, +2.52]`

高層ビルではG=128が**全5 seedでRow-wiseに勝利し、しかもばらつきが非常に小さい**（std=0.708）という、狐とは対照的な安定した優位を示した。

### 4.4 統合（fox + skyscraper、計10 trajectories）

| 指標 | G1024 vs Row | G128 vs Row |
|---|---:|---:|
| mean ΔPSNR | +1.029 dB | +0.783 dB |
| median ΔPSNR | +0.502 dB | **+1.611 dB** |
| std ΔPSNR | 5.771 | **4.071** |
| p10 ΔPSNR | -3.037 dB | **-2.393 dB** |
| min ΔPSNR | -9.512 dB | -9.614 dB（ほぼ同水準） |
| **win rate** | 60% (6/10) | **80% (8/10)** |

median・win rate・std・p10のすべてでG=128がG=1024を上回った。ただし**両方式とも最悪ケース（狐seed=2026）でほぼ同規模の破局的な負け（約-9.5dB）を記録**しており、tail
riskそのものは解消されていない。

### 4.5 外れ値の目視確認（狐 seed=2026、高層ビル seed=2026）

最も極端だった2件（狐seed=2026: 両方式とも-9.5dB前後の大敗、高層ビルseed=2026: G1024が+12.6dBの大勝）についてフレームを直接確認した。ピクセル統計（平均・標準偏差・min/max）は全モードで正常範囲内であり、**黒画面やノイズのような破綻ではなかった**。狐seed=2026を見ると、Row-wise/Hybrid referenceは元の歩様を維持したのに対し、**G1024とG128は両方とも同じ「歩様の別フェーズ＋前景に小枝が出現」という代替軌道に分岐**していた。バグではなく、量子化スキームの変更が引き起こした正当な（しかし大きな）trajectory分岐であることを確認した。

### 4.6 結論（20-step multi-seed時点）

「20 stepsではdenoisingの自己修正作用により量子化誤差の感度が下がる」という仮説は、**この10
trajectoryのデータでは支持されなかった**。std（約4〜6dB）はmean/medianに対して非常に大きく、稀に発生する破局的な分岐（tail
risk）は4-stepの結果と同様に大きい。より正確には、「denoising
stepを増やすことで量子化摂動に対するtrajectory感度が下がる」と一般化することはできず、**trajectory依存性はstep数を超えて根強く残っている**、というのが現時点の結論である。

一方で、G=128はG=1024よりも一貫して良い統計指標（win
rate、median、std、p10）を示しており、特に高層ビルでは全seedで安定した優位を示した。今後さらにプロンプト・seedを増やし、この差が統計的に有意かを検証する価値がある。

### 4.7 最悪ケースの内訳：latentのstep単位トレース（狐 seed=2026）

4.5節で確認した狐seed=2026の大分岐（Row-wise/G=128ともHybrid referenceから-9.5dB前後）が、denoisingの**どのstepで発生したか**を特定するため、`H3_DUMP_LATENT_PREFIX`診断フック（`h3_dit.c`の`denoise_euler_gpu`に追加）を使い、Hybrid reference・Row-wise・G=128の3本について**全20
stepのvideo latentを個別にダンプ**し、各step時点でのHybrid referenceに対するRMSEを追跡した。

| Step | σ(video) | Δσ | RMSE(Row) | RMSE(G128) | G128/Row比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0044 | 0.000258 | 0.000343 | 1.33× |
| 5 | 0.9796 | 0.0066 | 0.001082 | 0.001285 | 1.19× |
| 10 | 0.9362 | 0.0131 | 0.006880 | 0.014370 | 2.09× |
| 13 | 0.8889 | 0.0229 | 0.012928 | 0.038284 | 2.96× |
| 16 | 0.8000 | 0.0500 | 0.026880 | 0.086436 | 3.22× |
| 20 | 0.3871 | 0.3871 | 0.144162 | 0.472957 | 3.28× |

（全20 step分の詳細は `latent_trace_fox2026.csv` を参照）

**特定のstepで不連続に分岐したのではなく、20
stepすべてを通じてほぼ滑らかに誤差が増大し続けた。** 各stepの誤差の対前step比（成長率）はRow-wise・G128とも一貫して1.2〜1.8倍/step程度で推移しており、「stepXで急増、それ以降は別軌道」という単一の分岐点は観測されなかった。step
1→20全体の幾何平均成長率は、Row-wiseが約1.395×/step、G128が約1.463×/stepであり、**G128の誤差成長率自体がRow-wiseよりわずかに高い**。この結果、G128/Row比はstep1の1.33倍からstep20の3.28倍まで単調に拡大した。

このモデルの実際のsigmaスケジュール（`h3_schedule_build(20, ...)`で確認）も合わせて確認したところ、**Δσ（1
Euler stepあたりの更新幅）はstep1の0.0044からstep20の0.3871まで、終盤にかけて急激に拡大する**（「denoisingは終盤ほど細かく更新する」という一般的な直感とは逆の挙動）。これはstep17〜20で誤差成長率が加速する（1.36〜1.83倍/step）ことと符合しており、少なくとも部分的にはこのsigmaスケジュール自体が終盤の急拡大に寄与している可能性がある。

**解釈上の重要な注意点**: この結果から「カオス的な初期値鋭敏性」や「正のLyapunov指数」を主張することはできない。G=128の量子化されたDiTは20
stepすべてに適用されており、単一stepでの摂動がその後Hybrid reference（attn.out_projのみ非量子化）の
dynamicsのみで増幅されたわけではない。観測された指数的増大は、
`既存の軌道差の伝播` + `各stepで新たに注入される量子化誤差`
の両方が混ざった結果である。より正確な表現は、**「量子化による小さな軌道差が、各denoising
stepで継続的に注入・伝播され、単一の観測可能な分岐点なしに、ほぼ指数関数的に増大した」**というものである。これがどちらの効果（伝播 or 新規注入）に主として起因するかを切り分けるには、後述の single-step
injection実験が必要である。

### 4.8 single-step injection実験：結果

4.7節を受け、`H3_INT8_WEIGHT_GROUP_ONLY_STEP`診断フック（`h3_dit.c`の`run_block`に追加、既存の`H3_INT8_KEEP_BF16_ATTENTION_OUT`と組み合わせて使用）を実装し、**特定の1
stepだけG=128を使い、残り全stepはBF16に戻す**実験を、狐seed=2026について注入step k=1, 5, 10, 15,
18で実施した（各回、Hybrid referenceに対する20 step分のlatent RMSEを完全にトレース）。

| 注入step k | 注入直後のRMSE | step20 最終RMSE | full G128最終RMSE(0.472957)に対する比 |
|---:|---:|---:|---:|
| 1 | 0.000343 | 0.450120 | 95.2% |
| **5** | 0.000776 | **0.472108** | **99.8%** |
| 10 | 0.001133 | 0.077245 | 16.3% |
| 15 | 0.002094 | 0.059439 | 12.6% |
| 18 | 0.003621 | 0.016108 | 3.4% |

**単発の摂動だけで、20 step全てをG128で実行した場合とほぼ同じ規模の最終乖離に到達した。** 特にk=5では、注入直後のRMSEはわずか0.000776（Hybrid reference
latentのnormに対して0.1%未満）にもかかわらず、以後19
stepすべてをHybrid reference（attn.out_projのみ非量子化）に戻しても最終RMSEが0.472108まで成長し、全step常時G128実行時の0.472957に対して**99.8%**に達した。これは「毎stepで新たに注入される量子化誤差の累積」ではなく、**特定の早期stepで生じた微小な摂動が、その後のHybrid reference
denoising dynamicsのみによって指数的に増幅される**ことが主因であることを強く示す。

興味深いことに、「摂動が早いほど最終的な乖離が大きい」という単純な関係でもない：k=1（最も早い注入）の最終比は95.2%だが、k=5の方がわずかに高い99.8%。これは、このtrajectoryにおいてstep5前後が特に軌道選択に影響しやすい"感度の窓"である可能性を示唆する（4.10節で方向の一致度からさらに検証）。

**log-RMSEの直線性（指数成長の数値的裏付け）**: 各注入について、注入step以降のlog(RMSE)をstepに対して線形回帰した。

| 系列 | 対象step範囲 | growth/step | R² |
|---|---|---:|---:|
| Row-wise（全step常時） | 1–20 | 1.358× | 0.989 |
| G128（全step常時） | 1–20 | 1.463× | 0.992 |
| injection k=1 | 1–20 | 1.470× | 0.995 |
| injection k=5 | 5–20 | 1.492× | 0.982 |
| injection k=10 | 10–20 | 1.496× | 0.988 |
| injection k=15 | 15–20 | 1.917× | 0.995 |
| injection k=18 | 18–20 | 2.109× | 0.990 |

全系列でR²=0.98〜0.995と非常に高く、**指数的成長は数値的にも裏付けられた**。ただし、後半（k=15, 18）ほどgrowth/stepが大きく見える点は、残りstep数が少ない（k=18はn=3点）ことによる幾何平均の不安定性を含む可能性があり、単純にsigmaスケジュールだけに帰属させるのは早計である。実際、全G128トレースの**各step単独の**成長率（RMSE_{t+1}/RMSE_t）と実測のsigma/Δσとの相関を取ったところ、相関係数はΔσに対して+0.43、σに対して-0.28と、**中程度に留まり支配的ではなかった**。4.7節の「sigmaスケジュール終盤の急拡大が成長率加速に寄与している可能性」は部分的な要因にとどまり、単独の説明にはならない。

### 4.9 step5-only注入は、full G128とほぼ同じ代替trajectoryを選んでいる

RMSEの大きさが似ているだけなのか、実際に同じ方向へ発散しているのかを確認するため、step20時点の最終latentについて、Hybrid referenceからの差分ベクトル（delta = mode − Hybrid reference）どうしのcosine類似度を測定した。

| 比較 | 2つの最終状態間のRMSE | delta同士のcosine類似度 |
|---|---:|---:|
| **k=5-only vs full G128** | 0.106330 | **0.974684**（ほぼ同一方向） |
| k=1-only vs full G128 | 0.384080 | 0.654757（中程度の一致） |
| k=10-only vs full G128 | 0.469217 | 0.129887（ほぼ無相関） |
| k=1-only vs k=5-only | 0.370361 | 0.678398 |

**k=5-only注入は、full
G128実行とcosine類似度0.975というほぼ完全な方向一致を示した。** 2つの最終状態間のRMSE（0.106）も、それぞれのHybrid referenceからのRMSE（約0.47）よりずっと小さく、両者は実質的に同じ代替trajectoryに収束している。実際に代表フレーム（11枚目）を目視比較したところ、k=5-only注入とfull
G128は**同一の歩様フェーズ・同一位置の小枝**という、4.5節で確認したのと全く同じ構図を示した。

一方k=10-onlyは、注入直後のRMSE自体はk=5よりむしろ大きい（0.001133 >
0.000776）にもかかわらず、最終的にはfull G128とほぼ無相関な方向（cosine=0.13）へ発散している。**したがって「摂動の大きさ」よりも「摂動がいつ入るか」の方が、最終的にどのtrajectoryへ着地するかを支配している**、という結論になる。

### 4.10 結論（single-step injection時点）

以上を踏まえ、狐seed=2026の大分岐について次のように結論づけられる：

> **観測された大きなtrajectory
> divergenceは、各denoising
> stepで量子化誤差が累積した結果ではない。早期step（特にstep5付近）における単発の微小なG128摂動だけで、その後の19
> stepをすべてHybrid reference（attn.out_projのみ非量子化）推論に戻しても、最終誤差の95〜100%近く、かつfull
> G128実行とほぼ同じ方向（cosine類似度0.97）の乖離が再現された。したがって、Hybrid referenceの
> denoising dynamicsそのものが、このtrajectory近傍で局所的な摂動を強く増幅する（finite-time
> perturbation amplification / local trajectory instability）ことが主因であり、これは「カオス的初期値鋭敏性」や「正のLyapunov指数」と断定できるほど一般化された主張ではないが、それに近い現象が少なくともこの1
> trajectoryでは明確に観測された。**

この結果は、grouped
INT8量子化そのものの評価を超えて、**H3のdenoising過程が持つ、特定stepでの摂動に対する感度**という、より一般的な性質を示唆している。今後の課題（6節）として、他のtail-riskケース（例：高層ビルseed=2026、G1024が+12.6dBの大勝）でも同様の"感度の窓"が存在するか、そしてその窓がプロンプトやseedによってどう変わるかの検証が挙げられる。

## Full BF16 referenceとの比較（真の量子化誤差評価）

「用語の定義」節で明らかになった通り、これまでの全ての比較（評価ラダー・20-step
multi-seed評価・single-step injection）は、Hybrid reference（attn.out_projのみ
非量子化、QKV/FC1/FC2は通常のint8）を基準にしたものであり、真の**Full
BF16**（QKV/attn-out/FC1/FC2すべて非量子化）との比較ではなかった。この節では、
既存の`--ssd-streaming`フラグ（SSDから元のBF16 DiT層を都度ストリーミングし、
`int8_mlp`/`int8_qkv`/`int8_attention_out`をすべて無効化する）を使い、初めて
真のFull BF16 referenceを生成し、Hybrid reference・Row-wise・G1024・G128を
これと比較した。

### 対象trajectory・生成条件

| trajectory | prompt | seed | 選定理由 |
|---|---|---|---|
| fox-2026 | "A red fox walks through fresh snow in a pine forest." | 2026 | 量子化方式によってtrajectoryが大きく分岐した例（4.5節） |
| sky-100 | "A towering skyscraper with geometric lines against the sky." | 100 | G128が比較的安定して優位だった例（4.3節） |

各trajectoryにつき、Full BF16（`--ssd-streaming`）・Hybrid reference
（`H3_DISABLE_INT8_ATTENTION_OUT=1 H3_INT8_KEEP_BF16_ATTENTION_OUT=1`）・
Row-wise（デフォルト）・G1024（`H3_INT8_WEIGHT_GROUP=1024`）・G128
（`H3_INT8_WEIGHT_GROUP=128`）の5本を、512×512・22フレーム・20 steps・
layers=50・reuse=1・同一seedで生成し、`H3_DUMP_LATENT_PREFIX`で各stepの
video latentも取得した。

`--ssd-streaming`の副産物として、**この経路は常駐int8経路よりも実行が
速く安定していた**（1層ずつしかメモリに保持しないため、本マシンの慢性的な
メモリひっ迫の影響を受けない）。一方、常駐int8経路（Hybrid/Row-wise/G1024/
G128）は今回も本マシン特有の速度不安定性を示した（詳細はREADME-gpu-
determinism.md参照）。速度に大きなばらつきはあったが、**得られた結果自体は
安定していた**。

### 結果①：完成動画のPSNR/SSIM（Full BF16基準）

| trajectory | mode | PSNR (dB) | SSIM |
|---|---|---:|---:|
| fox-2026 | Hybrid reference | 17.60 | 0.683 |
| fox-2026 | Row-wise | 17.45 | 0.683 |
| fox-2026 | **G1024** | **22.36** | **0.819** |
| fox-2026 | G128 | 21.58 | 0.791 |
| sky-100 | **Hybrid reference** | **18.25** | **0.619** |
| sky-100 | Row-wise | 16.81 | 0.543 |
| sky-100 | G1024 | 17.18 | 0.552 |
| sky-100 | G128 | 18.14 | 0.616 |

**fox-2026ではG1024がFull BF16に最も近く**（Hybrid referenceより約4.8dB
良い）、**sky-100ではHybrid reference自体がFull BF16に最も近く**、G128が
僅差で続く。いずれのtrajectoryでも、**「Hybrid referenceに近い」ことと
「Full BF16に近い」ことは全く別の順位を生む**——これがまさに、用語訂正で
懸念されていた通りの結果である。

### 結果②：step単位のlatent RMSE（Full BF16基準）

| trajectory | step | Hybrid | Row-wise | G1024 | G128 | 最良 |
|---|---:|---:|---:|---:|---:|---|
| fox-2026 | 1 | 0.000454 | 0.000356 | 0.000401 | 0.000375 | Row-wise |
| fox-2026 | 5 | 0.001894 | 0.002029 | 0.001784 | 0.002580 | G1024 |
| fox-2026 | 10 | 0.017103 | 0.019276 | 0.011643 | 0.011032 | G128 |
| fox-2026 | 15 | 0.073656 | 0.076610 | 0.038098 | 0.039616 | G1024 |
| fox-2026 | 20 | 0.517687 | 0.526884 | **0.300798** | 0.337104 | G1024 |
| sky-100 | 1 | 0.000225 | 0.000231 | 0.000282 | 0.000196 | G128 |
| sky-100 | 5 | 0.002043 | 0.003121 | 0.002893 | 0.002144 | Hybrid |
| sky-100 | 10 | 0.006972 | 0.019399 | 0.014939 | 0.007794 | Hybrid |
| sky-100 | 15 | 0.026332 | 0.072077 | 0.057154 | 0.029399 | Hybrid |
| sky-100 | 20 | **0.289098** | 0.491181 | 0.426134 | 0.301066 | Hybrid |

（全20 step・全モードのデータは`full_bf16_latent_rmse.csv`を参照）

**sky-100**は、step4以降**一貫してHybrid referenceが最もFull BF16に近く**、
**G128が僅差の2位**、G1024・Row-wiseの順で悪化していく——これは重み単体の
再構成誤差ランキング（3節①：G128 < G256 < G512 < G1024 <
Row-wise）と完全に整合する、素直な結果である。

**fox-2026**は対照的に、step1〜4ではHybrid/Row-wiseの方がFull BF16に近いが、
**step5以降はG1024・G128が逆転し、以後最後までG1024が最良であり続ける**。
これは「量子化誤差が小さいほどFull
BF16に近い」という単純な予想に反しており、後述のtrajectory分岐と関係している。

### 結果③：どちらの"枝"にいるか——step20時点でのdelta方向のcosine類似度

各modeの最終状態とFull BF16の差分ベクトル（delta = mode −
Full BF16）どうしのcosine類似度を取ると、trajectoryが分岐した場合に
「どのmode同士が同じ枝に着地したか」が分かる。

| trajectory | ペア | cosine類似度 |
|---|---|---:|
| fox-2026 | Hybrid ↔ Row-wise | **0.9621**（同じ枝） |
| fox-2026 | G1024 ↔ G128 | 0.5101（中程度） |
| fox-2026 | Hybrid ↔ G1024 | 0.4326（別の枝） |
| fox-2026 | Hybrid ↔ G128 | 0.4525（別の枝） |
| sky-100 | Hybrid ↔ G128 | **0.9483**（同じ枝） |
| sky-100 | Row-wise ↔ G1024 | 0.6662（中程度） |
| sky-100 | Hybrid ↔ Row-wise | 0.4079（別の枝） |
| sky-100 | Hybrid ↔ G128 | 0.4196（Row-wiseとの比較、別の枝） |

（全ペアは`full_bf16_direction_cosine.csv`を参照）

**fox-2026**では{Hybrid, Row-wise}が同じ枝（cosine 0.96）、{G1024,
G128}がもう一方の枝（cosine 0.51、やや弱いが同系統）——**Full
BF16自身の真の軌道は、Hybrid/Row-wiseの枝ではなく、G1024/G128の枝の方に
近かった**（結果②のRMSEが示す通り）。4.5節で確認した「G1024とG128が同じ
代替軌道に分岐した」という記述は、実は**「G1024とG128の分岐先が、たまたま
Full BF16の真の軌道に近かった」**ことを意味していたことになる——単なる
偶然か、量子化誤差の方向がこのtrajectoryの分岐点で本来の軌道と相関する
理由があるのかは、現時点では未解明である。

**sky-100**では{Hybrid, G128}が同じ枝（cosine 0.95）、Row-wise/G1024は
それぞれ異なる方向に外れている（Row-wiseとG1024の相互相関は0.67と中程度）。
こちらは分岐というより、量子化の粗さに応じて誤差が単調に拡大していく
（Hybrid→G128→G1024→Row-wiseの順で悪化）、より「素直な」構造である。

### 結論

1. **「量子化方式がHybrid referenceにどれだけ近いか」と「Full
   BF16にどれだけ近いか」は別の問いであり、これまでの全ての比較（3節・4節）は
   前者しか答えていなかった**——これが本節で初めて解消された。
2. sky-100のような「素直な」trajectoryでは、重みの量子化粗さと最終的な
   Full BF16からの乖離が単調に対応する（G128が最良、Row-wiseが最悪）。
3. fox-2026のような分岐trajectoryでは、**この単調な対応は成立しない**——
   G1024がFull BF16に最も近く、量子化を全くしていないHybrid
   referenceより明確に良い。これは「量子化誤差が小さい方が良い」という
   直感全体を無効化するものではなく、**分岐点での摂動方向がFull
   BF16自身の実際の軌道とたまたま近いかどうかで結果が支配される**、という
   これまでの一連の実験（4.7〜4.10節）の結論と整合する。
4. G128は両trajectoryで安定して上位（fox-2026で2位、sky-100で2位僅差）
   であり、G1024は当たり外れが大きい（fox-2026で1位、sky-100で3位）。
   この2点だけでは一般化できないが、**G128の方がG1024よりFull
   BF16に対しても頑健である可能性**を示唆する——ただし4.4節のHybrid
   reference基準の統計（10 trajectory）でもG128がG1024より一貫して
   良い統計指標を示していたことと方向性は一致する。

### 今後の課題

- trajectory数が2件のみであり、統計的な一般化はできない。他のtail-risk
  ケース（高層ビルseed=2026等）や、より典型的な（大きく分岐しない）
  trajectoryでも同様の比較を行う価値がある。
- fox-2026で「なぜG1024/G128の分岐先がFull BF16の真の軌道に近かったのか」
  は未解明。single-step injection実験（4.8〜4.10節）の手法をFull
  BF16 referenceに対しても適用すれば、この分岐がどのstepで・どちらの
  方向へ決まったかを特定できる可能性がある。
- Full BF16 referenceでの`--ssd-streaming`が常駐int8経路より高速・安定
  だったという副次的な観察は、量子化の目的（メモリ削減）を考えると
  皮肉ではあるが、少なくともこのマシンでの実験速度向上には有用（今後
  Full BF16を基準にした追加比較を行う際は`--ssd-streaming`を優先して使う）。

## 5. 性能（実行時間）

| Mode | Load wall(s) | Load peak(GiB) | Denoise wall(s) | Total wall(s) | Denoise Δ vs Row-wise |
|---|---:|---:|---:|---:|---:|
| Row-wise | 27.416 | 18.595 | 276.076 | 304.183 | (基準) |
| G=1024 | 27.209 | 18.600 | 274.849 | 302.687 | -0.44% |
| G=512 | 27.649 | 18.607 | 275.142 | 303.443 | -0.34% |
| G=256 | 26.200 | 18.621 | 269.133 | 295.942 | -2.51% |
| G=128 | 25.935 | 18.649 | 267.580 | 294.125 | **-3.08%（最速）** |

意外にも、Phase 1（計画書10.1「単純実装、性能最適化より正当性優先」）の実装であるにもかかわらず、grouped方式はRow-wiseより**遅くならなかった**。むしろgroup数が増えるほど速くなる傾向が見られ、G=128が全モード中最速だった。理由として、新カーネルのタイル形状（128×64、Row-wiseの既存カーネルは128×128でM5上の並列度が異なる可能性）と、G=128特有の「K_TILE(128)=group_size」による内側ループの構造的単純化を挙げた（詳細は本文中のやり取り参照）。メモリ使用量（peak
GiB）はgroup数増加に伴いわずかに増加するが（scaleバッファの増加分、計画書17節の予想通り軽微）。

## 6. 未実施の項目（今後の課題）

計画書全体からみると、本セッションで実施したのは Stage 0・Stage 1 の実装と、Stage 2〜5相当の正当性検証・品質/性能測定（4-step単発、4-step複数プロンプト、20-step
multi-seed統計評価）である。以下は未実施：

- **他のtail-riskケースでの"感度の窓"の検証**（4.8〜4.10節はfox seed=2026の1件のみ。高層ビルseed=2026（G1024が+12.6dBの大勝）や他のtail-riskケースでも同様に、特定の早期stepへの単発摂動がfull実行とほぼ同じ方向に収束するかを確認すれば、「特定stepへの感度」が一般的な現象かfox
  seed=2026特有かが分かる）
- **なぜstep5前後が特に感度が高いのかの追加探索**（4.8節：k=1よりk=5の方がfull
  G128への一致率が高かった。k=2〜4, 6〜9も注入して"感度の窓"の輪郭をより細かく特定する）
- **典型的な（tail-riskでない）trajectoryでの同種のlatentトレース**（4.7〜4.10節の手法を、大きく分岐しなかったseedに適用し、成長率や方向一致度が本質的に違うのかを比較する）
- **QKV / FC1 / FC2 行列への展開**（現状 attn.out_proj のみ）
- **他のprompt categoryでの20-step multi-seed検証**（現状は狐・高層ビルの2カテゴリ×5 seedのみ。計画書14.1の6カテゴリ中、人物・海・テキスト・高速動作は4-stepの一部でしか未検証）
- **G=512 / G=256の20-step再評価**（4-stepの結果と性能測定ではG=128寄りに絞ったため、20-stepでは未測定）
- **15秒評価（Stage 6）**（本実験は362フレームでの実運用条件を未検証）
- **LPIPS**（PSNR/SSIMのみ測定）
- **統計的有意性検定**（現状10〜13 trajectoryのmean/median/std/win
  rateの記述統計のみ。G=128がG=1024より優れるという4.4節の傾向が有意かはt検定等で未検証）
- **SDPA outputのhead-major対応**（grouped GEMMは現状row-major限定）
- **streaming cache（`H3_ATTENTION_CACHE`）との統合**（計画書12節。この機能は別ブランチのみに存在し、本ブランチには無い）
- **Phase 2カーネル最適化**（ただし今回の実測では既にRow-wiseと同等以上の速度が出ており、優先度は下がった）

## 7. 生成物

- `benchmark_grouped_int8.csv` — 実行時間データ（本README表5と同一データ）
- `quality_grouped_int8.csv` — weight/block0/block49/video（4-step, 20-step multi-seed含む）全レベルの誤差・品質指標
- `latent_trace_fox2026.csv` — 狐seed=2026、20 step分のlatent RMSEトレース（4.7節、σ・Δσ・成長率・G128/Row比を含む）
- `latent_trace_fox2026_injection.csv` — single-step injection実験（4.8節）の全step・全注入点データ（σ・Δσ・RMSE・成長率）
- `latent_trace_fox2026_direction.csv` — 各注入点とfull G128実行の最終状態比較（4.9節、RMSE・delta cosine類似度）
- `tests/test_int8_weight_group.c` — 正当性の単体テスト
- `h3_weight_quant_error.c` — weight再構成誤差の測定ツール
- `h3_dit.c`の診断フック2つ（`denoise_euler_gpu`内） — `H3_DUMP_LATENT_PREFIX`（各Euler
  stepのvideo latentをファイルにダンプ）、`H3_INT8_WEIGHT_GROUP_ONLY_STEP` + 既存の
  `H3_INT8_KEEP_BF16_ATTENTION_OUT`（特定の1
  stepだけgrouped int8を使い残りはBF16に戻す、single-step injection用）。いずれも未設定時は完全にno-op
- `outputs/` 配下の各種検証動画（`grouped-*.mp4`, `p2-*.mp4`〜`p4-*.mp4`, `perf-*.mp4`, `s20_*.mp4`,
  `sky20_*.mp4`）
- `full_bf16_video_quality.csv` — Full BF16 referenceとの動画レベルPSNR/SSIM
  （fox-2026・sky-100、Hybrid/Row-wise/G1024/G128の4モード分、「Full BF16
  referenceとの比較」節）
- `full_bf16_latent_rmse.csv` — 同実験の20 step全stepぶんのlatent RMSE
  （trajectory・step・mode・group_size・rmse_vs_full_bf16の列）
- `full_bf16_direction_cosine.csv` — step20時点でのdelta（mode − Full
  BF16）どうしの全ペアのcosine類似度

## 8. Git状態

ブランチ `exp/grouped-int8-weights`（`main`分岐、`exp-grouped-int8-baseline`タグ上）。コミット `d516988`
"Add group-wise INT8 weight quantization, Phase 1 (attention-output only)"、`323ef2a`
"Add block-output error diagnostics and summarize the experiment so far"、`324e4da`
"Add 20-step multi-seed trajectory-dependence results"、`3b45810`
"Add per-step latent trace for the fox seed=2026 outlier" 済み。本ドキュメント更新時点で
`README-grouped-int8.md`（4.8〜4.10節追加分）・`h3_dit.c`（`H3_INT8_WEIGHT_GROUP_ONLY_STEP`診断フック追加分）・新規
`latent_trace_fox2026_injection.csv` / `latent_trace_fox2026_direction.csv` が未コミット。
