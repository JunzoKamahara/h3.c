# Group-wise INT8 Weight Quantization — 実験まとめ

`exp/grouped-int8-weights`ブランチ（`main`から分岐、`exp-grouped-int8-baseline`タグの上）で実施した、DiT
weightのgroup-wise INT8量子化実験の中間まとめ。元の実験計画書「h3.c DiT Weight Group-wise INT8
Quantization 実験計画書」の Stage 0〜2 相当、および計画書に無かった追加の誤差伝播分析（評価ラダー①〜④）を実施した。

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

計画書には無かった追加分析として、「量子化誤差の小ささ」と「最終出力のBF16への近さ」が本当に対応するのかを、4段階に分けて切り分けた（すべて同一プロンプト "A red fox walks through fresh snow in a pine
forest."、seed=42、512×512、attn.out_proj projectionのみ変更）。

| 段階 | 内容 | RMSEによる順位（良い→悪い） |
|---|---|---|
| ①Weight単体（block 0/9/24/39/49平均） | BF16重みを量子化→逆量子化した値とBF16原本を比較 | **G128 < G256 < G512 < G1024 < Row-wise**（全ブロックで完全に単調） |
| ②Block 0通過後 | 1ブロック分のattention+MLP+residualを通した後の隠れ状態を比較 | G1024 < G128 < G256 < G512 < Row-wise（grouped 4方式全てがrow-wiseに勝つが、内部順序は崩壊） |
| ③Block 49通過後（50ブロック、denoising 1step） | 全50ブロックを通した後の隠れ状態を比較 | G512 < G128 ≈ G1024 < Row-wise < **G256**（grouped方式の中でG256が最下位に転落） |
| ④動画（4 steps全体） | 完成した動画フレームをBF16基準とPSNR/SSIMで比較 | プロンプト依存（下記参照） |

**誤差の相対的な大きさ（RelFrobErr）も段階を追うごとに急増**した：block0で約0.0045〜0.0048だったものが、block49では約0.054〜0.062（**約12倍**）に拡大。これは50層の非線形変換（attention/MLP/residual）を繰り返すことによる実質的な誤差増幅を示している。

### ④ 動画レベル（PSNR/SSIM、BF16基準、4 steps）

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
sizeがBF16に一貫して近いとは言えない**、というのが4-step評価時点での結論である。

ただしこの結論には重要な限界がある：4 stepsは1回のEuler更新が非常に大きく、量子化による小さな摂動がそのまま次stepの入力を大きく変えてしまう可能性がある（chaotic
amplification）。実運用に近い20 stepsではこの感度が下がるかもしれない、という仮説を次節で検証した。

## 4. 20-step × multi-seed検証：trajectory依存性の統計的評価

### 4.1 動機と設計

4-stepの結果が「量子化スキームの優劣」ではなく「4-step特有の軌道不安定性」を見ていただけである可能性を排除するため、実運用に近い**20
steps**で、同一プロンプト内で**複数seed**にわたる分布を測定した。比較対象はRow-wise / G=1024 / G=128の3方式に絞った（weight再構成誤差・実行速度の両方で対照的な特性を持つ両極端であり、G=512/G=256は前節までの結果で優先度が低いと判断）。

- プロンプト: 狐・雪、高層ビル の2つ
- seed: 42, 100, 123, 777, 2026 の5つ
- 各(プロンプト, seed)につき BF16基準 + Row-wise + G=1024 + G=128 の4本を生成（512×512、22フレーム、20
  steps、layers=50、reuse=1）
- 合計 2 prompts × 5 seeds × 4 runs = **40本の生成**を実行し、BF16基準に対するPSNR/SSIMを測定

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

最も極端だった2件（狐seed=2026: 両方式とも-9.5dB前後の大敗、高層ビルseed=2026: G1024が+12.6dBの大勝）についてフレームを直接確認した。ピクセル統計（平均・標準偏差・min/max）は全モードで正常範囲内であり、**黒画面やノイズのような破綻ではなかった**。狐seed=2026を見ると、Row-wise/BF16は元の歩様を維持したのに対し、**G1024とG128は両方とも同じ「歩様の別フェーズ＋前景に小枝が出現」という代替軌道に分岐**していた。バグではなく、量子化スキームの変更が引き起こした正当な（しかし大きな）trajectory分岐であることを確認した。

### 4.6 結論（20-step multi-seed時点）

「20 stepsではdenoisingの自己修正作用により量子化誤差の感度が下がる」という仮説は、**この10
trajectoryのデータでは支持されなかった**。std（約4〜6dB）はmean/medianに対して非常に大きく、稀に発生する破局的な分岐（tail
risk）は4-stepの結果と同様に大きい。より正確には、「denoising
stepを増やすことで量子化摂動に対するtrajectory感度が下がる」と一般化することはできず、**trajectory依存性はstep数を超えて根強く残っている**、というのが現時点の結論である。

一方で、G=128はG=1024よりも一貫して良い統計指標（win
rate、median、std、p10）を示しており、特に高層ビルでは全seedで安定した優位を示した。今後さらにプロンプト・seedを増やし、この差が統計的に有意かを検証する価値がある。

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
- `tests/test_int8_weight_group.c` — 正当性の単体テスト
- `h3_weight_quant_error.c` — weight再構成誤差の測定ツール
- `outputs/` 配下の各種検証動画（`grouped-*.mp4`, `p2-*.mp4`〜`p4-*.mp4`, `perf-*.mp4`, `s20_*.mp4`,
  `sky20_*.mp4`）

## 8. Git状態

ブランチ `exp/grouped-int8-weights`（`main`分岐、`exp-grouped-int8-baseline`タグ上）。コミット `d516988`
"Add group-wise INT8 weight quantization, Phase 1 (attention-output only)" と `323ef2a`
"Add block-output error diagnostics and summarize the experiment so far" 済み。本ドキュメント更新時点で
`README-grouped-int8.md` / `quality_grouped_int8.csv`（20-step multi-seedデータ反映分）が未コミット。
