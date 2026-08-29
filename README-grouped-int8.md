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

### ④ 動画レベル（PSNR/SSIM、BF16基準）

| プロンプト | 勝者(PSNR) | 勝者(SSIM) |
|---|---|---|
| 狐・雪(毛皮のテクスチャ) | G=1024 | G=1024 |
| 高層ビル(直線・幾何学) | Row-wise | Row-wise |

同一のG=1024設定でも、プロンプトが変わるだけで勝敗が完全に逆転した。

### 結論

計画書23節が最初から明示していた「group sizeが小さいほど良いとは仮定しない」という慎重な姿勢は、**4段階の測定すべてで実証された**。weight量子化の精度は理論通り単調に改善する一方、それがDiTという巨大な非線形システムを通過すると、量子化スキームの違いはむしろ「denoising軌道をどの方向にどれだけ摂動させるか」という、量子化誤差の大小とは別の効果として現れ、プロンプトや通過ブロック数によって勝者が入れ替わる。**特定のgroup
sizeがBF16に一貫して近いとは言えない**、というのが本実験時点での結論である。

## 4. 性能（実行時間）

| Mode | Load wall(s) | Load peak(GiB) | Denoise wall(s) | Total wall(s) | Denoise Δ vs Row-wise |
|---|---:|---:|---:|---:|---:|
| Row-wise | 27.416 | 18.595 | 276.076 | 304.183 | (基準) |
| G=1024 | 27.209 | 18.600 | 274.849 | 302.687 | -0.44% |
| G=512 | 27.649 | 18.607 | 275.142 | 303.443 | -0.34% |
| G=256 | 26.200 | 18.621 | 269.133 | 295.942 | -2.51% |
| G=128 | 25.935 | 18.649 | 267.580 | 294.125 | **-3.08%（最速）** |

意外にも、Phase 1（計画書10.1「単純実装、性能最適化より正当性優先」）の実装であるにもかかわらず、grouped方式はRow-wiseより**遅くならなかった**。むしろgroup数が増えるほど速くなる傾向が見られ、G=128が全モード中最速だった。理由として、新カーネルのタイル形状（128×64、Row-wiseの既存カーネルは128×128でM5上の並列度が異なる可能性）と、G=128特有の「K_TILE(128)=group_size」による内側ループの構造的単純化を挙げた（詳細は本文中のやり取り参照）。メモリ使用量（peak
GiB）はgroup数増加に伴いわずかに増加するが（scaleバッファの増加分、計画書17節の予想通り軽微）。

## 5. 未実施の項目（今後の課題）

計画書全体からみると、本セッションで実施したのは Stage 0・Stage 1 の実装と、Stage 2 の一部（正当性検証・予備的な品質/性能測定）に相当する。以下は未実施：

- **QKV / FC1 / FC2 行列への展開**（現状 attn.out_proj のみ）
- **他のprompt category**（計画書14.1の6カテゴリ中、実施したのは「動物・毛」「建築物・直線」の2つのみ）
- **20-step評価（Stage 5）・15秒評価（Stage 6）**（本実験は一貫して4stepsのみ。ステップ数を増やすとchaotic
  amplificationの影響が変わる可能性があり、傾向が変わりうる）
- **LPIPS**（PSNR/SSIMのみ測定）
- **SDPA outputのhead-major対応**（grouped GEMMは現状row-major限定）
- **streaming cache（`H3_ATTENTION_CACHE`）との統合**（計画書12節。この機能は別ブランチのみに存在し、本ブランチには無い）
- **Phase 2カーネル最適化**（ただし今回の実測では既にRow-wiseと同等以上の速度が出ており、優先度は下がった）

## 6. 生成物

- `benchmark_grouped_int8.csv` — 実行時間データ（本README表4と同一データ）
- `quality_grouped_int8.csv` — weight/block0/block49/video 全レベルの誤差・品質指標
- `tests/test_int8_weight_group.c` — 正当性の単体テスト
- `h3_weight_quant_error.c` — weight再構成誤差の測定ツール
- `outputs/` 配下の各種検証動画（`grouped-*.mp4`, `p2-*.mp4`, `perf-*.mp4`）

## 7. Git状態

ブランチ `exp/grouped-int8-weights`（`main`分岐、`exp-grouped-int8-baseline`タグ上）。コミット `d516988`
"Add group-wise INT8 weight quantization, Phase 1 (attention-output only)" 済み。本ドキュメント作成時点で
`h3_dit.c`（診断フック追加分）・`Makefile`・`h3_weight_quant_error.c`（新規）が未コミット。
