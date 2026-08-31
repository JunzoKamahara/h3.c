# GPU数値非決定性とDenoising軌道感度の切り分け実験

ブランチ: `exp/gpu-determinism-probe`（`exp/grouped-int8-weights` のコミット `b34e8ae`
"Add H3_INJECT_LATENT_* for controlled perturbation-scaling injection" から分岐）。

計画書: `H3.c - M5環境におけるGPU数値非決定性とDenoising軌道感度の切り分け実験計画書.md`
（ユーザー提供、`~/Downloads/`）。以下、計画書の章番号・Phase名で参照する。

## 背景（計画書2章の要約）

`exp/grouped-int8-weights` での20-step multi-seed検証・single-step
injection実験の過程で、**量子化を一切使わない2回の純粋BF16再実行**（fox / seed=2026 /
20 steps）が bit-identical にならないことが判明した：

| step | BF16 vs BF16再実行 RMSE | full G128 vs BF16 RMSE | 比率 |
|---:|---:|---:|---:|
| 1 | 0.000258 | 0.000343 | 75% |
| 3 | 0.000611 | 0.000603 | **101%（再実行ノイズがG128差を上回る）** |
| 10 | 0.006880 | 0.014370 | 48% |
| 20 | 0.144162 | 0.472957 | 30% |

この「GPU実行系そのもののrun-to-run noise floor」と「量子化による摂動」を分離しないまま
量子化方式を比較するのは統計的に妥当でない、というのが本実験の出発点。

## 実装した診断機構

`h3_dit.c` の `encode_forward()`（DiTの1step分の50 block forward pass）に、環境変数駆動の
record/replay型プローブを追加した（計画書6章「h3_mps_probeまたは既存h3バイナリへの
diagnostic mode」の後者を採用）。

- `H3_PROBE_OP` = `qkv` | `sdpa` | `attn_out` | `block`
- `H3_PROBE_STEP` — 対象の0-basedステップ
- `H3_PROBE_BLOCK` — 対象の0-basedブロック
- `H3_PROBE_REPEAT` — 繰り返し回数（既定100）
- `H3_PROBE_CSV_PATH` — 出力CSV（既定 `mps_op_determinism.csv`）
- `H3_PROBE_CASE` — CSVの `case` 列に付与する自由記述ラベル

`qkv` / `sdpa` / `attn_out` は、対象ブロックの**実際の**`run_block()`呼び出しが完了した
直後に割り込み、そのブロックが実際に使った実データ（実際のweight・実際のAdaLN出力・
実際のSDPA出力）を入力として、**同一のGPU呼び出しだけ**を`H3_PROBE_REPEAT`回再実行する。
呼び出しの分岐（int8 vs BF16、head-major vs row-major）は`run_block()`本体の分岐条件
（`dit->int8_qkv`, `dit->int8_attention_out`, `dit->int8_weight_group`,
`attention_input_quantized`等）をそのまま再現しており、"実際に使われているカーネル"を
確実にテストする（計画書5.2節「実際のH3で使用されるtensor shape・dtype・MPSGraph経路を
維持」の要件）。

`block`は対象ブロックの**実際の**`run_block()`呼び出しの直前に割り込み、その時点の
`dit->hidden`をスナップショットしたうえで、`attention_adaln_ready=0` /
`fuse_next_attention=0`に固定して`run_block()`自体を`H3_PROBE_REPEAT`回、毎回スナップ
ショットから再実行する（計画書11章 Phase B）。

各繰り返しについて、SHA256の代わりに高速な64-bit FNV-1aハッシュ（十分に衝突を検出できる
上、実装コストが低い）でrun0との完全一致を検出しつつ、RMSE / MAE / MaxAbs / cosine /
first-mismatch-indexも算出し、`mps_op_determinism.csv`（計画書32章のスキーマに準拠、
`zero_pad_mode`は今回未実装のため常に`n/a`、`sync_mode`は毎回`h3_gpu_submit()`で強制
同期しているため常に`forced`）へ追記する。

### 実装中に見つけた副次的なバグ2件（記録として残す）

1. **`h3_gpu_begin()`の呼び出し規律**：`run_block()`直後は前段のGPUコマンドが
   まだsubmitされていないため、repeat 0はその未submit状態を再利用できるが、repeat 1以降は
   自分のsubmitでセッションが終了しているため`h3_gpu_begin()`を呼び直す必要がある。
   最終的に「読み出しのたびに明示的に`h3_gpu_submit()`してから次のrepeatで`h3_gpu_begin()`」
   という規律に統一した。
2. **`qkv`の`input_hash`診断列が、block 0以外では無意味な値になっていた**：デフォルト設定
   では`fuse_int8_qkv_input`最適化により、block 1以降のQKV入力は前ブロックの融合
   gate+AdaLN+量子化ステップが書いた`dit->int8_activation`であり、`dit->mod_attention`
   自体は**書き込まれない**（block 0の値が残ったまま）。実際に`H3_DUMP_BLOCK_HIDDEN`
   （既存の検証済み機構）で`dit->hidden`をblock 0直後とblock 48直後で比較したところ
   RMSE=4014（大きく異なる、正常）だったのに対し、`dit->mod_attention`はblock 0と
   block 48で完全に同一バイト列だった。これは実際のQKV演算自体には影響しない
   （`run_block()`と同じ分岐選択によって実際には正しい入力`dit->int8_activation`が
   使われている）が、診断用の`input_hash`列は誤解を招くため、その条件下では
   ハッシュ計算をスキップするよう修正した。

## Phase A + B の結果

### 同一プロセス内repeat（block 0 = "unfused" QKV経路、`attention_input_quantized=0`）

| op | repeat | unique output hash | RMSE (run_i vs run_0) |
|---|---:|---:|---|
| qkv | 100 | 1 | 全て 0.000000000 |
| sdpa | 100 | 1 | 全て 0.000000000 |
| attn_out | 100 | 1 | 全て 0.000000000 |
| block（run_block全体） | 20 | 1 | 全て 0.000000000 |

### 同一プロセス内repeat（block 25 = "fused int8" QKV経路、`attention_input_quantized=1`——全50 blockのうち49個が通る本番のデフォルト経路）

| op | repeat | unique output hash | RMSE (run_i vs run_0) |
|---|---:|---:|---|
| qkv | 50 | 1 | 全て 0.000000000 |
| sdpa | 50 | 1 | 全て 0.000000000 |
| attn_out | 50 | 1 | 全て 0.000000000 |
| block（run_block全体） | 20 | 1 | 全て 0.000000000 |

### fresh process repeat（プロセスを毎回再起動、block 0, qkv）

12回の独立したプロセス起動で、`input_hash` / `output_hash` とも**全て同一**
（`output_hash = 1900045fc35b9a50`が12/12）。

### fresh process repeat（block 49、全50 block通過後、attn_out）

3回の独立したプロセス起動で、`input_hash`（`8e0ddb15c6aca758`）・`output_hash`
（`725876c719831aab`）とも**完全に一致**した。つまり、text
encoder・noise生成・patch投影・50 blockすべてのDiT
forward計算という、1step分の計算パイプライン全体を通しても、fresh
processでbit-identicalな結果が得られている。

## 解釈（計画書33章のObserved / Interpretation / Not establishedの型式で）

**Observed**：
QKV projection、SDPA、attention-out
projectionの各GPU演算は、単体では（同一プロセス内repeat・fresh process
repeatとも）観測した範囲でbit-identicalだった。これはblock
0の"unfused"経路（99個中1個相当）とblock
25の"fused int8"経路（99個中49個相当、本番のデフォルト経路）の両方で確認した。
run_block全体（AdaLN→QKV→RoPE→SDPA→attention-out→gate→MLP→gate、の1
block分の完全なフォワードパス）を同一の入力から繰り返した場合も、block 0・block
25とも20/20回すべてbit-identicalだった。

block 49（全50 blockのうち最後の1つ）でのfresh process
repeatでも、`input_hash`・`output_hash`とも3/3で完全一致した。これは、text
encoder・noise生成・patch投影・50 block分のDiT forward計算という、**1step分の
計算パイプライン全体**がfresh processでもbit-identicalであることを意味する。

**Interpretation**：
この結果は計画書10章のCase Cに強く該当する——**個々のMPSGraph演算、それらを1
blockぶん合成したcomposite実行、さらに全50
blockを通した1step分のフォワードパス全体まで、検証した範囲では完全に決定的**
であり、非決定性の原因を「GPUカーネル内部の非決定的reduction順序」に帰属させる
仮説（計画書3.1節の主仮説）は、少なくとも本実験の範囲では**支持されなかった**。

これにより、当初observedされていたBF16 vs BF16再実行の乖離（本README冒頭の表）の
入り口は、**DiTのforward計算そのものではなく、それ以降の経路**——Euler更新
（`h3_gpu_euler_bf16`）、あるいはlatentのpatchify/unpatchify・GPU-CPU間の
読み書き往復（`H3_DUMP_LATENT_PREFIX`が使うのと同じ経路）——に絞り込まれる。
計画書のPhase C以降の設計（Outcome C: 「Block / DiT
forwardまで決定的で、denoise loopでのみ差」→「scheduler / step state / RNG /
latent updateを重点調査」）がまさにこの状況に対応しており、次に調べるべきは
Euler更新とlatent読み書きの経路である。

**Not established（まだ確定していないこと）**：
- Euler更新（`h3_gpu_euler_bf16`）自体が決定的かどうか（未検証。次の最優先候補）。
- `H3_DUMP_LATENT_PREFIX`が使うpatchify/unpatchify・テンソル読み書き往復自体が
  余分な非決定性を持ち込んでいないか（往復コードはCPU側の純粋な再配置のみで、
  理論上は決定的なはずだが未確認）。
- Phase C（50 block通しのフォワードパスをsingle処理として複数回repeat）について、
  今回はfresh processでの確認（3回）に留まり、同一プロセス内での複数repeatは
  未実施。
- Phase D（5〜10stepの短縮trajectoryでのBF16
  pairwise比較によるnoise floorの定量化）は未実施。
- Phase E（α-scaling実験、等norm異方向実験）はこの切り分けが済むまで保留
  （計画書40章の原則どおり）。
- `H3_NAX=0`によるA/B比較、padding zero-clear診断、synchronization強制診断
  （計画書14〜16章）は未実施。

## 実施環境に関する注記

この一連の計測中、システムメモリが強い圧迫状態にあることが判明した
（`memory_pressure`: 空きページ4174、23GB中12GBがcompressor）。原因として、
本セッションの前段階（コンパクション前）で使われていたClaude
Codeプロセス（`--resume=f8cf9264-...`、PID 88151、8:31AMから起動）が、現在の
セッション（`--resume=da936d0a-...`、PID
90318、9:39AM起動）と並行して生き続けていることを`ps aux`で確認した。単一の
`h3`起動（モデルロード＋1
stepのdenoise）に通常90秒前後で済むところ、10分以上かかるケースが発生しており、
これは2つのエージェントセッションが同一マシンのメモリ・ディスクI/Oを奪い合っている
ことが原因である可能性が高い。以降の計測（Phase C・D・E）を安定した速度で行うには、
不要な方の古いセッションを終了することを検討されたい。

## Euler更新・latent state lifecycleのトレース（`H3_TRACE_DENOISE_STATE`）

Phase A/Bにより、DiTのforward計算（QKV・SDPA・attention-out・50 block通し）は
検証範囲で完全に決定的と分かった。次の焦点は、ユーザー提案に沿って、
`DiT最終出力（velocity/noise-prediction） → Euler更新 → 次stepのlatent`
のどこで最初に非決定性が現れるかを、denoise loopをstep単位でトレースして
特定することである。

### 実装

`denoise_euler_gpu()`に`H3_TRACE_DENOISE_STATE=<csvパス>`を追加した。
評価される各stepについて、

- `dit_out` — `encode_forward()`直後、Euler更新前の`dit->video_output_bf16`
  （DiTが出した生のvelocity/noise-prediction）
- `euler_out` — Euler更新（`h3_gpu_euler_bf16`）直後の`dit->video_input`
  （次stepの`latent_in`になる値そのもの）

について、FNV-1aハッシュとsum / sum_abs / L2 / min / maxをCSVに記録する。
`latent_in`は列として別出しにしていない——`euler_out[step]`は定義上
`latent_in[step+1]`と同一のため、2回読むのは冗長である（step
0のlatent_inは初期noise latentで、既存の`H3_DUMP_LATENT_PREFIX`と同じ手段で
別途確認できる）。

**実装上の注意（試行錯誤の記録）**：最初のバージョンは`latent_in`・`dit_out`・
`euler_out`それぞれに独立した`h3_gpu_submit()`+`h3_gpu_begin()`を挟み、
1 stepあたり3回の余分な同期を発生させていた。ある20-step実行で、この版が
step 1〜2の時点で`top`上「stuck」状態・19GB phys_footprintに達し、通常90秒で
終わる処理が15分以上進まなくなる現象を2回再現した。調査の結果、19GB
自体はこのモデルの通常のロードピーク（`benchmark_grouped_int8.csv`のLoad
peak(GiB)列と整合、`ps`のRSSが小さく見えるのはGPU共有メモリを含まないため）
であり、リークではなく、**このマシン（24GB統合メモリ）で他の常駐アプリと
同時にこのモデルを何度も読み込んだ結果の環境要因**である可能性が高いと
判断した。ただし念のため、余分な同期を`dit_out`用の1回だけに削減し
（`euler_out`は既存のstep終端の`finish`ロジックに相乗りする形に変更）、
安全側に倒した。

### 現状の検証状況

- 3-step実行（削減前の3同期版）で1回、正しく動作することを確認した：
  sigmaが単調減少し、各stepの`euler_out_hash`が次stepの`latent_in`
  として想定通り一致した（自己無矛盾性の確認）。
- 削減後（現行）のコードで、20-step・10-step・3-stepいずれの実行も、
  本セッション終盤にマシン全体が断続的に極端に遅くなる状態
  （`top`で`stuck`、あるいは1 stepの計算に数分〜10分以上かかる）に繰り返し
  遭遇し、2回分のトレースを比較するところまで到達できなかった。
  プロセスのRSS自体は毎回小さく安定しており、コード側に新たなリークがある
  兆候はない。前セクションで報告した「もう一つの古いセッション」を終了した
  直後は明確に改善したが、その後も別の要因（詳細未特定）で同様の遅延が
  複数回再発した。

### Not established

- Euler更新（`h3_gpu_euler_bf16`）自体、およびlatent state
  lifecycle（GPU→CPU→GPU往復）が2回の独立実行間で一致するかどうかは、
  **まだ実データで確認できていない**。実装は完了しているため、マシンの
  負荷が落ち着いた状態で2回実行しCSVをdiffすれば、次にすぐ得られる結果である。
- したがって、「DiT forward計算までは決定的、Euler以降が非決定性の入口」
  という前セクションの推論は、まだ検証段階であり確定していない。

### 次にやること（環境が安定してから）

```bash
H3_TRACE_DENOISE_STATE=runA.csv ./h3 ... --steps 20 ...
H3_TRACE_DENOISE_STATE=runB.csv ./h3 ... --steps 20 ...
diff runA.csv runB.csv   # 最初に食い違う行・列を特定
```
一致すればEuler経路も除外でき、残る容疑は「patchify/unpatchifyの往復」
「GPU⇄CPUのテンソル読み書き」自体に絞られる（計画書のPhase D該当）。

## CSV

- `mps_op_determinism.csv` — 上記すべてのop-level / block-level repeat結果
  （503行、block 0・block 25・fresh process分を統合）

## 参照

- 計画書: `H3.c - M5環境におけるGPU数値非決定性とDenoising軌道感度の切り分け実験計画書.md`
- 先行する量子化実験一式: [README-grouped-int8.md](README-grouped-int8.md)
  （`exp/grouped-int8-weights`ブランチ）
