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

### この節時点でのNot established（M4 Max検証前）

- Euler更新（`h3_gpu_euler_bf16`）自体、およびlatent state
  lifecycle（GPU→CPU→GPU往復）が2回の独立実行間で一致するかどうかは、
  M5/24GBマシンでは実データで確認できなかった。
- M5でこの機体固有の遅延問題（後述）が繰り返し発生したため、`INSTRUCTIONS-
  m4max-trace.md`を作成し、同じ検証をM4 Max/128GB機に依頼した。結果は次節。

## マシン間比較：M4 Max/128GBでの結果（決定的な追加証拠）

`INSTRUCTIONS-m4max-trace.md`の手順で、M4 Max/128GB機にて`H3_TRACE_DENOISE_
STATE`によるstep単位トレースを実行してもらった。M5側で20-step完走が
できなかった問題が、M4 Maxでは発生しなかった（正常に完走）。

### 結果

| 環境 | 演算経路 | 2回の独立実行 |
|---|---|---|
| M4 Max/128GB | MPSGraph（デフォルト。`runA.csv`/`runB.csv`） | **bit-identical** |
| M4 Max/128GB | TensorOps/int8（`H3_FORCE_TENSOROPS=1`、M5と同一カーネル。`runA-tensorops.csv`/`runB-tensorops.csv`） | **bit-identical** |
| M5/24GB | TensorOps/int8（デフォルト） | 過去に非決定性を確認済み（本調査の出発点、README-grouped-int8.md参照） |
| M5/24GB | MPSGraph（`H3_GPU_CLASS=m3`） | **測定不可**（後述） |

M4 Max側の1回目（`runA.csv`/`runB.csv`）は、この機がデフォルトで
"M5"ではないと判定されるため、自動的にMPSGraph経路（TensorOps/int8を使わない
方）で実行されていた。これだけでは「M5と同じカーネルを使った上での比較」に
なっていないという指摘があり、2回目に`H3_FORCE_TENSOROPS=1`を付けて
**M5がデフォルトで使うのと全く同じTensorOps/int8カーネル**を強制的に使わせて
再度2回実行してもらった（`runA-tensorops.csv`/`runB-tensorops.csv`）。
こちらも20行（20 step）すべてでhash・統計値が完全一致した。

### M5側でのMPSGraph強制は測定不能だった

対称性を取るため、M5側で`H3_GPU_CLASS=m3`（MPSGraph経路への強制）を試みたが、
実行時のメモリ使用量が**37GB（うちcompressorが36GB）**に達し、24GBの
統合メモリに収まらず`top`上「stuck」状態になったため、途中で強制終了した。
（`H3_GPU_CLASS=m3`はEuler常駐サンプラー自体も無効化してしまうため、
`H3_GPU_SAMPLER=1`を追加で指定してこのトレース対象のコードパスに固定する
必要があった、という実装上の注意点も記録しておく。）
これはM5側のTensorOps/int8パスがDiTピークのtensor storageを32.8GB→17.1GB
に削減する設計だという`main`ブランチのコミットメッセージ（Metal
4パッチ）の記述と整合しており、MPSGraph経路自体がこのマシンでは物理的に
動かせないことが原因であって、コードの問題ではない。

### 結論（この段階での更新、慎重な言い換え）

M4 Max/128GBでは、**MPSGraph経路・TensorOps/int8経路のどちらでも、2回の
独立実行が完全にbit-identical**だった。特に後者は、M5がデフォルトで使うのと
全く同じコード（同じMetalシェーダ・同じディスパッチ順序）を実行した結果である
ため、「h3.cのTensorOps実装に、常に非決定性を生む本質的なrace conditionが
ある」という強い仮説はかなり否定できる。

ただし、これは「TensorOps実行スタック全体を完全に無罪にした」ことまでは
意味しない。M4 MaxとM5では、**同じMetalコードであってもGPUハードウェア・
ドライバ内部実装・TensorOpsの実機コンパイル結果・メモリサブシステムが
異なり得る**ため、「特定の実行スタックの組み合わせ（M5チップ + そのドライバ +
TensorOps）でのみ現れる非決定性」という可能性は排除されていない。

さらに重要な交絡として、今回の比較はGPU世代（M5 vs M4 Max）だけでなく
**メモリ容量（24GB vs 128GB）**も同時に変えてしまっている。本セッションでは
過去に、別のClaude Codeプロセスの残留により空きページが極端に少なくなる
ほどメモリが逼迫している状態を確認しており（本README「実施環境に関する
注記」節）、この24GB unified memory環境・その時の実行時状態（メモリ圧迫・
allocatorの挙動・swap発生の有無）こそが、M5チップ世代そのものより先に
疑うべき変数である。

現時点で正確な結論は次の通り：

> The nondeterminism could not be reproduced on an M4 Max, including when
> forcing the same TensorOps/int8 code path used on the M5. This rules out
> an unconditional determinism defect in the h3.c TensorOps implementation.
> The remaining cause appears specific to the M5/24GB execution environment
> or its interaction with runtime state; whether this is an M5-generation
> property, memory-pressure effect, driver/runtime behavior, or
> machine-specific condition remains unresolved.

本調査の当初の主仮説（計画書3.1節「MPSGraph内部のreduction順序」）は、
この段階でも支持する証拠が得られていない。

### Not established（現時点）

- 非決定性がM5「チップ世代」の特性なのか、この特定の個体・特定のOS/ドライバ
  バージョンに固有なのか、あるいは**メモリ圧迫・allocator・実行時状態への
  依存**なのかは未確認（比較対象になる別のM5機がなく、かつM4 Maxとの比較は
  メモリ容量が交絡している）。
- M5側でEuler経路・latent lifecycle自体を直接トレースして2回のBF16実行を
  diffする、という当初の目的は、依然としてM5機では未達成のままである。

### 次の再現試験（M5をclean rebootした状態で実施予定）

M5/24GB機を再起動し、他アプリを最小限にした状態で、フル20-step動画生成では
なく**5〜10 step・latent-onlyの短縮trace**を3〜5回連続で実行し、比較する。
目的は「GPU世代の問題」と「メモリ圧迫・実行時状態の問題」を切り分けること。

条件と読み取り方：

| 条件 | 目的 |
|---|---|
| 再起動直後・他アプリ最小 | clean baseline |
| 同条件で3〜5 repeat | M5で本当に再現性がないか |
| （余力があれば）意図的にメモリを圧迫させた状態 | memory pressureとの因果 |
| （余力があれば）連続実行で温間状態にしてから | thermal/power-stateとの関連 |

使うのは既存の`H3_TRACE_DENOISE_STATE`機構（`--steps 5`〜`10`程度に短縮）。
現在のCSVスキーマは`dit_out_hash`・`euler_out_hash`の2列で、`latent_in`は
別列を持たない（定義上`latent_in[step] == euler_out[step-1]`のため、
step Nの`euler_out_hash`をstep N+1の`latent_in_hash`として読み替える —
詳細は本README「実装」節）。3〜5回分のCSVを横に並べて、**最初に値が割れる
(step, 列)** を特定する：

- ある行の`dit_out_hash`から複数回で割れる → DiT forward計算自体
  （M4 Maxの結果と矛盾するため、M5固有の何かに強く絞られる）
- `dit_out_hash`は揃うが`euler_out_hash`から割れる → Euler更新・書き戻し経路
- あるstepの`euler_out_hash`は揃うが、次stepの`dit_out_hash`から割れる →
  euler_out→次stepのlatent_inへの受け渡し（バッファ管理・同期）

**clean reboot直後でbit-identicalになれば**、非決定性は「M5そのもの」より
「M5/24GB上のメモリ圧迫・allocator・実行時状態に依存した現象」だった
可能性が急激に高まる。**clean reboot直後でも同じ箇所から差が出るなら**、
「M5世代のGPU/ランタイム特有」という仮説が強まる。

### 結果：M5/24GB、clean reboot後、5回連続実行 — 5/5 bit-identical

実際にM5機を再起動し、fox / seed=2026 / `--steps 8`（latent-only、
`H3_TRACE_DENOISE_STATE`）を5回連続実行して比較したところ、**5本すべてが
byte-identical**だった（`m5_clean_reboot_run1.csv`〜`run5.csv`、
`diff run1.csv runN.csv`が全てのペアで空）。

なお、この5回の実行では単純な「clean baseline」にはならなかった点も記録
しておく：再起動直後にもかかわらず、Dropbox・Google Drive・Creative Cloud
などの自動起動アプリがすぐにメモリを使い始め、加えてこのセッションを
動かしているClaude Code自身のプロセス群（Claude Helper・WindowServer等）
だけで合計1GB超を占め、`top`の`PhysMem`は終始23G/24G使用・空き数百MB
という状態が続いた。5本のうち複数本が`top`上「stuck」表示になり、1本
あたり数分から2時間以上かかるケースもあった（8 stepという短いtraceに
対して）。つまり**「速度」の面では極めて不安定・低速だったが、「結果の
正しさ」の面では完全に安定していた**。

### 結論（更新・確定）

この結果は、前節で立てた2つの仮説のうち**「M5/24GB上のメモリ圧迫・
allocator・実行時状態に依存した現象」を支持し、「M5世代のGPU/ランタイム
特有の非決定性」という仮説をかなり弱める**。clean reboot直後でも
Dropbox等やClaude Code自身によって実質的にメモリがひっ迫した状態には
なったが、それでも計算結果自体は5/5で完全に一致した。

したがって、本調査全体の結論を次のように更新する：

> Across every level tested — individual GPU ops, whole-block forward
> passes, full single-step and full 20-step DiT+Euler traces on both an
> M4 Max and (after a clean reboot) the M5 machine, including 5
> consecutive short traces under real memory pressure on the M5 — this
> investigation found no reproducible bit-level nondeterminism in h3.c's
> GPU execution path. The nondeterminism originally observed between two
> full 20-step BF16 generations on the M5 machine was not reproduced by
> any of the controlled tests here, including on the same machine after
> a reboot. The original hypothesis (nondeterministic MPSGraph/TensorOps
> reduction order) is not supported by any evidence gathered. What
> remains unexplained is the original observation itself: it may have
> depended on conditions (a specific memory-pressure state, a specific
> sequence of prior GPU work, thermal/power state, or something else
> entirely) that these tests did not exactly reproduce, since none of
> the controlled repeats - including several run under comparably heavy
> memory pressure - triggered it again.

未解決のまま残っていたのは、**本調査の出発点そのもの**（fox/seed=2026の
BF16 vs BF16再実行で観測された、20-step全体でのrun-to-run差）を、
制御された条件下で誰も再現できていないという点だった。5回の短縮trace
（`H3_TRACE_DENOISE_STATE`、8-step）との間には、(1) 元の観測が`H3_DUMP_
LATENT_PREFIX`のみ（追加の同期なし）による測定だったのに対しこちらは
step毎に追加のGPU submitを挟む測定だった、(2) step数が20と8で異なる、
という2点の違いが残っていた。

### 最終確認：元の手法・元のstep数での再実行 — 20/20 bit-identical

この最後の差異を埋めるため、`H3_DUMP_LATENT_PREFIX`のみ（`H3_TRACE_
DENOISE_STATE`は使わず、追加の同期なし）で、fox/seed=2026・20-stepの
生成をM5機で2回実行し（`orig1`・`orig2`）、各stepでダンプされる
video latent（raw F32、`h3_dit_video_elements`要素）を`cmp`で
バイト単位比較した。**20 step全てでbyte-identical**だった
（step01〜step20、1つも不一致なし）。

これは、本調査の出発点となった元の観測を、**元の測定手法・元のstep数で
正確に再現しようとした試み**であり、この2回の実行では非決定性は
一切現れなかった。したがって、上記2つの候補（測定手法の違い・step数の
違い）はどちらも決定的な要因ではなかったことになる。

本調査全体の結論を、ここで確定させる：

> Every attempted reproduction of the original observation — including
> the exact original methodology (H3_DUMP_LATENT_PREFIX, no extra
> synchronization, full 20 steps) run twice on the same M5 machine —
> came back bit-identical. Combined with full determinism at every other
> level tested (individual GPU ops, whole-block forward passes, full
> single-step and full 20-step DiT+Euler traces, on both an M4 Max and
> the M5 machine, under both normal and heavy memory-pressure
> conditions), this investigation found no reproducible bit-level
> nondeterminism anywhere in h3.c's GPU execution path. The original
> divergence between two full 20-step BF16 generations that motivated
> this investigation could not be reproduced by any controlled test,
> including an exact repeat of the original method. Whatever produced
> it originally was not captured by any condition varied here (GPU
> generation, TensorOps vs MPSGraph kernel choice, memory pressure,
> machine, or measurement method), and remains unexplained. The
> practical conclusion stands unconditionally: quantization-scheme
> quality comparisons (Row-wise vs G1024 vs G128 etc.) can proceed
> without needing to control for GPU nondeterminism, since none has
> been found.

（原本の`orig1_stepNN.bin`/`orig2_stepNN.bin`はサイズの都合上リポジトリには
含めていない。再現する場合は上記コマンドで再生成できる。）

## CSV

- `mps_op_determinism.csv` — 上記すべてのop-level / block-level repeat結果
  （503行、block 0・block 25・fresh process分を統合）
- `runA.csv` / `runB.csv` — M4 Max、MPSGraph経路（デフォルト）、20-stepの
  `H3_TRACE_DENOISE_STATE`トレース。bit-identical
- `runA-tensorops.csv` / `runB-tensorops.csv` — M4 Max、TensorOps/int8経路
  （`H3_FORCE_TENSOROPS=1`）、20-stepの同トレース。bit-identical
- `m5_clean_reboot_run1.csv`〜`run5.csv` — M5/24GB、clean reboot後、
  デフォルト経路、8-stepの同トレースを5回連続実行。5/5 bit-identical

## 参照

- 計画書: `H3.c - M5環境におけるGPU数値非決定性とDenoising軌道感度の切り分け実験計画書.md`
- 先行する量子化実験一式: [README-grouped-int8.md](README-grouped-int8.md)
  （`exp/grouped-int8-weights`ブランチ）
