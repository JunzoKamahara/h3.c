# M4 Max環境での作業指示書：20-step Euler状態トレース比較

対象ブランチ: `exp/gpu-determinism-probe`（このファイルもこのブランチにある）

## 背景

M5/24GB環境で、`H3_TRACE_DENOISE_STATE`によるstep単位のdit_out/euler_out
トレースを2回実行して比較しようとしたが、20-step生成の途中（step 2〜7あたり）で
マシン全体が`top`上「stuck」表示になり数分〜数十分進まなくなる現象に繰り返し
遭遇し、比較用データを2本そろえられなかった（マシン再起動後、トレース無しの
素の実行でも同じ現象が再現したため、原因はこのブランチのコードではなく
M5/24GB機のメモリ余裕不足である可能性が高いと判断している。詳細は
[README-gpu-determinism.md](README-gpu-determinism.md)の
「Euler更新・latent state lifecycleのトレース」節を参照）。

M4 Max/128GBならメモリ余裕が大きく、この問題を回避してトレースを完走できる
可能性が高い。本指示書の手順で2回分のトレースCSVを取得し、diffを取ってほしい。

## 前提

- このリポジトリを`git checkout exp/gpu-determinism-probe`した状態
- `make h3`でビルド済み（このブランチのコミットには、M4 Max用に
  Metal 4高速パスをハードウェア機能ベースで判定するよう修正した`main`の
  変更も既にマージ済み。デフォルト動作はM5専用ヒューリスティックがM4上では
  無効になるだけで、今回のトレース自体には影響しない）
- MiniMax-H3のモデルディレクトリ（`-d`オプションで渡すパス）が用意されていること

```bash
git clone https://github.com/JunzoKamahara/h3.c.git
cd h3.c
git checkout exp/gpu-determinism-probe
make h3
```

## 実行手順

同一設定（プロンプト・seed・step数など）で、`H3_TRACE_DENOISE_STATE`の出力先だけ
変えて2回実行する。`<MODEL_DIR>`はMiniMax-H3モデルディレクトリへのパスに置き換える。

```bash
H3_TRACE_DENOISE_STATE=runA.csv ./h3 -d <MODEL_DIR> \
  -p "A red fox walks through fresh snow in a pine forest." \
  --seed 2026 --width 512 --height 512 --frames 22 --steps 20 --layers 50 --reuse 1 \
  -o runA.mp4

H3_TRACE_DENOISE_STATE=runB.csv ./h3 -d <MODEL_DIR> \
  -p "A red fox walks through fresh snow in a pine forest." \
  --seed 2026 --width 512 --height 512 --frames 22 --steps 20 --layers 50 --reuse 1 \
  -o runB.mp4

diff runA.csv runB.csv
```

`runA.mp4`/`runB.mp4`自体は不要（トレースが目的なので、生成できたらそのまま
放置か削除で構わない）。

### 万一20-stepが重い場合

同じ現象がM4 Maxでも起きるようなら、`--steps 10`や`--steps 5`に落として
まず短い方で試してほしい（`README-gpu-determinism.md`にある通り、`--steps 3`
はM5側でも安定して完走できていた）。

## 結果の見方

`runA.csv`と`runB.csv`は次の列を持つ：

```
step,sigma,delta_sigma,
dit_out_hash,dit_out_sum,dit_out_sum_abs,dit_out_l2,dit_out_min,dit_out_max,
euler_out_hash,euler_out_sum,euler_out_sum_abs,euler_out_l2,euler_out_min,euler_out_max
```

- `dit_out` = DiTが出した生のvelocity/noise-prediction（Euler更新の直前）
- `euler_out` = Euler更新直後のlatent（次stepの`latent_in`と同一）

`diff runA.csv runB.csv`が**空**（差分なし）なら、DiT forward計算に続いて
Euler更新経路も含めて完全に決定的、という結論になる。

差分がある場合は、**最初に食い違う行**を見る：

- その行の`dit_out_hash`から違う → DiT forward計算自体に非決定性がある
  （これまでのPhase A/Bの結果と矛盾するので、詳しい追加調査が必要）
- `dit_out_hash`は一致するが`euler_out_hash`から違う → Euler更新
  （`h3_gpu_euler_bf16`）自体、あるいはlatentのGPU⇄CPU往復に非決定性がある
- ある行の`euler_out_hash`は一致するが、**次の行**の`dit_out_hash`から違う →
  「euler_outが次stepのlatent_inとして正しく引き継がれているか」の
  受け渡し部分（バッファ管理・同期）を疑う

## 報告してほしいこと

1. `diff runA.csv runB.csv`の出力（差分なしなら「差分なし」の一言でよい）
2. 20-stepが問題なく完走したか（完走までのおおよその時間も分かれば）
3. 差分があった場合は、`runA.csv`・`runB.csv`のファイルそのもの
   （両方送ってもらえれば、こちらで詳しく分析する）
