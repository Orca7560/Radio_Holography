# スキャニング効果解析

`scanning_effect.py` は、相関結果の時系列と SKD を時刻で対応付け、走査に伴う実効的な時間遅れを推定します。

## 入力

- 相関結果 CSV: `Epoch, Amp, Phase, SNR` 列を持つファイル
- SKD: `$SKED` 節に、時刻・積分時間・Az/El オフセットが記載されたファイル

`Epoch` は `2026/184 08:22:03.01`、SKD 時刻は `26184082203.01` の形式に対応しています。

## 実行

```bash
python scanning_effect.py correlation_summary.csv schedule.skd --outdir scan_2026184
```

標準設定では、時刻を 6 ms 以内で対応付け、SNR 3 以上のデータを使い、\(-1000\)〜\(+1000\) ms の実効遅れを 5 ms 刻みで探索します。必要に応じて、例えば次のように変更できます。

```bash
python scanning_effect.py correlation_summary.csv schedule.skd \
  --match-tolerance 0.01 --min-snr 2 --max-lag-ms 300 --lag-step-ms 1
```

## 原理

Az 正方向・負方向の走査で作った振幅マップが最も一致する遅れ \(\tau\) を選びます。補正は各サンプルごとに

\[
\theta_{\rm corr}=\theta_{\rm SKD}-v\tau
\]

として加えます。したがって、正方向と負方向の走査には自動的に逆向きの補正が入ります。`Best scan lag` と `Typical Az correction` が、求めたい補正量です。

## 出力

- `matched_and_corrected.csv` — SKD 座標、推定走査速度、補正座標・補正量を含む全サンプル
- `lag_search.csv` — 試行した遅れと正逆走査間の不一致
- `scan_rows.csv` — 自動識別した各走査列と速度
- `scanning_diagnostics.png` — 遅れ探索曲線と補正前後の振幅・位相マップ

## 注意

遅れ推定には、十分な SNR を持つ Az 正方向・負方向の両方の走査列が必要です。提示された SKD 抜粋は Az 正方向の一列だけなので、この部分だけでは遅れは数値化できません。全ラスタの SKD と相関時系列を入力してください。
