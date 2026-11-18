# 複数のbeam.txtの積分

同じアンテナ・周波数・偏波・指向基準で取得した複数のbeam.txtを指定する。

```bash
python3 integrate_beams.py \
  --in test_data/beam_1.txt test_data/beam_2.txt test_data/beam_3.txt \
  --out test_result/beam_integrated.txt --antenna 34

python3 Holography.py \
  --in test_result/beam_integrated.txt --out test_result/integrated \
  --antenna 34 --slice-beam --slice-aperture
```

出力は既存のbeam.txtと同じCSV形式のtxt。**この変更のHolography.pyと組み合わせること**。
従来の未較正beam.txtも引き続き入力できる。

## 計算方法

1. 各ファイルの中心ON点から振幅・位相の時間変動を求める。既存のHolographyと同じ関数で、ON振幅の線形補間値で割り、アンラップしたON位相の線形補間値を差し引く。ONの時間範囲外は端点値を使う。
2. 停止ON（中心座標かつ既定Length=10秒）を走査データから除く。中心通過点は走査データに残す。
3. 各測定を最寄りの共通Az/El格子に割り当てる。既定間隔は両軸1.5 arcmin、原点は(0, 0)。欠損格子は出力しない。位置の補間、ピーク位置合わせ、欠損値の補充は行わない。
4. 同じ格子の複素電場を `E = sum(Length_i * E_i) / sum(Length_i)` で平均する。振幅は`abs(E)`、位相は`arg(E)`。このため、位相の違う信号は相殺されることがある。振幅だけの平均やパワー平均とは異なる。

走査間隔が3 arcminなら`--grid-step 3 3`、格子原点を変えるなら`--grid-origin AZ EL`を指定する。格子間隔を誤ると異なる指向位置が同じ格子に入るので、観測の間隔に合わせる。座標は入力のPRD補正済みAz_Offset/El_Offsetを使う。積分はアンテナ34の位相反転前の規約を保ち、Holographyで一度だけ反転する。

較正用ONを選ぶオプションは`--on-length SEC`、`--on-amp-min AMP`、`--exclude-center-crossings`、`--on-phase-linear-fit`。既定では停止ONに加えて中心通過点も較正に使う。各ファイルに、時刻の異なる採用ONが2点以上必要。

## 出力列と注意点

- `Length`: 格子に寄与した測定の積分時間の合計。
- `Epoch`: 寄与した測定の最初の時刻。連続した積分区間の開始時刻を意味しない。
- `N_samples` / `N_files`: 各格子に寄与した測定数／ファイル数。全ファイルに共通しない格子も残る。
- `SNR`: frinZの平均雑音振幅の定義を前提に、独立な各測定の熱雑音を加重平均で伝播した値。ON基準の不確かさや観測間の共通誤差は含まない。
- `Beam_State=on_calibrated`、`Antenna_m`、`Grid_Az_arcmin`、`Grid_El_arcmin`: Holographyで二重ON補正を避け、積分時の格子を維持するための情報。削除しない。

積分済み入力ではHolographyのON補正オプションと`--mc-trials`は使用できない。ON点を含む元の時系列情報は積分で失われるため。積分後の通常解析・鏡面精度計算・作図は実行できる。

入力には正のLength・SNRと非負のAmpが必要。入力ファイルの重複（内容の一致を含む）、積分済みファイルの再積分、入力への上書きはエラーとする。既存出力を上書きする場合は`--overwrite`を指定する。

検証: `python3 -m unittest test_integrate_beams.py`
