# 複素ビームから開口面電場を推定する独立テスト

`bayesian_holography.py` は既存の `Holography.py`、`holography_mc.py`、シェルスクリプトを呼ばず、変更もしません。測定点の複素ビームとSNRから、開口面電場の事後分布を推定します。

**最初の研究用試作です。細かい格子で復元できたことと、実際にその分解能で測れたことは区別してください。** まず既知の鏡面を使うデモ、次に観測データでの事前分布依存性の確認を想定しています。

## 準備

```bash
python3 -m pip install numpy scipy pandas matplotlib scikit-image
python3 bayesian_holography.py
```

引数なしで全オプションと既定値を表示します。`--in/--input`、`--out/--output` に対応します。

## まず動作確認

```bash
python3 bayesian_holography.py --demo --out test_result/bayes_demo \
  --grid-size 16 --chains 4 --iterations 4000 --warmup 1500
```

非対称な振幅低下と鏡面変形を持つ模擬開口面を作り、複素ビームを計算して実部・虚部に独立な正規雑音を加え、そのデータから復元します。`--demo` は自動的に `--calibration none` になります。模擬鏡面は推定と同じ画素モデルです。このデモで確認できるのは実装の整合性であり、現実の光学系や格子化誤差への頑健性ではありません。

- `demo_beam.txt`: 入力にした雑音付き複素ビーム。
- `demo_truth.npz`: 正解の複素電場と鏡面変位。
- `demo_recovery.png`: 左が正解の振幅、右が推定した複素電場と正解の絶対差。
- `summary.json`: 正解のRMSと複素電場復元誤差も記録。

32 mの本格的な試行例:

```bash
python3 bayesian_holography.py --demo --out test_result/bayes_demo32 \
  --antenna 32 --grid-size 32 --chains 4 --iterations 10000 --warmup 2000
```

## 観測データ

```bash
python3 bayesian_holography.py \
  --in I26290YA/beam.txt --out I26290YA/bayesian_test \
  --antenna 32 --grid-size 32 \
  --chains 4 --iterations 10000 --warmup 2000 --seed 2026
```

34 mの場合は `--antenna 34` を指定します。既存の規約に合わせて**入力Phaseの符号を反転してから**ON補正と推定を行います。既に34 m用の位相反転を行ったファイルを再び与えないでください。

入力列は `Epoch, Length, Amp, Phase, SNR, Az_Offset, El_Offset`。位相は度、Lengthは秒、offsetは角分です。Epochは `2026/184 08:00:00.000` または秒までの表記を受け付けます。座標は渡された値をそのまま使用し、cos(El)を追加で掛けません。Az/El両方の座標を使うため、横スキャン・縦スキャンを区別する `--scan-axis` は不要です。必要なPRD座標補正・走査遅延補正は先に行ってください。SKD/PRDや相関処理はこのスクリプトでは読みません。

既にON補正された複素ビームを入力する場合:

```bash
python3 bayesian_holography.py --in calibrated_beam.txt --out bayes_calibrated \
  --calibration none --antenna 32
```

その場合も `Amp/SNR` が補正後の実部・虚部それぞれの雑音標準偏差になるよう、入力を用意してください。

## 何を推定するか

未知量は、開口内の各画素の**複素電場**と、その空間的な滑らかさを制御する2つのパラメータです。振幅や鏡面変位を固定していません。

測定点を格子補間せず、各観測座標で以下の式からビームを予測します。

```text
B(theta) = (1 / supported_area) integral A(x,y)
           exp[-2 pi i (theta_x*x + theta_y*y)/lambda] dx dy

d_i = B(theta_i) + n_R,i + i n_I,i
n_R,i, n_I,i ~ Normal(0, sigma_i^2)
sigma_i = Amp_i / SNR_i
```

フーリエの符号は既存スクリプトのビーム→開口面IFFTに対応します。規格化は開口の支持領域の面積で行い、一様な電場A=1ならB(0)=1です。画素内の電場を一定とし、画素面積の積分にはsinc因子を含めます。

滑らかさの事前分布は開口内の隣接画素グラフで定義します。中心遮蔽を挟んだ画素同士には直接の辺を作りません。グラフLaplacianの固有値をl_j、固有基底での複素電場をc_jとして、

```text
Re(c_j), Im(c_j) ~ Normal(0, 1/q_j)
q_j = (1 + ell^2 * l_j)^2 / tau^2
log(ell) ~ Normal(log(length_prior_m), length_log_sd^2)
log(tau) ~ Normal(log(scale_prior), scale_log_sd^2)
```

- ell [m]: 滑らかさの長さ。大きいと細かい変化を抑えます。
- tau: 電場の固有基底係数の尺度。画素ごとの振幅標準偏差そのものではありません。
- 実部・虚部を対称に扱う正規分布です。振幅の正値性に別の制約を掛けず、複素電場として推定します。
- アンテナ円外と指定した中心正方形をゼロとする**既知の支持領域**を課します。円の離散化には画素中心を使い、厳密な円の縁ではなく、選択された正方形画素の集合をモデル化します。
- 中心遮蔽を固定したくない実験には `--center-block-size 0` を使えます。

特定の照度分布や既知の鏡面変形を正解として与える事前分布ではありません。一方で「複素電場が滑らか」という仮定にも復元結果は依存します。

## MCMCの中身

このモデルでは、ellとtauを固定すると、全画素の複素電場の条件付き事後分布が多変量正規分布になり、行列計算で求められます。

1. 複素電場を解析的に積分消去した尤度を使い、`log(ell), log(tau)` をMetropolis–Hastings法でサンプリングします。
2. 各保持ステップで、得られたellとtauに対応する条件付き事後分布から**全画素の複素電場を同時に**生成します。
3. 各試行の開口面から位相・鏡面変位・鏡面RMSを計算します。

2パラメータだけの不確かさを出しているわけではありません。全画素の事後分布も保存します。全画素を一つずつランダムウォークさせる必要はないため、強い画素間相関も行列計算で扱えます。

ウォームアップ中に提案分布の尺度と共分散を適応し、保持期間では固定します。複数チェーンは異なる乱数で逐次実行します。各チェーン完了時にチェックポイントを保存しますが、この版には途中再開オプションはありません。

## ON補正と雑音についての範囲

既定は中心座標の点をON候補とし、通過点も含めます。Amp閾値はありません。

- `--exclude-center-crossings`: `Length=--on-length` の停止ONだけで補正。
- `--on-length 10`: 停止ONの積分時間。
- `--on-amp-min 0.4`: 任意のAmp閾値。
- `--exclude-on-time "2026/184 08:26:01.000"`: 補正から除く開始時刻。繰り返し指定可。
- `--on-phase-linear-fit`: 位相を時間に対する一次回帰で補正。未指定はON間の線形補間。

振幅はON間の線形補間値で除算し、位相は補間/回帰した値を差し引きます。SNR由来のsigmaも同じ振幅規格化で変換します。停止ONはビームモデルへの入力から外し、中心通過点は走査データとして残します。ON範囲外では補間の端点値を使います（位相一次回帰指定時は直線を延長）。

**この版の事後分布はON補正量を固定した条件でのものです。** ON点の雑音から生じる共通の較正誤差、装置のドリフトの不確かさ、通過点を較正にも使うことによる依存関係は含めません。既存の雑音付加MCの代替となる「全誤差」ではありません。ON複素利得も未知量とする同時推定は次段階です。

SNRが0の点は、同じLength（許容差1e-6秒）のSNRが正の点から求めた生データのsigma=Amp/SNRの**算術平均**で補います。中央値やRMSではありません。補った点は平均の計算に再利用せず、正常な点のsigmaは変更しません。積分時間の異なる停止ONとoffsetは混ぜません。該当時刻・採用sigma・参照点数はログに残し、input_calibration.csvにはraw_quadrature_sigma（ON振幅規格化前）とquadrature_sigma_imputed（補完の有無）を保存します。quadrature_sigmaは従来どおりON振幅規格化後の値です。SNRが負の場合、または同じ積分時間の参照点がない場合は停止します。正のSNRに対するAmpが0の場合も従来どおり停止します。既知の雑音がある場合に限り `--noise-sigma VALUE` で全点共通の**生データ単位**の実部・虚部雑音SDを明示できます。低SNR点を無条件に切り捨てる処理はしません。

積分区間の重なる測定点は雑音が独立でないため、この版ではエラーにします。重なりのない開始刻みで作ったbeam.txtを使ってください。天候・大気・装置・RFI等による相関雑音もこの尤度には含みません。

## 積分中の動き

既定は測定点を瞬間のビーム値として扱います。積分によるスキャン方向のぼけもモデルに含めたい場合:

```bash
python3 bayesian_holography.py --in beam.txt --out bayes_motion \
  --antenna 32 --motion-average
```

`Az_Rate_arcmin_s, El_Rate_arcmin_s` 列を使い、各点のLengthの間を等速で動いた場合のビームをGauss–Legendre積分で平均します。画素のsinc因子も時刻ごとに計算します。積分点数は最大位相変化に応じて8以上に増やします。

**offset座標が積分区間の中点での座標であることが前提です。** 入力の座標付与規約を確認してください。加減速・折返し・複雑なPRD軌道は表現しません。デモは静止測定のため、`--demo --motion-average` は使えません。

## 格子を細かくする実験

```bash
python3 bayesian_holography.py --in beam.txt --out bayes_fine \
  --antenna 32 --grid-size 48 --chains 4 --iterations 10000 --warmup 2000
```

32 mでgrid-size=32なら画素幅1 m、48なら0.667 mです。これは計算格子の幅であり、分解能の保証ではありません。Nyquist範囲外の観測点が入る粗すぎる格子では停止します。補間で細かいビームを作ってから独立データのように入力することはしないでください。

`beam_mean.png` は推定した開口面から連続的な角度座標で予測したビームです。表示の細かさだけなら `--beam-grid 201` で変更できます。これで開口面の未知量や測定情報は増えません。

## 出力と読み方

すべて指定した出力ディレクトリ内に保存します。PNGに対応するPDFは `pdf/` にまとめます。点・線のグラフは黒、タイトルと凡例は付けません。

| ファイル | 内容 |
|---|---|
| `bayesian_holography.log` | 進捗、警告、RMSの結果、エラー |
| `input_calibration.csv` | 使用点・ON選択・補正量・補正後の複素値・sigma |
| `chain_XX.npz` | チェーンごとの固有基底係数とlogパラメータ。実空間への変換は再計算したsmooth_basisが必要 |
| `posterior_trace.csv` | チェーン番号、保持試行番号、ell、tau、全開口/円環RMS |
| `posterior_aperture.npz` | 支持領域マスク、座標軸、全画素の複素電場試行、鏡面誤差試行・平均・SDなど |
| `aperture_mean.png` | 左:事後の振幅平均のdB、右:位相の円周平均。振幅平均と複素平均の絶対値は別物 |
| `aperture_uncertainty.png` | 振幅SD、位相の円周SD |
| `surface_posterior.png` | tilt除去後の鏡面誤差の平均・SD |
| `surface_rms_posterior.png` | 左:全開口、右:半径2–12 mのRMS分布 |
| `mcmc_trace.png` | 上からell、tau、全開口RMS、円環RMSのチェーン |
| `posterior_beam.npz` | 連続座標で計算した複素ビーム平均と振幅SD |
| `beam_mean.png` | ビームの複素平均の振幅dBと位相 |
| `beam_uncertainty.png` | ビーム振幅SD。`--predict-draws` 試行を使う近似 |
| `beam_residuals.csv` | 測定点での複素予測値・残差/sigma |
| `summary.json` | 条件、乱数seed、採択率、収束指標、RMSの中央値と信用区間 |

NPZ内の電場・鏡面誤差配列は支持領域内だけを並べた値です。例えば `image[support] = field_mean` で2次元配置に戻せます。全試行はチェーン順に連結しています。

鏡面変位は位相をマスク付き2次元アンラップした後、32 mなら半径2–14 m、34 mなら2–15 mで平面をフィットして除き、`lambda*phase/(4*pi)` から求めます。**従来スクリプトの±135度による分岐補正とは異なる**ため、同じRMSになることを想定しないでください。低振幅で位相が不定な画素が多いときは、鏡面誤差への変換自体が不安定になります。

RMSは全開口（円周に触れる画素を除外）と半径2–12 mの両方を保存し、どちらも中心3 m四方に触れる画素を除きます。支持領域の中心遮蔽サイズとは別に、この3 mのRMS除外を行います。前後の物理的な変位符号、正面から見た左右の対応は相関系・座標規約の外部確認が必要です。

RMSの16–84%区間や2.5–97.5%区間は**このモデルと事前分布の下でのベイズ信用区間**です。SDを試行数の平方根で割った値ではありません。

`rank_split_rhat` はrank正規化/folded split R-hatの大きい方、`bulk_ess` はrank変換系列の自己相関から見積もった有効標本数です。目安はR-hat<=1.01、ESS>=400です。ell、tauと2種類のRMSについて診断し、満たさなければ警告します。全画素の収束を保証する指標ではありません。チェーンが同じ状態に留まる場合も、低採択率や軌跡を必ず確認してください。

## 計算コストと研究用の確認

未知画素数をP、測定点数をNとして、初期の行列作成が概ねO(N P^2)、各提案のCholesky分解がO(P^3)です。格子辺を2倍にするとPは約4倍、1提案のコストは概ね64倍になります。良いPCでも最初は32程度から確認してください。NumPy/SciPyのBLASが並列化するので、環境に応じて `OPENBLAS_NUM_THREADS=4` 等で調整できます。

- チェーン数、試行数を増やし、収束指標と軌跡を確認する。
- `--length-prior-m 1`, `3`, `6` などを変えて、細部とRMSが事前分布に依存しすぎないかを見る。
- seedを変えて再計算する。同じseedなら同じ条件での再現性確認に使える（ライブラリやBLAS環境差はあり得る）。
- 異なる格子、異なる模擬鏡面・SNRでも検証する。デモと同じ格子だけではモデル誤差を検出できない。
- `beam_residuals.csv` の実部/虚部の残差を確認する。標準化残差の平均二乗が約1に近いかは適合の目安だが、パラメータ推定による自由度の減少もある。

出力に既存のチェーンやsummary.jsonがある場合は上書きせず停止します。比較試行は別の `--out` に保存してください。

## 検証

```bash
python3 -m unittest discover -s tests -p test_bayesian_holography.py -v
```

周辺尤度をデータ空間で直接計算した値との一致、複素条件付き標本の共分散、フーリエの符号・画素積分、積分中の等速移動、収束診断、ON補正・34 m位相反転・重複積分の拒否を検証します。


## Holography.pyと同じ配色・測定ビームとの比較

物理量マップの配色と表示範囲をHolography.pyにそろえます。

- 振幅dB: inferno、既定 -45〜0 dB（`--db-min` で下限を変更）。
- 位相: twilight、-180〜180度、カラーバーは60度刻み。
- 鏡面誤差の平均: coolwarm、既定 -3〜3 mm（`--surface-limit-mm`）。
- 標準偏差: viridis、下限0。SDには正負がないため発散型配色を使いません。

色や表示範囲を変えるだけで、保存した数値や推定モデルを変更することはありません。旧版は振幅を-80 dBまで表示していたため、Holography.pyより弱い外側の構造まで強調されていました。

**既存の結果から描き直す（MCMC再実行なし）:**

```bash
python3 bayesian_holography.py --replot results --out results_view
```

Windowsのコマンドプロンプトでも同じ一行のコマンドを使えます。入力先には `summary.json`, `posterior_aperture.npz`, `posterior_beam.npz`, `beam_residuals.csv`, `posterior_trace.csv` が必要です。元の結果を保護するため、`--out` は別ディレクトリにします。大きな全電場試行配列を読み込む必要はありません。元のNPZやsummary.jsonは変更せず、新しい図・診断ファイルだけを保存します。新規解析でも同じ図・診断を自動生成します。

追加する比較図:

| ファイル | 読み方 |
|---|---|
| `beam_data_model_db.png` | 左:ON補正後の測定振幅、右:同じ測定座標での推定振幅。両方とも測定振幅の最大値を共通0 dB基準とし、モデルの利得誤差を別規格化で消しません |
| `beam_data_model_db_zoom.png` | 上の比較を中心で拡大。`--zoom-size 24` は各軸±12角分 |
| `beam_data_model_phase.png` | 左:測定位相、右:同じ測定座標の推定位相。低振幅の位相は不定になり得ます |
| `beam_fit_slices.png` | 左列:Az方向、右列:El方向。上:振幅dB、下:位相。黒丸:測定、黒実線:推定。凡例・タイトルは付けません |
| `beam_standardized_residuals.png` | 左:実部、右:虚部の(測定−推定)/sigma。色範囲±5で、CSV/指標は切りません |
| `beam_residual_histogram.png` | 左:実部、右:虚部の標準化残差。ヒストグラムと標準正規分布の線を比較 |
| `beam_fit_quality.json` | 全点・中心半径12角分以内・外側の残差指標。ピーク振幅比や全点での測定/推定ピーク座標も記録 |

比較マップは補間せず、元の測定座標に四角いマーカーを描きます。空白は未観測の場所です。主走査軸や格子へのスナップによる違いを持ち込まないための診断です。断面は横断座標の絶対値が `--slice-half-width`（既定0.75角分）以内、主座標が `--zoom-size/2` 以内の点を使います。完全に同じ横断座標でない点も入るので、実際の幅・点数は `plot_settings.json` に記録します。推定線は測定座標での予測を座標順に結ぶもので、追加データを生成するものではありません。

`--motion-average` を使った解析では、測定座標比較は積分中の動きを含む訓練時の予測です。一方、連続的な `beam_mean.png` は瞬間のビームの予測なので、この2種類を区別してください。

`mean_squared_standardized_quadrature_residual` は実部・虚部を合わせた残差二乗平均です。雑音と同程度なら概ね1に近づきますが、パラメータ推定による自由度の減少を補正したreduced chi-squareではありません。2を超えると目安として警告します。大きい場合はON補正・座標/遅延・積分平均・支持領域・事前分布などを調べ、単にMCMCの試行数を増やすだけで改善するとは考えないでください。

## 実際に変えているパラメータを確認する

- **Metropolis–Hastingsで探索する量**: `log(ell)` と `log(tau)` の2つ。
- **各保持ステップで同時生成する量**: 有効な開口内P画素の複素電場係数（実部P個・虚部P個、合計2P個）。ell/tauに対応する条件付き多変量正規分布から生成し、画素間の相関を保持します。
- **固定する量**: 測定ビームと座標、測定雑音sigma、ON補正量、アンテナ径・中心遮蔽、周波数。
- **推定した電場から計算する量**: 位相、振幅、鏡面変位、RMS。RMSやビーム幅自体をランダムに動かしてフィットしているわけではありません。

これは測定したAmp/Phaseを雑音の範囲で振り直す通常の誤差伝播MCとは異なります。測定値は固定し、それを説明できる未知の開口面電場の分布を求めます。

`inference_parameters.json` に未知数の個数、探索方法、固定量、事前分布設定を保存します。`parameter_summary.csv` にはell/tauとRMSの役割・中央値・16/84パーセンタイルを保存します。`mcmc_trace.png` の最初の2段が探索パラメータ、下の2段が電場から計算したRMSです。
