# Az / El ラスター走査の解析

主走査軸（各走査列内で連続して変化する座標）を `--scan-axis az` または
`--scan-axis el` で指定する。省略時は従来のAz走査となる。

```bash
./YI_Holography.sh I26184Y --scan-axis el
./autoHolo.sh I26184Y --scan-axis el
```

`YI_Holography.sh` は同じ軸指定を `corr_fringe.py`、
`group_up_txt_prd.py`、`scanning_effect.py`、`Holography.py` に渡す。
`--after-corr` でも同じ指定が必要。独立実行する場合も各コマンドに指定する。
例えば相互相関による別手法は次のように試せる。

```bash
python3 scanning_effect_xcorr.py beam.txt schedule.skd --scan-axis el --outdir xcorr_el
```

El走査のPRD座標は、既定ではPRDのEl位置と時刻から駆動速度を算出する。
PRD間の指令速度と実際の本走査速度が異なる場合は
`--el-drive-speed ARCMIN_S` を `YI_Holography.sh` または
`group_up_txt_prd.py` に指定する。折返し中はPRDの座標を補間する。

往路・復路は指定軸の増加方向・減少方向として判定する。
`best_lags.csv` の `lag_increasing_ms` / `lag_decreasing_ms` は
指定した走査軸に対する結果。Az・El座標はそれぞれ同じ列名で出力する。
`Holography.py --beam` と `--slice-beam` の断面方向も主走査軸に従う。

観測時のEl走査PRD・SKDと、初回 `frinZ` の出力について、実観測で
往復方向、PRDの速度、ON点の判定を確認してから鏡面精度を解釈する。
