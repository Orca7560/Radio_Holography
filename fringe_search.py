"""
fringe_search.py — fringe_search用XMLのdelayをRes-Delayが収束するまで自動調整する。

概要:
  対象ディレクトリ内の `*_step*_fringe_search.xml` を1つずつ処理し、各XMLに
  対して gico3 → fringe を繰り返し実行して、fringeが返すRes-Delay（残差delay）
  が実質0になるまでXML中の <clock key="K">/delay を更新する。

処理の流れ（各XMLごと）:
  1. delay=0で基準測定を行い、基準のRes-Delayを取得する。
  2. 基準のRes-Delay絶対値をtrial幅として、+方向と-方向を1回ずつ試し、
     Res-Delayがより小さくなった側の符号を採用する。
     （どちらも改善しなければそのXMLはスキップし、delayは0に戻す。）
  3. 採用した符号方向にdelayを積み増しながら再測定を繰り返し、Res-Delayが
     0になるか、反復回数が MAX_ITERATIONS（既定20回）に達するまで続ける。
  4. 収束したら、対応する本番用XML（`_fringe_search.xml` を除いたファイル名、
     例: `xxx_step01_fringe_search.xml` → `xxx_step01.xml`）のdelayを、
     収束した累積delay（秒換算）で更新する。
  5. fringe_search用XML自身のdelayは常に0に戻して終了する。

前提条件:
  - gico3, fringe コマンドにPATHが通っていること。
  - 実行ディレクトリ（またはworkdir配下）に `./raw`, `./stepcor` があり、
    gico3が `./stepcor` 以下に `YAMAGU32_YAMAGU34_<timestamp>_*<label>.cor`
    という命名の.corファイルを生成すること。
  - 各fringe_search用XMLに `process/epoch`, `process/skip`, `stream/label`,
    `clock[@key='K']/delay` のタグが存在すること。

実行例:
  python fringe_search.py
  python fringe_search.py --workdir I26184Y

出力:
  実行ディレクトリに `fringe_search_result.log`（タブ区切り）を出力する。
  列: xml_file, iterations, res_delay[sample], cumulative_delay[sec],
      amp[%], snr
  収束しなかった場合は iterations 以降が "FAILED"/"N/A" になる。
"""
import sys
import argparse
import glob
import subprocess
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import re
import numpy as np

# 書き込むdelay値の符号を指定する変数 ('+' または '-')
# 収束処理の最大反復回数（無限ループ防止）
MAX_ITERATIONS = 20

# ログファイル名
LOG_FILENAME = "fringe_search_result.log"


def run_external_command(command_list, capture=False):
    """外部コマンドを実行する。capture=Trueの場合、標準出力を文字列として返す。"""
    print(f"[INFO] 実行コマンド: {' '.join(command_list)}")
    try:
        result = subprocess.run(
            command_list,
            check=True,
            capture_output=capture,
            text=True
        )
        if capture:
            return result.stdout
    except FileNotFoundError:
        print(f"[ERROR] コマンドが見つかりません: {command_list[0]}。パスが通っているか確認してください。")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] コマンドの実行に失敗しました (終了コード: {e.returncode}): {' '.join(command_list)}")
        if capture:
            return e.stdout
    return None


def parse_fringe_output(output):
    """fringeの出力からRes-Delay, Amp, SNRの値を抽出する。

    出力フォーマット例（列インデックスは0始まり）:
      [0] Epoch(年/日)  [1] Epoch(時刻)  [2] Label  [3] Source  [4] Length
      [5] Amp  [6] SNR  [7] Phase  [8] Noise-level  [9] Res-Delay  [10] Res-Rate ...

    戻り値: (res_delay, amp, snr) のタプル。抽出失敗時はNoneを含む。
    """
    if not output:
        return None, None, None

    lines = output.splitlines()
    for line in lines:
        if re.match(r'^\s*\d{4}/\d{3}', line):
            parts = line.split()
            if len(parts) > 9:
                res_delay = None
                amp = None
                snr = None
                try:
                    res_delay = float(parts[9])
                except ValueError:
                    print(f"  [WARN] Res-Delayの値の変換に失敗しました: {parts[9]}")
                try:
                    amp = float(parts[5])
                except ValueError:
                    print(f"  [WARN] Ampの値の変換に失敗しました: {parts[5]}")
                try:
                    snr = float(parts[6])
                except ValueError:
                    print(f"  [WARN] SNRの値の変換に失敗しました: {parts[6]}")
                return res_delay, amp, snr
    return None, None, None


def update_delay_in_xml(filename, new_delay_value, sign):
    """XMLファイル内のdelay値を更新する（有効数字6桁の指数表記）"""
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        delay_element = root.find("clock[@key='K']/delay")

        if delay_element is None:
            print(f"  [WARN] '{filename}' 内に <clock key='K'>/delay タグが見つかりません。")
            return

        # 小数点以下5桁の指数表現（＝整数部1桁＋小数部5桁で有効数字6桁）
        formatted_delay = f"{sign}{abs(new_delay_value):.5e}"
        delay_element.text = formatted_delay

        if sys.version_info >= (3, 9):
            ET.indent(tree, space="  ")

        tree.write(filename, encoding='UTF-8', xml_declaration=True)
        print(f"  [SUCCESS] '{filename}' のdelayを '{formatted_delay}' に更新しました。")

    except ET.ParseError:
        print(f"  [ERROR] XMLファイルの解析に失敗しました: {filename}")
    except FileNotFoundError:
        print(f"  [ERROR] 更新対象のXMLファイルが見つかりません: {filename}")


def run_gico3(xml_file):
    """gico3を実行する"""
    gico3_command = ["gico3", "--schedule", xml_file, "--raw-file", "./raw", "--cor-file", "./stepcor"]
    run_external_command(gico3_command)


def find_cor_file(xml_file):
    """fringe_search用xmlに対応する.corファイルを1つ取得する（複数マッチ時は最初の1つを使用）"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        epoch_str = root.find('process/epoch').text.strip()
        skip_sec = int(root.find('process/skip').text)
        label = root.find('stream/label').text.strip()

        base_datetime = datetime.strptime(epoch_str, '%Y/%j %H:%M:%S')
        scan_start_datetime = base_datetime + timedelta(seconds=skip_sec)
        timestamp_for_file = scan_start_datetime.strftime('%Y%j%H%M%S')

        file_pattern = f"./stepcor/YAMAGU32_YAMAGU34_{timestamp_for_file}_*{label}.cor"
        cor_files = sorted(glob.glob(file_pattern))

        if not cor_files:
            print(f"  [WARN] 対応する.corファイルが見つかりませんでした: {file_pattern}")
            return None

        if len(cor_files) > 1:
            print(f"  [INFO] 複数の.corファイルが見つかりました。最初の1つを使用します: {cor_files[0]}")

        return cor_files[0]

    except (ET.ParseError, AttributeError, ValueError) as e:
        print(f"  [ERROR] XML処理中にエラーが発生しました: {xml_file} ({e})")
        return None


def run_one_measurement(xml_file):
    """gico3とfringeを1回実行し、(Res-Delay, Amp, SNR)を返す。"""
    run_gico3(xml_file)
    cor_file = find_cor_file(xml_file)
    if cor_file is None:
        print("  [ERROR] .corファイルが見つかりません。")
        return None
    result = parse_fringe_output(run_external_command(["fringe", "--in", cor_file], capture=True))
    if result[0] is None:
        print("  [WARN] fringeの結果からRes-Delayを抽出できませんでした。")
        return None
    print(f"  [INFO] Res-Delay: {result[0]} sample, Amp: {result[1]}, SNR: {result[2]}")
    return result

def set_delay_samples(xml_file, delay_samples, sign):
    update_delay_in_xml(xml_file, abs(delay_samples) / 1024000000.0, sign)

def converge_delay(xml_file):
    """+/- の試行結果から補正符号を自動決定してdelayを収束させる。"""
    update_delay_in_xml(xml_file, 0.0, "+")
    iterations = 1
    print("  [INFO] --- 反復 1回目: delay=0 の基準測定 ---")
    base = run_one_measurement(xml_file)
    if base is None:
        return None
    res_delay, amp, snr = base
    base_abs = abs(res_delay)
    if base_abs <= 0:
        return {"iterations": iterations, "res_delay": res_delay, "amp": amp,
                "snr": snr, "cumulative_delay_sample": 0.0, "delay_sign": "+"}

    trial_delay = base_abs
    chosen_sign = None
    chosen = None
    for sign in ("+", "-"):
        if iterations >= MAX_ITERATIONS:
            break
        iterations += 1
        print(f"  [INFO] --- 反復 {iterations}回目: {sign}方向の試行 ---")
        set_delay_samples(xml_file, trial_delay, sign)
        trial = run_one_measurement(xml_file)
        if trial is None:
            return None
        if abs(trial[0]) < base_abs:
            chosen_sign, chosen = sign, trial
            break
        update_delay_in_xml(xml_file, 0.0, "+")

    if chosen_sign is None:
        print("  [WARN] + / - のどちらもRes-Delayを小さくできませんでした。このstepをスキップします。")
        update_delay_in_xml(xml_file, 0.0, "+")
        return None

    cumulative_delay_sample = trial_delay
    res_delay, amp, snr = chosen
    while abs(res_delay) > 0 and iterations < MAX_ITERATIONS:
        cumulative_delay_sample += abs(res_delay)
        set_delay_samples(xml_file, cumulative_delay_sample, chosen_sign)
        iterations += 1
        print(f"  [INFO] --- 反復 {iterations}回目 ({chosen_sign}方向) ---")
        measured = run_one_measurement(xml_file)
        if measured is None:
            return None
        res_delay, amp, snr = measured

    if abs(res_delay) > 0:
        print(f"  [WARN] 最大反復回数({MAX_ITERATIONS}回)に達しましたが収束しませんでした。")
        return None
    print(f"  [SUCCESS] 収束しました。符号={chosen_sign}, 累積delay={cumulative_delay_sample} sample")
    return {"iterations": iterations, "res_delay": res_delay, "amp": amp,
            "snr": snr, "cumulative_delay_sample": cumulative_delay_sample,
            "delay_sign": chosen_sign}


def write_log(log_path, log_entries):
    """収束結果をログファイルに書き出す"""
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("xml_file\titerations\tres_delay[sample]\tcumulative_delay[sec]\tamp[%]\tsnr\n")
        for entry in log_entries:
            f.write(
                f"{entry['xml_file']}\t"
                f"{entry['iterations']}\t"
                f"{entry['res_delay']}\t"
                f"{entry['cumulative_delay_sec']}\t"
                f"{entry['amp']}\t"
                f"{entry['snr']}\n"
            )
    print(f"\n[INFO] ログファイルを出力しました: {log_path}")


def main(workdir="."):
    """メインの処理を実行する関数"""
    try:
        os.chdir(workdir)
        print(f"[INFO] '{os.path.abspath(workdir)}' を作業ディレクトリにします。\n")
    except FileNotFoundError:
        print(f"[ERROR] ディレクトリ '{obs_code}' が見つかりません。")
        sys.exit(1)

    fringe_xml_files = sorted(glob.glob('*_step*_fringe_search.xml'))

    if not fringe_xml_files:
        print("[WARN] fringe search用のXMLファイルが見つかりませんでした。")

    log_entries = []

    print("--- delay収束処理を開始します ---")
    for xml_file in fringe_xml_files:
        print(f"[INFO] 処理中のXMLファイル: {xml_file}")

        result = converge_delay(xml_file)

        if result is not None:
            # 収束した最終delay（sample）を秒に変換してstep用xmlに反映（xml書き込み時のみ除算する）
            target_xml_file = xml_file.replace('_fringe_search.xml', '.xml')
            final_delay_sec = result["cumulative_delay_sample"] / 1024000000.0
            update_delay_in_xml(target_xml_file, final_delay_sec, result["delay_sign"])

            log_entries.append({
                "xml_file": xml_file,
                "iterations": result["iterations"],
                "res_delay": result["res_delay"],
                "cumulative_delay_sec": f"{result['delay_sign']}{abs(final_delay_sec):.5e}",
                "amp": result["amp"],
                "snr": result["snr"],
            })
        else:
            log_entries.append({
                "xml_file": xml_file,
                "iterations": "FAILED",
                "res_delay": "N/A",
                "cumulative_delay_sec": "N/A",
                "amp": "N/A",
                "snr": "N/A",
            })

        # fringe_search用xmlのdelayは常に0に戻しておく
        update_delay_in_xml(xml_file, 0.0, "+")
        print("")

    print("--- 全ての処理が完了しました ---")

    write_log(LOG_FILENAME, log_entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "gico3とfringeを繰り返し実行し、fringe_search用XMLのdelayを"
            "Res-Delayが収束するまで自動調整します。"
        ),
        epilog=(
            "実行例:\n"
            "  python fringe_search.py\n"
            "  python fringe_search.py --workdir I26184Y\n\n"
            "対象ディレクトリ内の *_step*_fringe_search.xml をすべて処理し、"
            "結果を fringe_search_result.log に出力します。\n"
            "gico3 / fringe コマンドにPATHが通っている必要があります。"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--workdir", default=".", metavar="DIR",
                        help="処理対象の観測ディレクトリ。未指定時はカレントディレクトリ。")
    args = parser.parse_args()
    main(args.workdir)
