import sys
import glob
import subprocess
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import re
import numpy as np

# 書き込むdelay値の符号を指定する変数 ('+' または '-')
DELAY_SIGN = '+'

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


def converge_delay(xml_file):
    """fringe_search用xmlに対してgico3->fringeを反復し、Res-Delayが0sample以下になるまでdelayを収束させる。

    継続/停止の判定、および累積は常にsample単位の生値で行い、
    XMLファイルへ書き込むときにのみ1024000000で除算して秒に変換する。

    戻り値: 収束成功時は dict(iterations, res_delay, amp, snr, cumulative_delay_sample)
            失敗時は None
    """
    # 反復開始前にfringe_search用xmlのdelayを0に初期化する
    update_delay_in_xml(xml_file, 0.0, DELAY_SIGN)
    cumulative_delay_sample = 0.0

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"  [INFO] --- 反復 {iteration}回目 ---")

        run_gico3(xml_file)

        cor_file = find_cor_file(xml_file)
        if cor_file is None:
            print("  [ERROR] .corファイルが見つからないため、この反復を中止します。")
            return None

        fringe_command = ["fringe", "--in", cor_file]
        fringe_output = run_external_command(fringe_command, capture=True)

        res_delay, amp, snr = parse_fringe_output(fringe_output)

        if res_delay is None:
            print("  [WARN] fringeの結果からRes-Delayを抽出できませんでした。")
            return None

        print(f"  [INFO] Res-Delay: {res_delay} sample, Amp: {amp}, SNR: {snr}")

        # 継続/停止の判定は読み取ったsample値そのままで行う
        judge_delay = np.abs(res_delay)
        if judge_delay <= 0:
            print(f"  [SUCCESS] 収束しました（Res-Delay <= 0 sample）。累積delay: {cumulative_delay_sample} sample")
            return {
                "iterations": iteration,
                "res_delay": res_delay,
                "amp": amp,
                "snr": snr,
                "cumulative_delay_sample": cumulative_delay_sample,
            }

        # 収束していない場合、sample単位のまま累積する
        cumulative_delay_sample += res_delay
        # fringe_search用xmlに書き戻すときだけ1024000000で割って秒に変換する
        update_delay_in_xml(xml_file, cumulative_delay_sample / 1024000000.0, DELAY_SIGN)

    print(f"  [WARN] 最大反復回数({MAX_ITERATIONS}回)に達しましたが収束しませんでした。このstepをスキップします。")
    return None


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


def main(obs_code):
    """メインの処理を実行する関数"""
    try:
        os.chdir(obs_code)
        print(f"[INFO] '{obs_code}' ディレクトリに移動しました。\n")
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
            update_delay_in_xml(target_xml_file, final_delay_sec, DELAY_SIGN)

            log_entries.append({
                "xml_file": xml_file,
                "iterations": result["iterations"],
                "res_delay": result["res_delay"],
                "cumulative_delay_sec": f"{DELAY_SIGN}{abs(final_delay_sec):.5e}",
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
        update_delay_in_xml(xml_file, 0.0, DELAY_SIGN)
        print("")

    print("--- 全ての処理が完了しました ---")

    write_log(LOG_FILENAME, log_entries)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"使用法: python {script_name} <観測コード>")
        sys.exit(1)

    observation_code = sys.argv[1]
    main(observation_code)
