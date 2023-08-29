import json
import os
import glob
import time
from tqdm import tqdm
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options


REQUIRED_DATA_SPLIT = ["train" , 'val', 'test']
df = pd.read_excel("dataset/class_selected_30.xlsx")


SELECTED_CLASSES = """
what_are_your_hobbies
thank_you
what_is_your_name
change_user_name
time
no
spelling
current_location
weather
reminder
distance
translate
how_old_are_you
calendar
tell_joke
what_song
date
play_music
yes
goodbye
repeat
where_are_you_from
alarm
fun_fact
traffic
make_call
calculator
next_song
todo_list
change_volume
""".split( "\n")
SELECTED_CLASSES = [s.strip() for s in SELECTED_CLASSES if len(s) >= 2]


def read_json_data(json_path):
    if os.path.isdir(json_path):
        json_data_list = glob.glob(json_path + "/*.json")

        json_data = dict()
        for json_data_path in json_data_list:
            json_data.update(json.load(open(json_data_path)))

    if json_path.endswith(".json"):
        json_data = json.load(open(json_path, "r"))
    else:
        raise Exception("Can't read input")

    return json_data


def webdriver_sess_init():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--incognito")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    return driver


def translate(itm_cnt, driver, sentence, target_lang="bn"):
    try:
        target_url = f"https://translate.google.co.in/?sl=auto&tl={target_lang}&text={sentence}&op=translate"
        driver.get(target_url)
        time.sleep(3)
        output_text = driver.find_element(By.CLASS_NAME, "HwtZe").text

        return output_text

    except Exception as e:
        return "TRANSLATION_FAILED"


def translate_proc(driver, json_data, target_lang="bn", save_dir="selected_classes"):
    os.makedirs(save_dir, exist_ok=True)
    itm_cnt = 0
    bn_data = []
    split = json_data[0]["intent"]
    if split not in SELECTED_CLASSES:
        return

    if os.path.exists(f"{save_dir}/bn_{split}.json"):
        print(f"{save_dir}/bn_{split}.json Exists")
        print("Skipping this Class")
        return
    for data in tqdm(json_data):
        sentence = data["text"]
        try:
            z = translate(
                itm_cnt=itm_cnt,
                driver=driver,
                sentence=sentence,
                target_lang=target_lang,
            )
            bn_data.append({**data, "bn_text": z})
            itm_cnt += 1
        except Exception as e:
            print(e)
            bn_data.append({{**data, "bn_text": "TRANSLATION_FAILED"}})
            itm_cnt += 1

    with open(f"{save_dir}/bn_{split}.json", "w", encoding="utf-8") as f:
        print("saving", f"{save_dir}/bn_{split}.json")
        json.dump(bn_data, f, ensure_ascii=False)


def main():
    json_paths = glob.glob("./CLINIC150/*json")
    for json_path in tqdm(json_paths):
        clinic_json_data = read_json_data(json_path)
        driver = webdriver_sess_init()
        time.sleep(2)
        translate_proc(driver, clinic_json_data)

        driver.close()


if __name__ == "__main__":
    main()
