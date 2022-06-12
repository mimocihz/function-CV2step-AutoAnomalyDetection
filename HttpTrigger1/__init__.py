import logging

import azure.functions as func

import requests
import json
import os
import pandas as pd
from PIL import Image, ImageDraw
from azure.storage.blob import BlockBlobService
from dotenv import load_dotenv


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # 環境変数セット
    #load_dotenv('.env')
    CV_URL = os.getenv("CV_URL")
    CV_URL_CLS = os.getenv("CV_URL_CLS")
    CV_API_KEY = os.getenv("CV_API_KEY")
    ACCOUNT_NAME = os.getenv("ACCOUNT_NAME")
    ACCOUNT_KEY = os.getenv("ACCOUNT_KEY")
    CONTAINER_NAME_INPUT = os.getenv("CONTAINER_NAME_INPUT")
    CONTAINER_NAME_RESULTS = os.getenv("CONTAINER_NAME_RESULTS")
    CONTAINER_NAME_CROP = os.getenv("CONTAINER_NAME_CROP")

    # Custom Vision Object Detection: 検知モデルの接続情報セット
    url=CV_URL
    headers={
            'Prediction-Key':CV_API_KEY,
            'content-type':'application/octet-stream'
            }

    # Custom Vision Classification: 分類モデルの接続情報セット
    url_class=CV_URL_CLS
    headers_class={
            'Prediction-Key':CV_API_KEY,
            'content-type':'application/octet-stream'
            }

    # Blob Storage 接続情報
    # ストレージアカウント名
    accountname = ACCOUNT_NAME 
    # アカウントキー
    accountkey = ACCOUNT_KEY
    # コンテナ名
    container_name_input = CONTAINER_NAME_INPUT
    container_name_results = CONTAINER_NAME_RESULTS
    container_name_crop = CONTAINER_NAME_CROP
    # Blob Storage 操作オブジェクト
    blob_service_client = BlockBlobService(account_name=accountname, account_key=accountkey)
    # 分析画像ファイル取得
    img_file_name = req.params.get('filename')
    img_in_path = os.path.expanduser("~/input")
    if not os.path.exists(img_in_path):
                os.makedirs(os.path.expanduser("~/input"))
    img_file_path = os.path.join(img_in_path, img_file_name)
    blob_service_client.get_blob_to_path(container_name_input, img_file_name, img_file_path)


    # 分析結果出力先
    img_out_path = os.path.expanduser("~/results")
    if not os.path.exists(img_out_path):
                os.makedirs(os.path.expanduser("~/results"))

    # 分析結果書き戻し画像出力先
    out_draw_fname = os.path.splitext(os.path.basename(img_file_path))[0] + '_result.jpg'
    out_draw_path = os.path.join(img_out_path, out_draw_fname)

    # 解析対象の画像セット
    img = Image.open(img_file_path)
    draw = ImageDraw.Draw(img)

    # 結果判定一覧
    df = pd.DataFrame(columns = ["file_name", "判定", "正常確率", "異常確率"])
    result_list_fname = os.path.splitext(os.path.basename(img_file_path))[0] + '_result_list.csv'
    result_list_path = os.path.join(img_out_path, result_list_fname)
    print(result_list_path)

    # Custom Vision Object Detection: 検知モデルの呼び出し
    response = requests.post(url, data=open(img_file_path,"rb"), headers=headers)
    print(response.status_code)
    print(response.elapsed)
    result = response.json()

    # 調整項目
    threshold = 0.9
    margin = 0.1

    for i, detection_target in enumerate(result["predictions"]): 

        if detection_target['probability'] > threshold: # 指定確率以上の画像のみ出力対象とする
            # 座標情報取得
            y=int(detection_target['boundingBox']['top']*img.height)
            x=int(detection_target['boundingBox']['left']*img.width)
            h=int(detection_target['boundingBox']['height']*img.height)
            w=int(detection_target['boundingBox']['width']*img.width)

            left = (x + int(margin*w/2))
            top = (y + int(margin*h/2))
            right = (x+w - int(margin*w/2))
            bottom = (y+h - int(margin*h/2))

            # 切取画像ファイル名セット
            tag_name = '_' + detection_target['tagName'] + '_'
            out_fname = os.path.splitext(os.path.basename(img_file_path))[0] + tag_name + str(i) + '.jpg'
            out_path = os.path.join(img_out_path, out_fname)

            # 切取画像保存
            #img.crop((x, y, x+w, y+h)).save(out_path, quality=95)
            img.crop((left, top, right, bottom)).save(out_path, quality=95)
            blob_service_client.create_blob_from_path(container_name_crop, out_fname, out_path)
            
            # 切取画像を分類モデルにかける
            response_class = requests.post(url_class, data=open(out_path,"rb"), headers=headers_class)
            print("*****")
            print(out_fname)
            print(response_class.json())
            result_class = response_class.json()

            # 分類モデルの判定を記録
            j = 0
            for j, target_class in enumerate(result_class["predictions"]):
                if target_class['tagName'] == "正常":
                    normal_prob = target_class['probability']
                elif target_class['tagName'] == "異常":
                    abnormal_prob = target_class['probability']

            # 正常・異常枠を元画像に書き戻し
            # 異常確率0.85以上の場合は赤枠
            if abnormal_prob >= 0.85:
                draw.rectangle((left, top, right, bottom),outline=(255,0,0),width=10)
                judge = "異常"
            # 0.5 < 異常確率 < 0.85の場合は黄枠
            elif abnormal_prob > 0.5 and abnormal_prob < 0.85:
                draw.rectangle((left, top, right, bottom),outline=(255,255,0),width=10)
                judge = "異常疑"
            # それ以外は白枠
            else:
                draw.rectangle((left, top, right, bottom),outline=(255,255,255),width=10)
                judge = "正常"
            
            # 結果判定一覧に行追加
            df.loc[i]=[out_fname, judge, normal_prob, abnormal_prob]
            #print(df)

    # 分析結果画像保存
    img.save(out_draw_path,quality=95)
    blob_service_client.create_blob_from_path(container_name_results, out_draw_fname, out_draw_path)

    # 結果判定一覧をCSVファイル出力
    df.to_csv(result_list_path, header=True, index=True)
    blob_service_client.create_blob_from_path(container_name_results, result_list_fname, result_list_path)

    return func.HttpResponse(f"finish")