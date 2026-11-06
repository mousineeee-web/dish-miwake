import hashlib
import json
import logging
import threading

import streamlit as st
from PIL import Image, UnidentifiedImageError

from classifier import MODEL_NAME, load_runtime
from food_catalog import CATALOG_VERSION, DISHES, all_prompts
from image_input import MAX_FILE_SIZE, open_uploaded_image
from open_set import CONFIG, FoodRecognizer


LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_model_runtime():
    return load_runtime()


@st.cache_resource(show_spinner=False)
def get_recognizer(runtime_signature):
    return FoodRecognizer(*get_model_runtime())


@st.cache_resource(show_spinner=False)
def get_inference_lock():
    """Bound peak CPU/memory use when several visitors submit together."""
    return threading.Lock()


st.set_page_config(page_title="料理みわけ", page_icon="🍽️", layout="centered")

st.title("料理みわけ")
st.write(f"{len(DISHES)}種類の料理候補を比較します。判断できない画像には、無理に料理名を付けません。")
st.caption("寿司・ラーメン・蘭州牛肉麺・ピザなどの候補比較版です。画像を登録して追加学習する仕組みではありません。")
st.caption("公開デモ：画像は実行サーバーへ送信して処理します。アプリは画像をファイル保存したり、追加学習や外部の画像認識APIへ送信したりしません。個人情報を含む写真はアップロードしないでください。")
with st.expander(f"対象{len(DISHES)}種類と利用上の注意"):
    st.write('・'.join(row[1] for row in DISHES))
    st.info("すべての料理に対応するものではありません。料理以外・対象外・曖昧な画像の拒否を試みますが、誤認や見逃しは残ります。栄養・アレルギー・食べられるかは判断しません。")

with st.sidebar:
    st.subheader("推定条件")
    language = 'English'
    st.caption("候補文：English（表示は日本語）")
    st.caption("devで精度の低かった日本語候補は、通常画面では使用しません。言語比較はCLIで実行できます。")
    st.caption(f"モデル：{MODEL_NAME}")
    st.caption(f"{CATALOG_VERSION}｜対象外・不確実な入力を保留する試作版")

uploaded_file = st.file_uploader(
    "料理画像を選択",
    type=["jpg", "jpeg", "png"],
    help="JPGまたはPNG、10 MB以下・1200万画素以下の静止画を使用してください。",
)

if uploaded_file is None:
    st.info("画像を選ぶと、ここにプレビューが表示されます。")
elif uploaded_file.size > MAX_FILE_SIZE:
    st.error("画像が10 MBを超えています。小さい画像を選んでください。")
else:
    try:
        image = open_uploaded_image(uploaded_file.getvalue())
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error("画像を読み込めませんでした。10 MB以下・1200万画素以下のJPGまたはPNGの静止画を選んでください。")
    else:
        st.image(image, caption=uploaded_file.name, width="stretch")
        st.success("画像を読み込みました。")
        st.write(f"画像サイズ：{image.width} × {image.height} px")
        st.caption(f"{len(DISHES)}種類の料理候補と、料理以外・範囲外の食品候補を比較します。scoreは正解確率ではありません。")

        if st.button("料理名を推定", type="primary", width="stretch"):
            try:
                with st.spinner("初回はモデルの読み込みに時間がかかります…"):
                    signature = hashlib.sha256(json.dumps([all_prompts('English'), all_prompts('日本語'), CONFIG], ensure_ascii=False).encode()).hexdigest()
                    with get_inference_lock():
                        result = get_recognizer(signature).predict(image, language)
            except Exception:
                LOGGER.exception("Dish inference failed")
                st.error("推定を実行できませんでした。少し待ってから再試行してください。初回のモデル取得やサーバーの空きメモリが原因の場合があります。")
            else:
                st.subheader("推定結果")
                if result.decision == 'accepted':
                    st.success(f"料理の候補：「{result.predictions[0].label}」")
                    st.caption(result.reason)
                    for rank, prediction in enumerate(result.predictions[:3], 1):
                        st.progress(min(max(prediction.score, 0.0), 1.0),
                                    text=f"{rank}位　{prediction.label}　対応score {prediction.score:.3f}")
                else:
                    st.warning("判定を保留しました：" + result.reason)
                    st.info("料理を1品だけ大きく写した写真や、別の角度の写真を試してください。")
                    if result.decision == 'uncertain' and result.predictions:
                        with st.expander("参考候補（確定結果ではありません）"):
                            for prediction in result.predictions[:3]:
                                st.write(f"{prediction.label}：対応score {prediction.score:.3f}")

                st.caption(
                    f"前処理＋推論時間：{result.elapsed_seconds:.2f}秒。"
                    "scoreは用意した候補文との対応度であり、正解確率の保証ではありません。"
                )
                with st.expander("モデル入出力を確認"):
                    st.code(
                        "\n".join(
                            [
                                f"pixel_values: {result.pixel_values_shape}",
                                f"input_ids: {result.input_ids_shape}",
                                "raw_logits: "
                                + str([round(value, 4) for value in result.raw_logits]),
                                f"prompt_language: {language}",
                                f"catalog_version: {result.catalog_version}",
                                f"decision: {result.decision}",
                            ]
                        )
                    )
                    st.json(result.diagnostics)
