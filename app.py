import hashlib
import json
import logging
import threading

import streamlit as st

from classifier import MODEL_NAME, load_runtime
from food_catalog import (CATALOG_VERSION, CATALOG_GROUPS, DISHES, DRINK_KEYS,
                          all_prompts)
from image_input import (MAX_FILE_SIZE, SUPPORTED_FORMAT_LABEL, ImageUploadError,
                         decode_uploaded_image)
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
st.write(f"{len(DISHES)}種類の食べ物・飲み物を比較します。曖昧な画像では、名前を無理に確定しません。")
st.caption("料理に加えて、フルーツ・お茶や水などの飲み物・タイ料理に対応した候補比較版です。画像を登録して追加学習する仕組みではありません。")
st.caption("候補の追加はモデルの追加学習ではありません。見た目が似た食品や透明な飲み物は取り違える場合があります。")
st.caption("公開デモ：画像は実行サーバーへ送信して処理します。アプリは画像をファイル保存したり、追加学習や外部の画像認識APIへ送信したりしません。個人情報を含む写真はアップロードしないでください。")
with st.expander(f"対象{len(DISHES)}種類と利用上の注意"):
    for group, rows in CATALOG_GROUPS:
        st.markdown(f"**{group}（{len(rows)}種類）**")
        st.write('・'.join(row[1] for row in rows))
    st.info("すべての食品に対応するものではありません。対象外・曖昧な画像の拒否を試みますが、誤認や見逃しは残ります。栄養・アレルギー・食材・飲食の安全性は判断しません。")

with st.sidebar:
    st.subheader("推定条件")
    language = 'English'
    st.caption("候補文：English（表示は日本語）")
    st.caption("devで精度の低かった日本語候補は、通常画面では使用しません。言語比較はCLIで実行できます。")
    st.caption(f"モデル：{MODEL_NAME}")
    st.caption(f"{CATALOG_VERSION}｜対象外・不確実な入力を保留する試作版")

uploaded_file = st.file_uploader(
    "食べ物・飲み物の写真を選択",
    type=None,  # Do not let a filename/MIME filter reject valid phone exports.
    max_upload_size=MAX_FILE_SIZE // (1024 * 1024),
    help=f"{SUPPORTED_FORMAT_LABEL}。20 MiB以下・5000万画素以下。拡張子ではなく実際の内容を確認します。",
)
st.caption("対応：" + SUPPORTED_FORMAT_LABEL + "。撮影方向・透明背景を自動調整します。")
st.caption("大きな写真は長辺2048 px以内へ縮小。アニメーション・複数ページは先頭の1枚、HEIFは主画像のみ使用します。RAW・SVG・PDF・動画は対象外です。")

if uploaded_file is None:
    st.info("画像を選ぶと、ここにプレビューが表示されます。")
elif uploaded_file.size > MAX_FILE_SIZE:
    st.error("画像が20 MiBを超えています。小さい画像を選んでください。")
else:
    try:
        decoded = decode_uploaded_image(uploaded_file.getvalue())
        image = decoded.image
    except ImageUploadError as error:
        st.error(str(error))
    else:
        st.image(image, caption=uploaded_file.name, width="stretch")
        st.success("画像を読み込みました。")
        st.caption(f"検出形式：{decoded.detected_format}｜元画像：{decoded.original_size[0]} × {decoded.original_size[1]} px")
        if decoded.resized:
            st.caption("表示・推定用に縮小しました。端末内の元の写真は変更していません。")
        st.write(f"画像サイズ：{image.width} × {image.height} px")
        st.caption(f"{len(DISHES)}種類の候補と、非食品・範囲外の食品候補を比較します。scoreは正解確率ではありません。")

        if st.button("名前を推定", type="primary", width="stretch"):
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
                    st.success(f"食べ物・飲み物の候補：「{result.predictions[0].label}」")
                    st.caption(result.reason)
                    if result.predictions[0].key in DRINK_KEYS:
                        st.warning("飲み物は見た目の候補です。写真だけでは透明な液体の中身・成分・糖分・温度・飲用の安全性を確認できません。")
                    for rank, prediction in enumerate(result.predictions[:3], 1):
                        st.progress(min(max(prediction.score, 0.0), 1.0),
                                    text=f"{rank}位　{prediction.label}　対応score {prediction.score:.3f}")
                else:
                    st.warning("判定を保留しました：" + result.reason)
                    st.info("食べ物・飲み物を1つだけ大きく写した写真や、別の角度の写真を試してください。")
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
