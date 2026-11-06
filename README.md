# 料理みわけ

写真から、60種類の食べ物・飲み物の候補を比較する小さなStreamlitアプリです。
対象外・候補が近い画像では、無理に名前を付けずに判定を保留します。

`open60-v3`：従来の料理31種に、フルーツ12種・飲み物8種・タイ料理9種を追加した試作版です。
候補文を追加したzero-shot比較であり、モデルを追加学習したものではありません。
変更点と検証の扱いは [EXPANSION_NOTES.md](EXPANSION_NOTES.md) を参照してください。

**試作版です。誤認・誤保留が残ります。scoreは正解確率ではありません。**
栄養、アレルギー、食材、食品の安全性の判断には使わないでください。

## Streamlit Community Cloud 部署（给维护者）

1. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)，点击 **Create app**。
2. 选择 **Yup, I have an app**，填写：

   | 项目 | 填写内容 |
   |---|---|
   | Repository | `mousineeee-web/dish-miwake` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | Advanced settings → Python version | **3.12** |
   | Secrets | **留空**，不需要 API Key 或付费识别接口 |

3. 点击 **Deploy**。首次安装依赖及下载模型需要等待。
4. 页面打开后，上传一张自己有权使用的食物照片，点击「名前を推定」。
   **只有实际推理成功，才算完成部署验证**。首页打开不代表模型已加载。

可选子域名需要以平台当时可用的名称为准；仓库准备完成不代表已取得公网应用地址。

[官方部署说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

### 冷启动与资源限制

- 模型约813 MB，第一次推理时自动从Hugging Face获取，保存在 `.model_cache/`，不放进GitHub。
- 权重和Python运行时的内存占用并不相同。免费平台可能加载缓慢或内存不足，不能保证多人同时使用时的可用性。
- 应用缓存模型，并将同一进程的推理串行化，限制同时推理的内存峰值；繁忙时需要排队。
- 休眠或重启后可能需要重新加载。发表前先访问并试一次，另保留本地版本和备用视频。
- 如果安装失败，先确认Python选的是 **3.12**；Linux x86_64的CPU wheel使用CPython 3.12。
- 如果首页能开但推理失败，请在 **Manage app → Logs** 查看加载或资源错误。
- 本仓库没有替你创建Streamlit账号、同意付费条款或执行云端部署。

## 使い方

1. 写真を選択します。JPEG・PNG・HEIC/HEIF・WebP・AVIF・GIF・TIFF・BMP・JPEG 2000に対応します。20 MiB以下・5000万画素以下に制限しています。
2. 「名前を推定」を押します。初回はモデルの取得・読み込みに時間がかかります。
3. 候補名と対応score、または判定保留の理由を確認します。

画面の表示は日本語、比較する候補文は英語です。60種類の範囲は画面の「対象60種類と利用上の注意」で確認できます。
画像を登録して新しい料理を学習する仕組みではありません。

### 統一された画像入力（2026年11月6日更新）

選択画面で拡張子・MIMEを制限せず、サーバーで**実際のバイト列**を許可した画像decoderだけに渡します。
同じ写真がカメラからHEIC、相冊からJPEGとして渡された場合も、拡張子の大文字・小文字や別名に依存しません。
JFIF/JPE、HIF、TIF、JP2/J2K、拡張子なし・誤った拡張子でも、内容が対応する画像なら読み込みます。
ファイル名を書き換えて形式変換した扱いにはしません。

- JPEG・PNG・WebP・AVIF・GIF・TIFF・BMP/DIB・JPEG 2000は固定したPillowで読み込みます。
- HEIC/HEIFは固定した `pillow-heif==1.8.1` で主画像を読み込みます。depth/auxiliary画像は使用しません。
- EXIFの撮影方向を反映し、RGB・CMYK・グレースケール・パレットをRGBへ正規化します。透明部分は白背景へ合成します。
- 16 bitグレースケールは8 bitへ縮小します。HEIF/AVIFのHDRはdecoderの8 bit変換を使用します。色校正やHDR編集を保証する機能ではありません。
- アニメーション（GIF/APNG/WebP/AVIF）・複数ページTIFFは先頭の1枚だけを使用します。HEIFは先頭ではなく主画像です。
- 4032×3024、5712×4284、8064×6048（約4877万画素）は画素数上限内です。長辺2048 pxを超える写真は縦横比を維持して縮小します。
- 元画像の上限は20 MiB・5000万画素。ヘッダー確認後にdecodeし、同一プロセス内のdecodeを直列化します。codecによっては縮小前に元画像全体をメモリへ展開するため、利用者数に対する無制限の可用性は保証しません。
- RAW（DNG/ProRAW等）・JPEG XL・SVG・PDF・動画、破損ファイルは対応外です。JPEG/PNGへ実際に書き出してください。外部converterやshellは起動しません。
- 正規化したプレビューにはEXIF/GPS・XMPを引き継ぎません。端末内の元の写真は変更しません。

形式を広く受け付けることと、料理を正しく認識することは別です。入力互換性の修正はモデルの追加学習ではありません。
縮小・透明背景の合成・codecの非可逆圧縮でpixel値や推定結果が変わる場合があります。過去の正解率を全形式の精度として主張しません。

実装の根拠：[Pillowの画像形式](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)、
[Image.openの形式制限](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open)、
[pillow-heif](https://pillow-heif.readthedocs.io/en/stable/)、
[Streamlit file_uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)。

## モデルと判定

- [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)
- 固定revision：`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`
- 候補：60食品・飲料 + 5一般食品・飲料 + 6対象外食品 + 15非食品 = 86文
- モデルの追加学習は行わず、画像と候補文の対応logitを比較します。
- `open_set.py` がlogit差に基づいて受付・保留を判断します。未知画像を完全に拒否する保証はありません。
- 同じ蘭州牛肉麺でもラーメンと誤認する例があり、候補の多さは実用精度を保証しません。
- 元の `open31-v2` の31候補・説明は維持し、29候補を追加しました。一般的なfruit/drinkの文は対象外群から一般食品群へ移動しました。拒否閾値の数値とモデルrevisionは変更していません。
- 水・牛乳などは見た目の候補です。透明な別の液体・植物性ミルク・糖分・成分・温度・飲用の安全性は写真だけでは判断できません。お茶は緑茶・紅茶などを1候補にまとめ、茶葉の品種を断定しません。

## 写真とプライバシー

**公開サイトにアップロードした画像は、あなたの端末から実行サーバーへ送信されます。**

アプリのコードは画像をディスクへ保存せず、追加学習や外部の画像認識APIへの転送も行いません。
処理中はStreamlitのセッション／アップロード機構とメモリを使用します。即時消去やホスティング事業者の運用まで保証するものではありません。
機密情報・顔・個人情報を含む画像はアップロードしないでください。写真は自分が使用権を持つものを使ってください。

モデルファイルの初回取得では、実行サーバーからHugging Faceへ接続します。
公開用repositoryに、授業の報告、学生番号、評価用写真、model cache、認証情報は含めていません。

## ローカル実行

Python 3.12を使います。Linux x86_64はCPU専用PyTorchをインストールします。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

既存のモデルcacheを再利用する場合は `DISH_MODEL_CACHE` にcache directoryを指定できます。
cacheがすべて揃っていれば `HF_HUB_OFFLINE=1` で起動できます。

## テスト

```bash
python -m unittest discover -s tests -v
```

通常の自動テストはfake modelを使い、候補順・score処理・拒否規則・アップロード検証・UI分岐を確認します。
新しい写真での精度や、本番cloudのメモリを保証するテストではありません。

モデルの出典とlicenseは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。
