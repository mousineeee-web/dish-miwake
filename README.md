# 料理みわけ

料理の写真から、31種類の候補を比較する小さなStreamlitアプリです。
料理以外・対象外・候補が近い画像では、無理に料理名を付けずに判定を保留します。

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
4. 页面打开后，上传一张自己有权使用的料理照片，点击「料理名を推定」。
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

1. JPG・PNG・HEIC・HEIFの写真を選択します。10 MiB以下・2500万画素以下に制限しています。
2. 「料理名を推定」を押します。初回はモデルの取得・読み込みに時間がかかります。
3. 候補名と対応score、または判定保留の理由を確認します。

画面の表示は日本語、比較する候補文は英語です。31種類の範囲は画面の「対象31種類と利用上の注意」で確認できます。
画像を登録して新しい料理を学習する仕組みではありません。

### スマートフォンのHEIC写真

HEIC/HEIFは `pillow-heif==1.8.1` で自動的に8 bit RGBへデコードします。
拡張子をJPGへ変更する必要はありません。複数画像を含むHEIFでは主画像だけを使用し、
撮影方向を反映してEXIF/GPS・XMP情報をプレビューへ引き継ぎません。
一般的な4032×3024（約1219万画素）と5712×4284（約2447万画素）の写真は画素数制限内です。
4800万画素など2500万画素を超える写真や10 MiBを超えるファイルは、先に縮小してください。
深度画像・補助画像・動画の認識には対応しません。

入力形式を追加しただけで、候補文・モデル重み・拒否閾値を再学習／調整したものではありません。
旧写真での正解率をHEIC写真全体の精度として主張しません。

## モデルと判定

- [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224)
- 固定revision：`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`
- 候補：31料理 + 3一般料理 + 8対象外食品 + 15非料理 = 57文
- モデルの追加学習は行わず、画像と候補文の対応logitを比較します。
- `open_set.py` がlogit差に基づいて受付・保留を判断します。未知画像を完全に拒否する保証はありません。
- 同じ蘭州牛肉麺でもラーメンと誤認する例があり、候補の多さは実用精度を保証しません。
- この公開版でも、候補文・閾値・モデルrevisionは元の `open31-v2` から変更していません。

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
