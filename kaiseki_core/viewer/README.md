# kaiseki_core ビューアー (Streamlit)

xlsx 解析結果をブラウザで確認するための軽量ビューアー。

## ローカルで見る

```powershell
cd "G:\その他のパソコン\マイ ノートパソコン\解析"
streamlit run kaiseki_core\viewer\app.py
```

または `kaiseki_core\viewer\run.bat` をダブルクリック。

ブラウザで `http://localhost:8501` が開く。サイドバーから xlsx を選択。

## 外出先から見る (Cloudflare Quick Tunnel)

### 初回セットアップ
```powershell
winget install --id Cloudflare.cloudflared
```

### 起動
```powershell
kaiseki_core\viewer\run_tunnel.bat
```

コンソールに `https://<ランダム名>.trycloudflare.com` が表示される。これをスマホで開く。

- ランダムURLは起動の都度変わる（無料版の制約）
- PC を閉じる/Ctrl+C でトンネルが切れる
- 永続URLが欲しい場合は `cloudflared tunnel create <name>` で named tunnel を作成し、DNS設定が必要

### 注意
- Quick Tunnel は **誰でもURLを知っていればアクセス可能**。データに機密性があるなら Cloudflare Access (Zero Trust) でメール認証を追加するか、Tailscale 経由に切り替える
- スマホで Basic認証が欲しい場合は `streamlit-authenticator` を追加

## 表示される内容

xlsx 内の全シートをタブ表示:
- `Data` / `ChainData` — 冒頭500行のみプレビュー（件数が多いため）
- 集計シート — 全行 + 自動で棒グラフ
- `CZ成功率` — トップにメトリックカードで主要数値を表示

## 対応機種

`DEFAULT_OUTPUT_DIRS` に列挙された機種別フォルダの xlsx を自動検出:
- `ビッグドリーム解析Python\output`
- `デュオ解析Python\output`

新機種は `app.py` の `DEFAULT_OUTPUT_DIRS` に追加するか、起動時に `-- --output-dir <path>` で指定。

```powershell
streamlit run kaiseki_core\viewer\app.py -- --output-dir "C:\path\to\output"
```
