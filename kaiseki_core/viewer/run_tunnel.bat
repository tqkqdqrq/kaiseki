@echo off
REM Cloudflare Quick Tunnel 経由でビューアーを外部公開する。
REM 事前準備: cloudflared をインストール (winget install --id Cloudflare.cloudflared)
REM 起動するとランダムな *.trycloudflare.com URL が表示される。

cd /d "%~dp0\..\.."

REM Streamlit をバックグラウンド起動
start /b streamlit run kaiseki_core\viewer\app.py --server.headless true --server.port 8501

REM 数秒待ってからトンネル起動
timeout /t 4 /nobreak >nul

REM Cloudflare Tunnel (URL は標準出力に表示される)
cloudflared tunnel --url http://localhost:8501

pause
