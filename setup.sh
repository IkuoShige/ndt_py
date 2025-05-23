#!/bin/bash

# uv（Python仮想環境マネージャー）をインストール
echo "uvをインストールしています..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv環境変数を設定
export PATH="$HOME/.cargo/bin:$PATH"

# 仮想環境の作成
echo "Python仮想環境を作成しています..."
uv venv

# 仮想環境をアクティベート
source .venv/bin/activate

# 依存関係をインストール
echo "依存関係をインストールしています..."
uv pip install -r requirements.txt

echo "セットアップが完了しました！"
echo "今後は以下のコマンドで仮想環境をアクティベートしてください："
echo "source .venv/bin/activate"
