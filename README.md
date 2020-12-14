# 深層強化学習による振動制御

１質点系モデルにサンプル地震動を入力した際の応答を、深層強化学習を用いて制御する時刻歴応答解析プログラムです。各ステップごとに減衰定数を変化させて絶対応答加速度の最小化を目指します。

OpenSeesPyを用いていますが、こちらのコードをいじる場合には、DockerとVSCodeの使用がおすすめです。その場合には、こちらも参考にしてください。
https://github.com/kakemotokeita/openseespy-docker-vscode

## 振動制御の学習の概要
概要は、以下のリンクより、Google Colaboratoryでもご確認いただけます。

https://colab.research.google.com/github/kakemotokeita/dqn-seismic-control/blob/master/seismic_analysis.ipynb

## プログラム
強化学習を用いない通常の１質点系振動解析プログラムを確認したい方は、こちらからご確認いただけます。

https://github.com/kakemotokeita/dqn-seismic-control/tree/master/scripts/main.py

プログラムや記述に誤りがある場合や、改善点などありましたら、issues、PRなどからお知らせください。

