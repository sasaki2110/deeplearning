# DeepLarning入門
## PyTorchのチュートリアルをやっても頭に入ってこない
  基本的に何かが足りていない。
  その何かを埋めるために、これをやってみるか。
  
  https://euske.github.io/introdl/index.html
  
### 第1回 プログラマのためのPython入門
  とりあえずとばすか。

### 第2回 機械が「学習する」とはどういうことか?
  プログラマー的には、総当たりをするってことか。
  
  訓練データを用いて、特徴量 yi が取りうる値別に、 答えの件数をカウントし、一番多いのを、yi の 答えと仮置き
  それを、評価データを用いて、正しさ具合を評価すると。

  よりよくするために、特徴量 yi だけではなく、yi + yj や ji + yj + yk と特徴値を複合して精度を上げると

  やっぱりプログラマー視点じゃないから、頭に入ってこない・・・・・・

## よりプログラマー視点が強い、こっちを見てみよう
  @ITなら、プログラマーよりでしょう。
  https://atmarkit.itmedia.co.jp/ait/series/18508/


### 機械学習やディープラーニングってどんなもの？
  座学だけ。だけど、相当解りやすかった。

### Hello Deep Learning：ニューラルネットワークの作成手順
  ここではまたiris（アヤメ）を利用
  まあ、疑問点は残りながらも、大体の意味を理解して、一旦NNを作り上げた。

### データセット、多次元配列、多クラス分類
  まあ、出力層を３つ化も動いたは動いた。
  ただ、詳細解説が先延ばしになってるから、理解してない感が強い。

### ニューラルネットワークの内部では何が行われている？
  ここからが見たいところじゃないかな？
    重みとバイアスはどこにある？
    重みとバイアスを使った計算の実際
    重みとバイアスの更新
　
　重みとバイアスは各層の処理（例えば入力層のfc1()にある）

  重みとバイアスを使った計算の実際
    線形変換を実行している。
    fc1() = Linnear() でこれがされる。
    実際には x1 * w1 + x2 * w2 ・・・・・・・・・・・・ x4 * w4 + b と同じ事をしている。
  
  重みとバイアスの更新
    訓練して、lossを損失関数で求め、それをbackword()して、最適化関数をstep()すると、
    たしかに重みとバイアスは更新されている。

### ニューラルネットワークの学習でしていること
  つまりは
  ０．オプティマイザーの傾きを初期化（２回目以降でいいけど、１回目もやると）
  １．損失関数でロスを求める
  ２．ロスをバックワードlocc.backword() する事で、損失の現在地を偏微分し勾配ベクトル（傾きを求める）
  ３．最適化（オプティマイザー）を実行step()する事で、傾きの方向へ重みを更新する。

### 自分だけのLinearクラスを作ってみよう