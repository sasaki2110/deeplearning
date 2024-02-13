import csv
with open("./mushroom/agaricus-lepiota.data") as fp:
    table = list(csv.reader(fp))

import random


# 特徴量の整理
label = [
    "かさの形状 (b:ベル状, c:円錐, x:凸型, f:平面, k:こぶ状, s:凹型)",
    "かさの表面 (f:繊維質, g:みぞ状, y:うろこ状, s:つるつる)",
    "かさの色 (n:茶色, b:黄褐色, c:肉桂色, g:灰色, r:緑色, p:ピンク, u:紫色, e:赤色, w:白色, y:黄色)",
    "傷の有無 (t:あり, f:なし)",
    "におい (a:アーモンド, l:アニス, c:クレソート, y:生臭い, f:腐敗臭, m:カビ臭い, n:無臭, p:刺激臭, s:スパイシー)",
    "ひだの付き方 (a:密着, d:下がっている, f:フリー, n:刻み目)",
    "ひだの間隔 (c:近い, w:密集, d:離れている)",
    "ひだの大きさ (b:広い, n:狭い)",
    "ひだの色 (k:黒色, n:茶色, b:黄褐色, h:チョコレート色, g:灰色, r:緑色, o:オレンジ, p:ピンク, u:紫色, e:赤色, w:白色, y:黄色)",
    "柄の形状 (e:広がっている, t:狭まっている)",
    "柄の先端 (b:球状, c:棒状, u:カップ, e:等幅, z:枝状, r:根っこ状, ?:欠けている)",
    "柄の表面(リングの上) (f:繊維質, y:うろこ状, k:すべすべ, s:つるつる)",
    "柄の表面(リングの下) (f:繊維質, y:うろこ状, k:すべすべ, s:つるつる)",
    "柄の色(リングの上) (n:茶色, b:黄褐色, c:肉桂色, g:灰色, o:オレンジ, p:ピンク, e:赤色, w:白色, y:黄色)",
    "柄の色(リングの下) (n:茶色, b:黄褐色, c:肉桂色, g:灰色, o:オレンジ, p:ピンク, e:赤色, w:白色, y:黄色)",
    "覆いの形状 (p:部分, u:全体)",
    "覆いの色 (n:茶色, o:オレンジ, w:白色, y:黄色)",
    "リングの数 (n:なし, o:ひとつ, t:ふたつ)",
    "リングの形状 (c:網状, e:減衰, f:派手, l:大型, n:なし, p:垂下, s:ぴったり, z:帯状)",
    "胞子の色 (k:黒色, n:茶色, b:黄褐色, h:チョコレート色, r:緑色, o:オレンジ, u:紫色, w:白色, y:黄色)",
    "個体数 (a:豊富, c:群生, n:おびただしい, s:ところどころに, v:数本, y:孤立)",
    "生育地 (g:芝生, l:葉っぱ, m:牧草地, p:道端, u:都市, w:汚物, d:樹木)"
]

random.shuffle(table)
# 最初の7000件を訓練データとして使う。
train = table[:7000]
# 残りをテストデータとして使う。
test = table[7000:]

# answers:ある特徴量の答えを列挙した、一次元配列
# 一番多い答えを、この特徴量の正解として返す。
def findbest(answers):
    # 要素別の件数をカウントするカウンター
    countOf = {}

    # この特徴量の列挙された答えをループ
    for answer in answers:
        # この答えが、すでにカウントされているか？
        if answer in countOf:
            # カウントアップ
            countOf[answer] += 1
        else:
            # あらたに答えを、件数１として登録
            countOf[answer] = 1

    # とりあえず、答えの一番目を正解と仮おき
    
    best = answers[0]

    # カウンターに含まれた答えでループ
    for answer in countOf.keys():
        #print(f"answer = {answer} countOf[answer] = {countOf[answer]}")
        # その答えの件数が、仮おきされた答えの数よりおおければ
        if countOf[best] < countOf[answer] :
            # この答えを、新たに正解として仮おき
            best = answer

    # 最終的にもっとも件数の多かった答えを返す
    return best

# 特徴量の数だけ繰り返す
for i in range(22):
    for j in range(22):
        if i < j:

            # n番目（=i）の特徴量yを使った決定木を作成。
            # ここで feats は、特徴量yの値と「答えの列挙」をペアで格納したリスト。
            feats = {}
            for row in train:
                answer = row[0]  # 回答
                f1 = row[i+1]       # 特徴量y
                f2 = row[j+1]       # 特徴量y
                f = f1 + "," + f2   # 特徴量yの値と、特徴量yの値を連結したもの。         # 特徴量yの値と、特徴量yの値を連結したもの。
                if not (f in feats):
                    # キーが存在しないとき、空のリストから始める。
                    feats[f] = []
                feats[f].append(answer)

            #print(feats)

            # 各特徴量の取りうる値別の正解（つまり答えの列挙の中で、最も数が多いもの）を探し、この特徴量のルールとする。
            rule = {}
            for f in feats.keys():
                #print("key = ", f, "dataLengs = ", len(feats[f]), "best = ", findbest(feats[f]))
                rule[f] = findbest(feats[f])

            print("この特徴量", label[i], "\nが取りうる値別の正解は = ", rule)

            # できた決定木の正しさをテストデータを用いて測定

            score = 0
            for row in test:
                answer = row[0]  # 回答
                f1 = row[i+1]       # 特徴量yの値
                f2 = row[j+1]       # 特徴量yの値
                f = f1 + "," + f2   # 特徴量yの値と、特徴量yの値を連結したもの。         # 特徴量yの値と、特�         # 特徴量yの値と、特徴量yの値を連結したもの。
                if (f in rule) and (rule[f] == answer):
                    # 規則を使った結果、正しい回答を出せれば得点。
                    score = score + 1

            print("正しさ指数は = ", score / len(test) * 100, "%")
            print("\n")