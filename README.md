# chainerTrain

sample01.py

ANDを学習させるプログラム
教師付き、1000回で学習する。（SGD関数の引数がかなり効いている感じ）

sample02.py

XORを学習させるプログラム
教室付き、NNが１層では線形しか学習できないので、２層にしてある。

sample03.py

XORを学習するプログラム
結果を0,1の２値にするため回帰を使い、mean_square_errorでerrorを求めるようにしたが
なぜか自分の環境ではエラーで動かない。

sample05.py

XORを学習するプログラム
mean_square_errorを使って動く例。
プログラムがオブジェクト化していないため読みにくい。
optimizerの使い方が以前のsampleと異なっていたため、その練習の後が残っている。
（結局、学習はできるが学習速度が目に見えて違う）

sample07.py

sample05.pyを元に、基本的な手順を分かりやすいよう関数化。

sample08.py

sample01.pyを参考に3入力を作成してみた。

sample10

sample07.pyを元に作成。３次元の入力で任意の学習パターンができることを確認。
やはり、softmax_cross_emtropy()は線形のものしか対応できないようで、mean_squared_error（）だとうまくいく。


基本、他人のHPにあった例をそのまま理解した内容をコメントしている。

