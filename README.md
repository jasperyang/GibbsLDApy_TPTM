# GibbsLDApy_TPTM

在我的GibbsLDApy的项目基础上改的，做成的论文 Crowdsourced Time-sync Video Tagging using Temporal and Personalized Topic Modeling 的工程实现。

个人觉得这篇论文在构想上修改lda使之适应有时间关系并且有用户关系的评论是一个很有意思以及很有意义的创新，但是个人认为lda在弹幕分析或说是短文本分析已然式微。

使用方法就是我的

	test1.ipynb
    
因为是实验性的东西，用jupyter最方便。

如果没有安装jupyter也没有关系

	python LDA.py -est -alpha 0.5 -beta 0.1 -ntopics 100 -niters 1000 -savestep 100 -twords 20 -dfile test_data/dfile -split_num 10


