#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: Utils.py
@time: 3/7/17 10:56 PM
@desc:
'''

import Constants
import pandas as pd
from Strtokenizer import *
import jieba.posseg as pseg


# 采用词性过滤的方式来过滤对弹幕挖掘没有实际意义的词 具体可查 http://www.cnblogs.com/adienhsuan/p/5674033.html
POS_tag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj","ul","r", "eng"]


class Utils(object):

    danmu_dir = None
    # 下面这几个变量都是给TPTM准备的
    user = {}
    comments = []
    _comments = []
    user_num = 0
    comments_num = 0
    shots = None
    _shots = None
    com = None
    stopwords = None
    split_num = 0

    # @tested
    def __init__(self):
        # do nothing
        print("util init")

    # @tested
    def parse_args(self,argc,argv,pmodel):
        model_status = Constants.MODEL_STATUS_UNKNOWN
        dir = ""
        model_name = ""
        dfile = ""
        alpha = -1.0
        beta = -1.0
        K = 0
        niters = 0
        savestep = 0
        twords = 0
        withrawdata = 0
        split_num = 10

        i = 0
        while i < argc:
            arg = argv[i]
            if arg == "-est":
                model_status = Constants.MODEL_STATUS_EST
            elif arg == "-estc":
                model_status = Constants.MODEL_STATUS_ESTC
            elif arg == "-inf":
                model_status = Constants.MODEL_STATUS_INF
            elif arg == "-dir":
                i += 1
                dir = argv[i]
            elif arg == "-dfile":
                i += 1
                dfile = argv[i]
            elif arg == "-model":
                i += 1
                model_name = argv[i]
            elif arg == "-alpha":
                i += 1
                alpha = float(argv[i])
            elif arg == "-beta":
                i += 1
                beta = float(argv[i])
            elif arg == "-ntopics":
                i += 1
                K = int(argv[i])
            elif arg == "-niters":
                i += 1
                niters = int(argv[i])
            elif arg == "-savestep":
                i += 1
                savestep = int(argv[i])
            elif arg == "-twords":
                i += 1
                twords = int(argv[i])
            elif arg == "-split_num":
                i += 1
                pmodel.split_num = split_num = int(argv[i])
            elif arg == "-withrawdata":
                withrawdata = 1
            i += 1

        if model_status == Constants.MODEL_STATUS_EST :
            if dfile == "" :
                print("Please specify the input data file for model estimation! \n")
                return 1
            pmodel.model_status = model_status
            if K > 0 :
                pmodel.K = K
            if alpha >= 0.0 :
                pmodel.alpha = alpha
            else :
                pmodel.beta = 50.0/K
            if beta >= 0.0 :
                pmodel.beta = beta
            if niters > 0 :
                pmodel.niters = niters
            if savestep > 0 :
                pmodel.savestep = savestep
            if twords > 0 :
                pmodel.twords = twords
            pmodel.dfile = dfile
            idx = re.search('/[0-9a-zA-Z.]+$',dfile)
            if not idx :
                pmodel.dir = dir
            else :
                pmodel.dir = dfile[0:idx.start()+1]
                pmodel.dfile = dfile[idx.start()+1:]
                print("dir = ",pmodel.dir,'\n')
                print("dfile = ",pmodel.dfile,'\n')
        if model_status == Constants.MODEL_STATUS_ESTC :
            if dir == '' :
                print("Please specify model diractory!\n")
                return 1
            if model_name == '' :
                print("Please specify model name upon that you want to continue estimating!\n")
                return 1
            pmodel.model_status = model_status
            if dir[len(dir)-1] != '/' :
                dir += '/'
            pmodel.dir = dir
            pmodel.model_name = model_name
            if niters > 0 :
                pmodel.niters = niters
            if savestep > 0 :
                pmodel.savestep = savestep
            if twords > 0 :
                pmodel.twords = twords
            # read <model>.others file to assign values for ntopics, alpha, beta, etc.
            if self.read_and_parse(pmodel.dir + pmodel.model_name + pmodel.others_suffix,pmodel) :
                return 1
        if model_status == Constants.MODEL_STATUS_INF :
            if dir == '' :
                print("Please specify model diractory!\n")
                return 1
            if model_name == '' :
                print("Please specify model name for inference!\n")
                return 1
            if dfile == '' :
                print("Please specify the new data file for inference!\n")
                return 1
            pmodel.model_status = model_status
            if dir[len(dir) - 1] != '/':
                dir += '/'
            pmodel.dir = dir
            pmodel.model_name = model_name
            pmodel.dfile = dfile
            if niters > 0:
                pmodel.niters = niters
            else :
                pmodel.niters = 20
            if twords > 0:
                pmodel.twords = twords
            if withrawdata > 0 :
                pmodel.withrawstrs = withrawdata
            # read <model>.others file to assign values for ntopics, alpha, beta, etc.
            if self.read_and_parse(pmodel.dir + pmodel.model_name + pmodel.others_suffix, pmodel):
                return 1
        if model_status == Constants.MODEL_STATUS_UNKNOWN :
            print("Please specify the task you would list to perform (-est/-estc/inf)!\n")
            return 1

        return 0

    # @tested
    def read_and_parse(self,filename,pmodel):
        # open file <model>.others to read:
        # alpha=?
        # beta=?
        # ntopics=?
        # ndocs=?
        # nwords=?
        # citer=?  # current iteration (when the model was saved)
        fin = open(filename)
        if not fin :
            print("Cannot open file ",filename," \n")
            return 1
        line = fin.readline()
        while line :
            strtok = Strtokenizer(line,'= \t\r\n')
            count = strtok.count_tokens()
            if count != 2 :
                continue
            optstr = strtok.token(0)
            optval = strtok.token(1)
            if optstr == 'alpha' :
                pmodel.alpha = float(optval)
            elif optstr == 'beta' :
                pmodel.beta = float(optval)
            elif optstr == 'ntopics' :
                pmodel.K = int(optval)
            elif optstr == 'ndocs' :
                pmodel.M = int(optval)
            elif optstr == 'nwords' :
                pmodel.V = int(optval)
            elif optstr == 'liter' :
                pmodel.liter = int(optval)
            line = fin.readline()
        fin.close()
        return 0

    # @tested
    def generate_model_name(self,iter):
        model_name = 'model-'
        buff = ''
        if 0 <= iter and iter < 10 :
            buff += '0000' + str(iter)
        elif 10 <= iter and iter < 100 :
            buff += '000' + str(iter)
        elif 100 <= iter and iter < 1000:
            buff += '00' + str(iter)
        elif 1000 <= iter and iter < 10000:
            buff += '0' + str(iter)
        else :
            buff += str(iter)

        if iter >= 0 :
            model_name += buff
        else :
            model_name += 'final'
        return model_name

    # @tested
    # 冒泡排序...-_-
    def sort(self,probs,words):
        for i in range(len(probs)-1) :
            for j in range(i+1,len(probs)) :
                if probs[i] < probs[j] :
                    tempprob = probs[i]
                    tempword = words[i]
                    probs[i] = probs[j]
                    words[i] = words[j]
                    probs[j] = tempprob
                    words[j] = tempword
        return 0

    # @tested
    # 归并排序...  本来vect用的是 vector<pair<int,double>> 的数据结构,在python里面我就用 list[dict,dict,...]这种方式来代替了
    def quicksort(self,vect,left,right):
        l_hold = left
        r_hold = right
        pivotidx = left
        pivot = vect[pivotidx]      # pivot 是 dict

        while left < right :
            while list(vect[right].values())[0] <= list(pivot.values())[0] and left < right :       # 这里有个强制转换成list的trick,本来是view,这是个更加dynamic的数据结构
                right -= 1
            if left != right :
                vect[left] = vect[right]
                left += 1
            while list(vect[left].values())[0] >= list(pivot.values())[0] and left < right :
                left += 1
            if left != right :
                vect[right] = vect[left]
                right -= 1
        vect[left] = pivot
        pivotidx = left
        left = l_hold
        right = r_hold

        if left < pivotidx :
            self.quicksort(vect,left,pivotidx-1)
        if right > pivotidx :
            self.quicksort(vect,pivotidx+1,right)
        return 0



    # @tested
    '''
    Api_name : CountKey
    Argc :  filename (读取文件名)
            resultName (输出文件名)
    Func :  对输入文件统计里面所有单词词频
    '''
    # 读取弹幕信息并且分割
    # encoding: utf-8

    # 统计关键词及个数 （根据文件）
    def CountKey(self, fileName, resultName):
        try:
            # 计算文件行数
            lineNums = len(open(fileName, 'rU').readlines())

            # 统计格式 格式<Key:Value> <属性:出现个数>
            i = 0
            table = {}
            source = open(fileName, "r")
            result = open(resultName, "w")

            while i < lineNums:
                line = source.readline()
                line = line.rstrip()
                # print line

                words = line.split(" ")  # 空格分隔
                # print str(words).decode('string_escape') #list显示中文
                
                # 字典插入与赋值
                for word in words:
                    if word != "" and table.has_key(word):  # 如果存在次数加1
                        num = table[word]
                        table[word] = num + 1
                    elif word != "":  # 否则初值为1
                        table[word] = 1
                i = i + 1

            # 键值从大到小排序 函数原型：sorted(dic,value,reverse)
            dic = sorted(table.iteritems(), key=lambda asd: asd[1], reverse=True)
            word_fre = pd.DataFrame(dic)
            for i in range(len(dic)):
                # print 'key=%s, value=%s' % (dic[i][0],dic[i][1])
                result.write("<" + dic[i][0] + ":" + str(dic[i][1]) + ">\n")
            return word_fre

        except Exception as e:
            print('Error:'+str(e))
        finally:
            source.close()
            result.close()
            # print 'END\n\n'

    # @tested
    '''
    Api_name : cut_word
    Argc : line (文件中的一行)
    Func : 对 line 进行分词
    '''
    def cut_word(self,line):
        return [ word  for word,flag in pseg.cut(line)  \
                  if word not in self.stopwords and flag not in \
                  POS_tag]


    # @tested
    '''
    Api_name : mktrncdocs
    Argc : None
    Func : 利用 shots 生成一个符合原始LDA需求的输入文档格式
    '''
    def mktrncdocs(self):
        # 生成 trndocs.dat 文件
        # 该文件就是视频的剪切 -----> 分成了 split_num 份数，每一份代表一篇文档
        f = open('test_data/trndocs.dat','w')
        f.write(str(self.comments_num)+'\n')
        for i in range(self.split_num):
            for j in range(len(self.shots[i])):
                for k in range(len(self.shots[i][j])):
                    f.write(self.shots[i][j][k].encode('utf-8')+' ')
                f.write('\n')

        # 这个生成的是给 LDA 用的 一个 shot 一行
        f = open('test_data/trndocs2.dat', 'w')
        for shot in self.shots:
            for comments in shot:
                for comment in comments:
                    f.write(comment.encode('utf-8') + ' ')
            f.write('\n')


    '''
    用来读取源弹幕文件，生成新的数据，并且生成符合LDA的训练数据
    data:
            user : dict
            comments : list
            user_num : number
            comments_num : number
            shots : list
            com : list
            stopwords : dict
    '''
    def process_source(self,danmu_dir,split_num):

        self.danmu_dir = danmu_dir.split('/')[1].split('.')[0]
        self.split_num = split_num
        # 数据和停用词
        danmu = open(danmu_dir)

        # stopwords 读取
        self.stopwords = {}.fromkeys([line.rstrip().decode('utf-8') for line in open('data/stopwords.txt')])
        
        new_stop_words = [
                    " ","the","of","is","and","to","in","that","we","for",\
                    "an","are","by","be","as","on","with","can","if","from","which","you",
                    "it","this","then","at","have","all","not","one","has","or","that","什么","一个"
                    ]
        for word in new_stop_words:
            self.stopwords[word.decode('utf-8')] = None

        self.comments = []
        self._comments = []
        # 读取文件，分析后存储到 user 和 comments
        for line in danmu.readlines()[:-1]:
            start = line.find('p=')
            stop = line.find('">')
            sub1 = line[start + 3:stop]
            t = sub1.split(',')[0]
            sub1 = sub1.split(',')[6]
            start = line.find('">')
            stop = line.find('</d>')
            sub2 = line[start + 2:stop].decode('utf-8')
            self.comments.append((float(t), sub2))
            self._comments.append((float(t), sub1, sub2))

        # 排序，分割comments ---> shots
        self.comments = sorted(self.comments)
        self._comments = sorted(self._comments,key=lambda _comments : (_comments[0],_comments[2]))
        spli = (self.comments[-1][0] - self.comments[0][0]) / self.split_num
        self.shots = []
        self._shots = []
        for i in range(self.split_num):
            self.shots.append([x[1] for x in self.comments if x[0] > i * spli and x[0] <= (i + 1) * spli])
            self._shots.append([[x[1], x[2]] for x in self._comments if x[0] > i * spli and x[0] <= (i + 1) * spli])
        self.shots[0].insert(0, self.comments[0][1])
        self._shots[0].insert(0, [self._comments[0][1], self._comments[0][2]])

        self.com = self.shots[:]  # 复制 shots ,因为后面会对shots处理，而后面还需要这个shots原来的数据

        for i in range(self.split_num):
            self.shots[i] = map(self.cut_word, self.shots[i])
        
        
        # 过滤掉一个字的单词
        location = []
        for shot in self.shots:
            for comment in shot:
                count3 = 0
                for word in comment:
                    if len(word) <= 1:
                        comment.pop(count3)
                    count3 += 1
       
        
        # 去掉分词后变成空值的单词 以及 短语少于两个的
        count = 0
        location = []
        for shot in self.shots:
            step = 0
            count2 = 0
            counter = []
            for comment in shot:
                if len(comment) < 2:
                    location.append((count, step-count2)) # 这里有个小修改，减去count2是因为pop的机制问题
                    counter.append(step-count2)
                    count2 += 1
                step += 1
            for i in counter:
                shot.pop(i)
            count += 1
            

        # 弹出那些在 _shot 里对应的分词后变为空的comment
        for lo in location:
            self._shots[lo[0]].pop(lo[1])

        # 弹出那些在 com 里对应的分词后变为空的comment , 这个是必须要做的，否则后面用到com的地方会溢出
        for lo in location:
            self.com[lo[0]].pop(lo[1])
           
        

        for uc_all in self._shots:
            for uc in uc_all:
                temp = []
                if not self.user.has_key(uc[0]):
                    temp.append(uc[1])
                    self.user[uc[0]] = temp
                else:
                    self.user[uc[0]].append(uc[1])

        # 统计user的个数 , 现在统计的是这个文档里的user，后期要做成对所有文档的统计量，还要能支持增量
        self.user_num = len(self.user)

        # comments的数量
        self.comments_num = 0
        for i in range(self.split_num):
            self.comments_num += len(self.shots[i])

        # 生成训练文本
        self.mktrncdocs()

        # 生成每个片段所有单词的文档，并且统计词频
        f = open('data/comments.txt', 'w')
        for i in range(self.split_num):
            for x in self.shots[i]:
                for word in x:
                    f.write(word.encode('utf-8') + ' ')
            f.write('\n')
        f.close()

        self.word_fre = self.CountKey('data/comments.txt', 'data/comments_map')
