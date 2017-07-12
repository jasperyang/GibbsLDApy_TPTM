#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: TPTM.py
@time: 5/18/2017 11:00 PM
@desc:   这个部分是功能性拓展，基于danmuLDA的实现部分
        1. 读取B站格式弹幕
        2. 生成各类矩阵 delta_s,delta_c等等
        3. 更新 lambda_s和 x_u_c_t 的函数
        4. 生成给原始LDA的alpha向量
'''

import eulerlib
import numpy as np
import math
import pandas as pd
from scipy import stats as st  # 用户正态函数的pdf
from multiprocessing import Pool,Manager # 可以使用多核的进程池,这个value是一个 shared memory变量,用于多进程里的变量共享
import time
import pickle
from Utils import *
import threading


lock = Manager().Lock()
manager = Manager()
result_list = manager.list() # 这个是专门在计算delta的多线程中使用的一个全局变量，用于保存如下格式 (i,j,cos)
pool_size = 8

# @tested
'''
Api_name : lgt
Argc : y
Func : lgt函数
'''
def lgt(y):
    return math.log(1 + math.exp(y))


# @tested
'''
Api_name : dlgt
Argc : y
Func : lgt求导
'''
def dlgt(y):
    return 1 / ((1 + math.exp(y)) * np.log(10))



# @tested
'''
Api_name : delta_s
Argc : i (index)
    j (index)
Func : 利用result生成delta，delta就是shot与shot之间的余弦距离
       useless!
'''
def do_delta_s(i, j, result_s_i, result_s_j,lock):
    numerator = np.sum(result_s_i.time * result_s_j.time)
    denominator = pow(np.sum(pow(result_s_i.time, 2)), 0.5) * pow(np.sum(pow(result_s_j.time, 2)), 0.5)
    if denominator != 0:
        cos = numerator / denominator
    else:
        cos = 0
    with lock:
        result_list.append((i, j, cos))


# 配合 multiprocessing pool 对多参数的要求添加的函数
def multi_do_delta_s(args):
    do_delta_s(*args)


# @tested
'''
Api_name : delta_c
Argc : i (index)
       j (index)
Func : 利用result生成delta，delta就是shot与shot之间的余弦距离
       useless!
'''


def do_delta_c(i, j, result_c_i, result_c_j,lock):
    merged = pd.merge(result_c_i, result_c_j, how='inner', on=0)
    numerator = np.sum(merged.time_x * merged.time_y)
    denominator = pow(np.sum(pow(result_c_i.time, 2)), 0.5) * pow(np.sum(pow(result_c_j.time, 2)), 0.5)
    if denominator != 0:
        cos = numerator / denominator
    else:
        cos = 0.0
    if cos != 0.0:
        with lock:
            result_list.append((i, j, cos))


# 配合 multiprocessing pool 对多参数的要求添加的函数
def multi_do_delta_c(args):
    do_delta_c(*args)


# @tested
'''
Api_name : calculate_lambda_s
Argc : shot
       start
Func : 计算关于输入shot的lambda值
       # 进程函数 --> 计算 yita_lambda_s
'''

def calculate_lambda_s(shot, start, total_topic,lambda_s,shots,comment_all,M_pre_c,eur,word2id,column,nw,yita,M_pre_s,z,lock):
    for topic in range(total_topic):
        result = 0
        lam_s = lambda_s[shot][topic]
        for comment in range(len(shots[shot])):
            x_u = comment_all.ix[comment + start, 2 + total_topic + 2 + total_topic + topic]
            m_pre_c = M_pre_c[comment + start][topic]
            t1 = x_u * dlgt(x_u * lam_s + m_pre_c)
            t2 = []
            for t in range(total_topic):
                t2.append(lgt(
                    comment_all.ix[comment + start, 2 + total_topic + 2 + total_topic + t] * lam_s + M_pre_c[comment + start][t]))
            t2 = sum(t2)
            t3 = t2
            t2 = eur.phi(t2)
            t3 = eur.phi(t3 + len(shots[shot][comment]))
            n_tc = 0
            for t in z[start+comment]:
                if t == topic :
                    n_tc += 1
            t4 = eur.phi(lgt(x_u * lam_s + m_pre_c) + n_tc)
            t5 = eur.phi(lgt(x_u * lam_s + m_pre_c))
            result += t1 * (t2 - t3 + t4 - t5)
        with lock:
            # print(yita * (-(lam_s + M_pre_s[shot][topic]) / (lam_s * lam_s) + result))
            result_list.append((shot,topic,lambda_s[shot][topic] - yita * (-(lam_s + M_pre_s[shot][topic]) / (lam_s * lam_s) + result)))

def multi_calculate_lambda_s(args):
    calculate_lambda_s(*args)


# @tested
'''
Api_name : calculate_x_u_c_t
Argc : i
       start
Func : 计算关于输入每个用户的topic分布
       # x_u_c_t 的更新代码
       # 注意 ：这里的 comment_all 已经排过序了，和上面的不一样
'''

def calculate_x_u_c_t(i, start,total_topic,user_ct,comment_all_sort,x_u_c_t,eur,shots,word2id,nw,column,yita,lambda_s,z):
    for topic in range(total_topic):
        result = 0
        for j in range(start, start + user_ct.ix[i, 0]):
            shot = comment_all_sort.ix[j,2 + total_topic+1]
            lambda_s_t = lambda_s[shot][topic]
            m_pre_c_t = comment_all_sort.ix[j, 2 + total_topic + 2 + topic]
            x_u = x_u_c_t.iat[i, topic + 1]
            t1 = lambda_s_t * dlgt(x_u * lambda_s_t + m_pre_c_t)
            t2 = []
            for k in range(total_topic):
                t2.append(lgt(comment_all_sort.ix[j, 2+total_topic+2+total_topic+k] * lambda_s[shot][k] + comment_all_sort.ix[
                                       j, 2+total_topic+2+k]))
            t3 = eur.phi(sum(t2) + len(
                shots[int(comment_all_sort.ix[j, ['shot']])][int(comment_all_sort.ix[j, ['com']])]))
            t2 = eur.phi(sum(t2))
            n_tc = 0
            for t in z[j]:
                if t == topic :
                    n_tc += 1
            t4 = eur.phi(lgt(x_u * lambda_s_t + m_pre_c_t) + n_tc)
            t5 = eur.phi(lgt(x_u * lambda_s_t + m_pre_c_t))
            result += t1 * (t2 - t3 + t4 - t5)
            result_list.append((i,topic+1,x_u_c_t.iat[i, topic + 1] - yita * (-x_u / (comment_all_sort.ix[j, topic + 2] ** 2) + result))) # 这里会除一个sigma_u的平方，所以这个sigma_u不能设置成0.1，这样迭代没几次就会溢出了
            

def multi_calculate_x_u_c_t(args):
    calculate_x_u_c_t(*args)



'''
TPTM 的工具包

    updated : 1 取单个元素 iloc 改成 iat ，速度较快
                          loc  改成 at              -----> 主要加快了更新 x_u_c_t 的速度
                又做了更新，iat 改成 ix 了，取值较快
              2 更新了delta的计算方式，大幅节省内存，并且节约计算时间 -------> merge
'''


class TPTM(object):
    # 定义变量 默认数据 全部的类变量
    user = {}
    comments = []
    _comments = []
    _user = {}
    split_num = 10
    total_topic = 10
    iteration = 0
    yita = 0.005
    eur = None
    nw = None  # model 传进来的矩阵
    gamma_s = 0.5  # 我自己设的
    gamma_c = 0.3  # 论文中做实验得到的最好的值
    sigma_s = 0.1  # 应该是每个片段的都不一样，但是这里我认为其实每个片段的topic分布没有统计可能性，不合理，都设成一样的了

    user_num = 0
    comments_num = 0
    stopwords = None
    shots = None
    com = None
    delta_c = None
    delta_s = None
    result_s = None
    result_c = None
    M_pre_c = None
    M_pre_s = None
    comment_all = None
    comment_all_sort = None
    user_sigma = None
    pi_c = None
    x_u_c_t = None
    column = None
    word2id = None
    word_fre = None
    user_ct = None
    alpha_c = None
    z = None # lda的z矩阵

    ut = None  # Utils

    '''无参初始化，这是不能用的'''

    def __init__(self):
        # 总的topic个数
        self.total_topic = 10
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        print('danmu project init')
        # do nothing

    # @tested
    '''
    Api_name : init
    Argc :  splitnum (视频片段分割个数)
            total_topic (主题个数)
            pool_num (线程池的容量)
            nw_metric (单词对应topic的分布的矩阵)
            ut (Utils)
            z (lda)
            gamma_s
            gamma_c
            sigma_s
    Func :  danmu 初始化
    '''

    def __init__(self, split_num, total_topic,eur_maxnum, nw_metric, ut,z, gamma_s=0.1, gamma_c=0.2, sigma_s=0.1):
        self.user = {}
        self.comments = []
        self.split_num = split_num
        # 总的topic个数
        self.total_topic = total_topic
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        # 欧拉函数的定义
        self.eur = eulerlib.numtheory.Divisors(eur_maxnum)  # maxnum
        self.nw = nw_metric
        self.gamma_s = gamma_s
        self.gamma_c = gamma_c  # 一般设置 0.3
        self.sigma_s = sigma_s  # 应该是每个片段的都不一样，但是这里我认为其实每个片段的topic分布没有统计可能性，不合理，都设成一样的了，自己定一个1吧
        self.z = z
        
        self.ut = ut
        self.user = self.ut.user
        self.comments = self.ut.comments
        self.user_num = self.ut.user_num
        self.comments_num = self.ut.comments_num
        self.shots = self.ut.shots
        self.com = self.ut.com
        self.stopwords = self.ut.stopwords
        self.word_fre = self.ut.word_fre

        # 下面这两个原本是在 preprocessing.py 中生成

        self.MkFakeSigma()

        # 每一个用户的user-topic分布
        # sigma_u_t 是每个用户对于每一个topic的sigma值
        # 从文件里面读取每一个用户的每一个topic的sigma值
        # 每一行一个用户 (顺序就是下面生成的 user_ 中的顺序)
        self.user_sigma = pd.read_csv('data/sigma_u_t.csv')
        self.user_sigma = self.user_sigma.drop(['Unnamed: 0'], 1)
        self.user_sigma.fillna(1)

        # word2id 读取
        f = open('test_data/wordmap.txt')
        f.readline()
        wordmap = []
        line = f.readline()
        while line:
            temp = line.split(' ')
            wordmap.append([temp[0], temp[1][:-1]])
            line = f.readline()
        self.word2id = pd.DataFrame(wordmap)  # 读取单词对应id的表
        self.column = list(self.word2id)[0]  # 这个是因为第一行是单词的个数，会变成index，下面转换成字典后出现二级索引，所以做了处理
        self.word2id = self.word2id.set_index(0).to_dict()[1]
        print('danmu project init')

    # @tested
    '''
    Api_name : MkFakeSigma
    Argc : None
    Func : 制造一个假的 user-topic 分布
           这个函数很鸡肋，因为真的没什么用。。。
    '''

    def MkFakeSigma(self):
        user_num = len(self.user)
        f = open('data/sigma_u_t.csv', 'w')
        f.write(',user')
        for i in range(self.total_topic):
            f.write(',topic' + str(i))
        f.write('\n')
        for key in self.user.keys():
            f.write(',' + key)
            for j in range(self.total_topic):
                f.write(',1')
            f.write('\n')

    # @tested
    '''
    Api_name : preprocessing
    Argc : danmu_dir
    Func : 读取弹幕文件并分词处理，之后生成所需数据，这是重要的预处理
    生成数据：
             user : dict
             shots : list
             com : list
             lambda_s : array
             delta_c : array
             delta_s : array
             result_s : list
             result_c : list
             M_pre_c : array
             M_pre_s : array
             comment_all : dataframe
             comment_all_sort : dataframe
             pi_c : array
             x_u_c_t : dataframe
             user_ct : dataframe
    '''

    def preprocessing(self):

        self.com = self.ut.com

        '''
        # 计算每一个shot里面的所有的单词的词频 ------->   缺点：执行速度实在太慢了，后期需要修改 , 这一部分要执行十五分钟左右
        self.result_s = []
        for i in range(self.split_num):
            shot_word_fre = self.word_fre.copy()
            shot_word_fre['time'] = 0
            for x in self.shots[i]:
                for word in x:
                    index = shot_word_fre[self.word_fre[0] == word.encode('utf-8')].index
                    shot_word_fre.ix[index, 'time'] = shot_word_fre.ix[index, 'time'] + 1
            shot_word_fre = shot_word_fre.drop(1, 1)
            self.result_s.append(shot_word_fre)

        # 计算每一个comment的词频向量  -----------> 现在的办法是每个 comment 都有一个完整的词向量，便于后面的计算，问题是这样很占内存资源
        # 不按照每一个shot分片后内部的comment之间的delta计算，所有的comment进行计算
        self.result_c = []
        for i in range(self.split_num):
            for j in range(len(self.shots[i])):
                shot_word_fre = self.word_fre.copy()
                shot_word_fre['time'] = 0
                for word in self.shots[i][j]:
                    index = shot_word_fre[self.word_fre[0] == word.encode('utf-8')].index
                    shot_word_fre.ix[index, 'time'] = shot_word_fre.ix[index, 'time'] + 1
                shot_word_fre = shot_word_fre.drop(1, 1).reset_index().drop('index', 1)
                shot_word_fre = shot_word_fre[shot_word_fre.time > 0]
                self.result_c.append(shot_word_fre)
        

        # 这部分计算也十分耗时，我改成多线程了 ,速度很快了
        # 计算delta<s,_s> : 这里用的是词频向量 余弦值    -----> 下三角矩阵，后面方便计算
        # 从后面的shot往前计算
        self.delta_s = np.zeros((self.split_num, self.split_num))
        seq = range(self.split_num)
        # 修改 time 的数据类型 to float64
        for shot in self.result_s:
            shot.time = shot.time.astype('float64')

        delta_s_pool = Pool(pool_size)
        del result_list[:]  # 使用这个全局变量前先初始化
        lst_vars = []

        start_time = time.time()  # 下面的多线程开始执行的时间
        seq.reverse()
        for i in seq:
            for j in range(i):
                lst_vars.append((i, j, self.result_s[i], self.result_s[j],lock))
        delta_s_pool.map(multi_do_delta_s, lst_vars)
        delta_s_pool.close()
        delta_s_pool.join()
        print('calculate delta_s %d second' % (time.time() - start_time))
      
        
        # 赋值操作
        for x in result_list:
            self.delta_s[x[0]][x[1]] = x[2]

        # 计算delta<c,_c> : 这里用的是词频向量 余弦值    -----> 下三角矩阵，后面方便计算
        # 从后往前
        # 这里是不按照每个shot分开然后计算里面的comment
        seq = range(len(self.result_c))
        # 修改 time 的数据类型 to float64
        for i in seq:
            self.result_c[i].time = self.result_c[i].time.astype('float64')

        # list存储
        self.delta_c = np.zeros((len(self.result_c), len(self.result_c)))

        delta_c_pool = Pool(pool_size)
        del result_list[:]  # 使用这个全局变量前先初始化
        lst_vars = []

        start_time = time.time()  # 下面的多线程开始执行的时间
        for i in seq:
            for j in range(i):
                lst_vars.append((i, j, self.result_c[i], self.result_c[j],lock))
        delta_c_pool.map(multi_do_delta_c, lst_vars)
        delta_c_pool.close()
        delta_c_pool.join()
        print('calculate delta_c %d second' % (time.time() - start_time))

        # 赋值操作
        for x in result_list:
            self.delta_c[x[0]][x[1]] = x[2]
           
        
        '''
        
        # 计算 delta_s 和 delta_c ,之前理解错误,这个距离就是相差多少的距离
        self.delta_s = np.zeros((self.split_num, self.split_num))
        seq = range(self.split_num)
        seq.reverse()
        for i in seq:
            for j in range(i):
                self.delta_s[i][j] = np.exp(-self.gamma_c * (i-j))
        
        self.delta_c = np.zeros((self.comments_num, self.comments_num))
        seq = range(self.comments_num)
        seq.reverse()
        for i in seq:
            for j in range(i):
                self.delta_c[i][j] = np.exp(-self.gamma_c * (i-j))
        
        np.save('data/progress/delta_c.npy', self.delta_c)
        np.save('data/progress/delta_s.npy', self.delta_s)
        
        
        # 利用上面的用户对应评论的字典 make 一个 dataframe
        user_ = pd.DataFrame()
        temp1 = []
        temp2 = []
        for key in self.user.keys():
            for i in range(len(self.user[key])):
                temp1.append(key)
                temp2.append(self.user[key][i])
        user_['user'] = temp1
        user_['comment'] = temp2

        # 处理得到一个大表，里面包括所有评论以及评论的人，和每个人对应的所有的topic的sigma值
        # 这里处理之后好像有点问题，有些用户没有，下面我直接就都填充0.1了
        comment_per_shot = []
        for i in range(self.split_num):
            temp = pd.DataFrame(self.com[i])
            u = []
            tem = pd.DataFrame()
            for j in range(len(temp)):
                user_id = user_[user_.comment == temp.iat[j, 0]].iat[0, 0]
                index = temp[temp[0] == temp.iat[j, 0]].index[0]
                temp.drop(index)
                u.append(user_id)
                a = self.user_sigma[self.user_sigma.user == user_id].iloc[:, 1:]
                tem = tem.append(a)
            tem = tem.reset_index().drop(['index'], 1)
            temp['user'] = pd.DataFrame(u)
            temp = temp.join(tem)
            comment_per_shot.append(temp)
            
        
        # 有了上面的矩阵后，计算论文中提到的 M_pre_s 以及 M_pre_c
        # 需要两个衰减参数 gamma_s 以及 gamma_c
        # M_pre_s 比较好计算，M_pre_c 比较复杂一点，因为涉及到了每一个shot
        self.M_pre_s = np.zeros((self.split_num, self.total_topic))  # 行：shot个数    列：topic个数
        self.lambda_s = np.zeros((self.split_num, self.total_topic))
        
        # 先初始化 M_pre_s[0] 以及 lambda_s[0]
        mu = 0  # 初始的 M_pre_s[0] 都是0
        self.lambda_s = np.random.normal(size=(self.split_num,self.total_topic))
        
        for i in range(1, self.split_num):
            for topic in range(self.total_topic):  # 先循环topic
                numerator = 0
                denominator = 0
                for j in range(i):
                    numerator += np.exp(-self.gamma_s * self.delta_s[i][j]) * self.lambda_s[j][topic]
                    denominator += np.exp(-self.gamma_s * self.delta_s[i][j])
                self.M_pre_s[i][topic] = numerator / denominator
        
        '''# 这里不去管是不死符合lambda_s的分布要求了
        # 从 第1的开始
        self.lambda_s[0] = np.random.normal(mu, self.sigma_s, self.total_topic)
        for i in range(1, self.split_num):
            for topic in range(self.total_topic):  # 先循环topic
                numerator = 0
                denominator = 0
                for j in range(i):
                    numerator += np.exp(-self.gamma_s * self.delta_s[i][j]) * self.lambda_s[j][topic]
                    denominator += np.exp(-self.gamma_s * self.delta_s[i][j])
                self.M_pre_s[i][topic] = numerator / denominator
                s = np.random.normal(self.M_pre_s[i][topic], self.sigma_s, 1)
                self.lambda_s[i][topic] = np.random.normal(self.M_pre_s[i][topic], self.sigma_s,1)
        '''

        # 所有的 comment 的一个 dataframe ,comment-user_id-topic0,1,2...99 ，后面的topic分布是user_id的
        self.comment_all = pd.concat(comment_per_shot).reset_index().drop('index', 1)
        # 给那些没有topic分布的用户填充1 ----> 缺失值（就是生成用户的topic分布表没有生成全）
        self.comment_all = self.comment_all.fillna(1)
        self.comment_all = self.comment_all.rename(columns={0: 'comment'})

        # 生成 pi_c 和 M_pre_c 不同于上面，因为这里是对每个shot的面的comment进行操作
        # 先初始化 M_pre_c[0] 和 第0个 shot 里的第一个 comment 对应的 pi_c[0]
        self.M_pre_c = np.zeros((len(self.comment_all), self.total_topic))  # 行：shot个数    列：topic个数
        self.pi_c = np.zeros((len(self.comment_all), self.total_topic))
        self.alpha_c = np.ones((len(self.comment_all), self.total_topic))*0.5  # 初始化 alpha_c : 0.5
        
        ''' # 初始化 pi_c 的任务去除掉了
        for i in range(self.total_topic):
            self.pi_c[0][i] = self.lambda_s[0][i] * np.random.normal(mu, self.comment_all.iat[0, i + 2]) + self.M_pre_c[0][i]

        start = 0  # shot 之间的位移
        for q in range(self.split_num):
            if q == 0:
                for i in range(1, len(self.com[q])):
                    for topic in range(self.total_topic):  # 先循环topic
                        numerator = 0
                        denominator = 0
                        for j in range(i):
                            numerator += np.exp(-self.gamma_c * self.delta_c[i][j]) * self.pi_c[j][topic]
                            denominator += np.exp(-self.gamma_c * self.delta_c[i][j])
                        self.M_pre_c[i][topic] = numerator / denominator
                        self.pi_c[i][topic] = self.lambda_s[q][topic] * np.random.normal(mu, self.comment_all.iat[i, topic + 2]) + \
                                              self.M_pre_c[i][topic]
                start += len(self.com[q])
            else:
                for i in range(start, start + len(self.com[q])):
                    for topic in range(self.total_topic):  # 先循环topic
                        numerator = 0
                        denominator = 0
                        for j in range(i):
                            numerator += np.exp(-self.gamma_c * self.delta_c[i][j]) * self.pi_c[j][topic]
                            denominator += np.exp(-self.gamma_c * self.delta_c[i][j])
                        self.M_pre_c[i][topic] = numerator / denominator
                        self.pi_c[i][topic] = self.lambda_s[q][topic] * np.random.normal(mu, self.comment_all.iat[i, topic + 2]) + \
                                              self.M_pre_c[i][topic]
                start += len(self.com[q])
        '''

         # 将 comment_all 升级成一个新的大表 comment_all_sort 结构为 {comment,user_id,user_id的topic,shot,com,该comment属于的topic分布,该comment的user的topic分布},有了这个表，后面的处理会很方便
        temp = []
        for i in range(self.split_num):
            for j in range(len(self.shots[i])):
                t = pd.DataFrame({'shot':[i],'com':[j]})
                temp.append(t)
        a1 = pd.concat(temp)
        a1 = a1.reset_index().drop('index', 1)
        self.comment_all = pd.concat([self.comment_all, a1], axis=1)
        self.comment_all = pd.concat([self.comment_all, pd.DataFrame(self.M_pre_c)], axis=1) # comment_all 不 sort 版的留给更新 lambda_s 用
        self.comment_all_sort = self.comment_all.sort_values('user')  # 按照 user 排序
        # 生成 user-topic 分布的 dataframe
        self.x_u_c_t = np.zeros((len(self.comment_all_sort), self.total_topic))
        for i in range(len(self.comment_all_sort)):
            for topic in range(self.total_topic):
                self.x_u_c_t[i][topic] = np.random.normal(mu, self.comment_all_sort.iat[i, topic + 2], 1)
        user_id = self.comment_all_sort.drop_duplicates('user')['user'].reset_index().drop('index', 1)
        self.x_u_c_t = user_id.join(pd.DataFrame(self.x_u_c_t))
        
        self.comment_all = pd.merge(self.comment_all,self.x_u_c_t,how='left',on='user')
        self.comment_all_sort = self.comment_all.sort_values('user')  # 按照 user 排序
        

        # 每个人评论的次数的dataframe
        self.user_ct = pd.DataFrame(self.comment_all_sort.groupby('user').count()['topic0'])

        # 保存重要数据，矩阵和dataframe到 data/progress文件夹里
        properties = open('data/progress/properties.xml', 'w')
        properties.write('user_num:' + str(self.user_num) + '\n')
        properties.write('comments_num:' + str(self.comments_num) + '\n')
        properties.write('iteration:' + str(self.iteration) + '\n')
        output = open('data/progress/comments.pkl', 'wb')
        pickle.dump(self.comments, output, -1)
        output = open('data/progress/user.pkl', 'wb')
        pickle.dump(self.user, output)
        output = open('data/progress/shots.pkl', 'wb')
        pickle.dump(self.shots, output, -1)
        output = open('data/progress/com.pkl', 'wb')
        pickle.dump(self.com, output, -1)
        np.save('data/progress/lambda_s.npy', self.lambda_s)
        output = open('data/progress/result_s.pkl', 'wb')
        pickle.dump(self.result_s, output, -1)
        output = open('data/progress/result_c.pkl', 'wb')
        pickle.dump(self.result_c, output, -1)
        np.save('data/progress/M_pre_s.npy', self.M_pre_s)
        np.save('data/progress/M_pre_c.npy', self.M_pre_c)
        self.comment_all.to_csv('data/progress/comment_all.csv', encoding='utf-8')
        self.comment_all_sort.to_csv('data/progress/comment_all_sort.csv', encoding='utf-8')
        np.save('data/progress/pi_c.npy', self.pi_c)
        self.x_u_c_t.to_csv('data/progress/x_u_c_t.csv', encoding='utf-8')
        self.user_ct.to_csv('data/progress/user_ct.csv', encoding='utf-8')

    # @tested
    '''
    Api_name : load_stage_data
    Argc : None
    Func : 
    '''

    def load_stage_data(self):
        print('loading')
        pro = []
        properties = open('data/progress/properties.xml')
        for line in properties.readlines():
            pro.append(line.split(':')[1])
        self.user_num = int(pro[0][:-1])
        self.comments_num = int(pro[1][:-1])
        self.iteration = int(pro[2][:-1])
        input = open('data/progress/comments.pkl', 'rb')
        self.comments = pickle.load(input)
        input = open('data/progress/user.pkl', 'rb')
        self.user = pickle.load(input)
        input = open('data/progress/shots.pkl', 'rb')
        self.shots = pickle.load(input)
        # self.com = pickle.load(input)
        self.lambda_s = np.load('data/progress/lambda_s.npy')
        self.delta_c = np.load('data/progress/delta_c.npy')
        self.delta_s = np.load('data/progress/delta_s.npy')
        input = open('data/progress/result_s.pkl', 'rb')
        self.result_s = pickle.load(input)
        input = open('data/progress/result_c.pkl', 'rb')
        self.result_c = pickle.load(input)
        self.M_pre_s = np.load('data/progress/M_pre_s.npy')
        self.M_pre_c = np.load('data/progress/M_pre_c.npy')
        self.comment_all = pd.read_csv('data/progress/comment_all.csv')
        self.comment_all = self.comment_all.drop('Unnamed: 0', 1)
        self.comment_all_sort = pd.read_csv('data/progress/comment_all_sort.csv')
        self.comment_all_sort = self.comment_all_sort.drop('Unnamed: 0', 1)
        self.pi_c = np.load('data/progress/pi_c.npy')
        self.x_u_c_t = pd.read_csv('data/progress/x_u_c_t.csv')
        self.x_u_c_t = self.x_u_c_t.drop('Unnamed: 0', 1)
        self.user_ct = pd.read_csv('data/progress/user_ct.csv')
        self.user_ct.set_index('user',inplace=True)
        print('loading finished!')


    # @tested
    '''
     Api_name : update_lambda_s
     Argc : iteration (现阶段迭代的次数)
     Func : 更新所有shot的lambda_s分布
     '''

    def update_lambda_s(self, iteration):
        self.iteration = iteration
        self.yita = 0.0005 / pow(2, (self.iteration % 10))
        start_time = time.time()  # 下面的多线程开始执行的时间
        start = 0  # 初始化，用于控制在哪一个shot里面
        lst_vars = []
        del result_list[:]
        _pool = Pool(pool_size)
        for shot in range(len(self.shots)):
            lst_vars.append((shot, start, self.total_topic,self.lambda_s,self.shots,self.comment_all,self.M_pre_c,self.eur,self.word2id,self.column,self.nw,self.yita,self.M_pre_s,self.z,lock))
            start += len(self.shots[shot])  # start 增加位移，移动一个shot
        _pool.map(multi_calculate_lambda_s, lst_vars)
        _pool.close()
        _pool.join()

        print('updating lambda_s before assigning %d second' % (time.time() - start_time))
        # 更新
        for item in result_list:
            self.lambda_s[item[0]][item[1]] = item[2]
        # 保存每一轮迭代会改变的数据
        properties = open('data/progress/properties.xml', 'w')
        properties.write('user_num:' + str(self.user_num) + '\n')
        properties.write('comments_num:' + str(self.comments_num) + '\n')
        properties.write('iteration:' + str(self.iteration) + '\n')
        np.save('data/progress/lambda_s.npy', self.lambda_s)
        np.save('data/progress/M_pre_s.npy', self.M_pre_s)
        np.save('data/progress/M_pre_c.npy', self.M_pre_c)
        self.x_u_c_t.to_csv('data/progress/x_u_c_t.csv', encoding='utf-8')
        print('updating lambda_s %d second' % (time.time() - start_time))

    # @tested
    '''
     Api_name : update_x_u_c_t
     Argc : iteration (现阶段迭代的次数)
     Func : 更新所有用户的topic分布
     '''

    def update_x_u_c_t(self, iteration):
        self.iteration = iteration
        self.yita = 0.0005 / pow(2, (self.iteration % 10))
        start_time = time.time()  # 下面的多线程开始执行的时间
        start = 0  # 初始化，用于控制在哪一个shot里面
        del result_list[:]
        start = 0
        # 这里和更新lambda_s不一样，因为用户多，但是每个用户的评论量少，用多进程反而慢
        for i in range(len(self.user_ct)):
            calculate_x_u_c_t(i, start,self.total_topic,self.user_ct,self.comment_all_sort,self.x_u_c_t,self.eur,self.shots,self.word2id,self.nw,self.column,self.yita,self.lambda_s,self.z)
            print(i)
            start += self.user_ct.ix[i, 0]
        
        # 更新
        for item in result_list:
            self.x_u_c_t.iat[item[0],item[1]] = item[2]
        
        l = []
        for i in range(len(self.comment_all.columns)-(2+self.total_topic)):
            l.append((2+self.total_topic)+i)
        self.comment_all.drop(self.comment_all.columns[l],1)
        self.comment_all = pd.concat([self.comment_all, pd.DataFrame(self.M_pre_c)], axis=1)
        self.comment_all = pd.merge(self.comment_all,self.x_u_c_t,how='left',on='user')
        self.comment_all.fillna(0)
        self.comment_all_sort = self.comment_all.sort_values('user')
        self.comment_all_sort.fillna(0)
        
        # 保存每一轮迭代会改变的数据
        properties = open('data/progress/properties.xml', 'w')
        properties.write('user_num:' + str(self.user_num) + '\n')
        properties.write('comments_num:' + str(self.comments_num) + '\n')
        properties.write('iteration:' + str(self.iteration) + '\n')
        np.save('data/progress/lambda_s.npy', self.lambda_s)
        np.save('data/progress/M_pre_s.npy', self.M_pre_s)
        np.save('data/progress/M_pre_c.npy', self.M_pre_c)
        self.x_u_c_t.to_csv('data/progress/x_u_c_t.csv', encoding='utf-8')
        print('updating x_u_c_t %d second' % (time.time() - start_time))

    # @tested
    '''
     Api_name : update_Mpre_s
     Argc : None
     Func : 更新 M_pre_s
    '''

    def update_Mpre_s(self, iteration):
        start_time = time.time()  # 下面的多线程开始执行的时间
        self.iteration = iteration
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        for i in range(0, self.split_num):
            for topic in range(self.total_topic):  # 先循环topic
                numerator = 0
                denominator = 0
                for j in range(i):
                    numerator += self.delta_s[i][j] * self.lambda_s[j][topic]
                    denominator += self.delta_s[i][j]
                self.M_pre_s[i][topic] = (numerator+1) / (denominator+1)
        np.save('data/progress/M_pre_s.npy', self.M_pre_s)
        print('updating M_pre_s %d second' % (time.time() - start_time))

    # @tested
    '''
     Api_name : update_Mpre_c
     Argc : None
     Func : 更新 M_pre_c
    '''

    def update_Mpre_c(self, iteration):
        start_time = time.time()  # 下面的多线程开始执行的时间
        self.iteration = iteration
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        start = 0  # shot 之间的位移
        for q in range(self.split_num):
            for i in range(start, start + len(self.com[q])):
                for topic in range(self.total_topic):  # 先循环topic
                    numerator = 0
                    denominator = 0
                    for j in range(i):
                        numerator += self.delta_c[i][j] * self.pi_c[j][topic]
                        denominator += self.delta_c[i][j]
                    self.M_pre_c[i][topic] = (numerator+1) / (denominator+1)
            start+=len(self.com[q])
        l = []    
        for i in range(len(self.comment_all.columns)-(2+self.total_topic)):
            l.append((2+self.total_topic)+i)
        self.comment_all.drop(self.comment_all.columns[l],1)
        self.comment_all = pd.concat([self.comment_all, pd.DataFrame(self.M_pre_c)], axis=1)
        self.comment_all = pd.merge(self.comment_all,self.x_u_c_t,how='left',on='user')
        self.comment_all.fillna(0)
        self.comment_all_sort = self.comment_all.sort_values('user')
        self.comment_all_sort.fillna(0)
        np.save('data/progress/M_pre_c.npy', self.M_pre_c)
        print('updating M_pre_c %d second' % (time.time() - start_time))
        
        
    # @tested
    '''
     Api_name : Get_alpha_c
     Argc : None
     Func : 获取 alpha_c 的值（对于每一条评论的topic分布）
    '''
    def Get_alpha_c(self):
        # 先更新 pi_c
        start = 0
        for q in range(self.split_num):
            for i in range(start, start + len(self.com[q])):
                for topic in range(self.total_topic):  # 先循环topic
                    self.pi_c[i][topic] = self.lambda_s[q][topic] * self.comment_all.ix[i,2+self.total_topic+2+self.total_topic+topic] + self.M_pre_c[i][topic]
            start+=len(self.com[q])
        np.save('data/progress/pi_c.npy', self.pi_c)
        self.alpha_c = self.pi_c.copy()
        for i in range(self.comments_num):
            for j in range(self.total_topic):
                self.alpha_c[i][j] = lgt(self.alpha_c[i][j]/10)
        return self.alpha_c

