# -*- coding: utf-8 -*-
# @Time : 2022/11/24 22:09
# @Author : Wang Hui
# @File : test
# @Project : News_Recommender
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from trian import trian_item_word2vec

plt.rc('font',family="SimHei",size=13)

import os,gc,re,warnings,sys
warnings.filterwarnings("ignore")

path=r'D:\Study\Code\Python\数据\\' #设置路劲

###train
trn=pd.read_csv(path+'train_click_log.csv')
trn_click=trn.iloc[:30000,:]
item=pd.read_csv(path+'articles.csv')
item_df=item.iloc[:30000,:]
item_df=item_df.rename(columns={'article_id': 'click_article_id'})   #此处重命名，方便后续match
item_emb=pd.read_csv(path+'articles_emb.csv')
item_emb_df=item_emb.iloc[:30000,:]

###test
tst=pd.read_csv(path+'testA_click_log.csv')
tst_click=tst.iloc[:30000,:]
#对每个用户的点击时间戳进行排序
trn_click['rank']=trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
tst_click['rank']=tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
#计算用户文章的点击次数，并添加新的一列count
trn_click['click_cnts']=trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
tst_click['click_cnts']=tst_click.groupby(['user_id'])['click_timestamp'].transform('count')
trn_click=trn_click.merge(item_df,how='left',on=['click_article_id'])
trn_click.head()
trn_click.info()
trn_click.describe()   #用于对数据进行统计学估计，count:行数 mean:平均值 std:标准差 min:最小值 25%:第一四分位数 50%:第二四分位数 75%:第三四分位数 max:最大值
# print(s.mean())
trn_click.user_id.nunique()  #训练集中的用户数量
trn_click.groupby('user_id')['click_article_id'].count().min()  #训练集里面每个用户至少点击两篇文章
# plt.figure()
# plt.figure(figsize=(15,20))
# i=1
# for col in ['click_article_id', 'click_timestamp', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type', 'rank', 'click_cnts']:
#     plot_envs=plt.subplot(5,2,i)  #表示画布中子图的个数，前两个参数表示子图的行列，最后一个参数表示图片所处的位置
#     i=i+1
#     v=trn_click[col].value_counts().reset_index()[:10]
#     fig=sns.barplot(x=v['index'],y=v[col])
#     for item in fig.get_xticklabels():  #返回刻度列表信息
#         item.set_rotation(90)     #刻度旋转90度，避免数据挤到一起
#     plt.title(col)
# #调整子图间距
# plt.tight_layout()
# plt.show()

s2=trn_click['click_environment'].value_counts()   #查看不同点击环境的次数
s1=trn_click['click_deviceGroup'].value_counts()   #查看不同设备的使用情况
print(s1)
print(s2)
tst_click=tst_click.merge(item_df,how='left',on=['click_article_id'])    #两个表匹配融合
# print(tst_click.head())
tst_click.user_id.nunique()
tst_click.groupby('user_id')['click_article_id'].count().min()   #注意测试集中有只点击过一次文章的用户
#新闻数据浏览
item_df.head().append(item_df.tail())
item_df['words_count'].value_counts()     #单词数出现的频率
item_df['category_id'].nunique()      #文章主题
# print(item_df['category_id'].hist())
# print(item_df.shape)

#新闻文章嵌入向量表示
item_emb_df.head()
item_emb_df.shape
#####merge
user_click_merge = trn_click.append(tst_click)
#用户重复点击
user_click_count = user_click_merge.groupby(['user_id', 'click_article_id'])['click_timestamp'].agg({'count'}).reset_index()
user_click_count[:10]
user_click_count[user_click_count['count']>7]
user_click_count['count'].unique()
#用户点击新闻次数
user_click_count.loc[:,'count'].value_counts()
def plot_envs(df, cols, r, c):
    # plt.figure()
    plt.figure(figsize=(10, 5))
    i = 1
    for col in cols:
        plt.subplot(r, c, i)
        i += 1
        v = df[col].value_counts().reset_index()
        fig = sns.barplot(x=v['index'], y=v[col])
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(col)
    plt.tight_layout()
    plt.show()
# 分析用户点击环境变化是否明显，这里随机采样10个用户分析这些用户的点击环境分布
sample_user_ids = np.random.choice(tst_click['user_id'].unique(), size=10, replace=False)
sample_users = user_click_merge[user_click_merge['user_id'].isin(sample_user_ids)]
cols = ['click_environment','click_deviceGroup', 'click_os', 'click_country', 'click_region','click_referrer_type']
# for _, user_df in sample_users.groupby('user_id'):
#     plot_envs(user_df, cols, 2, 3)
#用户点击新闻数量的分布
user_click_item_count = sorted(user_click_merge.groupby('user_id')['click_article_id'].count(), reverse=True)
# plt.plot(user_click_item_count)
#点击次数在前50的用户
# plt.plot(user_click_item_count[:50])
# plt.show()
#点击次数排名在[25000:50000]之间
plt.plot(user_click_item_count[25000:50000])

#新闻点击次数分析
item_click_count=sorted(user_click_merge.groupby('click_article_id')['user_id'].count(),reverse=True)
# plt.plot(item_click_count)
# plt.show()
#点击次数最多的前一百篇新闻
# plt.plot(item_click_count[:100])
# #点击次数排在3500到最后的新闻排布
# plt.plot(item_click_count[3500:])   #这些新闻可以视为冷门新闻

#新闻共现频次：两篇新闻连续出现的次数
tmp=user_click_merge.sort_values('click_timestamp')
tmp['next_item']=tmp.groupby(['user_id'])['click_article_id'].transform(lambda x:x.shift(-1))
union_item = tmp.groupby(['click_article_id','next_item'])['click_timestamp'].agg({'count'}).reset_index().sort_values('count', ascending=False)
union_item[['count']].describe()
# #画个图直观地看一看
# x = union_item['click_article_id']
# y = union_item['count']
# plt.scatter(x, y)
# plt.show()

#新闻文章信息
#不同类型的新闻出现的次数
plt.plot(user_click_merge['category_id'].value_counts().values)
plt.plot(user_click_merge['click_timestamp'].value_counts().values[150:])

#用户查看文章的长度的分布
#通过统计不同用户点击新闻的平均字数，这个可以反映用户是对长文更感兴趣还是对短文更感兴趣
plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(),reverse=True))

#用户点击新闻的时间分析
item_w2v_emb_dict=trian_item_word2vec(user_click_merge)

# 随机选择5个用户，查看这些用户前后查看文章的相似性
sub_user_ids = np.random.choice(user_click_merge.user_id.unique(), size=5, replace=False)
sub_user_info = user_click_merge[user_click_merge['user_id'].isin(sub_user_ids)]

sub_user_info.head()

#将训练得到的词向量进行可视化
def get_item_sim_list(df):
    sim_list = []
    item_list = df['click_article_id'].values
    for i in range(0, len(item_list)-1):
        emb1 = item_w2v_emb_dict[str(item_list[i])] # 需要注意的是word2vec训练时候使用的是str类型的数据
        emb2 = item_w2v_emb_dict[str(item_list[i+1])]
        sim_list.append(np.dot(emb1,emb2)/(np.linalg.norm(emb1)*(np.linalg.norm(emb2))))
    sim_list.append(0)

    return sim_list

for _, user_df in sub_user_info.groupby('user_id'):
    item_sim_list = get_item_sim_list(user_df)
    plt.plot(item_sim_list)
    plt.show()
