#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import pickle
import pandas as pd
from _datetime import datetime
from datetime import timedelta
import math
import numpy as np

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07",
                "2016-03-14", "2016-03-21", "2016-03-28", "2016-04-04", "2016-04-11", "2016-04-15"]

def user_basic_feature():
    """得到基本用户信息：user_id, 年龄范围（7个选项）,性别，等级"""
    dump_path = './cache/user_basic.pk1'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path,'rb'))
    else:
        user = pd.read_csv(user_path,encoding='gbk')
        #print(user.info())
        user['age'] = user['age'].map(convert_age)
        #print(user.head())

        #del user['user_reg_tm'] 考虑把等级和注册时间用成一个特征，比如升级速率（购买频率）

        age_df = pd.get_dummies(user["age"],prefix="age")
        sex_df = pd.get_dummies(user["sex"],prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"],prefix="user_lv_cd")
        user = pd.concat([user['user_id'],age_df,sex_df,user_lv_df],axis = 1)
        pickle.dump(user,open(dump_path,'wb'))
    return user


def convert_age(age_str):
    if age_str == '-1':
        return 0
    elif age_str == '15岁以下':
        return 1
    elif age_str == '16-25岁':
        return 2
    elif age_str == '26-35岁':
        return 3
    elif age_str == '36-45岁':
        return 4
    elif age_str == '46-55岁':
        return 5
    elif age_str == '56岁以上':
        return 6
    else:
        return -1


def product_basic_feature():
    """得到商品基本信息特征：sku_id, a2, a2, a3, cate, brand"""
    dump_path = './cache/product_basic.pk1'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path,'rb'))
    else:
        product = pd.read_csv(product_path,encoding='gbk')
        #print(product.head())
        #print(product.info())
        pickle.dump(product,open(dump_path,'wb'))
    return product


def comment_feature(start_date,end_date):
    """得到时间范围内商品评论信息：sku_id, comment_num, has_bad_comment, bad_comment_rate"""
    dump_path = './cache/comments_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path,'rb'))
    else:
        comments = pd.read_csv(comment_path)

        comment_date_end = end_date
        comment_date_start = comment_date[0]

        #选出合适的评论截至日期，只要到这个日期的评论信息即可
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_start = date
                break
        comments = comments[(comments.dt >= comment_date_start) & (comments.dt < comment_date_end)]
        del comments['dt']
        # print(comments.info())
        pickle.dump(comments,open(dump_path,'wb'))
    return comments


def get_action_1():
    action1 = pd.read_csv(action_1_path)
    return action1

def get_action_2():
    action2 = pd.read_csv(action_2_path)
    return  action2

def get_action_3():
    action3 = pd.read_csv(action_3_path)
    return action3

def get_all_actions():
    """所有的action信息 填充model_id的NaN为-1
    user_id, sku_id, time, model_id, type, number"""
    dump_path = './cache/all_action.pk1'
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        action1 = get_action_1()
        action2 = get_action_2()
        action3 = get_action_3()
        actions = pd.concat([action1, action2, action3])
        actions = actions.fillna({'model_id': -1})

        pickle.dump(actions,open(dump_path,'wb'))
    return actions

def get_action_cate(cate):
    """得到所有时间范围内某类商品action信息"""
    dump_path = './cache/all_action-cate%s.pk1' %cate
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = get_all_actions()
        actions = actions[actions.cate == cate]  # 选择具体商品的数据类别

        pickle.dump(actions, open(dump_path,'wb'))

    #print(actions.info())
    #print(actions.head(20))
    return actions


def get_action(start_date,end_date,cate):
    """得到时间范围内的所有action信息：user_id, sku_id, model_id, time, type, number
    cate指定具体商品类别"""
    dump_path = './cache/action-cate%s_%s_%s.pk1' %(cate, start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path,'rb'))
    else:
        actions = get_action_cate(cate)
        actions = actions[ (actions.time >= start_date) & (actions.time < end_date)] #符合时间要求的数据

        pickle.dump(actions,open(dump_path,'wb'))
    #print(actions.info())
    #print(actions.head(20))
    return actions


def action_accumulate_feature(start_date, end_date, cate):
    """累计统计距离预测开始日期前，不同时间段各个行为的数量作为特征，这里选用的是前1天，2天，3天，5天....30天"""
    dump_path = './cache/action_accumulate_%s_%s.pk1'%(start_date,end_date)
    if os.path.exists(dump_path):
        actions_acc = pickle.load(open(dump_path,'rb'))
    else:
        actions_acc = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            days = i
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions_acc is None:
                actions_acc = action_feature(start_days, end_date, cate, days)
            else:
                actions_acc = pd.merge(actions_acc, action_feature(start_days, end_date, cate, days), how='outer',
                                   on=['user_id', 'sku_id'])
        pickle.dump(actions_acc,open(dump_path,'wb'))

    #print(actions_acc.info())
    #print(actions_acc.head(10))
    return actions_acc

def action_feature(start_date, end_date, cate, days):
    """得到时间范围内某类商品的行为统计信息，user_id， sku_id, 以及这段时间各个行为的数量
        这个函数主要用于计算距离预测日期开始前前1，2，3，5，7....30天各个行为的累计数量，作为累计特征"""
    actions = get_action(start_date, end_date, cate)
    actions = actions[['user_id','sku_id','type']]
    df = pd.get_dummies(actions['type'], prefix='days%s-action' %days)
    actions = pd.concat([actions, df], axis=1)
    del actions['type']
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    #print(actions.info())
    #print(actions.head(10))

    return actions


def user_accumulate_feature(start_date, end_date, cate):
    """得到时间范围内用户对某类（cate）商品的行为统计：user_id, 这段时间行为1，2，3，5，6转化为购买行为的转化率"""
    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio']

    dump_path = './cache/user_accumulate_%s_%s.pk1' %(start_date, end_date)
    if os.path.exists(dump_path):
        actions_user = pickle.load(open(dump_path,'rb'))
    else:
        actions_user = get_action(start_date, end_date, cate)
        df = pd.get_dummies(actions_user['type'], prefix='action')
        actions_user = pd.concat([actions_user['user_id'], df], axis=1)
        actions_user = actions_user.groupby(['user_id'], as_index=False).sum()

        actions_user['user_action_1_ratio'] = actions_user['action_4'] / actions_user['action_1']
        actions_user['user_action_2_ratio'] = actions_user['action_4'] / actions_user['action_2']
        actions_user['user_action_3_ratio'] = actions_user['action_4'] / actions_user['action_3']
        actions_user['user_action_5_ratio'] = actions_user['action_4'] / actions_user['action_5']
        actions_user['user_action_6_ratio'] = actions_user['action_4'] / actions_user['action_6']
        actions_user = actions_user[feature]

        #print(actions_user.info())
        #print(actions_user.head(100))
        pickle.dump(actions_user,open(dump_path,'wb'))
    return actions_user


def product_accumulate_feature(start_date,end_date,cate):
    """得到某类商品的行为转化率统计: sku_id, 行为1，2，3，5，6转化为购买的转化率"""
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio']
    dump_path = './cache/product_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions_product = pickle.load(open(dump_path,'rb'))
        #print(actions_product.head(20))
    else:
        actions_product = get_action(start_date,end_date,cate)
        df = pd.get_dummies(actions_product['type'], prefix='action')
        actions_product = pd.concat([actions_product['sku_id'], df], axis=1)
        actions_product = actions_product.groupby(['sku_id'],as_index = False ).sum()

        actions_product['product_action_1_ratio'] = actions_product['action_4'] / actions_product['action_1']
        actions_product['product_action_2_ratio'] = actions_product['action_4'] / actions_product['action_2']
        actions_product['product_action_3_ratio'] = actions_product['action_4'] / actions_product['action_3']
        actions_product['product_action_5_ratio'] = actions_product['action_4'] / actions_product['action_5']
        actions_product['product_action_6_ratio'] = actions_product['action_4'] / actions_product['action_6']

        actions_product = actions_product[feature]# select these column

        #print(actions_product.info())
        #print(actions_product.head(50))

        pickle.dump(actions_product,open(dump_path,'wb'))
    return actions_product


def get_labels(start_date, end_date, cate):
    """得到时间范围内购买类该商品的用户信息：user_id, sku_id, lable（1——购买，0——不购买）"""
    dump_path = './cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        labels = pickle.load(open(dump_path,'rb'))
    else:
        actions = get_action(start_date,end_date,cate)
        actions = actions[actions.type == 4]
        actions = actions.groupby(['user_id','sku_id'], as_index=False).sum()
        actions['label'] = 1
        labels = actions[['user_id','sku_id','label']]
        #print(labels.head())
        #print(labels.info())
        pickle.dump(labels,open(dump_path,'wb'))
    return labels



def make_train_set(train_start_date,train_end_date, label_start_date, label_end_date, cate):
    """构造训练集，train_start_date 和 train_end_date用于指定训练集取的开始和结束日期，长度为30天。
        label_start_date和label_end_date用于指定训练集对应的label，未来5天用户的购买情况（1——购买，0——不购买）
        cate为商品类别
        函数返回users——用户商品id组，labels——是否购买，feature——所有的特征数据"""
    user_basic = user_basic_feature()
    product_basic = product_basic_feature()

    # 用户和商品的累积属性考虑前三个月数据，由于这里只能从02-01开始，所以无论训练集取哪一段的行为，都从02-01开始累积到训练集结束日期
    start_date = "2016-02-01"
    user_acc = user_accumulate_feature(start_date,train_end_date,cate)
    product_acc = product_accumulate_feature(start_date,train_end_date,cate)

    comment = comment_feature(train_start_date,train_end_date)
    labels = get_labels(label_start_date, label_end_date, cate)
    action_acc = action_accumulate_feature(train_start_date,train_end_date,cate)


    user_feature = pd.merge(user_acc, user_basic, how='right', on='user_id')
    product_feature = pd.merge(product_acc, product_basic, how='right', on='sku_id')
    product_feature = pd.merge(product_feature, comment, how='left', on='sku_id')
    feature = pd.merge(user_feature,action_acc, how ='left', on ='user_id')
    feature = pd.merge(feature,product_feature,how ='left', on= 'sku_id')
    feature = pd.merge(feature, labels, how='left', on=['user_id', 'sku_id'])

    feature = feature.fillna(value=0)

    users = feature[['user_id','sku_id']].copy()
    labels = feature['label'].copy()
    del feature['label']

    #write train set to file
    #feature['user_id'] = feature['user_id'].astype(int)
    #feature['sku_id'] = feature['sku_id'].astype(int)
    #feature.to_csv('./set/train_data%s_%s.csv'%(train_start_date,train_end_date),index=False,encoding='utf-8')
    #labels.to_csv('./set/train_labels%s_%s.csv'%(train_start_date,train_end_date),index=False,encoding='utf-8')

    del feature['user_id']
    del feature['sku_id']

    return users,feature,labels


def make_data_set(start_date, end_date, cate):
    """构造数据集，start_date 和 end_date用于指定训练集取的开始和结束日期，长度为30天。预测接下来五天购买情况
        cate为商品类别
        函数返回users——用户商品id组，labels——是否购买，feature——所有的特征数据"""
    user_basic = user_basic_feature()
    product_basic = product_basic_feature()

    # 用户和商品的累积属性考虑前三个月数据，由于这里只能从02-01开始，所以无论训练集取哪一段的行为，都从02-01开始累积到训练集结束日期
    start = "2016-02-01"
    user_acc = user_accumulate_feature(start, end_date, cate)
    product_acc = product_accumulate_feature(start, end_date, cate)

    comment = comment_feature(start_date, end_date)
    action_acc = action_accumulate_feature(start_date, end_date, cate)

    user_feature = pd.merge(user_acc, user_basic, how='right', on='user_id')
    product_feature = pd.merge(product_acc, product_basic, how='right', on='sku_id')
    product_feature = pd.merge(product_feature, comment, how='left', on='sku_id')
    feature = pd.merge(user_feature, action_acc, how='left', on='user_id')
    feature = pd.merge(feature, product_feature, how='left', on='sku_id')

    feature = feature.fillna(value=0)

    users = feature[['user_id', 'sku_id']].copy()

    # write data set to the file
    # feature['user_id'] = feature['user_id'].astype(int)
    # feature['sku_id'] = feature['sku_id'].astype(int)
    # feature.to_csv('./set/dataSet%s_%s.csv'%(train_start_date,train_end_date),index=False,encoding='utf-8')

    del feature['user_id']
    del feature['sku_id']

    return users, feature

def report(pred, label):
    """测评召回率和准确率"""

    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print('F11=' + str(F11))
    print('F12=' + str(F12))
    print('score=' + str(score))


if __name__ == '__main__':
    cate = 8
    train_start_date = '2016-03-02'
    train_end_date = '2016-04-01'
    label_start_date = '2016-04-01'
    label_end_date = '2016-04-05'
    make_train_set(train_start_date,train_end_date,label_start_date,label_end_date,cate)
