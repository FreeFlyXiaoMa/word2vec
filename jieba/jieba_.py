# -*- coding: utf-8 -*-
# @Time     :2019/10/28 19:24
# @Author   :XiaoMa
# @File     :jieba_.py

import jieba
seg_list=jieba.cut("我来到北京清华大学",cut_all=True)
#全模式
print('全模式：'+'/'.join(seg_list))

#精确模式
seg_list=jieba.cut("我来到北京清华大学",cut_all=False)
print('精确模式：'+'/'.join(seg_list))

#默认模式
seg_list=jieba.cut('他来到了网易杭研大厦')
print('默认：',','.join(seg_list))

#搜索引擎模式
seg_list=jieba.cut_for_search('小明毕业于中国科学院计算所，后在日本京都大学深造')
print('搜索引擎模式：',','.join(seg_list))

