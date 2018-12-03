"""
Pre-processing csv files to feed them into our proposed model

1. Dataset
종족	범주형			0	2	Race
아군베이스에 있는 적 드론 수	수치형	sameArea		0	4	DrInMyBase
전체 저글링 수	수치형	completeCount , total 		0	30	AllLing
적 본진 저글링 수	수치형	sameArea		0	30	EmBaseLing
적 앞마당 저글링 수	수치형	sameArea		0	30	EmFrontLing
내 본진 저글링 수	수치형	sameArea		0	30	MyBaseLing
내 앞마당 저글링 수	수치형	sameArea		0	30	MyFrontLing
적 앞마당 밖 저글링 수	수치형	sameArea		0	30	OtherLing
#TIME
전체 해처리 수[전체, 완성, 미완성]	수치형	레어 포함(Complete, Incomplete, AllCount)		0	5	AllHat	AllHatCom	AllHatUnCom
적 본진 해처리 수[전체, 완성, 미완성]	수치형	레어 포함(Complete, Incomplete, AllCount)		0	5	EmBaseHat	EmBaseHatCom	EmBaseHatUnCom
적 앞마당 해처리 수[전체, 완성, 미완성]	수치형	레어 포함(Complete, Incomplete, AllCount)		0	5	EnFrontHat	EnFrontHatCom	EnFrontHatUnCom
내 본진 근처 해처리 수[전체, 완성, 미완성]	수치형	레어 포함(Complete, Incomplete, AllCount)		0	1	NearMeHat	NearMeHatCom	NearMeHatUnCom
상대 본진 가스 존재 여부	범주형			0	1	EmGas
내 본진 가스 존재 여부	범주형			0	1	MyBaseEmGas
적 본진과의 거리가 머냐 가깝냐 여부	범주형	적 본진과 내 본진의 거리가 128타일 이상 1, 0	값 수정필요할 지도..	0	1	EmIsClose
적군 베이스 위치 확실여부	범주형	적 본진 찾은적 있냐		0	1	CheckEmBase
3분전 완성된 해처리 수	수치형			0	3	HatBefore3M
앞마당 건설중인 해처리의 HP	수치형	0 ~ 1250		0	1250	FrontHatHP


## 181130
Em -> en
앞에 모두 소문자 처리
Map, Y 추가
ememyBaseIsVisible 추가
"""
import pandas as pd
import numpy as np
import arglist
import glob
import util
import pickle

print('start preprocess!')

total_df = pd.DataFrame()

n_train_files = arglist.n_train_files

for n, f in enumerate(glob.glob(arglist.raw_files_dir + 'train/' + '*' + arglist.raw_file_ext)):

    print('{} file is preprocessing'.format(n))

    if n > n_train_files:
        break

    df = pd.read_csv(f, sep=',')
    util.encode_onehot()

    # string ->

    # min max normalization
    df = util.normalize_columns_dataframe(df, arglist.l_columns_with_min_max)

    # remove Time column
    df.drop(['time'], axis=1)
    total_df = total_df.append(df)

util.dump_pickle_even_when_no_exist(arglist.pickle_file, total_df, arglist.pickle_dir + 'train/' )




print('start preprocess!')

total_df = pd.DataFrame()

n_train_files = arglist.n_train_files

for n, f in enumerate(glob.glob(arglist.raw_files_dir + 'test/' + '*' + arglist.raw_file_ext)):

    print('{} file is preprocessing'.format(n))

    if n > n_train_files:
        break

    df = pd.read_csv(f, sep=',')
    util.encode_onehot()

    # string ->

    # min max normalization
    df = util.normalize_columns_dataframe(df, arglist.l_columns_with_min_max)

    # remove Time column
    df.drop(['time'], axis=1)
    total_df = total_df.append(df)

util.dump_pickle_even_when_no_exist(arglist.pickle_file, total_df, arglist.pickle_dir + 'test/' )
