import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

# raw data import
raw_data = pd.read_csv("C:/Projects/10. RAW/data.txt", sep='\t', encoding='CP949')

# 데이터 확인
raw_data.head()
raw_data.info()
print(raw_data.columns.tolist())
raw_data.shape

# 코드 명세서 import
ifrs_code = pd.read_csv("C:/Projects/10. RAW/K-IFRS_code.txt", sep='\t', encoding='CP949')

# 데이터 확인
ifrs_code.head()
print(ifrs_code.columns.tolist())

# 부도기업리스트 import
budo_data = pd.read_csv("C:/Projects/10. RAW/부도기업리스트.txt", sep='\t', encoding='CP949')

# 데이터 확인
budo_data.head()
budo_data.info()
# 확인 결과 사업자번호 클래스가 float으로 나와 integer로 변환
budo_data['사업자번호'] = budo_data['사업자번호'].astype('Int64')
budo_data.head()
budo_data.info()
#변환 완료

# 정상기업리스트 import
normal_data = pd.read_csv("C:/Projects/10. RAW/정상기업리스트.txt", sep='\t', encoding='CP949')

# 데이터 확인
normal_data.head()
normal_data.info()


# 부도 정보를 raw data에 추가하여 target data 생성
"""
target_data = raw_data.copy()
for name in target_data['회사명']:
    for budo in budo_data['회사명']:
        if name == budo:
            target_data.loc[target_data['회사명'] == name,'budo_date'] = budo_data.loc[budo_data['회사명'] == budo,'첫 발생'].to_string()[7:]
            break
#또는
target_data = pd.merge(raw_data, budo_data[['회사명','첫 발생']], on='회사명', how='left')
"""
# 결과 학인
"target_data.loc[target_data['budo_date'].isna() == False, ['회사명', 'budo_date']]"
# 한 회사에만 정보가 추가됨. 회사명이 아닌 거래소코드로 결합

# target_data의 거래소코드값 수정
target_data = raw_data.copy()
target_data['거래소코드_new'] = target_data['거래소코드'].map(lambda x: 'A' + str(x).zfill(6))

# 부도 정보 추가
target_data = pd.merge(target_data, budo_data[['거래소 코드', '발생연도']], left_on='거래소코드_new', right_on='거래소 코드'
                       , how='left')
target_data = target_data.drop('거래소 코드', axis=1)
target_data['발생연도'].dtypes
# 확인 결과 발생연도가 float으로 나와 integer로 변환
target_data['발생연도'] = target_data['발생연도'].astype('Int64')
target_data['발생연도'].dtypes

# 결과 학인
target_data[target_data['발생연도'].isna() == False]

# 개수 확인
len(target_data[target_data['발생연도'].isna() == False]['거래소코드_new'].unique())
# 타켓 데이터의 부도 회사 390개

len(budo_data['거래소 코드'].unique())
# 부도 데이터의 부도 회사 490개
# 100개 회사가 결합되지 않음. 결합되지 않은 회사 확인

# 변수 생성
budo_budo = budo_data['거래소 코드'].unique()
target_budo = target_data[target_data['발생연도'].isna() == False]['거래소코드_new'].unique()

# 차이 확인
diff = list(set(budo_budo) - set(target_budo))
diff_df = budo_data[budo_data['거래소 코드'].isin(diff)]
# 확인 결과 3건을 제외한 97건은 SPC임.
# 3건(한국자산신탁(주), 한국토지신탁, 에이비온) 확인 필요

# 1, 2, 3년내 부도 컬럼 생성
"""
target_data['budo_gap'] = 0
target_data['budo_in'] = 0
target_data['budo_gap'] = pd.to_numeric(target_data['회계년도'].str[:4]) - target_data['발생연도']
target_data['budo_gap'] = target_data['budo_gap'].astype('Int64')
target_data['budo_in'] = target_data['budo_in'].astype('Int64')
for i, row in target_data.iterrows():
    if row['budo_gap'] == 1:
        target_data.loc[i, 'budo_in'] = 1
    elif row['budo_gap'] == 1:
        target_data.loc[i, 'budo_in'] = 2
    elif row['budo_gap'] == 2:
        target_data.loc[i, 'budo_in'] = 3
    else:
        target_data.loc[i, 'budo_in'] = 0
"""
# 진행시 에러 발생
"""
Traceback (most recent call last):
  File "<input>", line 2, in <module>
  File "pandas\_libs\missing.pyx", line 360, in pandas._libs.missing.NAType.__bool__
TypeError: boolean value of NA is ambiguous
"""
# 에러 확인 결과 budo_gap을 산출하는 과정에서 데이터 타입에 오류가 생기는 것으로 판단.

# budo_gap의 NaN 처리 진행
target_data['budo_gap'] = pd.to_numeric(target_data['회계년도'].str[:4]) - target_data['발생연도']
target_data['budo_gap'] = target_data['budo_gap'].fillna(-9999)
for i, row in target_data.iterrows():
    if row['budo_gap'] == 0:
        target_data.loc[i, 'budo_in'] = 1
    elif row['budo_gap'] == -1:
        target_data.loc[i, 'budo_in'] = 2
    elif row['budo_gap'] == -2:
        target_data.loc[i, 'budo_in'] = 3
    elif row['budo_gap'] > 0:
        target_data.loc[i, 'budo_in'] = 99  # 기부도
    else:
        target_data.loc[i, 'budo_in'] = -9999 # 정상

target_data['budo_gap'].dtype

# 년도별 부도 추세 확인
# 변수 확인
target_data[['발생연도', 'budo_gap']]
graph_raw = target_data.loc[((target_data['budo_in'] == 1) | (target_data['budo_in'] == 2) | (target_data['budo_in'] == 3)), ['발생연도', 'budo_in']].reset_index(drop=True)
graph_df = pd.DataFrame(data={'d1': [0], 'd2': [0], 'd3': [0]}, index=sorted(graph_raw['발생연도'].unique()))
a = 1
for column in graph_df.items():
    for i in graph_df.iterrows():
        print(i)

for i, row in graph_df.iterrows():
    if i == graph_raw.loc[0, '발생연도']:
        if graph_raw['budo_in'] == 1:
            row = 1

    print(row)

graph_raw.loc[:,'발생연도']

year = np.array(raw_dic['발생연도'].unique()).tolist()
future = np.array(raw_dic['budo_in'].unique()).astype('int8').tolist()

for oc_yr, size in zip(year, future):
    raw_dic_sub =raw_dic[raw_dic['발생연도'] == oc_yr]
    plt.plot(list(raw_dic_sub.발생연도), list(raw_dic_sub.budo_in), linewidth=size)
# 수십번의 values must be a 1D array 에러 해결방법: list()로 x, y 값을 처리
# 하지만 다른 예시에서는 2D series에서도 작동되었던 것으로 보아 확인 필요


for i in graph_df.iteritems():
    print(i)


"""
raw_dic = raw_dic[['발생연도', 'budo_in']]
raw_dic['budo_in'] = raw_dic['budo_in'].astype('Int64')

raw_dic.plot(kind='bar', x='발생연도', y=['budo_in'== 1])

plt.plot(raw_dic.발생연도, raw_dic.budo_in)
stack = raw_dic.stack()
stack_df = pd.DataFrame(stack).reset_index()
stack_df
x = raw_dic['발생연도'].values
x = np.atleast_1d(x)
y = raw_dic['budo_in'].values
y = np.atleast_1d(y)

raw_dic.plot.line(x='발생연도', y='budo_in')
plt.plot(x, y)



my_dict = Counter(y)
labels, values = zip(*Counter(raw_dic).items())
indexes = np.arange(len(labels))
plt.bar(indexes, values, 1)
plt.xticks(indexes + 0.5, labels)
plt.show()
"""