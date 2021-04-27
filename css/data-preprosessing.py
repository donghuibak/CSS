import pandas as pd
import numpy as np

# raw data import
raw_data = pd.read_csv("C:/Users/DH/Desktop/202101/Project/Data/data.txt", sep='\t', encoding='CP949')

# 데이터 확인
raw_data.head()
raw_data.info()
print(raw_data.columns.tolist())
raw_data.shape

# 코드 명세서 import
ifrs_code = pd.read_csv("C:/Users/DH/Desktop/202101/Project/Data/K-IFRS_code.txt", sep='\t', encoding='CP949')

# 데이터 확인
ifrs_code.head()
print(ifrs_code.columns.tolist())

# 부도기업리스트 import
budo_data = pd.read_csv("C:/Users/DH/Desktop/202101/Project/Data/부도기업리스트.txt", sep='\t', encoding='CP949')

# 데이터 확인
budo_data.head()
budo_data.info()

# 정상기업리스트 import
normal_data = pd.read_csv("C:/Users/DH/Desktop/202101/Project/Data/정상기업리스트.txt", sep='\t', encoding='CP949')

