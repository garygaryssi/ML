import pandas as pd
import matplotlib.pyplot as plt
import seaborn

# 데이터 불러오기
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# .head(self, n=5) DataFrame 내의 처음 n줄의 데이터만 출력
# print(df_train.head())
# print(df_test.head())

# 정보가 비어 있는 칼럼도 존재함. 컬럼 12개 존재
# print(df_train.info())
# print(df_test.info())


# .corr() : 상관관계를 나타내는 메소드
# Pclass와 Survived간의 상관관계가 있음을 추론할 수 있다.
# print(df_train.corr())

# Pclass-Survived 그래프
# 생존자 수 확인
# .value_counts() : value를 counts해줌
# print(df_train["Survived"].value_counts())

# 생존 그룹과 그렇지 않은 그룹으로 나눈다.
survive = df_train.loc[df_train["Survived"] == 1].copy()
dead = df_train.loc[df_train["Survived"] == 0].copy()


# 히스토 그램
plt.hist(survive["Pclass"], alpha=0.5, label='Survived')
plt.hist(dead["Pclass"], alpha=0.5, label='Dead')
plt.legend(loc='best')
plt.show()

print(survive["Pclass"].value_counts(sort=False))
print(dead["Pclass"].value_counts(sort=False))

# bar 차트
survive["Pclass"].value_counts(sort=False).plot(kind="bar", label="Survived", color='blue', alpha=0.5)
dead["Pclass"].value_counts(sort=False).plot(kind="bar", label="dead", color="red", alpha=0.5)

plt.xlabel(["1st Class", "Business", "Economy"])
plt.legend(loc="best")
plt.show()

# seaborn 차트
seaborn.countplot(data=df_train, x="Pclass", hue="Survived")
plt.show()

# seaborn 스타일 darkgrid
df_train["Pclass"] = df_train["Pclass"].replace(1, "1st").replace(2, "Business").replace(3, "Economy")
df_train["Survived"] = df_train["Survived"].replace(1, "Alive").replace(0, "Dead")

seaborn.set_theme(style="darkgrid")
seaborn.countplot(data=df_train, x="Pclass", hue="Survived")
plt.show()

