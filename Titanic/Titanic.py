import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

# print(survive["Pclass"].value_counts(sort=False))
# print(dead["Pclass"].value_counts(sort=False))

# bar 차트
survive["Pclass"].value_counts(sort=False).plot(kind="bar", label="Survived", color='blue', alpha=0.5)
dead["Pclass"].value_counts(sort=False).plot(kind="bar", label="dead", color="red", alpha=0.5)

plt.xlabel(["1st Class", "Business", "Economy"])
plt.legend(loc="best")
plt.show()

# seaborn 차트
seaborn.countplot(data=df_train, x="Pclass", hue="Survived")
plt.show()

# seaborn test03
df_train["Pclass"] = df_train["Pclass"].replace(1, "1st").replace(2, "Business").replace(3, "Economy")
df_train["Survived"] = df_train["Survived"].replace(1, "Alive").replace(0, "Dead")

seaborn.set_theme(style="darkgrid")
seaborn.countplot(data=df_train, x="Pclass", hue="Survived")
plt.show()

# 학습을 위한 데이터 전처리

df_train = pd.read_csv("train.csv")

gt_train = df_train["Survived"]


df_train = df_train.drop(["Name", "PassengerId", "Ticket", "Fare", "Cabin", "Survived"], axis=1)

df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean()) # .fillna() 빈 곳을 채움
df_train["Embarked"] = df_train["Embarked"].fillna("S")

df_train["Sex"] = df_train["Sex"].map({"male" : 0, "female" : 1})

df_train["Embarked"] = df_train["Embarked"].map({"Q" : 0, "C" : 1, "S" : 2})

# 모델 생성

csf = RandomForestClassifier()
csf.fit(df_train, gt_train)

df_test = pd.read_csv("test.csv")

pId = df_test["PassengerId"] # baseline에 맞추기위해 빼둔다

df_test = df_test.drop(["Name", "PassengerId", "Ticket", "Fare", "Cabin"], axis=1) # 0 : 행,  1 : 열

df_test["Age"] = df_test["Age"].fillna(df_train["Age"].mean()) # .fillna 빈곳을 채움
df_test["Embarked"] = df_test["Embarked"].fillna("S") # .fillna 빈곳을 채움

df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})

df_test["Embarked"] = df_test["Embarked"].map({"Q": 0, "C": 1, "S": 2})

test_result = csf.predict(df_test)

print(test_result)

# dataFrame 만들기

submit = pd.DataFrame({"PassengerId": pId, "Survived": test_result}) # gender_submission 과 같은결과

# submit.to_csv("submit.csv", index=False) # index값 제거

test_gt = pd.read_csv("groundtruth.csv")

hit = 0
miss = 0

for i in range(len(test_result)):
    if test_result[i] == test_gt["Survived"][i]:
        hit += 1
    else:
        miss += 1

print(hit, miss, round(hit/(hit+miss), 2))

# print(submit)
