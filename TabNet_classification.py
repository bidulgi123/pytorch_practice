import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.datasets import load_iris

# Iris 데이터 불러오기
iris = load_iris()
X = iris.data  
y = iris.target  

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# TabNet 모델 초기화 및 학습
clf = TabNetClassifier(
    #optimizer
    optimizer_fn=torch.optim.Adam)
clf.fit(X_train=X_train, y_train=y_train
    #validation data
    #eval_set=[(X_test, y_test)]
    #배치 사이즈
    ,batch_size=32,
    #ghost batch nomalization에 사용되는 ghost(github에는 미니 라고 되어있음) 배치 크기
    #virtual_batch_size=4
    #eval_metric(평가 함수)
    eval_metric=['accuracy'],
    #loss function(손실 함수)
    loss_fn=nn.CrossEntropyLoss(),
    #5번 이상 개선 없을 시 조기 종료, 0은 조기 종료 미설정
    patience=0,
    #최대 에포크
    max_epochs=50,
    #남은 마지막 배치 삭제 여부
    drop_last=False)

# 모델 성능 평가
test_preds = clf.predict(X_test)
print(test_preds)
print(y_test)
#https://github.com/dreamquark-ai/tabnet