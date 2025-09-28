import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score, get_scorer_names, accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, KFold
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('season-2425.csv')
data.head()
dat = data.copy()
dat['Date'] = pd.to_datetime(dat['Date'], format='%d/%m/%y')
dat= dat.sort_values('Date').reset_index(drop=True)
teams = list(set(list(data['HomeTeam']) + list(data['AwayTeam'])))
features_list = []

for idx in range(len(data)):
    if idx < 10:  # 跳过前10场比赛(历史数据不足)
        continue
        
    current_match = data.iloc[idx]
    home_team = current_match['HomeTeam']
    away_team = current_match['AwayTeam']
    
    # 获取该比赛前的历史数据
    history_data = data.iloc[:idx]
    # 计算主队近5场数据
    home_recent = history_data[(history_data['HomeTeam'] == home_team) | 
                              (history_data['AwayTeam'] == home_team)].tail(5)
    
    # 主队作为主场的比赛
    home_as_home = home_recent[home_recent['HomeTeam'] == home_team]
    # 主队作为客场的比赛  
    home_as_away = home_recent[home_recent['AwayTeam'] == home_team]
    if len(home_recent) > 0:
        # 进球数
        home_goals_for = (home_as_home['FTHG'].sum() + home_as_away['FTAG'].sum()) / len(home_recent)
        home_goals_against = (home_as_home['FTAG'].sum() + home_as_away['FTHG'].sum()) / len(home_recent)
        # 射门数
        home_shots = (home_as_home['HS'].sum() + home_as_away['AS'].sum()) / len(home_recent)
        home_shots_on_target = (home_as_home['HST'].sum() + home_as_away['AST'].sum()) / len(home_recent)
        # 角球数
        home_corners = (home_as_home['HC'].sum() + home_as_away['AC'].sum()) / len(home_recent)
        # 犯规数
        home_fouls = (home_as_home['HF'].sum() + home_as_away['AF'].sum()) / len(home_recent)
        # 胜率
        home_wins = len(home_as_home[home_as_home['FTR'] == 'H']) + len(home_as_away[home_as_away['FTR'] == 'A'])
        home_win_rate = home_wins / len(home_recent)
    else: # 没有历史比赛数据使用平均值
        home_goals_for, home_goals_against = 1.5, 1.5
        home_shots, home_shots_on_target = 12.0, 4.0
        home_corners, home_fouls = 5.0, 11.0
        home_win_rate = 0.33
    # 计算客队近5场数据
    away_recent = history_data[(history_data['HomeTeam'] == away_team) | 
                              (history_data['AwayTeam'] == away_team)].tail(5)
    
    away_as_home = away_recent[away_recent['HomeTeam'] == away_team]
    away_as_away = away_recent[away_recent['AwayTeam'] == away_team]
    # 计算客队统计
    if len(away_recent) > 0:
        away_goals_for = (away_as_home['FTHG'].sum() + away_as_away['FTAG'].sum()) / len(away_recent)
        away_goals_against = (away_as_home['FTAG'].sum() + away_as_away['FTHG'].sum()) / len(away_recent)
        away_shots = (away_as_home['HS'].sum() + away_as_away['AS'].sum()) / len(away_recent)
        away_shots_on_target = (away_as_home['HST'].sum() + away_as_away['AST'].sum()) / len(away_recent)
        away_corners = (away_as_home['HC'].sum() + away_as_away['AC'].sum()) / len(away_recent)
        away_fouls = (away_as_home['HF'].sum() + away_as_away['AF'].sum()) / len(away_recent)
        away_wins = len(away_as_home[away_as_home['FTR'] == 'H']) + len(away_as_away[away_as_away['FTR'] == 'A'])
        away_win_rate = away_wins / len(away_recent)
    else:
        away_goals_for, away_goals_against = 1.5, 1.5
        away_shots, away_shots_on_target = 12.0, 4.0
        away_corners, away_fouls = 5.0, 11.0
        away_win_rate = 0.33
    features = [
        home_goals_for,           # 主队近期场均进球
        home_goals_against,       # 主队近期场均失球
        away_goals_for,           # 客队近期场均进球  
        away_goals_against,       # 客队近期场均失球
        home_win_rate,            # 主队近期胜率
        away_win_rate,            # 客队近期胜率
        home_shots,               # 主队近期射门
        away_shots,               # 客队近期射门
        home_shots_on_target,     # 主队近期射正
        away_shots_on_target,     # 客队近期射正
        home_corners,             # 主队近期角球
        away_corners,             # 客队近期角球
        home_fouls,               # 主队近期犯规
        away_fouls,               # 客队近期犯规
        1                         # 主场优势(固定为1)
    ]
    features_list.append(features + [current_match['FTR']])  # 添加目标变量

feature_names = [
    'home_goals_for', 'home_goals_against', 'away_goals_for', 'away_goals_against',
    'home_win_rate', 'away_win_rate', 'home_shots', 'away_shots',
    'home_shots_on_target', 'away_shots_on_target', 'home_corners', 'away_corners',
    'home_fouls', 'away_fouls', 'home_advantage'
]

features_df = pd.DataFrame(features_list, columns=feature_names + ['target'])

# 分离特征和目标变量
X = features_df[feature_names].copy()
y = features_df['target'].copy()

# 编码目标变量 H->2, A->0, D->1  
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X['goals_diff'] = (X['home_goals_for'] - X['home_goals_against']) - (X['away_goals_for'] - X['away_goals_against'])
X['form_diff'] = X['home_win_rate'] - X['away_win_rate']
X['shots_diff'] = X['home_shots'] - X['away_shots']
X['shots_on_target_diff'] = X['home_shots_on_target'] - X['away_shots_on_target']

# 效率特征
X['home_shot_accuracy'] = X['home_shots_on_target'] / np.maximum(X['home_shots'], 1)
X['away_shot_accuracy'] = X['away_shots_on_target'] / np.maximum(X['away_shots'], 1)
X['shot_accuracy_diff'] = X['home_shot_accuracy'] - X['away_shot_accuracy']

X['home_attack_efficiency'] = X['home_goals_for'] / np.maximum(X['home_shots_on_target'], 1)
X['away_attack_efficiency'] = X['away_goals_for'] / np.maximum(X['away_shots_on_target'], 1)
X['attack_efficiency_diff'] = X['home_attack_efficiency'] - X['away_attack_efficiency']

#分割训练测试集
split_point = int(len(X) * 0.8)
X_train = X.iloc[:split_point].copy()
X_test = X.iloc[split_point:].copy()
y_train = y_encoded[:split_point].copy()
y_test = y_encoded[split_point:].copy()

#数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#使用逻辑回归
logreg_model = LogisticRegression(
    random_state=50,
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs'
)

logreg_model.fit(X_train_scaled, y_train)

#训练集预测
y_train_pred = logreg_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

# 测试集预测
y_test_pred = logreg_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

# 预测概率
y_test_proba = logreg_model.predict_proba(X_test_scaled)
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


cm = confusion_matrix(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)

from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_test_pred, average=None)
target_names = label_encoder.classes_

avg_precision = precision.mean()
avg_recall = recall.mean() 
avg_f1 = f1.mean()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel('predicted result')
plt.ylabel('real result') 
plt.title('Premier League match result prediction (Logistic Regression) - Confusion Matrix')
plt.tight_layout()
plt.show()
