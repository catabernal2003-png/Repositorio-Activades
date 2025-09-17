import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io

def train_and_evaluate():
	data = pd.read_csv("data.csv")
	X = data.drop("HeartDisease", axis=1)
	y = data["HeartDisease"]
    
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	logistic_model = LogisticRegression()
	logistic_model.fit(X_train, y_train)
	y_pred = logistic_model.predict(X_test)
	conf_matrix = confusion_matrix(y_test, y_pred)
	accuracy = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, output_dict=True)
	report_text = classification_report(y_test, y_pred)
	return conf_matrix, accuracy, report, report_text

def plot_confusion_matrix(conf_matrix):
	plt.figure(figsize=(8,6))
	sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.title('Matriz de Confusión')
	buf = io.BytesIO()
	plt.tight_layout()
	plt.savefig(buf, format='png')
	plt.close()
	buf.seek(0)
	return buf