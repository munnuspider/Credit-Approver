from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# train models using logistic regression and naive bayes
# to test model performance:
# precision used to test false positives: TP/(TP+FP)
# CM and accuracy is general and F1 is a balance of precision and recall
# recall is for fale negatives: TP/(TP+FN)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score
log_model = LogisticRegression()
log_model.fit(x_train_scaled, y_train)
y_pred = log_model.predict(x_test_scaled)

# eval
print("Logistic Regression Model: ")
print("Precision", precision_score(y_test, y_pred))
print("Accuracy", accuracy_score(y_test, y_pred))
print("F1 score", f1_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("CM", confusion_matrix(y_test, y_pred))


# KNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train_scaled, y_train)
y_pred = knn_model.predict(x_test_scaled)

# eval
print("KNN: ")
print("Precision", precision_score(y_test, y_pred))
print("Accuracy", accuracy_score(y_test, y_pred))
print("F1 score", f1_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("CM", confusion_matrix(y_test, y_pred))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

NB_model = GaussianNB()
NB_model.fit(x_train_scaled, y_train)
y_pred = NB_model.predict(x_test_scaled)

# eval
print("Naive Bayes: ")
print("Precision", precision_score(y_test, y_pred))
print("Accuracy", accuracy_score(y_test, y_pred))
print("F1 score", f1_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print("CM", confusion_matrix(y_test, y_pred))

# with feature eng, logistic regression seems to have improved the most
# naive bayes are still the best
# due to one hot encoding and several data attributes - knn can't deal with uneven classes very well
