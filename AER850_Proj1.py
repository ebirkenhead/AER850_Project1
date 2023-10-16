#Imported libraries are shown under the first step they are needed for in order to attempt to better show why each library has been imported

# Step 1: Data Processing
import pandas as pd
data_frame = pd.read_csv('Project 1 Data.csv') #reads the data into the variable name "data_frame"

#Step 2: Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Extract the x, y, and z components of "data_frame"
x = data_frame['X']
y = data_frame['Y']
z = data_frame['Z']

#Plot the figure in a 3D line plot
figure = plt.figure()
ax = figure.add_subplot(111, projection='3d')

#Set figure legend, titles, axis etc
ax.plot(x, y, z, label = 'x and y versus z')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Dataset Visualization")
plt.legend()

plt.show() #Show the plot


#Step 3: Correlation Analysis

#Pearson correlation
corr = data_frame.corr()

#Correlation matrix using pearson correlation analysis
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
plt.title('Correlation Matrix')


#Step 4: Classification Model Development / Engineering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#Sets the feature and target variables
X = data_frame[['X', 'Y', 'Z']]
y = data_frame[['Step']]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, random_state = 40) #Split the data into test and train strands

#Perform classifier function, and grid searches on the data

#Decision Tree Classifier
dtc = DecisionTreeClassifier()
tree_params = [{'max_depth': list(range(10, 15)), 'min_samples_split': list(range(0,14)), 'min_samples_leaf': list(range(0,5))}]
classi_A = GridSearchCV(dtc, tree_params, cv = 10, scoring='accuracy')


#Random Forest Classifier
rfc = RandomForestClassifier()
forest_params = [{'n_estimators': list(range(0, 20))}]
classi_B = GridSearchCV(rfc, forest_params, cv = 10, scoring='accuracy')


#SVC
sv = SVC()
sv_params = [{'C': [0.1, 1, 10], 'kernel':['rbf'], 'gamma':['auto']}]
classi_C = GridSearchCV(sv, sv_params)


#Step 5: Model Performance Analysis
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix #Import required scoring systems

#"classifiers" Used as an umbrella variable for the classifiers imported in step 4
classifiers = [('Decision Tree Classifier', DecisionTreeClassifier(), sv_params),
                ('Random Forest Classifier', RandomForestClassifier(), forest_params),
                ('SVC', SVC(), sv_params)]

#A function was created to cycle through the different classifiers and scoring systems
def evaluation(model, X_train, y_train, X_test, y_test): 
    model.fit(X_train, y_train)
    y_prediction = model.predict(X_test) #Created y_prediction value to test results
    
    #Used y_prediction value in scoring systems
    f1 = f1_score(y_test, y_prediction, average='weighted')
    prec = precision_score(y_test, y_prediction, average='weighted')
    acc = accuracy_score(y_test, y_prediction)
    
    confusion = confusion_matrix(y_test, y_prediction) #Confusion matrix created for classifier
    
    print('f1: ', f1, 'precision score: ', prec, 'accuracy_score: ', acc) 
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

for name, model, param_grid in classifiers:
    evaluation(model, X_train, y_train, X_test, y_test)



#Step 6: Model Evaluation
import joblib

#Create joblib file
file = 'joblib_model_bestChoice.joblib'

#Hard input the outputted values from step 5
dtc_pred = [1, 1, 1]
rfc_pred = [0.98815, 0.989098, 0.9883721]
sv_pred = [0.73145, 0.726986, 0.7616279]

#peform the F1 score analysis on the y_test value and the precited values
dtc_f1 = f1_score(y_test, dtc_pred, average='macro')
dtc_prec = f1_score(y_test, rfc_pred, average='macro')
dtc_acc = f1_score(y_test, sv_pred, average='macro')

#Save the score analyses into one umbrealla variable
model_scores =  {"Decision Tree": (dtc_f1, dtc_prec, dtc_acc)}
best_model = dtc.fit(X_train, y_train)

#dump the values into the joblib file
joblib.dump(model_scores[best_model], file)

#test data
data = [[9.375, 3.0625, 1.51],[6.995, 5.125, 0.3875],[0, 3.0625, 1.93],[9.4, 3, 1.8],[9.4, 3, 1.3]]

#Perform the test
testing = model_scores[best_model].predict(data)

#End

























