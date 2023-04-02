import streamlit as st
st.title("AI to find Iris flower species")
st.subheader("Enter the Iris flower's properties and find out it's species")

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np




a=st.number_input(label_visibility="visible",label="Enter sepal length of an Iris flower (in cms)")
b=st.number_input(label_visibility="visible",label="Enter sepal width of an Iris flower (in cms)")
c=st.number_input(label_visibility="visible",label="Enter petal length of an Iris flower (in cms)")
d=st.number_input(label_visibility="visible",label="Enter petal width of an Iris flower (in cms)")

# st.write(type(box_a))

knn=KNeighborsClassifier(n_neighbors=1)
iris_dataset= load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

X_new=np.array([[a,b,c,d ]])

knn.fit(X_train,y_train)



smth=knn.score(X_test,y_test)

accuracy=smth*100

def qwerty():
    prediction=knn.predict(X_new)
    
    if (prediction == 0):
        st.write("The predicted species is Iris Setosa")        
    elif (prediction == 1):
        st.write("The predicted species is Iris Versicolor")      
    else:
        st.write("The predicted species is Iris Verginica")
    
st.button(label="Find the Species",on_click=qwerty)

