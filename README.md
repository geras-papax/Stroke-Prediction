# Stroke-Prediction---Random-Forest

Project for the Data Mining Course of the [Department of Computer Engineering & Informatics](https://www.ceid.upatras.gr/en)

## Description 

### Data

1. [Healthcare - Dataset](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv).
2. [Attributes - Dataset](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/healthcare-dataset-stroke-data/Attribute%20Information.docx).

### Part I

Dataset Analysis - With the **Attributes-Dataset**, we created some graphs that show the relationships between Attributes.

### Part II

We try to solve the Empty Values problem of the **healthcare dataset** with 4 methods:

1. **Remove** columns with empty values
2. **Replace** empty values with mean value of the column
3. [**Linear Regression**](https://en.wikipedia.org/wiki/Linear_regression)
4. [**K-Nearest-Neighbors**](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

### Part III

Using [**Random Forest**](https://en.wikipedia.org/wiki/Random_forest), we predict if a patient is prone or not to have a stroke, for every new dataset created in **Part II**.
<br /> For the classification we [split the data](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) as follows: **75% train**, **25% test**.
<br /> Finally, we measure the accuracy of our model using these performance metrics: [**f1 score**](https://en.wikipedia.org/wiki/F-score), [**precision and recall**](https://en.wikipedia.org/wiki/Precision_and_recall).


## Results:

### Dataset Analysis

![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/multiplots.png)

#### Prediction Results of each method

![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/remove.png)
![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/mean.png)
![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/linear%20regression.png)
![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/knn.png)

### Performance Results (f1 score, precision and recall)

#### n_estimators = 5
![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/scores.png)
#### n_estimators = 10 
![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/scores-10.png)
#### n_estimators = 20
![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/scores-20.png)
#### n_estimators = 100
![ ](https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/blob/main/graphs/scores-100.png)




## Team
- [Michail Karavokyris](https://github.com/karavokyrismichail)
- [Gerasimos Papachronopoulos](https://github.com/geras-papax)
