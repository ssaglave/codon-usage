# Codon-usage Project
This repository is implemented as a group project for the **Introduction to Data Science** course. The goal is to develop a robust classification model that can accurately predict an organism's kingdom based on its codon usage patterns. This will provide deeper insights into genomic adaptations and taxonomic relationships, contributing to our understanding of evolutionary biology and molecular genetics.

## Table of Contents
* [Dataset Overview](#overview)
* [About the project](#about)
* [Project Pipeline](#pipeline)
* [Tools, Libraries, and Frameworks](#tools)
* [Usage](#usage)
* [Team](#team)
* [References](#references)
* [Conclusion](#conclusion)

### <a name="overview"></a> Dataset Overview
The dataset is taken from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/577/codon+usage) 
The dataset comprises 65 numerical and 4 categorical attributes. It contains 13,028 instances, of which 80% (10,352) are classified under key kingdom types such as bacteria, vertebrates, plants, and viruses, while the remaining 20% (2,676) represent other kingdom types.

<img width="222" alt="image" src="https://github.com/user-attachments/assets/e8edf7ca-efbc-46bd-ab44-5701f17ded46" />


### <a name="about"></a> About the project
This project aims to analyse DNA codon usage frequencies across various biological organisms from diverse taxa. The dataset includes multivariate features, such as kingdom classification, DNA type, species ID, codon frequencies, and species names. The primary focus is on the *KINGDOM* column, which serves as the target variable. This column contains a range of sub-species categories, including 'arc' (archaea), 'bct' (bacteria), 'phg' (bacteriophage), 'plm' (plasmid), 'pln' (plant), 'inv' (invertebrate), 'vrt' (vertebrate), 'mam' (mammal), 'rod' (rodent), 'pri' (primate), and 'vrl' (virus). The findings from this study could enhance our knowledge of how codon usage varies among organisms, shedding light on the underlying biological processes that drive genomic diversity and evolutionary strategies.

### <a name="pipeline"></a> Project Pipeline
* **Data Pre-processing:** Handle missing values, normalize codon frequencies, and encode categorical features like kingdom and DNA type. 
 
* **Feature Selection:** Identify which features play a significant role in describing dependent variables by using a correlation plot. 
 
* **Model Selection:** Choose suitable machine learning algorithms for classification. 
 
* **Model Training:** Train the selected models on the dataset, splitting it into training and testing sets. 
 
* **Model Evaluation:** Evaluate model performance using metrics like accuracy, precision, recall, and F1 score to assess classification results. 
 
* **Results Interpretation:** Interpret findings to understand the biological significance of identified patterns and relationships among diverse organisms. 


### <a name="tools"></a> Tools, Libraries, and Frameworks
* Matplotlib, Seaborn for Visualization
* Python libraries: NumPy, Pandas for Data Preprocessing 
* Scikit-learn for model building and evaluation
* Machine learning frameworks: TensorFlow, Keras
* Jupyter Notebook for code development and analysis


### <a name="usage"></a> Usage
**Data Pre-processing:**
<br>Essential libraries for data manipulation, visualization, and preprocessing are imported. Dataset is read from a csv file. Dataset overview, descriptive statistics and dataset dimensions are reviewed.

**1. Exploratory Data Analysis (EDA)**
<br>We observed that the 'Kingdom' variable exhibits significant class imbalance, which could potentially impact the reliability and interpretability of further analysis. Seaborn distribution plots and histograms were utilized to visualize the distribution of the 'Kingdom' variable, providing insights into its frequency and overall data spread.

<img width="821" alt="image" src="https://github.com/user-attachments/assets/73b04fc0-4775-4f82-8914-ace4202abfc5" />

<img width="599" alt="image" src="https://github.com/user-attachments/assets/8803afc0-8c0c-454b-b356-e930a4aafcdc" />

**2. Data Cleaning**
<br>Identified columns containing numeric data and computed the correlation matrix for numerical columns to explore relationships between variables. Visualizing heatmap we came to know it is symmetric about the diagonal, as expected in a correlation matrix. 

<img width="596" alt="image" src="https://github.com/user-attachments/assets/0658884d-6e38-4809-91ef-6ebe49d08ab1" />

**3.	Outliers Handling:**
<br>•	Used IQR method to detect outliers and plotted boxplots and histograms for numerical columns.
<br>•	Boxplots highlight outliers in key columns – in this analysis, we found there are a significant number of outliers(29568 in number). Removing a large portion of these outliers would considerably reduce the dataset size, negatively impacting further analysis. Training a model on a smaller dataset would not be ideal for accurate classification or prediction. 
<br>•	Therefore, we replaced the outliers using mean imputation to handle outliers.

<img width="1200" alt="image" src="https://github.com/user-attachments/assets/bced3d2a-7b6f-46bb-b827-673888da8177" />

**4. Handling data imbalance and normalizing data:**
<br>The data appears to be highly imbalanced. To address this, we balanced the training data using SMOTE (Synthetic Minority Over-Sampling Technique). Then, we scaled numerical data using MinMaxScaler. Label encoding was applied to the "Kingdom" column, converting categorical values into numerical form for model compatibility .The dataset is clean, with no missing values and categorical values.

<img width="1217" alt="image" src="https://github.com/user-attachments/assets/adfbc7f3-80c2-4cf2-8794-2fface6085c9" />


**Feature Engineering:**
<br>Used PCA with scatter plot results to visualize the variance explained by principal components or other patterns for reduced dimensions. But since new PCA components are created using all features, there could be a possibility of data loss and would require transforming input features to PCA components again for prediction. Hence, to select and reduce feature list as we have a high number of features in this dataset so we used two other techniques for feature selection and elimination.

<img width="1030" alt="image" src="https://github.com/user-attachments/assets/75098a17-f9ed-4023-996b-50d7b4616893" />
<img width="914" alt="image" src="https://github.com/user-attachments/assets/132ae9a5-cb06-4b9a-bb43-033ef3cbbe5b" />

**Feature selection using lasso regression:**
<br>Used KFold cross validation using GridSearch for best param value of learning rate alpha for lasso regression model and used that to perform feature selection. Plotted bar chart to check feature importance and based on lasso coefficients selected features with high importance and reduced feature set from 64 to 46.

<img width="910" alt="image" src="https://github.com/user-attachments/assets/cbabe48e-e47f-4474-94fa-7a789dc272b2" />

**Feature extraction using bidirectional elimination:**
<br>We used `SequentialFeatureSelector` from the `mlxtend` library to identify important features using bi-directional elimination.  A `LinearRegression` model is chosen as the estimator for selecting features. The process targets between 1 and 46 features. Initially we selected about that 23 features in bi-directional elimination but as it making the model underfit the data so we decided to eliminate only 6 features in this step and go with 40 features.

<img width="955" alt="image" src="https://github.com/user-attachments/assets/25c04960-bff5-42d2-9f5d-5bf8a65805a5" />


**Model selection and training:**
<br>Trained few basic and few advanced machine learning models to perform multi-class classification:
<br><br>**Basic models used:**
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Random Forest (RF)  
4. Support Vector Machine (SVM) with both linear and RBF (non-linear) kernels.  
5. Naive Bayes (NB) 

<br>**Advanced models used:**
<br>6. XGBoost
<br>7. Extreme Learning Machine (ELM)
<br>8. A simple deep learning model with two layers
<br>9. An ensemble model

**Hyperparamter tuning:**
<br>•	Boosts performance and prevents overfitting and underfitting.
<br>•	Enhances accuracy and training efficiency.
<br>•	Tested Hyperparameters: Neuron counts (ELM), Learning rates (Neural Network), and Dropout layers (Deep Learning model)
<br>•	Utilized K-Fold Cross Validation with GridSearch to identify optimal parameter values.


**Model Evaluation:**
<br>Accuracy for all models on training and testing data is:

![image](https://github.com/user-attachments/assets/5077219a-538a-4899-9e36-a20206c18426)


**Results:**
<br>As we can see in the above models following are the top 3 performing models:
<br>XGBoost Classifier, SVM Classifier with Non-Linear kernel and Random Forest Classifier/Ensemble. But having 100% accuracy on training data means the model is very generalised, i.e., the model has learned patterns so broadly that it struggles to accurately predict specific details or nuances in new data. 
<br>Hence, the best model for this dataset is SVM using a non-linear kernel, which gives a training accuracy of 97% and a testing accuracy of 87%.

<img width="1027" alt="image" src="https://github.com/user-attachments/assets/00afc405-5531-401b-80ea-b1898777c8cb" />
<img width="1267" alt="image" src="https://github.com/user-attachments/assets/1852af7d-45f3-4fa0-b44f-cb56ce863316" />


### <a name="team"></a> Team
1. Suvarna Sangram Aglave (Team Lead)
2. Kush Rakesh Mehta 
3. Sanika Nirmal Sahuji 
4. Keerthana Bathini 

### <a name="references"></a> References
* https://en.wikipedia.org/wiki/Codon_usage_bias
* https://www.sciencedirect.com/science/article/abs/pii/0167779988900467
* https://www.pnas.org/doi/10.1073/pnas.2410003121

### <a name="conclusion"></a> Conclusion
**Final Results:**
* Each model's performance is summarized, highlighting their strengths and weaknesses.
* XGBoost provides robust results with manageable computation times.
* ELM offers quick training but has limitations with larger datasets.
* The deep learning model achieves competitive accuracy but requires longer training time.

Though both XGBoost and ensemble method give 88% accuracy on test data but have 100% accuracy on training data, so the model is very generalised i.e. the model has learned patterns so broadly that it struggles to accurately predict specific details or nuances in new data. Hence, the best model for this dataset is SVM using non-linear kernel which gives a training accuracy of 97% and testing accuracy of 87%.

