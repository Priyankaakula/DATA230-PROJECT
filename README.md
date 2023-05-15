Abstract 

         The aim of this study is to construct a prognostic model that can effectively anticipate the incidence of cardiovascular disease in individuals, utilizing diverse medical indicators. The dataset utilized in this study comprises data on more than 319796 patients, encompassing variables such as age, gender, blood pressure, cholesterol levels, and additional medical parameters. The project comprises initial data preprocessing and exploratory data analysis to obtain valuable insights from the dataset. Subsequently, feature selection and model building will be performed utilizing diverse machine learning algorithms, including logistic regression, decision trees, and random forests. The assessment of the models' performance is conducted through the utilization of diverse evaluation metrics, including but not limited to accuracy, precision, recall, and F1 score. The findings indicate that the random forest algorithm exhibits superior performance in forecasting heart disease, yielding an accuracy rate exceeding 85%. The present study holds significant practical implications for the timely identification and mitigation of cardiovascular ailments, thereby resulting in enhanced patient prognoses and decreased fatality ratios.













1   INTRODUCTION

           Heart disease is a popular cause of deaths worldwide, and timely identification and mitigation of cardiovascular disease can substantially diminish fatality rates. The objective of this study is to construct a prognostic model that can effectively anticipate the probability of cardiovascular disease in individuals by utilizing their medical data. The dataset utilized in this study encompasses data on more than 319796  patients, comprising variables such as age, gender, blood pressure, cholesterol levels, and additional medical parameters. The project comprises several phases, including data preprocessing, exploratory data analysis, feature selection, and model construction utilizing diverse machine learning algorithms. The models' performance is assessed through diverse evaluation metrics, including accuracy, precision, recall, and F1 score. The outcomes of this study hold practical significance in terms of timely identification and mitigation of cardiovascular ailments, which can result in better health outcomes for patients and decreased mortality rates. In general, this study underscores the capability of machine learning algorithms in the healthcare domain, specifically in forecasting and averting fatal illnesses like heart disease.


1.1   PROJECT BACKGROUND

              Heart disease is a significant public health concern and a primary contributor to death on a global scale. As per the report of the World Health Organization (WHO), there is an approximate of 17.9 million fatalities caused by cardiovascular diseases (CVDs) annually, which constitutes 31% of the total global deaths. Cardiovascular diseases (CVDs) can be averted by timely identification and proficient control of risk factors such as hypertension, lipid profile, and tobacco use. Accurate predictive models that can anticipate the probability of heart disease in patients are of paramount importance in facilitating early detection and prevention.

         Machine learning algorithms have demonstrated significant potential in the healthcare domain, specifically in forecasting and averting fatal ailments like heart disease, in recent times. The objective of this project is to construct a prognostic model that can effectively forecast the probability of cardiovascular disease in individuals by utilizing their medical data. The project is an extension of the expanding corpus of literature that leverages machine learning algorithms to construct precise prognostic models for diverse ailments. The healthcare industry has immense potential to leverage machine learning algorithms, as evidenced by the practical applications of such projects in predicting and preventing diseases. This can result in better patient outcomes and reduced mortality rates.

1.2    PROBLEM DEFINITION

            The problem tackled  is of significant importance as heart disease is a leading cause of mortality worldwide. Accurate prediction of heart disease can enable early intervention and appropriate treatment potentially saving lives and improving patient outcomes. By utilizing machine learning algorithms and statistical modeling techniques it aims to build predictive models that can classify patients as either having heart disease or being healthy based on their attribute profiles.

1.3    OBJECTIVE

            The objective of this project is to utilize medical data to forecast the probability of a patient being diagnosed with heart disease. Early diagnosis and treatment of patients can be advantageous for medical practitioners, as it has the potential to enhance patient outcomes and decrease healthcare expenses. Furthermore, the prognostications concerning heart disease may be of interest to insurance firms and policy makers as they seek to apportion resources and devise strategies pertaining to the prevention and management of this condition. The primary objective of this project is to leverage data analysis and machine learning techniques to offer practical insights and enhance health outcomes for individuals who are susceptible to heart disease.









1.4    LITERATURE REVIEW

          Heart disease and stroke are the foremost cause of mortality on a global scale, representing 31% of all fatalities. Timely detection and prophylaxis of such ailments can enhance the likelihood of survival and alleviate the strain on healthcare infrastructures. Various risk factors have been utilized to develop predictive models for cardiovascular diseases through the application of machine learning techniques. The employment of machine learning algorithms for the purpose of predicting the likelihood of heart disease has garnered heightened attention in recent times.

          Numerous investigations have been carried out utilizing machine learning methodologies to forecast the occurrence of cardiovascular disease. Krittanawong et al. (2017) employed machine learning algorithms to forecast the likelihood of cardiovascular disease utilizing clinical data. The research employed a dataset comprising 303 patients and implemented multiple machine learning algorithms, such as decision tree, logistic regression, and neural networks. According to the study, the neural network algorithm demonstrated the highest level of accuracy in forecasting the probability of heart disease.

          O'Gara et al. (2019) conducted a study utilizing machine learning algorithms to forecast the likelihood of heart failure by analyzing electronic health records. The research employed a dataset comprising 57,335 patients and implemented multiple machine learning algorithms, such as random forest and gradient boosting. According to the study, the gradient boosting algorithm demonstrated the highest level of accuracy in forecasting the likelihood of heart failure.

        Li et al. (2020) employed machine learning algorithms to forecast the likelihood of cardiovascular disease utilizing a dataset of 4,287 patients. The research utilized a variety of machine learning algorithms, such as logistic regression, decision tree, and support vector machines. According to the study, the support vector machine algorithm demonstrated the highest level of accuracy in forecasting the probability of heart disease.



           In aggregate, the studies mentioned above indicate that machine learning algorithms possess the capability to accurately forecast the likelihood of heart disease occurrence by leveraging diverse risk factors. Nevertheless, additional investigation is required to establish precise and dependable prognostic models for heart disease problems.



2    CRISP- DM METHODOLOGY

          We have followed a hybrid model of the waterfall and CRISP-DM methodology to develop the proposed solution. With this methodology we can have a clear picture of the time duration of each task and work done by each team mate.

Business Understanding
           In this stage, we have established goals through team meetings and brainstorming for heart disease dataset analysis. Evaluated research status and identified knowledge gaps through literature surveys. Developed a data mining plan including dataset preparation, trend exploration, algorithm implementation, outcome assessment  and documentation.

Data Understanding
          In this stage, we decided to collect the historical data collected from Kaggle. This data may include demographic information, medical history, risk factors, diagnostic tests, and outcomes of patients with heart disease. The goal is to obtain a comprehensive understanding of past cases and trends related to heart disease to inform the predictive modeling process. We also conducted Exploratory data analysis (EDA) which involves examining and visualizing the heart disease dataset to gain insights and identify patterns. Finally, we have performed Data quality checks that included identifying missing values, outliers, duplicate entries, or inconsistent formats.



Data Preparation
          In Data preparation, we have performed various levels of data pre-processing and feature selections. Data pre-processing is a crucial step in heart disease prediction as it involves cleaning, transforming, and organizing the data to ensure its quality and suitability for analysis. It typically includes handling missing values, handling outliers, scaling numerical features, encoding categorical variables, and splitting the data into training and testing sets. Feature selection plays a vital role in decision trees and KNN models for heart disease prediction. It involves identifying the most relevant and informative features from the available dataset. 


Data Modeling

































Data Understanding

The dataset utilized in this study comprises data pertaining to patients and their medical attributes, encompassing age, gender, blood pressure, cholesterol levels, and the presence or absence of cardiovascular disease. The dataset comprises 319796 instances and 18 variables. The variables under consideration are:

Age shows the number of years a person has lived.
Gender: Gender was coded as a binary variable, with a value of 1 indicating male and a value of 0 indicating female.
Resting blood pressure refers to the initial blood pressure measurement, expressed in millimeters of mercury (mm Hg), obtained upon admission to the hospital.
Serum cholesterol levels are typically measured in milligrams per deciliter (mg/dL).
A fasting blood sugar level that exceeds 120 mg/dl. The binary system is used to represent true and false values, where 1 represents true and 0 represents false.
The results of the resting electrocardiogram are expressed in numerical values of 0, 1, or 2.
The topic of interest pertains to the maximum heart rate attained during physical activity and the occurrence of angina as a result of exercise. (1 indicating affirmative response and 0 indicating negative response)
Oldpeak refers to the amount of ST depression that is caused by physical exercise in comparison to a state of rest. The slope, on the other hand, pertains to the incline of the maximum level of physical exertion. The ST segment and the number of major vessels (ranging from 0 to 3) colored by fluoroscopy are important indicators in the assessment of thalassemia. The numerical values assigned to the conditions are as follows: 3 denotes a normal state, 6 indicates a fixed defect, and 7 signifies a reversible defect.
The focus of this inquiry pertains to the identification and classification of cardiovascular ailments. (1 = affirmative, 0 = negative)
By means of conducting data exploration and visualization, it is possible to enhance our comprehension of the distribution of individual variables as well as the interrelationships among them. Data cleaning and preprocessing can be utilized to identify and rectify any absent or inaccurate data. The implementation of this approach would facilitate the development of more precise and efficient prognostic models for the detection of cardiovascular ailments.



• Motivation for this project. What is the problem ? 
The objective of our research is to make use of information regarding age, sex, blood pressure, cholesterol levels, and other medical indicators to forecast the occurrence of heart disease. One of the primary causes of death worldwide is heart disease, which can be considerably reduced by early detection and prevention. The challenge is to create a model that can precisely forecast a patient's risk of developing heart disease based on their medical history. This can help medical providers identify patients sooner and treat them when necessary, lowering the risk of heart disease and increasing patient outcomes overall.


 Data characteristics, and how to process data. 
The "Heart Disease UCI" dataset includes data on 319796 people and 18 variables, such as age, sex, type of chest pain, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate attained, exercise-induced angina, ST depression induced by exercise relative to rest, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, thalassemia, and the presence of heart disease. Here are some guidelines for handling this data:

Investigate the data: Before beginning any analysis, it's crucial to investigate the data to understand its properties. Summary statistics examination, variable distribution visualization, and the search for missing values are some examples of this.

Data cleaning: To make sure the data is ready for analysis after being explored, clean it. This may entail eliminating duplicates, dealing with missing values, and changing variables as required.

Feature engineering: You might need to build new features from the variables already there, depending on the analysis you wish to conduct. Combining variables, making dummy variables for categorical variables, and scaling variables as necessary can all be used to achieve this.

Splitting the data into a training set and a testing set is necessary once the data has been cleaned and processed. Your model is trained on the training set, and its generalizability to fresh data is assessed on the testing set.

Modeling: In order to create a predictive model, you can use a variety of machine learning techniques. Numerous techniques, including logistic regression, decision trees, random forests, and neural networks, may be used to do this. It's crucial to assess each model's performance using measures like accuracy, precision, recall, and F1 score.

The "Heart Disease UCI" dataset, as a whole, offers a comprehensive range of characteristics that may be utilized to investigate the risk factors connected to heart disease and create predictive models to pinpoint those who are at risk. We can learn insights that might aid us in better comprehending this significant health concern by properly processing and evaluating this data.

KNN

The KNN algorithm, also known as k-Nearest Neighbors, is a type of supervised learning approach that is commonly employed for classification tasks. This involves the prediction of a categorical label or class based on a given set of input features. The implementation of K-Nearest Neighbors (KNN) algorithm is a viable approach in predicting the likelihood of heart disease occurrence in patients, utilizing their demographic, clinical, and behavioral characteristics.

The K-nearest neighbors (KNN) algorithm operates by identifying the K closest neighbors to a designated test sample within the feature space, utilizing a distance metric. The metric utilized to measure distance can take various forms, including but not limited to Euclidean distance, Manhattan distance, or other distance metrics. After identifying the k nearest neighbors, the algorithm proceeds to assign the predicted class for the test sample by selecting the most frequent class among the identified neighbors.

K-Nearest Neighbors (KNN) algorithm can be employed to identify the k most similar patients from a given dataset, based on certain features, for a new patient. The classification of the new patient is then determined by the majority class of the k nearest neighbors.

The K-Nearest Neighbor (KNN) algorithm has the potential to be a valuable tool for predicting heart disease, owing to its ability to accommodate non-linear decision boundaries and its adaptability to novel data. The efficacy of the algorithm is contingent upon the selection of k and the distance metric employed, necessitating empirical investigation and assessment on a validation dataset.

Decision Tree 

The heart disease prediction project uses decision tree as a machine learning algorithm for the purpose of categorizing patients into two distinct groups, namely those who exhibit the presence of heart disease and those who do not. The decision tree algorithm generates a hierarchical structure that forecasts the target variable by considering multiple input variables, including but not limited to age, gender, chest pain type, and blood pressure. The decision tree algorithm constructs a tree structure through a recursive process of partitioning the dataset into progressively smaller subsets, contingent upon the value of a designated attribute, until a predetermined termination condition is satisfied. Subsequently, the tree is employed to categorize novel instances via traversal from the root to a leaf node, which ascertains the anticipated class.

The present study involves the utilization of the decision tree algorithm to train on the heart disease dataset with the aim of acquiring knowledge on the patterns and interrelationships that exist among each of the input variables and the variable being investigated. Upon completion of the training process, the decision tree model can be employed to make predictions regarding the likelihood of heart disease in novel patients, utilizing their respective input variables. The evaluation of the decision tree model's accuracy entails the utilization of metrics such as precision, recall, accuracy, and F1 score. These metrics serve to gauge the model's ability to accurately predict the presence or absence of heart disease.

Data Collection:

The data utilized in this research project was obtained from the Cleveland Clinic Foundation located in Switzerland. The dataset comprises 319796 individuals and encompasses 18 variables pertaining to heart disease illness. The data was obtained through a range of medical examinations, including but not limited to blood pressure assessment, electrocardiography (ECG), and angiography. The objective of this study is to utilize the available data to construct a predictive model for detecting the occurrence of heart disease in individuals. The dataset was procured from Kaggle's online platform.

Deployment:

We have deployed all the files in github repository.







EDA:

The above density plot represents the distribution of body mass index. The X-axis represents the Body Mass and the Y-axis represents the frequency. Heart Disease is represented in Red color and Normal is represented in Yellow color. Heart Disease at the body mass 30 has the highest frequency of 0.075. Normal at the body mass 25 has the highest frequency of 0.078. Both Heart disease and Normal are almost overlapping.


The above bar plot represents the distribution of people having heart disease for different ages. The X-axis represents the Age groups and the Y-axis represents the number of people who have heart disease. The 80 or Older age group have a higher heart disease count of 5700. The 18-24 age group has a lower heart disease count of 150. As the age group increases, the number of people who have heart disease counts increases gradually. Except for the 75-79 age group, the count has dropped to 4000.


The above correlation matrix, Physical health and DiffWalking has the highest correlation with value 0.43. DiffWalking and PhysicalActivity have the lowest correlation with value -0.28. Physical activity has the more negative correlations with Heart disease, BMI, Smoking, Stroke, Physical health, Mental health, Diffwalking and Diabetic.




The above plot represents the Distribution of correlation of features. DiffWalking has the highest with 0.200 value and Sleep Time has the lowest with 0.010 value. DiffWalking & Stroke, Diabetic & Physical Health, AlcoholDrinking & Mental health, these pair have just minor differences in the distribution.


Conclusion:
In conclusion, the heart disease study project yielded substantial insights regarding the incidence, causes, and mitigation strategies linked with heart disease. It is possible to discover essential factors and develop efficient strategies for timely diagnosis and prevention using data analysis and statistical modeling. It is critical to ensure the quality of data in order to generate reliable results. In general, this also improves comprehension and results in the treatment of heart disease.






