#Thomas Beaupre, Yunhwan Jeong
#Vijayalakshmi Ramasamy, Ph.D.
#CSCI - 412 Data Mining and Machine Learning

# Abstract
“Fossil-fuel combustion by-products are the world’s most significant threat to children’s health and future and are major contributors to global inequality and environmental injustice” (Perera, 2015). While we still live in a world where we are heavily reliant on fossil-fueled vehicles, understanding how fuel types can help impact those emissions is critical to understanding how to approach and limit environmental injustice. While new methods such as hybrid cars or electric cars hope to heavily reduce or completely eliminate our reliance on fossil fuels in this sector, we still understand that they are years away from rolling out a completely new set of vehicles around the world. Therefore to make an impact now, we can create an algorithm in which users can understand their pollution based on their fuel type usage and by creating this awareness, hope to limit individual human pollution by making smarter choices in fuel types. The primary aim of this research is to investigate the relationship between the various fuel types for internal combustion engines compared to the pollution they produce. By comparing various data sources we want to describe to the readers the impact of fuel choices on pollution and give key insights into making more sustainable and environmentally positive choices. To achieve our goal we want to collect data based on fuel consumption and their respective pollution levels. We want to intake a wide range of data to ensure unbiased fair results. By this we mean we want data from numerous cities and locations around the world. While our algorithm and results have not been developed yet after accessing prior research we understand that data is important in making informed decisions on our fuel consumption. A study by the Government of Canada helped to compare brands and vehicle models by their fuel consumption and carbon emissions (Hien & Kor, 2022). While that data may “make evidence-based recommendations to both vehicle users and producers to reduce their environmental impacts”, the algorithm we hypothesize will start to be able to rate not only how the car performs in emissions but how the fuel does as well (Hien & Kor, 2022). By highlighting the importance of fuel choices and the there influence on pollution levels we hope to promote a transition into cleaner more reliable and renewable forms of energy in use for transportation. 

# Dataset
Data being used is sourced from the EIA as a monthly dataset from the USA transportation sector. The data set can be found in section 2.2 at:
*   https://www.eia.gov/totalenergy/data/monthly/
*   https://www.eia.gov/totalenergy/data/monthly/#environment

Download for data raw:
https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T02.05&freq=m

*   https://www.eia.gov/totalenergy/data/browser/xls.php?tbl=T02.05&freq=m
*   https://www.eia.gov/totalenergy/data/browser/index.php?tbl=T11.05#/?f=M&start=200001
*   https://www.eia.gov/totalenergy/data/browser/index.php?tbl=T11.07#/?f=M

Download for modified data:https://docs.google.com/spreadsheets/d/11AQ37wdHwwgXyAPWROtKe-Dsx9sb8LTaygj88tlAPPM/edit?usp=sharing
Download for modified data:https://drive.google.com/file/d/13OwKKJ8ksppC3Vw6MB49SlvLTTfGhAwk/view?usp=sharing

# Literature Review
“Fossil-fuel combustion by-products are the world’s most significant threat to children’s health and future and are major contributors to global inequality and environmental injustice” (Perera, 2015). As global pollution levels continue to skyrocket, predictions towards the future are crucial to help convince individuals to make better, more eco-conscious choices. We want to expand the available knowledge on the topic of fuel choices and the environmental impacts specific fuels have within the transportation sector. Some key themes we cover within this review include differing ways of gathering CO2 data, and common algorithms available for CO2 predictions.

#Different Ways to Gather CO2 Data

Study 1 - “Machine Learning Application for Automotive Emission Prediction”
The Machine Learning Application for Automotive Emission Prediction, examines different ways to gather CO2 data to make their predictions. They had a unique approach of gathering their data instead of using readily available data they opted to gather data directly for active vehicles using the vehicle sensors, they were able to collect this data using the OBD II scanner tool. They then had an app to upload the data named Torque Pro. This technique is very interesting as they created their own dataset. This provided accurate data as it was directly measured from the source of pollution. 

Study 2 - “Predicting CO2 Emissions from Traffic Vehicles for Sustainable and Smart Environment Using a Deep Learning Model”
Predicting CO2 Emissions from Traffic Vehicles for Sustainable and Smart Environment Using a Deep Learning Model took a different approach to their data collection. They used an official record of CO2 emissions based on features. Which is an open data portal from the Canadian government. The information spans over several years and has over 12 columns by 7385 rows. This is a very common approach since the data is available online and is reliable. 

Study 3 - “Air pollution prediction using machine learning techniques”
Within Air pollution prediction using machine learning techniques they used a very interesting technique for data collection, they used monitoring stations that were specifically placed to ensure coverage and were spaced 3 km apart. This way of approaching predictions is interesting as it provides highly accurate data but comes at a large expense to keep the monitoring stations operational. Overall they had good predictions but this data collection was unfeasible for our project. 

Study 4 - “A new method for prediction of air pollution based on intelligent computation”
In A new method for prediction of air pollution based on intelligent computation, they talk about a combination of previous studies. Within this study, they combine both readily available datasets such as the KDD Cup 2018 in this case, and building pollution concentration capture stations. Using these two data sources they were able to congregate massive amounts of data to aid in their predictions. But with this massive data arose issues, in the article it states, “Despite the advantage of their large size, the limitations of such datasets include the possibility of missing values, that each concentration may show high and low value, and that the records for each station may not be equal”(Al-Janabi, Mohammad, Al-Sultan). Using mass data like this is unfeasible for our project but informs that having mass data as this study has doesn’t always equate to better predictions. 

Overview 
Within the three studies reviewed, we decided to follow Study 2’s methodology as the data is readily available, clean, and from a reputable source. We used data from EIA as they have monthly studies done which provide fuel source consumption by fuel type as well as another data set that calculates CO2 emissions from those fuel types.

#Algorithms Available For Best Predictions

Study 1 - “Artificial Intelligence-based CO2 Emission Predictive Analysis System”
Within the article of Artificial Intelligence-based CO2 Emission Predictive Analysis System, they speak on a variety of machine learning algorithms that may be used to make predictions. One key algorithm that is mentioned is the multivariable linear regression, this algorithm is explained as, “a technique that maps a linear relationship between several influential variables (independent) and one influenced outcome (dependent variable)”(Yeasmin). This is further explained as an extension to linear regression. Within the article, they explain that they use transmission, fuel type, distance, and consumption to make predictions for CO2

Study 2 - ”Air pollution prediction using machine learning techniques An approach to replace existing monitoring stations with virtual monitoring stations” 
In this article, one of the main topics mentioned is Ensemble learning. This is described within the article as, “The ensemble is a particular way to combine different models strategically to solve a particular problem” (Zhang and Ma, 2012). This way of prediction offers a way of eliminating individual weaknesses of algorithms and allows averages of individual algorithms to become the final result. This Is a good way of approaching the problem as it provides reliable results and is stated within the article to be, “By far, the ensemble methods have shown promising results compared to any other ML models” (Zhang and Ma, 2012).

Study 3 - “Forecasting Carbon Dioxide Emissions of Light-Duty Vehicles with Different Machine Learning Algorithms”
In Forecasting Carbon Dioxide Emissions of Light-Duty Vehicles with Different Machine Learning Algorithms, there is mention of gradient boosting, gradient boosting can be used to address certain regression issues. Gradient boosting is described as, “radient boosting adopts an additive form in which, when given a loss function ℒ(𝒴𝑖,𝐹𝑡), iteratively constructs a series of approximations Ft greedily”(Natarajan, Wadhwa, Preethaa and Paul). Gradient boosting can be used to improve learning while handling data from various sources well. 

Study 4 - “Monitoring urban transport air pollution and energy demand in Rawalpindi and Islamabad using leap model”
Within Monitoring urban transport air pollution and energy demand they speak of something called the LEAP model, they go on to elaborate that the LEAP model is, “an energy-planning system developed by the Stockholm Environment Institute, Boston (SEI-B)”(Shabbir, Ahmad). This LEAP model is an interesting idea as it allows individuals to make predictions without having to dive in-depth into machine learning to get results as the model is already created. 

Study 5 - “Air pollution prediction by using an artificial neural network model”
This article explores the possibility of using neural networks to make pollution predictions. more specifically they use an Artificial Neural Network model. Within this article they conclude that there predictions can be used and that ANN’s are a viable option for making predictions when it comes to pollution estimations. However they do say that, “Further research is recommended to compare the efficiency and potency of ANN with numerical, computational, and statistical models”(Maleki)

Overview

All three studies provide great insight on ways to progress with our model. We’d like to further explore ensemble learning as it will give us better results within our dataset. Outliers such as the 2020 pandemic causes severe miss predictions. These outliers cause thep model to believe there is a continuation of less consumption.
