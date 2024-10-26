# MTA-Fare-Evasion-Project
For this project, please install the necessary packages with the command:
pip install pandas matplotlib seaborn numpy pingouin scikit-learn
 
This project analyzes two data sets regarding bus fare evasion and subway fare evasion between 2019-Q1 and 2024-Q2. 
 
To start, I read and stored the data for both bus and subway fare evasions. I cleaned the data by removing the points from 2018 in the subway data set, since I wanted to compare it with the bus data based on time period, and the bus data started in 2019. I also separated the bus data based on Trip Type (Local, Express, SBS) to analyze each one separately.
 
Then, I moved on to exploratory data analysis, creating line graphs and bar graphs for the subway and bus data, with Time Period (Year-Quarter) as the independent variable and Fare Evasion (%) as the dependent variable. The graphs show that express bus fare evasions remain consistently low, whereas fare evasions for local and SBS buses have been steadily increasing since 2020-Q2 and 2020-Q3 (the height of the COVID-19 pandemic). Moreover, the graph for subway fare evasions shows that there was a sharp increase in fare evasions after the COVID-19 pandemic, and fare evasions have been roughly consistent since then. The line graph combining both subway and bus fare evasions places the subway data in context with the bus data, showing that subway fare evasions, along with express bus fare evasions, appear to remain relatively consistent while local and SBS bus fare evasions increase steadily.
 
After this, I moved onto statistical analysis. I created a new data frame to analyze bus and subway data together, and I ran linear regressions, with subway fare evasions as the independent variable and bus fare evasions as the dependent variable. I first conducted tests to find the correlation between subway fare evasions and each type of bus fare evasion. The correlation between subway and local bus fare evasions is r = 0.7175, showing that there is a moderately strong, positive correlation. The correlation between subway and express bus fare evasions is r = -0.3475, showing that there is a fairly weak, negative correlation. The correlation between subway and SBS bus fare evasions is r = 0.7867, showing that there is also a moderately strong, positive correlation.
 
After this, I conducted an ANOVA with a significance level of 0.05. I found that for the relationship between subway and local buses, p = 0.1654. For the relationship between subway and express buses, p = 0.884. And, for the relationship between subway and SBS buses, p = 0.529. So, nothing was statistically significant enough to conclude a relationship.

Finally, I decided to make a predictive model for Local bus fare evasion based on subway fare evasion since it had relatively significant results. I developed a Decision Tree, and then a Random Forest, from which I picked an individual tree.



Links to Data:
Bus Fare Evasion: data.ny.gov/Transportation/MTA-Bus-Fare-Evasion-Beginning-2019/uv5h-dfhp/about_data
Subway Fare Evasion: data.ny.gov/Transportation/MTA-NYCT-Subway-Fare-Evasion-Beginning-2018/6kj3-ijvb/about_data
