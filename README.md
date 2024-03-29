


# The STAARs of Texas High Schools
<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-013243.svg?logo=python&logoColor=blue"></a>
<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-2a4d69.svg?logo=numpy&logoColor=grey"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-8DF9C1.svg?logo=matplotlib&logoColor=blue"></a>
<a href="#"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-65A9A8.svg?logo=pandas&logoColor=white"></a>
<a href="#"><img alt="sklearn" src="https://img.shields.io/badge/sklearn-4b86b4.svg?logo=scikitlearn&logoColor=grey"></a>
<a href="#"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1560bd.svg?logo=scipy&logoColor=blue"></a>
<a href="#"><img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-1560bd.svg?logo=xgboost&logoColor=blue"></a>


<img
  src="scantron_A+.jpeg"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 200px">

## Project Description:
Our youth matters. In Texas, the State of Texas Assessment of Academic Readiness (STAAR) exam is used to measure student learning at the end of the school year. Scores on these exams are used to calculate school accountability ratings which ensures that only high performing schools stay open. We want to use the publically available data to identify key features of schools that have the largest impact on the STAAR exams. After exploration, we use a machine learning algorithm to predict the most likely STAAR exam outcome based on the features we identified as having the largest impact. The scope of this project is limited to Texas High Schools, but may be applied to other types of schools as well.

## Project Goals:
* Identify drivers of STAAR passing rates.
* Develop a deeper understanding of how a school's features influence STAAR passing rates.
* Build a model to accurately predict STAAR passing rates.
* Provide insights of which features influence STAAR passing rates.
* Empower schools with the knowledge and tools to improve.


## Executive Summary:
**Insights**
1. STAAR passing rate lower in schools with large proportion of economically disadvantaged students.
2. Schools with a higher number of teachers with 11+ years of experience have a significantly higher passing rate.
3. When comparing schools with above average economically disadvantaged student population, the schools with a higher STAAR passing rate have a lower total expenditure per student. 



## Initial Hypothesis:
Our initial hypothesis consists of predictions of significant drivers before exploration. We predict schools that have more experienced teachers and teachers with higher education will yield higher STAAR passing rates. Considering teachers are paid more, in most cases, as teachers possess more years of experience, schools with a higher average salary we predict will also have higher STAAR scores. Similarly the more time a teacher can dedicate to a student, the higher the STAAR score. With this logic we predict that smaller student to teacher ratios are more favorable. For school funding, we think that the more funding a school has the higher the STAAR scores will be because of the amount of resources are able to be allocated to the students.

## Project Plan:

* **Acquire the data:** "https://rptsvr1.tea.texas.gov/perfreport/tprs/tprs_srch.html"
  * Web scraping was used to acquire the data
  * The data is from Texas Education Agency (TEA).

* **Data prep for exploration:**
    * Schools that had special characters were removed from analysis
        * special characters (`*`, `-`, `?`, n/a)
    * Nulls were removed:
        * Nulls or reserved information was incoded into the special characters above and removed
    * All the percent signs, dollar signs, and commas were removed from values
    * Columns were combined into desired features
        * `high_edu` was generated from combining percent of teachers that have a masters or doctorate
        * Features for `teacher_exp_0to5`, `teacher_exp_6to10`, and `teacher_exp_11_plus` were generated from combining:
            * Beginning teachers and teachers with 1-5 years of experience into `teacher_exp_0to5`
            * Teachers with 11+ years of experience were combined into `teacher_exp_11_plus`
            * `teacher_exp_6to10` stayed the same. Teachers of 6-10 years of experience
    * There were an initial 1571 rows
        * The total number of rows after preperation and cleaning is 1391

* **Separate into train, validate, and test datasets**
 
* **Explore data to develop an understanding of what features affect a school's STAAR passing rate.**
   * Initial questions:
       * Do schools with more economically disadvantaged students have a lower average percent passing rate for STAAR exams?
       * Do schools with teachers that have more years of experience have a better average STAAR score passing rates?
       * Of schools with a large proportion of economically disadvantaged students, do the schools with higher average STAAR scores have higher expenses per student?
       * Do schools with above average economically disadvantaged students have significantly higher total expendature per student?
       * Is there a statisticaly significant correlation between the amount of extracurricular expendatures and algebra passing rates?
       * Is there a statisticaly significant correlation between student teacher ratio and all subjects passing rates?
       
* **Prep the data for modeling:**
    * Split data into X and y train
    * Scale all numeric data excluding target variables:
        * MinMaxScalar() was used to scale data
      
* **Develop a model to predict STAAR scores for `english_1`, `english_2`, `algebra`, `biology`, and `history`**
   * Regression models were used to predict STAAR scores
       * Linear Regression
       * Lasso Lars
       * Tweedie Regressor
       * Polynomial Regression
   * Evaluate models on train and validate data**
   * Select the best model based on the lowest RMSE and difference between in sample and out of sample data RMSE
   * Test the best model on test data
 
* **Draw conclusions**

## Data Dictionary:

* For a full glossary of all information provided by the TEA check this website:
    * https://tea.texas.gov/sites/default/files/comprehensive-tprs-glossary-2021.pdf


| **Feature** | **Definition** |
|:--------|:-----------|
|school_id| The id number of the school from TEA|
|english_1| English I, percent of students at approaches grade level or above for English I|
|english_2| English II, percent of students at approaches grade level or above for English II|
|algebra| Algebra, percent of students at approaches grade level or above for Algebra|
|biology| Biology, percent of students at approaches grade level or above for Biology|
|history| U.S. History, percent of students at approaches grade level or above for U.S. History|
|bilingual_or_english_learner| EB/EL Current and Monitored, percent of students in the dual-language program that enables emergent bilingual (EB) students/English learners (ELs) to become proficient in listening, speaking, reading, and writing in the English language through the development of literacy and academic skills in the primary language and English.|
|teacher_exp_0to5| Integer, number of teachers with 0-5 years of experience|
|teacher_exp_6to10| Integer, number of teachers with 6-10 numbers of experience|
|teacher_exp_11_plus| Integer, number of teachers with 11 or more years of experience|
|extracurricular_expend| The amount of funds (in dollars) spent on extracurriculuars per student|
|total_expend| The average total amount of funds (in dollars) spent per student|
|econdis| students that are from homes that are below the poverty line
|salary| Average Actual Salary, Average amount teachers are being paid in dollars|
|high_edu| Percent of teachers with a masters or doctorate degree|
|ratio| Count of the number of students per one teacher|



## Steps to Reproduce
1. Clone this repo
2. Use the function from acquire.py to scrape the data from the TEA website 
    * May take a few hours to web scrape.
3. Use the functions from prepare.py to prepare the data for exploration
4. Run the explore and modeling notebook
5. Run final report notebook


## Citation:
All data acquired from:
* https://rptsvr1.tea.texas.gov/perfreport/tprs/tprs_srch.html

Link to the final presentation slide deck:
* https://www.canva.com/design/DAFdeHe75pM/qSUa86WQ0Gtg3R7mET4Lyg/view?utm_content=DAFdeHe75pM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

Link to the Tableau dashboard:
* https://public.tableau.com/app/profile/andy.jensen8392/viz/CapstoneDashboard_16795800870340/Dashboard1?publish=yes
