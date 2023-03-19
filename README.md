<img
  src="scantron_A+.jpeg"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 200px">


# Capstone_readiness

# Project Description:
Our youth matters. In Texas, the State of Texas Assessment of Academic Readiness (STAAR) exam is used to measure student learning at the end of the school year. Scores on these exams are used to calculate school accountability ratings which ensures that only high performing schools stay open. We want to use the publically available data to identify key features of schools that have the largest impact on the STAAR exams. After exploration, we use a machine learning algorithm to predict the most likely STAAR exam outcome based on the features we identified as having the largest impact. The scope of this project is limited to Texas High Schools, but may be applied to other types of schools as well.

# Project Goals:
* Identify drivers of STAAR scores
* Build a model to predict STAAR scores
* Deliver results in a final notebook
* Deliver presentation to stakeholders

# Executive Summary:
Through the course of this project, I was able to accurately predict the `time_to_conflict`. There were drivers identified such as the Asia `region` having a significantlly longer `time_to_conflict` and internationalized intrastate conflicts having a significantly shorter `time_to_conflict`. That can be interpreted as the more parties involved the quicker the conflict can escalate. To any potential stakeholders I would recommend using this model as a supplement to predict conflict as situations develop and should not be solely relied upon.

# Initial Hypothesis:
Our initial hypothesis consists of predictions of significant drivers before exploration. We predict schools that have more experienced teachers and teachers with higher education will yield higher STAAR scores. Considering teachers are paid more, in most cases, as teachers possess more years of experience, schools with a higher average salary we predict will also have higher STAAR scores. Similarly the more time a teacher can dedicate to a student, the higher the STAAR score. With this logic we predict that smaller student to teacher ratios are more favorable. For school funding, we think that the more funding a school has the higher the STAAR scores will be because of the amount of resources are able to be allocated to the students.

# Project Plan:

* Acquire the data from "https://rptsvr1.tea.texas.gov/perfreport/tprs/tprs_srch.html"
  * Web scraping was used to acquire the data
  * The data is from Texas Education Agency (TEA).

* Data prep for exploration:
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

* Separate into train, validate, and test datasets
 
* Explore the train data in search of drivers of time_to_conflict
   * Answer the following initial questions
       * Is the average time to conflict for countries in Asia significantlly higher compared to all other regions?
       * Is the average time to conflict for countries in Africa and the Middle East significantlly lower than the average time to conflict for all regions?
       * Is the average time to conflict for countries that have an intrastate conflict over government significantlly greater than the average time to conflict for countries that have an interstate conflict over territory?
       * Is the average time to conflict for countries that have an internationalized intrastate conflict significantly less than the average time to conflict for all conflicts in the dataset?
       
* Prep the data for modeling:
    * Split data into X and y train
    * Scale all numeric data excluding target variables:
        * MinMaxScalar() was used to scale data
      
* Develop a model to predict STAAR scores for `english_1`, `english_2`, `algebra`, `biology`, and `history`
   * Regression models were used to predict STAAR scores
       * Linear Regression
       * Lasso Lars
       * Tweedie Regressor
       * Polynomial Regression
   * Evaluate models on train and validate data
   * Select the best model based on the lowest RMSE and difference between in sample and out of sample data RMSE
   * Test the best model on test data
 
* Draw conclusions

# Data Dictionary:

* For a full glossary of all information provided by the TEA check this website:
    * https://tea.texas.gov/sites/default/files/comprehensive-tprs-glossary-2021.pdf


| Feature | Definition |
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



# Steps to Reproduce
1. Clone this repo
2. Use the function from acquire.py to scrape the data from the TEA website !THIS TAKES A LONG TIME!
3. Use the functions from prepare.py to prepare the data for exploration
4. Run the explore and modeling notebook
5. Run final report notebook


# Citation:
All data acquired from:
* https://rptsvr1.tea.texas.gov/perfreport/tprs/tprs_srch.html
