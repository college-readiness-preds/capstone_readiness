# Capstone_readiness

# Project Description:
Our youth matters. In Texas we want to examine how well high school students are doing and what they are doing with their future. We will explore public high schools from around Texas and build a model to predict college, military, and career readiness.

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
        * Features for `ex_5`, `ex_10`, and `ex_plus` were generated from combining:
            * Beginning teachers and teachers with 1-5 years of experience into `ex_5`
            * Teachers with 11+ years of experience were combined into `ex_plus`
            * `ex_10` stayed the same. Teachers of 6-10 years of experience
    * There were an initial 1571 rows
        * The total number of rows after preperation and cleaning is XXXX

* Separate into train, validate, and test datasets
 
* Explore the train data in search of drivers of time_to_conflict
   * Answer the following initial questions
       * Is the average time to conflict for countries in Asia significantlly higher compared to all other regions?
       * Is the average time to conflict for countries in Africa and the Middle East significantlly lower than the average time to conflict for all regions?
       * Is the average time to conflict for countries that have an intrastate conflict over government significantlly greater than the average time to conflict for countries that have an interstate conflict over territory?
       * Is the average time to conflict for countries that have an internationalized intrastate conflict significantly less than the average time to conflict for all conflicts in the dataset?
       
* Prep the data for modeling:
    * encode columns to reduce the number of catagories:
        * `location` intiger 0-10 for top ten locations and other
        * `side_a` intiger 0-10 for top ten side_a's and other
        * `side_b` intiger 0-20 for top twenty side_b's and other
        * `start_date` intiger 0 or 1, 0 if before 2000 and 1 if after 2000
        * `time_to_conflict` 1= less than or equal to 30 days, 2= between 30 days and 1 year, 3= longer than a year
    * Dummies were encoded for:
        * `location`, `side_a`, `side_b`, `start_date`, `type_of_conflict`, `region`, `time_to_conflict`, and `incompatibility`
    * Dropped columns:
        * `territory_name` and `start_date2`
      
* Develop a model to predict the `time_to_conflict`
   * Use drivers identified in explore to build predictive models
       * Decision Tree
       * KNN
       * Random Forest
       * Linear Regression
   * Evaluate models on train and validate data
   * Select the best model based on highest accuracy and difference between in sample and out of sample data
   * Test the best model on test data
 
* Draw conclusions

# Data Dictionary:

* For a full glossary of all information provided by the TEA check this website:
    * https://tea.texas.gov/sites/default/files/comprehensive-tprs-glossary-2021.pdf


| Feature | Definition |
|:--------|:-----------|
|school_id| The id number of the school from TEA|
|eng1| English I, percent of students at approaches grade level or above for English I|
|eng2| English II, percent of students at approaches grade level or above for English II|
|algebra| Algebra, percent of students at approaches grade level or above for Algebra|
|biology| Biology, percent of students at approaches grade level or above for Biology|
|history| U.S. History, percent of students at approaches grade level or above for U.S. History|
|ebel| EB/EL Current and Monitored, percent of students in the dual-language program that enables emergent bilingual (EB) students/English learners (ELs) to become proficient in listening, speaking, reading, and writing in the English language through the development of literacy and academic skills in the primary language and English.|
|ex_5| Integer, number of teachers with 0-5 years of experience|
|ex_10| Integer, number of teachers with 6-10 numbers of experience|
|ex_plus| Integer, number of teachers with 11 or more years of experience|
|extra| The amount of funds (in dollars) spent on extracurriculuars per student|
|all_fund| The total amount of funds (in dollars) spent per student|
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
