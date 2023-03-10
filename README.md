# Capstone_readiness

# Project Description:
Our youth matters. In Texas we want to examine how well high school students are doing and what they are doing with their future. We will explore public high schools from around Texas and build a model to predict college, military, and career readiness.

# Project Goals:
* point
* point
* point
* point

# Executive Summary:
Through the course of this project, I was able to accurately predict the `time_to_conflict`. There were drivers identified such as the Asia `region` having a significantlly longer `time_to_conflict` and internationalized intrastate conflicts having a significantly shorter `time_to_conflict`. That can be interpreted as the more parties involved the quicker the conflict can escalate. To any potential stakeholders I would recommend using this model as a supplement to predict conflict as situations develop and should not be solely relied upon.

# Initial Hypothesis:
My first thoughts after looking at the dataset made me think that `region` and `type_of_conflict` were going to be the most significant drivers.

# Project Plan:

* Acquire the data from "https://ucdp.uu.se/downloads/index.html#armedconflict"
  * "UCDP/PRIO Armed Conflict Dataset version 22.1"
  * From Uppsala University in Sweden. Citation is at the bottom of the ReadMe.

* Data prep for exploration:
    * Unnecessary columns were dropped
        * total columns were reduced from 28 to 12 (11 original columns and 1 engineered column)
    * Target variable was engineered from the `start_date` and `start_date2` columns
        * This returned a value (in days) for how long from the beginning of the conflict it took for the conflict to accumulate at least 25 casualties
    * Nulls were valuable:
        * Nulls were in `side_a_2nd`, `side_b_2nd`, and `territory_name`
            * `side_a_2nd` and `side_b_2nd` were encoded to a `0` or `1` depending if they had an ally or not
            * `territory_name` nulls were changed to "government" since null values were indicating there was no territorial conflict and that it was a govermental conflict
    * There were an initial 2568 rows
        * The total number of rows was reduced to 294 because there were entries for each "episode" of the war (often on a yearly basis) and I just use the initial entry to predict the `time_to_conflict`

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
|ccmr| College, Career, or Military Ready (Annual Graduates)|
|eng1| End of English Course 1, percent of students at approaches grade level or above for English 1|
|eng2| End of English Course 2, percent of students at approaches grade level or above for English 2|
|algebra| Algebra, percent of students at approaches grade level or above for Algebra|
|biology| End of Biology, percent of students at approaches grade level or above for Biology|
|history| End of U.S. History, percent of students at approaches grade level or above for U.S. History|
|ebel| EB/EL Current and Monitored, percent of students in the dual-language program that enables emergent bilingual (EB) students/English learners (ELs) to become proficient in listening, speaking, reading, and writing in the English language through the development of literacy and academic skills in the primary language and English.|
|econdis| students that are from homes that are below the poverty line
|salary| Average Actual Salary, Average amount teachers are being paid in dollars|
|high_edu| Percent of teachers with a masters or doctorate degree|
|ratio| Count of the number of students per one teacher|
|attendance| Percent attendance for the school (annually)|



# Steps to Reproduce
1. Clone this repo
2. Use the function from prepare to prepare and obtain the data from the website
3. Run the explore and modeling notebook
4. Run final report notebook


# Citation:
All data acquired from:
* https://rptsvr1.tea.texas.gov/perfreport/tprs/tprs_srch.html
