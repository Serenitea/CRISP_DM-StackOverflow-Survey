
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

For this project, I was interestested in using Stack Overflow data from 2019 to better understand:

1. Which are currently the most commonly used programming languages?
2. How much does programming language affect income?
3. How has the prevalence of programming language usage changed over the last 5 years?
4. How do different types of developers differ in their choice of language?

## File Descriptions <a name="files"></a>

I compiled all of my steps into one notebook along with an HTML version.  Markdown cells were used to assist in walking through the thought process for individual steps. Almost all functions used in the notebook are stored in the 'data_preprocessing.py' and 'visuals.py' files.

Data folder - most of the original survey data and processed data are saved here. There are three files that were too large to be uploaded: the 2018 and 2019 survey results, and the preprocessed 2019 survey results. The survey results are publicly available at the links below.

Images folder - all data visualizations created are stored in this folder.

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@themaryzhou/who-uses-which-language-deriving-insights-from-the-stack-overflow-developer-survey-2019-cbc8718b948e).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credits to Stack Overflow for the data.  You can find the Licensing for the data and other descriptive information at the Stack Overflow website [here](https://insights.stackoverflow.com/survey).  Otherwise, feel free to use the code here as you would like!
