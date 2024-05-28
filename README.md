   # Dataset

The dataset used in this project comprises data from two primary sources:

1. **benign_website**: Extracted from the Majestic Million websites dataset.
2. **phishing_websites**: Obtained from PhishTank, a collaborative platform for sharing information about phishing URLs(20-03-2024 to 22-03-2024).

These datasets were merged and filtered to create the **merged_filtered_urls** dataset, which contains both benign and confirmed phishing websites.

The dataset used in this project consists of two types:

1. **Balanced Dataset**:
   - This dataset contains an equal ratio of benign(1000) and phishing websites(1000). Each class is represented in equal proportions, ensuring a balanced distribution of data.

2. **Unbalanced Dataset**:
   - The unbalanced dataset comprises 80% benign websites(1600) and 20% phishing websites(400) out of a total of 2000 websites used for the experiments. This imbalance reflects real-world scenarios where benign websites significantly outnumber phishing websites.

Both types of datasets were utilized to assess the performance of the models under different data distributions and to evaluate their effectiveness in handling imbalanced data.


# Feature Extraction

The `feature_train` file contains the extracted features of the URLs from the merged dataset. These features are essential for training machine learning models and analyzing website characteristics.

# Binary Feature Representation

The `binary_feature_train` file converts the extracted features into a binary format, facilitating efficient processing and analysis for machine learning tasks.

# Code

## 1. url_set_creation

The `url_set_creation.ipynb` script serves the purpose of creating a dataset of URLs by performing the following steps:

1. **Online Check**: The script verifies whether the website associated with each URL is online or not. If the website is unreachable or if its HTML content contains potentially harmful words, indicating a phishing attempt, the script marks the website as offline.

2. **Viewport Analysis**: It examines the HTML content of each website to determine if it contains a viewport tag indicative of mobile compatibility. Only websites with a viewport tag are retained for further processing, while others are discarded.

3. **Threaded Processing**: To enhance processing speed, the script employs threading for concurrently checking the status and characteristics of multiple URLs.

These steps ensure the creation of a dataset comprising URLs that are both online and potentially safe for further analysis or use.

## 2. feature_extraction

The `feature_extraction.ipynb` script is responsible for extracting features from URLs. It follows these key steps:

1. **Initial Feature Extraction**: Initially, over 40 features are extracted from each URL, encompassing various aspects such as domain appearances, presence of sensitive words, number of images and links, existence of forms, and more.

2. **Exploratory Data Analysis (EDA)**: Exploratory Data Analysis (EDA) is performed to identify important features. This involves creating Empirical Cumulative Distribution Function (ECDF) plots to visualize the distribution of feature values and discern significant patterns.

The following features were determined to be particularly influential based on the analysis of ECDF plots:

- `domain_appearances`: Number of times the domain appears in the URL.
- `title_contains_domain`: Whether the title contains the domain.
- `link_ratio`: Ratio of external links to internal links.
- `has_forms`: Presence of forms on the webpage.
- `form_landing`: Whether the form is located on the landing page.
- `num_images`: Number of images on the webpage.
- `num_links`: Number of links on the webpage.
- `num_iframes`: Number of iframes on the webpage.
- `num_subdomain_level`: Number of subdomains in the URL.
- `has_sensitive_word`: Presence of sensitive words in the webpage content.
- `num_terms_in_title`: Number of terms in the title of the webpage.
- `max_dom_depth`: Maximum depth of the DOM tree.
- `dom_node_count`: Count of DOM nodes on the webpage.
- `status`: Status of the website (online/offline).

By analyzing ECDF plots and focusing on these important features, the script ensures effective feature selection for subsequent analysis and modeling.

## 3. binary_threshold_feature

The `binary_threshold_feature.ipynb` script is responsible for converting extracted features into binary features by applying a binary threshold. This process involves transforming continuous or categorical features into binary values based on a specified threshold.

### Key Steps:

1. **Feature Transformation**: Initially, features extracted from URLs are in their original form, containing continuous or categorical values.

2. **Binary Thresholding**: The script applies a binary threshold to each feature, converting it into a binary representation. This thresholding process categorizes feature values as either 0 or 1 based on predefined criteria or statistical measures.

3. **Binary Feature Representation**: After thresholding, each feature is represented as a binary value, facilitating simpler and more efficient processing for subsequent analysis or modeling tasks.

By converting features into binary representations, the script enhances the interpretability and usability of the dataset, enabling clearer insights and improved performance in downstream tasks.

## phishing_url_detection

The `phishing_url_detection.ipynb` script is designed for detecting phishing URLs using various machine learning (ML) and deep learning (DL) models. It takes binary features extracted from URLs as input and evaluates the performance of different models based on metrics such as accuracy, precision, and true positive rate (TPR).

### ML Models:

1. **K-Nearest Neighbors (KNN)**:
   - Depth: 1 (resulting in maximum accuracy)

2. **Gradient Boosting**:
   - Parameters: max_depth=4, learning_rate=0.7

3. **Random Forest**:
   - Parameters: n_estimators=10

4. **Decision Trees**:
   - Parameters: max_depth=30

5. **Logistic Regression**:
   - Simple logistic regression

6. **CatBoost Classifier**:
   - Parameters: Learning rate = 0.1

7. **Support Vector Machine (SVM)**:
   - Parameters: gamma=0.1, kernel=['rbf', 'linear']

### DL Models:

1. **Multilayer Perceptron (MLP)**:
   - Number of layers: 3
   - Number of nodes per layer: 50
   - Activation functions: ReLU and sigmoid

2. **Recurrent Neural Network (RNN)**:
   - Number of cells: 64
   - Activation function: sigmoid
  
### Cross-Validation Setup
- Implement 10-fold cross-validation to evaluate the model's performance more reliably. Use the `cross_val_score` function from scikit-learn to perform cross-validation.

### Model Training
- Train the chosen machine learning model on the training dataset using the `fit` method.

### Model Evaluation
- Evaluate the trained model's performance using metrics such as accuracy, precision, tpr, or fpr. Calculate the average score across all folds of the cross-validation.

### Testing
- Test the trained model on a separate holdout dataset to assess its generalization performance. Calculate relevant evaluation metrics to ensure the model performs well on unseen data.


By evaluating these models with different configurations, the script aims to identify the most effective approach for phishing URL detection, providing insights into the performance and suitability of various ML and DL techniques for this task.


# Order of Running Files

To ensure the successful execution of the project, follow this recommended order of running the provided scripts:

1. **url_set_creation.ipynb**:
   - This script creates a dataset of URLs by checking website availability, identifying potentially harmful content, and filtering based on viewport tags.

2. **feature_extraction.ipynb**:
   - After creating the URL dataset, this script extracts features from the URLs, including domain appearances, presence of sensitive words, number of images and links, and more.

3. **binary_threshold_feature.ipynb**:
   - Once features are extracted, this script converts them into binary features using a specified threshold, enhancing the dataset for subsequent analysis.

4. **phishing_url_detection.ipynb**:
   - Finally, with the prepared dataset and binary features, this script implements various machine learning and deep learning models to detect phishing URLs and evaluates their performance using metrics like accuracy, precision, and true positive rate.

## Computing Environment Specifications

The code for this project was executed on Google Colab, utilizing the following computing environment specifications:

- **RAM Size**: 12.67 GB
- **Disk Size**: 107.7 GB

Running the code on Google Colab provided ample resources in terms of RAM and disk space, enabling efficient execution and analysis of the project.


