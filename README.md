# EnviroScan

EnviroScan: AI-Powered Pollution Source Identifier using Geospatial Analytics

---

Project Overview

Air pollution monitoring systems generally measure pollutant levels but do not identify the specific sources of pollution. This limitation makes it difficult for authorities and urban planners to implement targeted mitigation strategies.

The EnviroScan system uses machine learning, weather analytics, and geospatial data to identify the most likely source of pollution such as vehicular emissions, industrial activity, agricultural burning, waste burning, or natural causes.

The system integrates multiple data sources including:

- Air quality monitoring data
- Weather information
- Geospatial infrastructure features

Using this integrated dataset, the system predicts pollution sources and enables accurate pollution source prediction, supporting decision-making for environmental monitoring and urban planning.

---

Milestone 1 (Week 1–2)

This milestone includes:

- Module 1 – Data Collection
- Module 2 – Data Cleaning and Feature Engineering

---

Module 1: Data Collection from APIs and Location Databases

Objective:

The objective of this module is to collect air quality data, weather data, and geospatial environmental features from multiple sources. These datasets form the base for pollution source identification.

The following pollutant measurements are collected:
PM2.5, PM10, NO₂, SO₂, CO, O₃.
These are widely used indicators of urban air pollution.

---

1. Weather Data Collection

Weather information is collected using the OpenWeatherMap API.

The following weather parameters are retrieved:

- Temperature
- Humidity
- Wind Speed

Weather conditions significantly influence pollutant dispersion.

---

2. Geospatial Data Collection

Environmental context is extracted using OSMnx, which retrieves geospatial data from OpenStreetMap.

The following spatial features are collected:

- Road Networks
- Industrial Zones
- Agricultural Fields
- Waste Disposal Sites

These spatial features provide environmental context for identifying pollution sources.

---

3. Metadata Tagging

Each collected data point is tagged with the following metadata:

- Latitude
- Longitude
- Timestamp
- Monitoring station name

This ensures the dataset contains both spatial and temporal context.

---

4. Data Storage

The collected data is stored in a structured dataset:
enviro_scan_dataset.csv

This dataset is later used for preprocessing and machine learning.

---

Module 2: Data Cleaning and Feature Engineering

Objective:

The objective of this module is to preprocess the collected data and generate meaningful features for machine learning models.

---

1. Duplicate Removal

Duplicate records are removed to ensure dataset integrity.

Example:
"df.drop_duplicates()"

---

2. Handling Missing Values

API responses may sometimes contain missing values.

To address this:

- Missing pollutant values are replaced with realistic simulated values
- Missing values are handled using zero-value filling and noise-based simulation

This ensures dataset completeness.

---

3. Standardization of Timestamps and Data

All timestamps are converted into a standardized datetime format.

Time is represented in Indian Standard Time (IST).

Temporal features derived:

- Hour of Day
- Day of Week

These features help capture time-based pollution patterns.

---

4. Spatial Feature Engineering

Spatial proximity features are extracted using OpenStreetMap data through OSMnx.

Generated spatial attributes:

- road_count
- industry_count
- farmland_count
- dump_count

These features represent environmental infrastructure around monitoring locations.

---

5. Dataset Integration

All datasets are merged into a single feature-rich DataFrame containing:

- Pollution measurements
- Weather conditions
- Spatial features
- Temporal features
---

Milestone 2 (Week 3–4)

Milestone 2 focuses on pollution source labeling and machine learning model development.

Modules included:

- Module 3 – Source Labeling and Simulation
- Module 4 – Model Training and Source Prediction

---

Module 3: Source Labeling and Simulation

Objective:

This module assigns pollution source labels using rule-based heuristics derived from environmental knowledge.

Since real pollution source labels are not directly available, heuristic rules are used to simulate labeled training data.

---

Pollution Source Labeling Rules

i. Vehicular Pollution

Condition: High road density and high NO₂ concentration
Explanation: Vehicles emit nitrogen dioxide during fuel combustion.

ii. Industrial Pollution

Condition: Industrial zones nearby and high SO₂ concentration
Explanation: Industrial processes emit sulfur dioxide.

iii. Agricultural Pollution

Condition: Farmland nearby and high particulate matter
Explanation: Crop residue burning produces particulate emissions.

iv. Waste Burning

Condition: High PM2.5 levels (simulated burning conditions)
Explanation: Burning waste generates particulate pollution.

v. Natural Pollution

If none of the above conditions are satisfied, pollution is classified as natural.

---

Dataset Preparation

The labeled dataset is prepared and saved as:
enviro_scan_dataset.csv

This dataset serves as the training dataset for machine learning models.

---
Module 4: Model Training and Source Prediction

Objective:

The objective of this module is to train machine learning models capable of predicting pollution sources using environmental, spatial, and temporal features derived from the dataset.

---

1. Train-Test Split

The labeled dataset is divided into training and testing sets to evaluate model performance on unseen data.

- Training Data: 75%
- Testing Data: 25%

This ensures proper validation of the model’s predictive capability.

---

2. Feature Scaling

Feature scaling is applied using StandardScaler to normalize input features.

This ensures that all features contribute equally to the model and prevents bias due to differences in value ranges.

---

3. Machine Learning Models Used

The following classification models are implemented:

- Decision Tree : 
  A rule-based model that classifies data using hierarchical decision structures.

- Random Forest : 
  An ensemble learning algorithm that combines multiple decision trees to improve accuracy and reduce overfitting.

- XGBoost : 
  A gradient boosting algorithm known for high performance, efficiency, and scalability.

- Cross Validation : 
  To ensure model reliability and robustness, 5-Fold Cross Validation is applied.
---

4. Hyperparameter Tuning

Hyperparameter tuning is performed through controlled parameter selection and model comparison.

The following parameters are adjusted:

- Number of estimators (n_estimators)
- Maximum depth (max_depth)
- Learning rate (for XGBoost)

This helps improve model performance and ensures better generalization.

---

5. Model Evaluation Metrics

Model performance is evaluated using the following metrics:

- Accuracy → Measures overall correctness of predictions
- Precision → Measures correctness of positive predictions
- Recall → Measures ability to detect all relevant cases
- F1 Score → Harmonic mean of precision and recall
- Confusion Matrix → Provides detailed classification performance

These metrics help assess how well the model predicts pollution sources.

---

6. Model Selection

All models are compared based on:

- Accuracy scores
- Cross-validation performance

The best-performing model is selected as the final model for deployment.

---

7. Model Export

The final trained model is saved using joblib:

joblib.dump(final_model, "rf_model.pkl")

Additional files saved:

- scaler.pkl → used for feature scaling
- label_encoder.pkl → used for label encoding

---

Output:

- Trained Model: rf_model.pkl
- Scaler: scaler.pkl
- Label Encoder: label_encoder.pkl

These components are integrated into the Streamlit dashboard for real-time pollution source prediction.

---
Milestone 3 (Week 5–6)

Modules included:

- Module 5 – Geospatial Mapping and Heatmap Visualization
- Module 6 – Interactive Dashboard and Real-Time Monitoring

---

Module 5: Geospatial Mapping and Heatmap Visualization

Objective:

To visualize pollution data and predicted sources using geospatial mapping techniques, enabling identification of pollution hotspots and spatial distribution patterns.

---

1. Geospatial Map Integration

An interactive map is created using the Folium library.

The map displays pollution data across different locations using latitude and longitude coordinates.

The map is dynamically centered based on the available dataset.

---

2. Heatmap Visualization

Heatmaps represent pollution intensity using color gradients (blue → yellow → red),
where higher intensity indicates higher pollution levels.

---

3. Marker-Based Visualization

Location points are displayed using clustered markers.

Each marker includes:

- Tooltip (on hover) → shows PM2.5 value
- Popup (on click) → displays pollutant values:
  PM2.5, PM10, NO₂, SO₂, CO, O₃

---

4. Filtering Capability

The dashboard allows filtering based on:

- Location (city)
- Pollution source category

This enables interactive exploration of pollution data.

---

5. High-Risk Zone Identification

Locations with higher PM2.5 values are highlighted using color-coded markers, helping identify critical pollution zones.

---

6. Map Embedding

The map is embedded in the Streamlit dashboard using:
"st_folium()"

This allows real-time user interaction with the map.

---

Module 6: Interactive Dashboard and Real-Time Monitoring

Objective:

To develop an interactive web-based dashboard for real-time pollution monitoring, visualization, and AI-based source detection.

---

1. Dashboard Development

The dashboard is built using Streamlit.

It provides a user-friendly interface with multiple sections:

- Dashboard
- Source Detection
- Health Audit
- Dataset Explorer

---

2. Real-Time Data Integration

Air Quality Data (OpenAQ API):
Real-time pollutant data is fetched including PM2.5, PM10, NO₂, SO₂, CO, O₃.

Weather Data (OpenWeather API):
Real-time weather parameters include temperature, humidity, and wind speed.

If OpenAQ data is unavailable, the system automatically falls back to OpenWeather API, ensuring continuous real-time data availability.

---

3. Source Detection

Users can input latitude and longitude to analyze pollution at any location.

The system uses a trained machine learning model (Random Forest / XGBoost selected during training) to predict the pollution source.

---

4. Real-Time Metrics Display

The dashboard displays environmental metrics in real time:

- Pollutant concentrations
- Weather conditions
- Area features (road, industry, farmland counts)

These are displayed using structured metric cards for better readability.

---

5. Pollution Insights Visualization

Pie Chart (Source Distribution):
Shows contribution of different pollution sources based on PM2.5 values.

---

6. Alert System

The system provides real-time alerts:

- High pollution alert is triggered when PM2.5 exceeds 130
- Otherwise, the system displays safe air conditions

---

7. Dataset Explorer

Users can view the complete dataset in tabular format.
The dataset can be downloaded as a CSV file for further analysis.

---

8. Health Audit Module

This module identifies unsafe pollution conditions by:

- Displaying records where PM2.5 exceeds safe limits
- Highlighting pollution violations
- Providing system safety status

---

9. User Interaction Features

- Dropdown filters (city, source type)
- Real-time analysis input (latitude, longitude)
- Interactive map navigation

---

10. System Integration

The dashboard integrates:

- Dataset: enviro_scan_dataset.csv
- Real-time APIs: OpenAQ and OpenWeather

---



