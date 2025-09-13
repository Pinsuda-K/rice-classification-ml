#### rice-classification-ml
# Rice Variety Classification (Cammeo vs Osmancik)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Made with](https://img.shields.io/badge/Made%20with-Python%20%7C%20ScikitLearn-blue)
![Data](https://img.shields.io/badge/Data-Public%20(Kaggle)-orange)
![Language](https://img.shields.io/badge/English-lightgrey)

> Classifying Cammeo and Osmancik rice varieties using 7 morphological traits and machine learning models. This project explores both a baseline (Logistic Regression) and an ensemble method (Random Forest), comparing their performance and visualizing key insights. Originally explored in a group project, I later revisited and rebuilt the analysis independently, extended it with additional evaluation (10-run reliability check).

### Why I Built This
- Practice **end-to-end ML workflow** (EDA → training → evaluation → visualization)  
- Extend the work with **extra evaluation** (10-run statistics, logistic regression baseline)
  
This project gave me the opportunity to:
- Explore **7 morphological traits** of rice grains (Area, Perimeter, Axis Lengths, etc.)  
- Apply **Random Forest** and compare it with a **Logistic Regression baseline**  
- Practice **visual analysis** (scatter plots, boxplots, feature importance)  
- Repeated runs (with different seeds) improve **reliability of results**

---
### Dataset
- Source: Kaggle–Rice (Cammeo and Osmancik) 👉 https://www.kaggle.com/datasets/muratkokludataset/rice-dataset-commeo-and-osmancik
- Download: https://www.kaggle.com/api/v1/datasets/download/muratkokludataset/rice-dataset-commeo-and-osmancik
- Size: 3,810 rice grain samples
- Features (7):
  1. Area
  2. Perimeter
  3. Major Axis Length
  4. Minor Axis Length
  5. Eccentricity
  6. Convex Area
  7. Extent
- Target: Rice variety (Cammeo or Osmancik)
---

### Exploratory Data Analysis (EDA)
Class Distribution
Balanced Dataset wwith a slight dominance of Osmancik
![Rice Distribution](./images/pie_rice.png)
Feature Distributions
Boxplots reveal distinct separation in traits like Area and Axis Lengths
![Feature Distributions Boxplots](./images/boxplots_rice_looped.png)

Scatter Plots
Clear separability between varieties across multiple features
![Area vs Perimeter](./images/scatter_Area_vs_Perimeter.png)
![Area vs Major Axis Length](./images/scatter_Area_vs_Major_Axis_Length.png)
![Area vs Minor Axis Length](./images/scatter_Area_vs_Minor_Axis_Length.png)
![Area vs Convex Area](./images/scatter_Area_vs_Convex_Area.png)
![Area vs Extent](./images/scatter_Area_vs_Extent.png)
![Area vs Eccentricity](./images/scatter_Area_vs_Eccentricity.png)
---

### Methods
Tr





