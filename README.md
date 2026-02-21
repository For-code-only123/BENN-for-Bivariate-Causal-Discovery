# A-Unified-BENN-Framework-for-Bivariate-Causal-Discovery-with-Continuous-Discrete-and-Mixed-Variables

## 🚀 Running Real-World Experiments (Bivariate Pairwise Evaluation)

To reproduce our experiments on real-world datasets, please refer to the following instructions regarding data preprocessing, dataset-specific rules, and environment configuration. The main execution script is `real-experiments.py`.

### 1. Data Preprocessing & Type Identification

Before feeding data into the causal inference framework, the script automatically performs the following preprocessing steps:

* **Missing Value Handling:** Automatically replaces `?` characters with `NaN` and drops any rows containing missing values (`dropna`).
* **Automatic Variable Type Identification:**
* **Discrete Variables:** If a column's data type is `object` or `category`, or if the number of **unique values is less than or equal to 10** (`nunique() <= 10`), it is classified as a discrete variable. All discrete variables are transformed into numerical labels using `LabelEncoder`.
* **Continuous Variables:** Variables that do not meet the discrete criteria are classified as continuous and are uniformly converted to `float` types.


* **HSIC Independence Test:** Before inferring the causal direction, the code iterates through all variable pairs and applies the Hilbert-Schmidt Independence Criterion (HSIC) test using either RBF or Delta kernels. Only pairs that demonstrate statistically significant correlation () are passed to the BENN models for inference.

### 2. Dataset-Specific Rules

To ensure consistency with the original paper and benchmarks, we applied a hard-coded filtering rule for a specific dataset:

* **Algerian Forest Fires:** When processing the `algerian-forest-fires` dataset, the script automatically drops the `day` and `year` columns to prevent these time-based, non-endogenous features from interfering with causal inference.

### 3. Handling Mixed Data Pairs

For "mixed data pairs" consisting of one continuous variable and one discrete variable, the script reorganizes them before feeding them into the `BENN_Mixed` framework, regardless of their original order in the dataframe.

* **Fixed Ordering:** The model will always receive the inputs in a strict `(Continuous, Discrete)` order (i.e., `X_type="cont", Y_type="discrete"`). This ensures tensor dimension consistency and computational stability within the underlying mixed-network architecture.

### 4. Path Configuration Guide

Before running `real-experiments.py`, you **must** open the script, search for `\path`, and update them to your local absolute or relative paths. There are **5 paths** that need to be configured:

* **BENN Framework Core Files (4 paths):**
* `PATH_CONT`: Points to `BENN_Continuous.py`
* `PATH_DISC_1`: Points to `BENN_DIS.py`
* `PATH_DISC_2`: Points to `BENN_DS.py`
* `PATH_MIXED`: Points to `BENN_Mixed.py`


* **Dataset Location (1 path):**
* `base_dir`: Points to the root directory where you downloaded and stored the real-world datasets (e.g., your local `real-datasets` folder).



### 5. Code Organization Philosophy

You might notice that we did not split the underlying functions into numerous granular module files (like separate `utils.py`, `losses.py`, etc.). Instead, we provided **complete, standalone Python files** for continuous, discrete (two variants), and mixed data frameworks.

**Why did we structure it this way?**

* **Out-of-the-box Readability:** We aim to provide "standalone" code with maximum readability. Readers can understand the entire forward pass and loss computation logic within a single file, without having to jump back and forth between multiple directories and modules.
* **Lower Learning Curve:** The kernel function calculations and layer designs differ depending on the data type (continuous vs. discrete). Encapsulating them in their respective independent files allows researchers to focus entirely on the inference logic for a specific data type, making it incredibly easy to extract, modify, or integrate into your own projects.
