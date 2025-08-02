'''Machine Learning (ML) algorithms are broadly categorized into three main types, but some sources extend this to four or five based on learning style and applications. Here's a clear breakdown:

1. Supervised Learning
Definition: Learns from labeled data (input-output pairs).
Goal: Predict outcomes for new, unseen data.
Common Algorithms:

Regression (Predict continuous values):

Linear Regression

Polynomial Regression

Classification (Predict discrete categories):

Logistic Regression

Decision Trees

Random Forest

SVM (Support Vector Machines)

Naïve Bayes

Use Cases:
✔ Spam detection (classification)
✔ House price prediction (regression)

2. Unsupervised Learning
Definition: Finds patterns in unlabeled data.
Goal: Discover hidden structures or groupings.
Common Algorithms:

Clustering (Group similar data points):

K-Means

Hierarchical Clustering

DBSCAN

Dimensionality Reduction (Simplify data):

PCA (Principal Component Analysis)

t-SNE

Association Rules (Find item relationships):

Apriori

Use Cases:
✔ Customer segmentation (clustering)
✔ Anomaly detection (e.g., fraud)

3. Reinforcement Learning (RL)
Definition: Learns by trial and error using rewards/penalties.
Goal: Develop a strategy (policy) to maximize cumulative rewards.
Common Algorithms:

Q-Learning

Deep Q-Networks (DQN)

Policy Gradient Methods

Use Cases:
✔ Game AI (e.g., AlphaGo)
✔ Robotics and self-driving cars

4. Semi-Supervised Learning (Hybrid Approach)
Definition: Uses both labeled and unlabeled data (small labeled + large unlabeled datasets).
Goal: Improve accuracy with limited labeled data.
Techniques:

Self-Training

Co-Training

Use Cases:
✔ Medical imaging (where labeling is expensive)

5. Self-Supervised Learning (Emerging Type)
Definition: A subset of unsupervised learning where the data generates its own labels.
Goal: Learn representations without human-labeled data.
Examples:

Contrastive Learning (e.g., SimCLR)

Masked Language Models (e.g., BERT in NLP)

Use Cases:
✔ Pre-training large language models (LLMs)

Summary Table
Type	Data Requirement	Key Goal	Example Algorithms
Supervised	Labeled	Predict outcomes	Random Forest, SVM
Unsupervised	Unlabeled	Find patterns	K-Means, PCA
Reinforcement	Rewards/Penalties	Learn optimal actions	Q-Learning, DQN
Semi-Supervised	Mixed (some labels)	Leverage unlabeled data	Self-Training
Self-Supervised	Unlabeled (auto-labels)	Pre-train models	BERT, SimCLR
When to Use Which?
Supervised: When you have labeled data and clear predictions to make.

Unsupervised: For exploratory analysis or when labels are unavailable.

Reinforcement: For sequential decision-making tasks (e.g., robotics).

Semi-Supervised: When labeling data is expensive/time-consuming.

Self-Supervised: For pre-training models before fine-tuning.
'''