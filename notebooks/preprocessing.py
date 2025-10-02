# %%
import pandas as pd
import numpy as np

# %%
Df_dataset = pd.read_csv('/content/drive/MyDrive/Airline.csv')

# %%
Df_dataset.head()

# %%
Df_dataset.tail()

# %%
Df_dataset.dtypes

# %%
#Generate descriptive statistics for numeric columns
Df_dataset.describe()

# %%
Df_dataset['Age'].describe()


# %%
#Drop the columns
Df_dataset = Df_dataset.drop(['Unnamed: 0', 'id'], axis=1)


# %%
#Checks each column and counts the missing values for each column
Df_dataset.isnull().sum()


# %%
#Filter the columns which have missing columns greater than 0
Df_dataset.isnull().sum()[Df_dataset.isnull().sum() > 0]


# %%
#return a single number = the total count of missing values in just that one column
Df_dataset['Arrival Delay in Minutes'].isnull().sum()


# %%
#Replaces all null values with the mean of that column
Df_dataset['Arrival Delay in Minutes'] = Df_dataset['Arrival Delay in Minutes'].fillna(
    Df_dataset['Arrival Delay in Minutes'].mean()
)


# %%
#label encoding to convert categorical (text) columns into numbers
from sklearn.preprocessing import LabelEncoder

for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']:
    le = LabelEncoder()
    Df_dataset[col] = le.fit_transform(Df_dataset[col])


# %%
#how many times each unique value appears in the satisfaction column.
print(Df_dataset['satisfaction'].value_counts())


# %%
#analyze how satisfaction (target) varies across categories in another column.
def cross_tab_counts(df, column, target="satisfaction"):
    """
    Prints counts of target=1 (satisfied) and target=0 (not satisfied)
    for each category in the given column.

    Args:
        df (pd.DataFrame): The dataset
        column (str): Categorical column name
        target (str): Target column (default 'satisfaction')
    """
    categories = df[column].unique()
    for cat in categories:
        satisfied = df[(df[column] == cat) & (df[target] == 1)].shape[0]
        not_satisfied = df[(df[column] == cat) & (df[target] == 0)].shape[0]
        print(f"Category: {cat}")
        print(f"   Satisfied: {satisfied}")
        print(f"   Not satisfied: {not_satisfied}")
        print("-" * 40)


# %%
cross_tab_counts(Df_dataset, "Gender")


# %%
cross_tab_counts(Df_dataset, "Customer Type")


# %%
#pie chart visualization for categorical features
import matplotlib.pyplot as plt

def plot_pie(df, column, labels=None, colors=None, explode=None):
    """
    Plots a pie chart for a categorical column.

    Args:
        df (pd.DataFrame): Dataset
        column (str): Column to plot
        labels (list): Labels for pie slices (default uses unique values)
        colors (list): Colors for slices
        explode (tuple): Explode configuration for slices
    """
    counts = df[column].value_counts()
    values = counts.values
    labels = labels if labels else counts.index.astype(str)

    plt.figure(figsize=(6,6))
    plt.pie(values,
            labels=labels,
            colors=colors,
            explode=explode,
            autopct="%1.1f%%",  # shows percentage
            startangle=90)
    plt.title(f"Distribution of {column}")
    plt.axis("equal")
    plt.show()


# %%
# Gender distribution
plot_pie(Df_dataset, "Gender", labels=["Female", "Male"], colors=["pink", "blue"], explode=(0.05, 0))

# Customer type
plot_pie(Df_dataset, "Customer Type", colors=["lightblue", "lightgreen"])

# Travel type
plot_pie(Df_dataset, "Type of Travel", colors=["orange", "purple"])

# Class
plot_pie(Df_dataset, "Class", colors=["gold", "silver", "lightcoral"])

# Satisfaction
plot_pie(Df_dataset, "satisfaction", labels=["Not Satisfied", "Satisfied"], colors=["red", "green"], explode=(0.05, 0))


# %%
Df_dataset.to_csv("Airline_preprocessed.csv", index=False)


# %%



