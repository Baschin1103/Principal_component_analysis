# After the KNN-Classification I wanted to know which variables have the most relevance for the results.
# One approach for this is the Principal-Component-Analysis (PCA). It tries to create principal components (PC's) out of the variables so that less information gets lost.
# It is done here with the help of the sklearn-library.




# Import of necessary libraries.

import numpy as np
import sqlalchemy as sa
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D








# Dataset:
# The data is taken from a database I created before in another project (Players-Database-Hattrick). The connection is done with the library called sqlalchemy.

engine = sa.create_engine("postgresql://postgres:Password@localhost:5432/postgres")     # Password!!!

# I load all possible data from the database in this step. Two tables need to be joined in the query.

sql_query = "Select * from Spieler join Nationalität on Spieler.Nationalität_id = Nationalität.id left join Spezialität on Spieler.Spezialität_id = Spezialität.id;"

# The data is read into a dataframe from pandas.

df = pd.read_sql_query(sql_query, engine)

print('Database:\n', df)







# Defining X (independent variables).

X = df[['alter', 'tsi', 'gehalt', 'wochen_im_verein', 'erfahrung', 'form', 'kondition', 'verteidigung', 
        'spielaufbau', 'flügelspiel', 'passspiel', 'torschuss', 'standards', 'spezialität_name']].values


# Defining y (target variable).

y = df[['führungsqualitäten']]
y = np.array(y)
y = y.ravel()                   # Ravel of the data for better printing.

print('X:\n', X)
print('y:', y)







# Converting the X-data because the PCA needs numeric values for the procedure.

Le = LabelEncoder()

for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

print('X after encoding:\n', X)






# Splitting the data into a training- and testing-dataset. 
# 75% of the data will be reserved for training, 25% for testing.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)    






# Standardization of the data is very important before the PCA because to calculate the important variances, the variables need to be on the same scale.
# It means if we will calculate mean and standard deviation of standard scores it will be 0 and 1 respectively.

scaleStandard = StandardScaler()
X_train = scaleStandard.fit_transform(X_train)
X_test = scaleStandard.fit_transform(X_test)


print('X_train after standardisation:', X_train)






# PCA:

pca1 = PCA()
X_pca1 = pca1.fit_transform(X_train)            # Executing the PCA with the X_train-data.

evr = pca1.explained_variance_ratio_            # Calculating the explained variance ratios of the principal components.
evr = evr.reshape((-1, 1))

print('Explained variance for the principal components (descending order):\n', evr)


# How many principal-components are responsible for 95% of the variance in the data.

pca2 = PCA(0.95)
X_pca2 = pca2.fit_transform(X_train)
print('The number of principal-components responsible for 95% of the variance:', X_pca2.shape[1])


# The influence of the independent variables to the principal-components.

columns = ['Alt', 'Tsi', 'Geh', 'Wiv', 'Erf', 'For', 'Kondn', 'Vert', 
        'Spiela', 'Flügel', 'Pass', 'Tors', 'Stand', 'Spez']

print('Influence-values and -direction of the variables to the principal components:\n', pd.DataFrame(pca1.components_, columns=columns).round(2))


# Hint: After the KNN-Classification I wanted to know which variables have the most relevance for the results.
# The PCA calculates the most important variables (in descending order) for the two principal components: TSI, Gehalt, Erfahrung, Form, Alter and Kondition, but Kondition has a negative relationship.
# A negative relationship means that the Führungsqualitäten (target variable) will probably fall when the Kondition rises here.







# PCA with the three most important components and 3D-Plot:

pca3 = PCA(n_components = 3)
X_pca3 = pca3.fit_transform(X_train)


# Create the 3D scatter plot.
# Create a figure and a 3D axis.

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')      # The argument 111 specifies that it will be a single subplot occupying the entire figure area. projection='3d: This parameter sets the projection type to 3D, enabling the creation of three-dimensional plots.


colormap = plt.colormaps['brg']         # 'brg' is one of many colormaps you can choose. 
scatter = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=y_train, cmap=colormap)

# Set labels for the axes.

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Colorbar for the target-labels.

plt.colorbar(scatter, label='Führungsqualitäten')

# Show the plot.

plt.show()              # The plot shows no clear clusters of the target variable. As we saw in the KNN-Classification, the data is difficult because there are not strong dependencies between X and y. 


# The PC's with the most informations can be used as inputs for ML-models later. 
# This makes sense probably because it reduces the amount of dimensions (variables) and, so complexity while losing less data-information.
# It also reduces the amount of computing capacity.









