import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


df = pd.read_csv('creditcard.csv')

# df_example = df[['V1', 'V2', 'V3', 'Amount', 'Class']]

# plt.scatter(df_example['V1'], df_example['Amount'], c=df_example['Class'])
sns.scatterplot(data=df, x='V1', y='Amount', hue='Class')

df_data = df.drop('Class', axis=1)

# Oversampling the data
smote = SMOTE(random_state=101)
X, y = smote.fit_resample(df_data, df['Class'])

# Creating a new Oversampling Data Frame
df_oversampler = pd.DataFrame(X)
df_oversampler['Class'] = pd.DataFrame(y)

# Plot data after oversampling
sns.countplot(df_oversampler['Class'])
# sns.scatterplot(data=df_oversampler, x='V1', y='Amount', hue='Class')
# plt.scatter(df_example['V1'], df_example['Amount'], c=df_example['Class'])

# df_oversampler.to_csv('data.csv', index=True, index_label='Time')
plt.show()
