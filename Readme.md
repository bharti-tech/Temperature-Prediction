# **Minor Project On Multi LinearRegression** #
# **Sonam Kumari Bharti** #
<pre>

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

startup50 = pd.read_csv("https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv")
startup50



startup50.head()

startup50.shape

startup50.columns.tolist() # Storing all the Column Names in a List

startup50.describe()

startup50.info()

startup50.isnull().sum() # Checking the Missing Values in each Column of the DataFrame

c = startup50.corr()
c

sns.heatmap(c,annot=True,cmap='Blues')
plt.show()

sns.pairplot(startup50)
plt.show()

"""# **Multi LinearRegression**"""

startup50 = startup50.select_dtypes(include =['int64','float64']) #data cleaning 
startup50 #includes all the numerical columns

#data visualization 
import seaborn as sns 
sns.distplot(startup50['Profit'])

startup50.shape

x = startup50.iloc[:,0:3].values
y = startup50.iloc[:,-1].values

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

print(x_train.shape)
print(x_test.shape)

from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred

y_test

df1 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df1

df1.plot(figsize=(20,8),kind = 'bar')

df1.plot(figsize=(15,7))

sns.regplot(x='Actual',y='Predicted',data=df1)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)



</pre>
