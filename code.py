
import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\tanis\Downloads\datasets\Laptop prediction ML\laptops.csv")
df["Price"]=df["Price"]/100
df

df = df.drop(["Operating System Version"], axis=1)


df

for i in range(len(df["Screen Size"])):
    
    df["Screen Size"][i] = df["Screen Size"][i][:4]
    
df
# type(df["Screen Size"][0])

type(df["Screen Size"][0])

df["Screen Size"] = df["Screen Size"].apply(pd.to_numeric)

type(df["Screen Size"][0])

df

df.info()

df["RAM"] = df["RAM"].str.replace("GB","")
df["Weight"]= df["Weight"].str.replace("kg","")
df["Weight"]= df["Weight"].str.replace("s","")
df

df["RAM"] = df["RAM"].astype("int32")
df["Weight"]= df["Weight"].astype("float32")
df["Price"]= df["Price"].astype("int32")

df.info()

df.head()

import seaborn as sns

sns.distplot(df["Price"])


df["Manufacturer"].value_counts().plot(kind ="bar")

df

g =sns.barplot(x=df["Manufacturer"], y =df["Price"])
# g.set_ylim(0,300000)
import matplotlib.pyplot as plt
plt.xticks(rotation=90)
plt.show()

sns.barplot(x=df["Category"], y =df["Price"])
plt.xticks(rotation=90)
plt.show()

df['Touchscreen'] = df["Screen"].apply(lambda x: 1 if "Touchscreen" in x else 0)


df["Touchscreen"].value_counts()

df.head(20)


df['IPS'] = df["Screen"].apply(lambda x: 1 if "IPS" in x else 0)

df["IPS"].value_counts()

new = df["Screen"].str.split("x",n=1,expand = True)

new[0][-4:]

df['Y_res'] = new[1]

df

df['X_res']= new[0].str[-4:]

df.head(20)

df["X_res"]= df["X_res"].astype("int32")
df["Y_res"]= df["Y_res"].astype("int32")

df.info()

df.corr()["Price"]

df["PPI"] = (((df["X_res"]**2) + (df["X_res"]**2))**0.5)/df["Screen Size"]

df.head()

df.corr()["Price"]

df.drop(columns = ["X_res","Y_res","Screen"],inplace = True)

df

df["CPU name"] = df["CPU"].apply(lambda x : " ".join(x.split()[0:3]) )

df.head()


df["CPU name"].value_counts()

def fetch_processor(text):
    
    if text == "Intel Core i7" or text == "Intel Core i5" or text == "Intel Core i3" :
        return text
    elif text.split()[0] =="Intel":
        return "other Intel processor"
    else:
        return "AMD Processor"
    
    

df["CPU brand"] = df["CPU name"].apply(fetch_processor)

df.head(20)

df.drop(columns = ["CPU","CPU name"],inplace = True)
df.head()

df[' Storage'] = df[' Storage'].astype(str).replace('\.0', '', regex=True)
df[" Storage"] = df[" Storage"].str.replace('GB', '')
df[" Storage"] = df[" Storage"].str.replace('TB', '000')
new = df[" Storage"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage',' Storage'],inplace=True)

df.sample(10)

df.corr()["Price"]

df.drop(columns = ["Hybrid","Flash_Storage"],inplace = True)
df.head()

df["GPU brand"] = df['GPU'].apply(lambda x: x.split()[0] )
df.head()

df["GPU brand"].value_counts()

import matplotlib.pyplot as plt
sns.barplot(x=df["GPU brand"], y =df["Price"])
plt.xticks(rotation=90)
plt.show()

df.drop(columns = ["GPU"],inplace = True)
df.sample(10)

df.sample(10)

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

sns.barplot(x=df['Operating System'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df["Operating System"].value_counts()

def cat_os(inp):
    if inp == 'Windows':
        return inp
    elif inp == 'macOS' or inp == 'Mac OS':
        return 'Mac'
    else:
        return 'Others / No OS'

df['os'] = df['Operating System'].apply(cat_os)
df.sample(35)

df.drop(columns = ["Model Name"],inplace = True)
df.sample(10)

sns.barplot(x=df['os'],y=df['Price'],estimator= np.mean)
plt.xticks(rotation='vertical')
plt.show()

df.drop(columns = ["Operating System"],inplace = True)
df.sample(10)

df.drop(columns = ["Screen Size"],inplace = True)

y= np.log(df["Price"])
x = df.drop(columns = ["Price"])

y

x




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=2)

x_train

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#linear regression
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = KNeighborsRegressor(n_neighbors=3)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
]) 

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(x_train,y_train)

y_pred = pipe.predict(x_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))



