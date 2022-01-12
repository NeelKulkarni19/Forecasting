#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


Airline= pd.read_excel("F:/Dataset/AirlinesData.xlsx")


# In[6]:


Airline


# In[8]:


Airline.describe()


# In[9]:


plt.figure(figsize=(24,5))


# In[10]:


Airline.Passengers.plot()


# In[13]:


Airline["Date"] = pd.to_datetime(Airline.Month,format="%b-%y")


# In[14]:


Airline["month"] = Airline.Date.dt.strftime("%b")


# In[15]:


Airline["year"] = Airline.Date.dt.strftime("%y")


# In[16]:


Airline


# In[17]:


plt.figure(figsize=(12,8))


# In[18]:


HeatMonth= pd.pivot_table(data=Airline,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)


# In[20]:


sns.heatmap(HeatMonth,annot=True,fmt="g")


# In[21]:


plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=Airline)
plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=Airline)


# In[24]:


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
A = Airline["month"][0]
A[0:3]
Airline['Month']= 0


# In[25]:


for i in range(96):
    A = Airline["month"][i]
    Airline['month'][i]= A[0:3]


# In[26]:


Dummies = pd.DataFrame(pd.get_dummies(Airline['month']))


# In[28]:


Airline1=pd.concat([Airline.Passengers,Dummies],axis = 1)


# In[29]:


Airline1['t']= np.arange(1,97)


# In[30]:


Airline1["t_square"] = Airline1["t"]*Airline1["t"]
Airline1.columns
Airline1["log_Passengers"] = np.log(Airline1["Passengers"])
Airline1.rename(columns={"Passengers ": 'Passengers'}, inplace=True)
Airline1.Passengers.plot()


# In[31]:


Airline1


# In[32]:


plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=Airline)


# In[33]:


Airline.Passengers.plot(label="org")
for i in range(2,10,2):
    Airline["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[35]:


Airline.Passengers.plot()


# In[37]:


Train = Airline1.head(65)


# In[38]:


Test = Airline1.iloc[65:86,:]


# In[40]:


Predicdata = Airline1.tail(10)


# In[41]:


Airline2= Airline1.iloc[0:84,:]


# In[42]:


Train


# In[43]:


Predicdata


# In[44]:


import statsmodels.formula.api as smf 


# In[46]:


LinearModel = smf.ols('Passengers~t',data=Train).fit()
PredLinear =  pd.Series(LinearModel.predict(pd.DataFrame(Test['t'])))
rmseLinear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(PredLinear))**2))
rmseLinear


# In[48]:


Exp = smf.ols('log_Passengers~t',data=Train).fit()
PredExp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
RmseExp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(PredExp)))**2))
RmseExp


# In[49]:


Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
PredQuad = pd.Series(Quad.predict(Test[["t","t_square"]]))
RmseQuad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(PredQuad))**2))
RmseQuad


# In[50]:


AddSea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
Predaddsea = pd.Series(AddSea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(Predaddsea))**2))
rmse_add_sea


# In[51]:


add_sea_Quad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[52]:


Multiplicativesea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
PredictMultiplicativesea = pd.Series(Multiplicativesea.predict(Test))
rmseMultsea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(PredictMultiplicativesea)))**2))
rmseMultsea


# In[53]:


MultiplicativeAdditiveSea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
predMultiplicativeAdditiveSea = pd.Series(MultiplicativeAdditiveSea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(predMultiplicativeAdditiveSea )))**2))
rmse_Mult_add_sea


# In[54]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmseLinear,RmseExp,RmseQuad,rmse_add_sea,rmse_add_sea_quad,rmseMultsea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[55]:


predict_data


# In[56]:


Model = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Airline2).fit()


# In[58]:


pred_new  = pd.Series(MultiplicativeAdditiveSea.predict(predict_data))


# In[59]:


pred_new


# In[60]:


predict_data["forecasted_Passengers"] = pd.DataFrame(pred_new)


# In[61]:


predict_data


# In[ ]:




