#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import joblib
import streamlit as st


# In[37]:


df = pd.read_csv('C:\\Users\\USER-11\\Downloads\\ultimate_student_productivity_dataset_5000.csv')


# In[38]:


df.shape


# In[39]:


df.head()


# In[40]:


df.dtypes


# In[41]:


df.info


# In[42]:


df.describe()


# In[43]:


df.duplicated().sum()


# In[44]:


df.groupby('gender')['exam_score'].mean()


# In[45]:


sns.barplot(x='gender', y='exam_score', data=df)
plt.title("Average Exam Score by Gender")
plt.show()
plt.savefig("Average Exam Score by Gender")


# In[46]:


df.groupby('internet_quality')['productivity_score'].mean()


# In[47]:


df.groupby(['gender','study_hours'])['exam_score'].mean()


# In[48]:


df.groupby('burnout_level')['productivity_score'].mean()


# In[49]:


sns.lineplot(x='burnout_level', y='productivity_score', data=df)
plt.title("Burnout vs Productivity")
plt.show()
plt.savefig("Burnout vs Productivity")


# In[50]:


df.groupby('social_media_hours')['exam_score'].mean()


# In[51]:


sns.scatterplot(x='social_media_hours', y='exam_score', data=df)
plt.title("Social Media Hours vs Exam Score")
plt.show()
plt.savefig("Social Media Hours vs Exam Score")


# In[52]:


important_cols = ['study_hours','sleep_hours',
                  'social_media_hours',
                  'exam_score',
                  'productivity_score']

sns.heatmap(df[important_cols].corr(), annot=True, cmap='coolwarm')
plt.savefig('columns relations')


# In[53]:


x =  df.drop(['exam_score','part_time_job','internet_quality'],axis=1)
y = df['exam_score']


# In[54]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[55]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[56]:


numerical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[57]:


categorical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[58]:


preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols),
    ('cat',categorical_transformer,categorical_cols)
])


# In[59]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[60]:


model = Pipeline(steps=[
    ('pre',preprocessor),('reg',LinearRegression())
])


# In[61]:


model.fit(X_train,y_train)


# In[62]:


y_pred = model.predict(X_test)

print(f'accuracy:{r2_score(y_test,y_pred)*100:.2f}')


# In[63]:


from sklearn.ensemble import RandomForestRegressor


# In[64]:


model2 = Pipeline(steps=[
    ('pre',preprocessor),('reg',RandomForestRegressor(n_estimators=200,random_state=42))
     ])


# In[65]:


model2.fit(X_train,y_train)


# In[66]:


y_pred2 = model2.predict(X_test)
print(f'accuracy:{r2_score(y_test,y_pred2)*100:.2f}')


# In[67]:


joblib.dump(model,'LinearRgression')


# In[68]:


load = joblib.load('LinearRgression')
st.title('exam score prediction')
student_id = st.number_input('student_id')
age= st.number_input('age')
gender = st.selectbox('gender',['male','female'])
academic_level = st.selectbox('academic_level',['Postgraduate','High School','Undergraduate'])
study_hours = st.number_input('study_hours')
self_study_hours = st.number_input('self_study_hours')
online_classes_hours = st.number_input('online_classes_hours')
social_media_hours = st.number_input('social_media_hours')
gaming_hours = st.number_input('gaming_hours')
sleep_hours = st.number_input('sleep_hours')
screen_time_hours = st.number_input('screen_time_hours')
exercise_minutes = st.number_input('exercise_minutes')
caffeine_intake_mg = st.number_input('caffeine_intake_mg')
#part_time_job = st.number_input('part_time_job')
upcoming_deadline = st.number_input('upcoming_deadline')
#internet_quality = st.selectbox('internet_quality',['Good','Poor','Average'])
mental_health_score = st.number_input('mental_health_score')
focus_index = st.number_input('focus_index')
burnout_level = st.number_input('burnout_level')
productivity_score = st.number_input('productivity_scorem')

if st.button('predict'):
        data = pd.DataFrame({
              'student_id':[student_id],
                     'age':[age],
                  'gender':[gender],
          'academic_level':[academic_level],
             'study_hours':[study_hours],
        'self_study_hours':[self_study_hours],
    'online_classes_hours':[online_classes_hours],
      'social_media_hours':[social_media_hours],
            'gaming_hours':[gaming_hours],
             'sleep_hours':[sleep_hours],
       'screen_time_hours':[screen_time_hours],
        'exercise_minutes':[exercise_minutes],
      'caffeine_intake_mg':[caffeine_intake_mg],
          #'part_time_job ':[part_time_job],
       'upcoming_deadline':[upcoming_deadline],
       #'internet_quality ':[internet_quality],
     'mental_health_score':[mental_health_score],
             'focus_index':[focus_index],
           'burnout_level':[burnout_level],
     'productivity_score':[productivity_score]
    })
    
        
        prediction = load.predict(data)
        st.success(f'exam score prediction:{prediction[0]}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




