#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import cleaned dataset
import pandas as pd
data = pd.read_csv('clean.csv')


# In[2]:


##Taking subset for quick word embedding
#data = data.loc[6001:10000]


# In[3]:


#data


# In[4]:


##Lemmatization
##Replace all skills with their respective lemmatized grammar
import spacy
nlp = spacy.load('en_core_web_lg')
col = ['Skill1','Skill2','Skill3','Skill4','Skill5','Skill6','Skill7']
c = 0
for k in col:
    l = list(data[k].values)
    for i in range(len(l)):
        try:
            p = nlp(l[i])
            l[i] = " ".join([token.lemma_ for token in p])
            
            c = c + 1
            
        except:
            continue
        
    data[k] = data[k].replace(l)
    


# In[5]:


##Word Embedding


# In[6]:


#Make a set of all unique skills present in the entire dataset across all 7 columns

s1 = list(set(list(data['Skill1'].values)))
s2 = list(set(list(data['Skill2'].values)))
s3 = list(set(list(data['Skill3'].values)))
s4 = list(set(list(data['Skill4'].values)))
s5 = list(set(list(data['Skill5'].values)))
s6 = list(set(list(data['Skill6'].values)))
s7 = list(set(list(data['Skill7'].values)))

m = s1+s2+s3+s4+s5+s6+s7
skills = list(set(m))
print(len(skills))

##Prepare a skill_dct for the skill and its respective code value
skill_dct = {}

for i, val in enumerate(skills):
    skill_dct[val] = [i]

c = 0
##The following is a nested loop for checking similarity of each skill with the other
for i in range(len(skills)):
    if c == 2000:
        print("Done")
        c = 0
    c = c + 1
    #if len(skills[i]) < 5:
        #continue
    doc1 = nlp(skills[i])
    for j in range(i+1,len(skills)):
        #if len(skills[j]) < 5:
            #continue
        if (skills[i].__contains__(skills[j][0:3*int(len(skills[j])/4)])) or (skills[j].__contains__(skills[i][0:3*int(len(skills[i])/4)])):
            if i not in skill_dct[skills[j]]:
                skill_dct[skills[j]].append(i)
            if j not in skill_dct[skills[i]]:
                skill_dct[skills[i]].append(j)
            continue
        doc2 = nlp(skills[j])
        equality = doc1.similarity(doc2)
        
        #print(c)
        if equality >= 0.90:
            #Count variable to check the frequency of equality > 0.90
            #Codes of similar skills to be appended to value against the key in skill_dct
            if i not in skill_dct[skills[j]]:
                skill_dct[skills[j]].append(i)
            if j not in skill_dct[skills[i]]:
                skill_dct[skills[i]].append(j)
                

#print(c)


# In[ ]:


#print(c)


# In[ ]:


#data_enc includes integer codes of all columns on which machine learning model is to be trained
data_enc = data.copy()


# In[ ]:


print(skill_dct)


# In[ ]:


for key,val in skill_dct.items():
    #Single length value should be assigned as the code to key
    if type(val) == list:
        if len(val) == 1:
            skill_dct[key] = val[0]
        else:
            #Sorting lists in ascending order for comparison
            val.sort()
            skill_dct[key] = val


# In[ ]:


#skill_dct


# In[ ]:





# In[ ]:



#Search for similar skills in string key
def search(skill_dct,key,value):
    l = []
    for k,v in skill_dct.items():
        if (value==v):
            l.append(k)
    if len(l) > 0:
        return l
    else:
        return None
#Assign new codes to similar skills having same list content as value in skill_dct
code = len(skills) + 1

#same_sk records key-value pair of similar skills with the shortest length skill as the key
same_sk = {} 

#options_dct keeps a track of shortest length skill and its integer code. Used to interpret option in the frontend
options_dct = {}

for key,val in skill_dct.items():
    #Skills having similar pairs have a list as value
    if type(val) == list:
        #Search for similar skills in dict
        same = search(skill_dct,key,val)
        #If similar skills in str exist
        if same :
            if len(same)>=1:
                #Assign a unique integer code to all the similar skills unlike the list present earlier
                for elem in same:
                    skill_dct[elem] = code
                #Sorting and finding shortest skill to be the key in same_sk and options_dct 
                same.sort()
                short = same[0]
                same.pop(0)
                same_sk[short] = same
                options_dct[short] = code         
                code = code + 1
        else:
            #if no similar skills is found, this is rare or not even possible
            skill_dct[key] = code
            same_sk[key] = None
            options_dct[key] = code       
            code = code + 1
    else:
        #Skills having no similar skills and already have a single integer code as value in dct
        options_dct[key] = val
        same_sk[key] = None
                
 


# In[ ]:


#Check if any value left in dct which is of type list
for key,val in skill_dct.items():
    if val is list:
        print('spot')


# In[ ]:


#data_enc


# In[ ]:


##Replacing the 7 Skill columns with their encoded values

col = ['Skill1','Skill2','Skill3','Skill4','Skill5','Skill6','Skill7']
data_enc1 = data.copy()
for s in col:
    data_enc[s] = data_enc[s].replace(skill_dct)


# In[ ]:


#Making dictionary for assigning codes to every unique element in the columns
#using the dct to replace the current column in data_enc with their integer coded values

rolecat = list(set(list(data_enc['Role Category'].values)))
rolcat_dct = {}
for i,val in enumerate(rolecat):
    rolcat_dct[val] = i
data_enc['Role Category'] = data_enc['Role Category'].replace(rolcat_dct)

role = list(set(list(data['Role'].values)))
role_dct = {}
for i,val in enumerate(role):
    role_dct[val] = i
data_enc['Role'] = data_enc['Role'].replace(role_dct)

fun = list(set(list(data['Functional Area'].values)))
fun_dct = {}
for i,val in enumerate(fun):
    fun_dct[val] = i
data_enc['Functional Area'] = data_enc['Functional Area'].replace(fun_dct)

industry = list(set(list(data['Industry'].values)))
ind_dct = {}
for i,val in enumerate(industry):
    ind_dct[val] = i
data_enc['Industry'] = data_enc['Industry'].replace(ind_dct)

exp = list(set(list(data['Job Experience Required'].values)))
exp_dct = {}
for i,val in enumerate(exp):
    exp_dct[val] = i
data_enc['Job Experience Required'] = data_enc['Job Experience Required'].replace(exp_dct)

title = list(set(list(data['Job Title'].values)))
job_dct = {}
for i,val in enumerate(title):
    job_dct[val] = i
data_enc['Job Title'] = data_enc['Job Title'].replace(job_dct)


# In[ ]:


#Making json files for dct which will be required in generating viz

import json
with open('skills.json','w') as f:
    json.dump(skill_dct,f)

with open('functional_area.json','w') as f:
    json.dump(fun_dct,f)

with open('option.json','w') as f:
    json.dump(options_dct,f)

with open('role.json','w') as f:
    json.dump(role_dct,f)

with open('rolecat.json','w') as f:
    json.dump(rolcat_dct,f)

with open('industry.json','w') as f:
    json.dump(ind_dct,f)


# In[ ]:





# In[ ]:


#data_enc


# In[ ]:





# In[ ]:





# In[ ]:



for i in set(data['Functional Area'].values):
    data_f = data[data['Functional Area'] == i]
    data_enc_f = data_enc[data_enc['Functional Area'] == fun_dct[i]]
    x = len(data_f)
    if x<20 :
        data = data[data['Functional Area'] != i]
        data_enc = data_enc[data_enc['Functional Area'] != fun_dct[i]]
    elif x<100 :
        data = data.append([data_f]*10, ignore_index=False)
        data_enc = data_enc.append([data_enc_f]*10, ignore_index=False)
    elif x < 300:
        data = data.append([data_f]*3, ignore_index=False)
        data_enc = data_enc.append([data_enc_f]*3, ignore_index=False)
    elif x < 500:
        data = data.append([data_f]*2, ignore_index=False)
        data_enc = data_enc.append([data_enc_f]*2, ignore_index=False)
    elif x < 700:
        data = data.append([data_f], ignore_index=False)
        data_enc = data_enc.append([data_enc_f], ignore_index=False)
    elif x > 2700:
        data_f = data_f.iloc[::2]
        data_enc_f = data_enc_f.iloc[::2]
        ind = list(data_f.index.values)
        data.drop(ind, inplace=True)
        ind = list(data_enc_f.index.values)
        data_enc.drop(ind, inplace=True)


# In[ ]:





# In[ ]:


data.to_csv (r'C:\Users\Lenovo\Desktop\Code\Scripts\job_dataset.csv', index = False, header=True)
data_enc.to_csv (r'C:\Users\Lenovo\Desktop\Code\Scripts\job_dataset_encoded.csv', index = False, header=True)


# In[ ]:


#print(max(list(skill_dct.values())))


# In[ ]:




