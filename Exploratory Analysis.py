#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import datetime
import matplotlib
import os
import plotly.graph_objects as go
import itertools
from scipy.stats import pearsonr
import plotly.express as px


# In[2]:


#x_2012, y_2012, tms_2012 were part of RData file which was read and converted to .csv. 
#All data files are uploaded in the Data directory.

#x & y are dataframes each with 1471015 rows, 26 columns. Columns are named 'V1', 'V2', ..., 'V26' and correspond to 
#the 26 baboons. Each cell contains the x (or y) position of a baboon at a particular timestamp. Timestamps are stored in tms.
y = pd.read_csv("y_2012.csv") 
x = pd.read_csv("x_2012.csv")

#tms is a dataframe with 1471015 rows, 1 column. Each row contains a timestamp, and the index directly corresponds to the
#index of x and y.  The column is named 'x'.
tms = pd.read_csv("tms_2012.csv")

#demo contains age, sex, etc. information for each baboon
demo = pd.read_csv('IDs_2012.csv')   


# In[4]:


#look at the first five rows of x, y and tms: 
print(f'X: \n{x.head()}\n')
print(f'Y: \n{y.head()}\n')
print(f'tms: \n{tms.head()}')


# In[3]:


#When data is read from csv to pandas dataframe, it is read as string by default. The following line converts the 
#strings to datetime format (the 'pd.to_datetime' part), and add three hours to it to convert to East Africa Local time
#(the '+ pd.to_timedelta(...' part)
tms['x'] = pd.to_datetime(tms['x']) + pd.to_timedelta(([3] * len(tms)), unit = 'h')

#dates is a dataframe containing all the dates from the dataset
dates = pd.DataFrame(pd.to_datetime(tms['x']).dt.date)


#The next three lines are three lists initialized to be empty. These would contain one dataframe for each date.
#Essentially, x, y and tms are being broken up from one long dataframe spanning ~35 days into smaller dataframes spanning 1 day. 
x_days_full = []
y_days_full = []
tms_days_full = []

for day in dates.x.unique():
    ind = dates[dates['x'] == day].index.to_list()
    
    x_i = x.iloc[ind]
    y_i = y.iloc[ind]
    tms_i = tms.iloc[ind]
    
    x_days_full.append(x_i)
    y_days_full.append(y_i)
    tms_days_full.append(tms_i)


# In[4]:


#The next task is to temporally discretize each day to contain 1 data point every 10 seconds instead of one every second. 
#As an example, consider the first day and and see how it looks
#after resampling every 10 seconds. The dataframe of 43143 rows was reduced to 4315 rows.
t = tms_days_full[0][tms_days_full[0].index % 10 == 0]

print(f'{len(tms_days_full[0])} -> {len(t)}')

t

x_days_full[0].loc[t.index]


# In[5]:


import math


#Discretize every day's data
x_days = []
y_days = []
tms_days = []


for day in range(len(tms_days_full)):
    t = tms_days_full[day][tms_days_full[day].index % 10 == 0]
    
    x_days.append(x_days_full[day].loc[t.index].reset_index(drop = True))
    y_days.append(y_days_full[day].loc[t.index].reset_index(drop = True))
    
    tms_days.append(t.reset_index(drop = True))


# In[59]:


#Get a sense of how many baboons have data for how many days

fig = go.Figure()


counts = []
baboons_with_data = []
for day in range(35):
    b = []
    c = 0
    for baboon in x_days_full[day].columns:
        if baboon == 'centroid1':
            c += 0
        if len(x_days_full[day][x_days_full[day][baboon].notnull()]) > 0:
            c += 1
            b.append(baboon)
    counts.append(c)
    baboons_with_data.append(b)
    
s = set(baboons_with_data[0])
for day in range(len(baboons_with_data)):
    if day > 0:
        s = s.intersection(set(baboons_with_data[day]))
        
print(f"Baboons that have data on all 35 days: {s}")
    
fig.add_trace(go.Scatter(name = f'{day}.{baboon}', x=list(range(35)), y=counts,
                        mode='lines', marker = dict(size = 2)))
fig.update_layout(title = 'Number of Baboons with Non Null data (2012)',
                  xaxis_title='Day',
                   yaxis_title='Number of Baboons')
fig.show()


# In[6]:


#For analysis, we will discard the last and second last day of available data for each baboon.
#second_last_day_available is a dictionary that stores the second last day of each baboon. A dictionary is a mapping of 
#key: value pairs. In this case the keys are the baboons, and the values are the second last day of available data that the
#following loop accomplishes.

second_last_day_available = {}
for baboon in x_days_full[0].columns:
    for day in range(len(tms_days_full) - 1):

        not_null_indices = x_days_full[day+1][x_days_full[day+1][baboon].notnull()].index
        
        if len(not_null_indices) == 0 or day == len(tms_days_full) - 2:
            second_last_day_available[baboon] = day
            break


# In[7]:


# for day in range(len(tms_2min)):
#     dis = []
#     no_data = []

#     centroids_x_full = pd.DataFrame(x_days[day].mean(axis = 1))
#     centroids_y_full = pd.DataFrame(y_days[day].mean(axis = 1))

      
#     x_days[day]['centroid1'] = centroids_x_full
#     y_days[day]['centroid1'] = centroids_y_full


# In[7]:


#Interpolate Nans in the data. There might be some warnings displayed while executing this section-- that can be ignored.
#This takes about 20 seconds to run.

from scipy.interpolate import griddata

interpolated_x_days = []
interpolated_y_days = []

for day in range(len(tms_days)):
    x_ = pd.DataFrame()
    y_ =  pd.DataFrame()
    print(day)
    for baboon in x_days[day].columns: #[x for x in x_days[day].columns if x != 'dist']:
#         print(baboon)
        data = [x_days[day][baboon], y_days[day][baboon]]
        headers = ["x", "y"]
        
        
        data = pd.concat(data, axis=1, keys=headers)
        restore_index = data.index
        data.index = pd.to_datetime(tms_days[day]['x'])

        not_null_indices = data[data['x'].notnull()].index
        if len(not_null_indices) > 0:
            l1 = min(not_null_indices)
            l2 = max(not_null_indices)

            to_interpolate = data[(data.index >= l1) & (data.index <= l2)]
            to_interpolate.resample('D').mean()

            to_interpolate['x_i'] = to_interpolate['x'].interpolate()
            to_interpolate['y_i'] = to_interpolate['y'].interpolate()

            x_col = pd.concat([data[data.index < l1], to_interpolate, data[data.index > l2]], axis = 0)['x_i']
            x_col = x_col.rename(baboon)
            y_col = pd.concat([data[data.index < l1], to_interpolate, data[data.index > l2]], axis = 0)['y_i']
            y_col = y_col.rename(baboon)
        else:
            x_col = data['x']
            y_col = data['y']
            x_col = x_col.rename(baboon)
            y_col = y_col.rename(baboon)

        x_ = pd.concat([x_, x_col], axis = 1)
        y_ = pd.concat([y_, y_col], axis = 1)
        
    x_.index = restore_index
    y_.index = restore_index
    interpolated_x_days.append(x_)
    interpolated_y_days.append(y_)


# In[8]:


#Compute the group centroids at every timestamp every day

for day in range(len(tms_days)):
    dis = []
    no_data = []
#     print(day)
    centroids_x = pd.DataFrame(interpolated_x_days[day].mean(axis = 1))
    centroids_y = pd.DataFrame(interpolated_y_days[day].mean(axis = 1))

      
    interpolated_x_days[day]['centroid1'] = centroids_x
    interpolated_y_days[day]['centroid1'] = centroids_y

    
interpolated_x_days


# In[10]:


#In this part, we will calculate the spots where the baboons started from and returned to on each day for each baboon.
#For baboon B on day D, we will average the first 10 non-null coordinates of B on day D and label that at the position from
#where they started. This will also be the spot where they would spend the night on day D-1, except on the last day of the
#dataset. For the last day, the return site is simply the point where they were last found. 

#first_found_x (y) and last_found_x (y) are nested lists. Each element of first_found_x (y) and last_found_x (y)
#is a list corresponding to a day, containing the positions for each baboon. 

first_found_x = []
first_found_y = []

last_found_x = []
last_found_y = []

for day in range(len(tms_days)):
    ffx = []
    ffy = []
    
    
    
    for baboon in [x for x in interpolated_x_days[day].columns if x != 'centroid1']:
        not_null_indices = interpolated_x_days[day][interpolated_x_days[day][baboon].notnull()].index
        
        if len(not_null_indices) == 0:
            ffx.append(0)
            ffy.append(0)
           
            
        else:
            sx = 0
            sy = 0
            ct = 0
            for i in range(10):
                if i <= len(not_null_indices):
                    sx += interpolated_x_days[day].loc[not_null_indices.to_list()[i], baboon]
#                     print(sx)
                    sy += interpolated_y_days[day].loc[not_null_indices.to_list()[i], baboon]
                    ct += 1
#                     print(ct)
                    
            ffx.append(sx/ct)
            ffy.append(sy/ct)
            
    first_found_x.append(ffx)
    first_found_y.append(ffy)
    
    
for day in range(len(tms_days)):
    lfx = []
    lfy = []
    b = 0
    for baboon in [x for x in interpolated_x_days[day].columns if x != 'centroid1']:
        not_null_indices = interpolated_x_days[day][interpolated_x_days[day][baboon].notnull()].index
        
        
        if len(not_null_indices) == 0:
            lfx.append(0)
            lfy.append(0)
        
        else:
            
            if day < len(tms_days) - 1:
                not_null_indices_next = interpolated_x_days[day + 1][interpolated_x_days[day + 1][baboon].notnull()].index
                
                lfx.append(first_found_x[day+1][b])
                lfy.append(first_found_y[day+1][b])
                
#                 if len(not_null_indices_next) > 0:
#                     lfx.append(first_found_x[day+1][b])
#                     lfy.append(first_found_y[day+1][b])
                
#                 else:
#                     lfx.append(interpolated_x_days[day].loc[max(not_null_indices), baboon])
#                     lfy.append(interpolated_y_days[day].loc[max(not_null_indices), baboon])
            else:
                    lfx.append(interpolated_x_days[day].loc[max(not_null_indices), baboon])
                    lfy.append(interpolated_y_days[day].loc[max(not_null_indices), baboon])
                
            
        b += 1
        
    last_found_x.append(lfx)
    last_found_y.append(lfy)

    


# In[12]:


#This section takes about an hour to run.

#Each of distances, distances_from_start, distances_from_return and discrete_distances contain an element for every day. 
#The daily elements will be dictionaries, with baboons (including centroid) as keys and dataframes of [time, dist] as the values.
'''
distances = []  #contains the distances travelled every 5 minutes, with a sliding window of 10 seconds.
distances_from_start = [] #contains the distances the baboons are at from where they started
distances_from_return = []  #contains the distances the baboons are at from where they would return to
discrete_distances = [] #This contains the distances travelled every 10 seconds, with no sliding window. Used in a plot later.

for day in range(len(tms_days)):
    
    #initializing empty dictionaries to be appended to the lists
    d = {}
    d2 = {}
    d3 = {}
    d4 = {}
    
    b = 0 #Will be used to access first_found and last_found positions
    print(day)
    
    for baboon in interpolated_x_days[day].columns:
        if baboon != 'centroid1' and day >= second_last_day_available[baboon]: #We don't want data for an individual past their second_last day
            b += 1
            continue
        
        #notnull contains all the indices with non-null data on this day for this baboon
        notnull = interpolated_x_days[day][interpolated_x_days[day][baboon].notnull()].index
        
        #initializing the dataframes that would be values for this baboon in the distionaries
        df = pd.DataFrame(columns = ['time', 'distance'])
        df2 = pd.DataFrame(columns = ['time', 'distance'])
        df3 = pd.DataFrame(columns = ['time', 'distance'])
        df4 = pd.DataFrame(columns = ['time', 'distance'])
        
        if len(notnull) > 0:

            if baboon == 'centroid1':
                startx_ = interpolated_x_days[day].loc[notnull[0], baboon] 
                starty_ = interpolated_y_days[day].loc[notnull[0], baboon] 
                
                r_startx = interpolated_x_days[day].loc[notnull[len(notnull) -1], baboon] 
                r_starty = interpolated_y_days[day].loc[notnull[len(notnull) -1], baboon] 
            else:
                startx_ = first_found_x[day][b] 
                starty_ = first_found_y[day][b] 
                
                r_startx = last_found_x[day][b] 
                r_starty = last_found_y[day][b]
            
            initial_x = startx_ 
            initial_y = starty_ 
            
            r_initial_x = r_startx 
            r_initial_y = r_starty
            
            d_from_start = 0
            for i in range(len(notnull)):
                ind = notnull[i]

                currentx = interpolated_x_days[day].loc[ind, baboon] 
                currenty = interpolated_y_days[day].loc[ind, baboon] 
                
                if i+30 <= len(notnull)-1: #Since the dataset is 10 second discretized, rows every 30 index are 5 minutes apart
                    #The if stateent checks if the row 5 minute away from the current row is outside the dataset. if not, 
                    #designate the end index to be the index 5 minutes away from the current one.
                    end_ind = notnull[i + 30]
                else:
                    #otherwise, designate the end index as the last available index with non null data.
                    end_ind = notnull[len(notnull) - 1]
                    
                df.loc[i, 'time'] = tms_days[day].loc[ind, 'x']
                df2.loc[i, 'time'] = tms_days[day].loc[ind, 'x']
                df3.loc[i, 'time'] = tms_days[day].loc[ind, 'x']
                df4.loc[i, 'time'] = tms_days[day].loc[ind, 'x']
                
                endx = interpolated_x_days[day].loc[end_ind, baboon] 
                endy = interpolated_y_days[day].loc[end_ind, baboon]
                
                
                dist = ((currentx - endx) ** 2 + (currenty - endy) ** 2) ** 0.5
                
                discrete_dist = ((startx_ - currentx) ** 2 + (starty_ - currenty) ** 2) ** 0.5
                
                d_from_start = ((initial_x - currentx) ** 2 + (initial_y - currenty) ** 2) ** 0.5
                d_from_r = ((r_initial_x - currentx) ** 2 + (r_initial_y - currenty) ** 2) ** 0.5
                
                df.loc[i, 'distance'] = dist
                df2.loc[i, 'distance'] = d_from_start   
                df3.loc[i, 'distance'] = d_from_r
                df4.loc[i, 'distance'] = discrete_dist
                startx_ = currentx
                starty_ = currenty
                
        d[baboon] = df
        d2[baboon] = df2
        d3[baboon] = df3
        d4[baboon] = df4
        b += 1
        
    distances.append(d)
    distances_from_start.append(d2)
    distances_from_return.append(d3)
    discrete_distances.append(d4)
'''    
    


# In[11]:


#Write distances to files, and read from file so that it won't take an hour to run. 

folder1 =  "Distances"
folder2 =  "Distances from start"
folder3 =  "Distances from return"
folder4 =  "Discrete distances"
parent_dir = r"C:\Namrata\OSU\Summer21\baboon"
path1 = os.path.join(parent_dir, folder1)
path2 = os.path.join(parent_dir, folder2)
path3 = os.path.join(parent_dir, folder3)
path4 = os.path.join(parent_dir, folder4)
try:
    os.mkdir(path1)
    os.mkdir(path2)
    os.mkdir(path3)
    os.mkdir(path4)
except FileExistsError:
    True


# In[27]:


'''
for day in range(len(tms_days)):
    for b in list(distances[day].keys()):
        f1 = open(f"{path1}/Day{day+1}Baboon{b}Distances.csv", 'w', newline = '')
        dist_to_write1 = distances[day][b]
        dist_to_write1.to_csv(f1, index = False)
        f1.close()
    
for day in range(len(tms_days)):
    for b in list(distances_from_start[day].keys()):
        f2 = open(f"{path2}/Day{day+1}Baboon{b}Distances from start.csv", 'w', newline = '')
        dist_to_write2 = distances_from_start[day][b]
        dist_to_write2.to_csv(f2, index = False)
        f2.close()
    
    
for day in range(len(tms_days)):
    for b in list(distances_from_return[day].keys()):
        f3 = open(f"{path3}/Day{day+1}Baboon{b}Distances from return.csv", 'w', newline = '')
        dist_to_write3 = distances_from_return[day][b]
        dist_to_write3.to_csv(f3, index = False)
        f3.close()
    
for day in range(len(tms_days)):
    for b in list(discrete_distances[day].keys()):
        f4 = open(f"{path4}/Day{day+1}Baboon{b}Discrete distances.csv", 'w', newline = '')
        dist_to_write4 = discrete_distances[day][b]
        dist_to_write4.to_csv(f4, index = False)
        f4.close()
'''
distances1 = []  #contains the distances travelled every 5 minutes, with a sliding window of 10 seconds.
distances_from_start1 = [] #contains the distances the baboons are at from where they started
distances_from_return1 = []  #contains the distances the baboons are at from where they would return to
discrete_distances1 = [] #This contains the distances travelled every 10 seconds, with no sliding window. Used in a plot later.

for day in range(len(tms_days)):
    d = {}
    for b in interpolated_x_days[day].columns:
        try:
            dtypes = {'time': 'str', 'distance': 'float'}
            d[b] = pd.read_csv(f"{path1}/Day{day+1}Baboon{b}Distances.csv", dtype=dtypes, parse_dates=['time'])
        except FileNotFoundError:
            continue
    distances1.append(d)
    
for day in range(len(tms_days)):
    d = {}
    for b in interpolated_x_days[day].columns:
        try:
            dtypes = {'time': 'str', 'distance': 'float'}
            d[b] = pd.read_csv(f"{path2}/Day{day+1}Baboon{b}Distances from start.csv", dtype=dtypes, parse_dates=['time'])
        except FileNotFoundError:
            continue
    distances_from_start1.append(d)
        
    
    
for day in range(len(tms_days)):
    d = {}
    for b in interpolated_x_days[day].columns:
        try:
            dtypes = {'time': 'str', 'distance': 'float'}
            d[b] = pd.read_csv(f"{path3}/Day{day+1}Baboon{b}Distances from return.csv", dtype=dtypes, parse_dates=['time'])
        except FileNotFoundError:
            continue
    distances_from_return1.append(d)
        
    
for day in range(len(tms_days)):
    d = {}
    for b in interpolated_x_days[day].columns:
        try:
            dtypes = {'time': 'str', 'distance': 'float'}
            d[b] = pd.read_csv(f"{path4}/Day{day+1}Baboon{b}Discrete distances.csv", dtype=dtypes, parse_dates=['time'])
        except FileNotFoundError:
            continue
    discrete_distances1.append(d)


# In[13]:


#distances_from_start and distances_from_return in the previous section contained the straight line distances.
#Here, we calculate the actual distances travelled, using the discrete_distances 
'''
distances_from_start_new = []
distances_from_return_new = []

for day in range(len(tms_days)):
    print(day)
    d2 = {}
    d3 = {}
    b = 0
    for baboon in interpolated_x_days[day].columns:
        if baboon != 'centroid1' and day >= second_last_day_available[baboon]:
            b += 1
            continue
        notnull = interpolated_x_days[day][interpolated_x_days[day][baboon].notnull()].index
        df2 = pd.DataFrame(columns = ['time', 'distance'])
        df3 = pd.DataFrame(columns = ['time', 'distance'])
        
        if len(notnull) > 0:
            for i in range(len(notnull)):
                
                ind = notnull[i]

                df2.loc[i, 'time'] = tms_days[day].loc[ind, 'x']
                
                ind2 = notnull[len(notnull) - i - 1,]
                df3.loc[len(notnull) - i - 1, 'time'] = tms_days[day].loc[ind2, 'x']

                if i > 0:
                    df2.loc[i, 'distance'] = df2.loc[i-1, 'distance'] + discrete_distances[day][baboon].loc[i, 'distance']
                else:
                    df2.loc[i, 'distance'] = discrete_distances[day][baboon].loc[i, 'distance']

                if i > 0:
                    df3.loc[len(notnull) - i - 1, 'distance'] = df3.loc[len(notnull) - i, 'distance'] + discrete_distances[day][baboon].loc[len(notnull) - i - 1, 'distance']
                else:
                    df3.loc[len(notnull) - i - 1, 'distance'] = discrete_distances[day][baboon].loc[len(notnull) - i - 1, 'distance']

        d2[baboon] = df2
        d3[baboon] = df3
    distances_from_start_new.append(d2)
    distances_from_return_new.append(d3)
'''


# In[13]:


folder5 =  "Distances from start new"
folder6 =  "Distances from return new"

parent_dir = r"C:\Namrata\OSU\Summer21\baboon"
path5 = os.path.join(parent_dir, folder5)
path6 = os.path.join(parent_dir, folder6)
try:
    os.mkdir(path5)
    os.mkdir(path6)
    
except FileExistsError:
    True


# In[28]:


#Write distances to files, and read from file so that it won't take an hour to run. 
'''
for day in range(len(tms_days)):
    for b in list(distances_from_start_new[day].keys()):
        f5 = open(f"{path5}/Day{day+1}Baboon{b}Distances from start new.csv", 'w', newline = '')
        dist_to_write5 = distances_from_start_new[day][b]
        dist_to_write5.to_csv(f5, index = False)
        f5.close()
    
    
for day in range(len(tms_days)):
    for b in list(distances_from_return_new[day].keys()):
        f6 = open(f"{path6}/Day{day+1}Baboon{b}Distances from return new.csv", 'w', newline = '')
        dist_to_write6 = distances_from_return_new[day][b]
        dist_to_write6.to_csv(f6, index = False)
        f6.close()
    
'''
    
distances_from_start_new1 = [] #contains the distances the baboons are at from where they started
distances_from_return_new1 = []  #contains the distances the baboons are at from where they would return to


    
for day in range(len(tms_days)):
    d = {}
    for b in interpolated_x_days[day].columns:
        try:
            dtypes = {'time': 'str', 'distance': 'float'}
            d[b] = pd.read_csv(f"{path5}/Day{day+1}Baboon{b}Distances from start new.csv", dtype=dtypes, parse_dates=['time'])
        except FileNotFoundError:
            continue
    distances_from_start_new1.append(d)
        
    
    
for day in range(len(tms_days)):
    d = {}
    for b in interpolated_x_days[day].columns:
        try:
            dtypes = {'time': 'str', 'distance': 'float'}
            d[b] = pd.read_csv(f"{path6}/Day{day+1}Baboon{b}Distances from return new.csv", dtype=dtypes, parse_dates=['time'])
        except FileNotFoundError:
            continue
    distances_from_return_new1.append(d)
        


# In[90]:


#do all baboons have the same length of distances for all days? No
# for day in range(len(tms_2min)):
#     centroid = len(distances[day]['centroid1'])
#     for baboon in x_2min[day].columns:
#         try:
#             if len(distances[day][baboon]) != centroid:
#                 print(day, baboon, len(distances[day][baboon]), centroid)
#         except KeyError:
#             continue


# In[15]:


#The following section is to find the start times and return times.

from scipy.signal import find_peaks
import numpy as np

peak1 = [] #list of dictionaries, one for each day. each dictionary contains start times for all baboons
peak2 = [] #Structure same as peak1, except this contains return times instead of start times.
for day in range(len(tms_days)):
    p1 = {}
    p2 = {}
    for baboon in interpolated_x_days[day].columns:
        if baboon != 'centroid1' and day >= second_last_day_available[baboon]:
            continue
        if len(distances1[day][baboon]) <= 3000:
            continue
        first = min(distances1[day][baboon].index)
        last = max(distances1[day][baboon].index)
        half = math.floor((last - first)/2)
        one_third = math.floor((last - first)/3)
        two_third = math.floor((last - first)*2/3)
        df1 = distances1[day][baboon].loc[first : one_third]

        df2 = distances1[day][baboon].loc[two_third : last]
        y1 = df1.set_index(df1['time'])
        del y1['time']
        y1 = np.asarray(y1['distance'])

        peaks, properties = find_peaks(y1, height = y1[np.argsort(y1)[-3]], prominence=1, width=1)
        if len(peaks) == 0:
            peaks, properties = find_peaks(y1, height = y1[np.argsort(y1)[-3]], prominence=0, width=0)
        if len(peaks) > 1:
            peaks = np.array(peaks[-1])
            for key in properties:
                properties[key] = np.array(properties[key][0])


        print(properties["left_ips"])
        if properties["left_ips"].size > 0:
            p1[baboon] = math.floor(properties["left_ips"])
        
        y2 = df2.set_index(df2['time'])
        del y2['time']
        y2 = np.asarray(y2['distance'])

        peaks2, properties2 = find_peaks(y2, height = max(y2), prominence=1, width=1) # changed y2[np.argsort(y2)[-3]] to max(y2) to fix day 3 (2) in baboons late/early plot 
        if len(peaks2) == 0:
            peaks2, properties2 = find_peaks(y2, height = y2[np.argsort(y2)[-3]], prominence=0, width=0)
        if len(peaks2) > 1:
            peaks2 = np.array(peaks2[-1])
            for key in properties2:
                properties2[key] = np.array(properties2[key][-1])

        print(properties2["left_ips"])
        if properties2["left_ips"].size > 0:
            p2[baboon] = math.floor(properties2["left_ips"]) + two_third
            
    peak1.append(p1)
    peak2.append(p2)


# In[16]:


#Visualize distances 

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing = 0.2)

for day in range(len(tms_days)):
#     print(day)
    for baboon in interpolated_x_days[day].columns:
        if baboon != 'centroid1' and day >= second_last_day_available[baboon]:
            continue

        if baboon != 'centroid1':
            fig.add_trace(go.Scatter(name = f'D{day}.{baboon}', x=distances1[day][baboon]['time'], y=distances1[day][baboon]['distance'],
                            mode='lines'),
                         row=1, col=1)
        if baboon == 'centroid1':
            fig.add_trace(go.Scatter(name = f'D{day}.{baboon}', x=distances1[day][baboon]['time'], y=distances1[day][baboon]['distance'],
                        mode='lines',
                        marker = {'color' : 'black'}),
                         row=1, col=1)

            fig.add_shape(name = 'Peak1', type="line",
                            x0=distances1[day][baboon].loc[peak1[day]['centroid1'], 'time'], y0=0, x1=distances1[day][baboon].loc[peak1[day]['centroid1'], 'time'], y1=700,
                            line=dict(
                                color="purple",
                                width=1,
                                dash="dot",
                            ),
                              row=1, col=1
                        )
        
            fig.add_shape(name = 'Peak2', type="line",
                            x0=distances1[day][baboon].loc[peak2[day]['centroid1'], 'time'], y0=0, x1=distances1[day][baboon].loc[peak2[day]['centroid1'], 'time'], y1=700,
                            line=dict(
                                color="purple",
                                width=1,
                                dash="dot",
                            ),
                              row=1, col=1
                        )
    

            
for day in range(len(tms_days)):
    for baboon in interpolated_x_days[day].columns:
        if baboon != 'centroid1' and day >= second_last_day_available[baboon]:
            continue
        fig.add_trace(go.Scatter(name = f'D{day}.{baboon}',x=distances_from_start1[day][baboon]['time'], y=distances_from_start1[day][baboon]['distance'],
                        mode='lines'),
                          row=2, col=1)

        if baboon == 'centroid1':
           
            fig.add_trace(go.Scatter(name = f'D_start{day}_{baboon}', x=distances_from_start1[day][baboon]['time'], y=distances_from_start1[day][baboon]['distance'],
                        mode='lines',
                        marker = {'color' : 'black'}),
                          row=2, col=1)


            fig.add_shape(name = 'Peak1', type="line",
                            x0=distances1[day][baboon].loc[peak1[day]['centroid1'], 'time'], y0=0, x1=distances1[day][baboon].loc[peak1[day]['centroid1'], 'time'], y1=2100,
                            line=dict(
                                color="purple",
                                width=1,
                                dash="dot",
                            ),
                              row=2, col=1
                        )
        
            
            fig.add_shape(name = 'Peak2', type="line",
                            x0=distances1[day][baboon].loc[peak2[day]['centroid1'], 'time'], y0=0, x1=distances1[day][baboon].loc[peak2[day]['centroid1'], 'time'], y1=2100,
                            line=dict(
                                color="purple",
                                width=1,
                                dash="dot",
                            ),
                              row=2, col=1
                        )
            
            
fig.update_xaxes(title='Time', showticklabels=True, row=1, col=1)
fig.update_xaxes(title='Time', row=2, col=1)
fig.update_yaxes(title='Distance', row=2, col=1)
fig.update_yaxes(title='Distance', row=1, col=1) 

fig.show()


# In[ ]:


#This takes about an hour to run 

import datetime

#duration: return - start (in hours)
#reverse: The naming suggests the time of direction reversal, but it is misleading. This variable stores the time when the 
#baboons stop moving farther from where they would return.
#max_dists: The maximum distance they would travel on the way back. Identify the time when they were farthest from the return
#site, and find the distance they travelled from that time until they reached (using distances_from_return_new)
#avg_speed: Average speed after return initiation upto the end of journey
#max_speed: Maximum speed after return initiation
#dist: Distance from return site at the time of return initiation

to_plot = pd.DataFrame(columns = ['day', 'site', 'durations', 'starts', 'reverse', 'max_dists', 'returns', 'avg_speed', 'max_speed', 'dist'])
to_plot_c = pd.DataFrame(columns = ['day', 'site', 'durations', 'starts', 'reverse', 'max_dists', 'returns', 'avg_speed', 'max_speed', 'dist'])

#the following two are for the line plots of distances from return site after return initiation
time_vs_dist = pd.DataFrame(columns = ['day', 'site', 'baboon', 'return_time', 'distance', 'speed'])
time_vs_dist_c = pd.DataFrame(columns = ['day', 'site', 'baboon', 'return_time', 'distance', 'speed'])

baboons_variations_return = dict.fromkeys(list(interpolated_x_days[0].columns))
for b in baboons_variations_return.keys():
    baboons_variations_return[b] = pd.DataFrame(columns = ['day', 'difference'])


durations = []
other_durations = []
max_dists = []
starts = []
returns = []
return_inits = []
rmax_dist = []
centroid_starts = []
centroid_returns = []
centroid_dur = []
other_centroid_dur = []
centroid_maxdist = []
centroid_return_inits = []
centroid_rmax_dist = []

avg_speed = []
max_speed = []
centroid_avg_speed = []
centroid_max_speed = []

times_to_plot = pd.DataFrame()
i = 0
j = 0
days = [0, 4, 5, 6, 7, 8, 9, 10, 13, 16]

A = [3, 8, 23, 27]
A_ = [16, 17, 19]
B = [0, 1, 6, 14]
C = [9, 10, 13, 24, 26, 28, 30, 32]
D = [4, 15, 18, 25, 33]
E = [2, 5, 7, 11, 12, 15, 29, 31]
L = [20, 21, 22]

day_site_dict = {}
for day in range(len(tms_days)):
    if day in A:
        day_site_dict[day] = 'A'
    elif day in A_:
        day_site_dict[day] = 'A_'
    elif day in B:
        day_site_dict[day] = 'B'
    elif day in C:
        day_site_dict[day] = 'C'
    elif day in D:
        day_site_dict[day] = 'D'
    elif day in E:
        day_site_dict[day] = 'E'
    else:
        day_site_dict[day] = 'L'
    

centroid_ret = []
ct = 0
r1 = 0
r2 = 0
for day in range(30):
    print(day)

    k = 0
    for baboon in interpolated_x_days[day].columns:
        if baboon != 'centroid1' and day >= second_last_day_available[baboon] - 1:
            continue
        if baboon != 'centroid1' and len(distances_from_return1[day][baboon]) > 0:

            try:
                distances_from_return_new1[day][baboon] = distances_from_return_new1[day][baboon].sort_values(by = ['time']).reset_index(drop = True)
#                 first = end_of_beg[day][baboon]
                this_baboon = baboons_variations_return[baboon]
                k = len(this_baboon)
                first = peak1[day][baboon]
                last = peak2[day][baboon]
                firstc = peak1[day]['centroid1']
                lastc = peak2[day]['centroid1']
                this_baboon.loc[k, 'day'] = day + 1
                this_baboon.loc[k, 'difference'] = ((distances_from_return_new1[day][baboon].loc[last, 'time']).hour + (distances_from_return_new1[day][baboon].loc[last, 'time']).minute /60 + (distances_from_return_new1[day][baboon].loc[last, 'time']).second /3600)  - ((distances_from_return_new1[day]['centroid1'].loc[lastc, 'time']).hour + (distances_from_return_new1[day]['centroid1'].loc[lastc, 'time']).minute /60 + (distances_from_return_new1[day]['centroid1'].loc[lastc, 'time']).second /3600)
                

                if (distances_from_return1[day][baboon].loc[last, 'time']).hour > 10:
                    
                    r = last
                    
                    while r <= max(distances1[day][baboon].index):
#                         print(day)
                        time_vs_dist.loc[r1, 'return_time'] = (distances_from_return_new1[day][baboon].loc[r, 'time']).hour + (distances_from_return_new1[day][baboon].loc[r, 'time']).minute /60 + (distances_from_return_new1[day][baboon].loc[r, 'time']).second /3600
                        time_vs_dist.loc[r1, 'distance'] = distances_from_return_new1[day][baboon].loc[r, 'distance']
                        time_vs_dist.loc[r1, 'speed'] = distances1[day][baboon].loc[r, 'distance']/5
                        time_vs_dist.loc[r1, 'day'] = day
                        time_vs_dist.loc[r1, 'baboon'] = baboon
                        time_vs_dist.loc[r1, 'site'] = day_site_dict[day]
                        r+= 1
                        r1+= 1
                    starts.append(datetime.datetime.time(distances_from_return_new1[day][baboon].loc[first, 'time']))#.hour + (distances_from_return[day][baboon].loc[first, 'time']).minute * 0.01)
#                     last = max(distances_from_return[day][baboon]['time'].index)
                    ct += 1
#                     last = max_neg[day][baboon]
                    if day in A:
                        to_plot.loc[i, 'site'] = 'A'
                    elif day in A_:
                        to_plot.loc[i, 'site'] = 'A_'
                    elif day in B:
                        to_plot.loc[i, 'site'] = 'B'
                    elif day in C:
                        to_plot.loc[i, 'site'] = 'C'
                    elif day in D:
                        to_plot.loc[i, 'site'] = 'D'
                    elif day in E:
                        to_plot.loc[i, 'site'] = 'E'
                    else:
                        to_plot.loc[i, 'site'] = 'L'
                    to_plot.loc[i, 'day'] = day
                    to_plot.loc[i, 'starts'] = (distances_from_return_new1[day][baboon].loc[first, 'time']).hour + (distances_from_return_new1[day][baboon].loc[first, 'time']).minute /60 + (distances_from_return_new1[day][baboon].loc[first, 'time']).second /3600
                    to_plot.loc[i, 'returns'] = (distances_from_return_new1[day][baboon].loc[last, 'time']).hour + (distances_from_return_new1[day][baboon].loc[last, 'time']).minute /60 + (distances_from_return_new1[day][baboon].loc[last, 'time']).second /3600
                    to_plot.loc[i, 'durations'] = (distances_from_return_new1[day][baboon].loc[last, 'time'] - distances_from_return_new1[day][baboon].loc[first, 'time']).total_seconds()/3600
                    to_plot.loc[i, 'avg_speed'] = distances1[day][baboon].loc[last:max(distances1[day][baboon].index)]['distance'].mean(axis = 0)/300
                    to_plot.loc[i, 'max_speed'] = max(distances1[day][baboon].loc[last:max(distances1[day][baboon].index)]['distance'])/300 
                    to_plot.loc[i, 'dist'] = distances_from_return_new1[day][baboon].loc[last, 'distance']
                    to_plot.loc[i, 'max_dists'] = distances_from_return_new1[day][baboon].loc[pd.to_numeric(distances_from_return1[day][baboon]['distance']).idxmax(), 'distance']
#                     to_plot.loc[i, 'max_dists'] = max(distances_from_return_new[day][baboon]['distance'])
                    change = list(distances_from_start1[day][baboon][distances_from_start1[day][baboon]['distance'] == max(distances_from_start1[day][baboon]['distance'])]['time'])[0]
                    to_plot.loc[i, 'reverse'] = change.hour + change.minute/60 + change.second/3600
                    i += 1
                    avg_speed.append(distances1[day][baboon].loc[last:max(distances1[day][baboon].index)]['distance'].mean(axis = 0)/300)
                    max_speed.append(max(distances1[day][baboon].loc[last:max(distances1[day][baboon].index)]['distance'])/300)
                    returns.append(datetime.datetime.time(distances_from_return_new1[day][baboon].loc[last, 'time']))#.hour + (distances_from_return[day][baboon].loc[last, 'time']).minute * 0.01)
                    dur = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") + (distances_from_return_new1[day][baboon].loc[last, 'time'] - distances_from_return_new1[day][baboon].loc[first, 'time'])).time()#.total_seconds()/3600
                    durations.append(dur)
                    max_dists.append(max(distances_from_return1[day][baboon]['distance']))
#                     if baboon in max_neg[day].keys():
#                         other_dur = (distances[day][baboon].loc[max_neg[day][baboon], 'time'] - distances[day][baboon].loc[end_of_beg[day][baboon], 'time']).total_seconds()/3600
#                         other_durations.append(other_dur)
#                         return_inits.append((distances_from_return_new[day][baboon].loc[max_neg[day][baboon], 'time']).hour + (distances_from_return_new[day][baboon].loc[max_neg[day][baboon], 'time']).minute * 0.01)
#                         rmax_dist.append(max(distances_from_return[day][baboon]['distance']))
                        
                    
                    
            except KeyError:
                continue
        
        if baboon == 'centroid1':
#             first = min(distances_from_start[day][baboon]['time'].index)
#             first = end_of_beg[day][baboon]
            distances_from_return_new1[day][baboon] = distances_from_return_new1[day][baboon].sort_values(by = ['time']).reset_index(drop = True)
            first = peak1[day][baboon]
            centroid_starts.append(datetime.datetime.time(distances_from_return1[day][baboon].loc[first, 'time']))#.hour + (distances_from_return[day][baboon].loc[first, 'time']).minute * 0.01)
#             last = max(distances_from_return[day][baboon]['time'].index)
#             last = max_neg[day][baboon]
            last = peak2[day][baboon]
    
            if day in A:
                to_plot_c.loc[j, 'site'] = 'A'
            elif day in A_:
                to_plot_c.loc[j, 'site'] = 'A_'
            elif day in B:
                to_plot_c.loc[j, 'site'] = 'B'
            elif day in C:
                to_plot_c.loc[j, 'site'] = 'C'
            elif day in D:
                to_plot_c.loc[j, 'site'] = 'D'
            elif day in E:
                to_plot_c.loc[j, 'site'] = 'E'
            else:
                to_plot_c.loc[j, 'site'] = 'L'
            to_plot_c.loc[j, 'day'] = day
            to_plot_c.loc[j, 'starts'] = (distances_from_return_new1[day][baboon].loc[first, 'time']).hour + (distances_from_return_new1[day][baboon].loc[first, 'time']).minute /60 + (distances_from_return_new1[day][baboon].loc[first, 'time']).second /3600
            to_plot_c.loc[j, 'returns'] = (distances_from_return_new1[day][baboon].loc[last, 'time']).hour + (distances_from_return_new1[day][baboon].loc[last, 'time']).minute /60 + (distances_from_return_new1[day][baboon].loc[last, 'time']).second /3600
            to_plot_c.loc[j, 'durations'] = (distances_from_return_new1[day][baboon].loc[last, 'time'] - distances_from_return_new1[day][baboon].loc[first, 'time']).total_seconds()/3600
            to_plot_c.loc[j, 'avg_speed'] = distances1[day][baboon].loc[last:max(distances1[day][baboon].index)]['distance'].mean(axis = 0)/300
            to_plot_c.loc[j, 'max_speed'] = max(distances1[day][baboon].loc[last:max(distances1[day][baboon].index)]['distance'])/300 
            to_plot_c.loc[j, 'dist'] = distances_from_return_new1[day][baboon].loc[last, 'distance']
            to_plot_c.loc[j, 'max_dists'] = distances_from_return_new1[day][baboon].loc[pd.to_numeric(distances_from_return1[day][baboon]['distance']).idxmax(), 'distance']
            change = list(distances_from_start1[day][baboon][distances_from_start1[day][baboon]['distance'] == max(distances_from_start1[day][baboon]['distance'])]['time'])[0]
            to_plot_c.loc[j, 'reverse'] = change.hour + change.minute/60 + change.second/3600
            j += 1


            centroid_avg_speed.append(distances1[day][baboon].loc[last:max(distances1[day][baboon].index)]['distance'].mean(axis = 0)/300)
            centroid_max_speed.append(max(distances1[day][baboon].loc[last:max(distances1[day][baboon].index)]['distance'])/300)
            centroid_returns.append(datetime.datetime.time(distances_from_return_new1[day][baboon].loc[last, 'time']))#.hour + (distances_from_return[day][baboon].loc[last, 'time']).minute * 0.01)
            dur = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") + (distances_from_return_new1[day][baboon].loc[last, 'time'] - distances_from_return_new1[day][baboon].loc[first, 'time'])).time()#.total_seconds()/3600
            centroid_dur.append(dur)
#             other_dur = (distances[day][baboon].loc[max_neg[day][baboon], 'time'] - distances[day][baboon].loc[end_of_beg[day][baboon], 'time']).total_seconds()/3600
#             other_centroid_dur.append(other_dur)
#             centroid_maxdist.append(max(distances_from_return[day][baboon]['distance']))
#             centroid_return_inits.append((distances_from_return_new[day][baboon].loc[max_neg[day][baboon], 'time']).hour + (distances_from_return_new[day][baboon].loc[max_neg[day][baboon], 'time']).minute * 0.01)
#             centroid_rmax_dist.append(max(distances_from_return[day][baboon]['distance']))

            r = last
            while r <= max(distances1[day][baboon].index):
                time_vs_dist_c.loc[r2, 'return_time'] = (distances_from_return_new1[day][baboon].loc[r, 'time']).hour + (distances_from_return_new1[day][baboon].loc[r, 'time']).minute /60 + (distances_from_return_new1[day][baboon].loc[r, 'time']).second /3600
                time_vs_dist_c.loc[r2, 'distance'] = distances_from_return_new1[day][baboon].loc[r, 'distance']
                time_vs_dist_c.loc[r2, 'speed'] = distances1[day][baboon].loc[r, 'distance']/5
                time_vs_dist_c.loc[r2, 'site'] = day_site_dict[day]
                time_vs_dist_c.loc[r2, 'baboon'] = baboon
                time_vs_dist_c.loc[r2, 'day'] = day
                r2+=1
                r+=1

                
            centroid_ret.append(to_plot_c.loc[j-1, 'returns'])

#beeps after completion
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)


# In[95]:


fig = px.histogram(to_plot, x = 'returns')
fig.update_layout(title = f"Distribution of return initiation times", xaxis_title = 'Time', yaxis_title = 'Count', 
                 font = dict(size = 18))
fig.show()


# In[96]:



to_plot = to_plot.sort_values(by = ['site'])
time_vs_dist_sorted = time_vs_dist.sort_values(by = ['site'])
time_vs_dist_c_sorted = time_vs_dist_c.sort_values(by = ['site'])
color_discrete_map = {"A": 'crimson', "A_": 'blue', "B": 'magenta', "C": 'turquoise', "D": 'orange', "E": 'teal', "L": 'purple'}
fig = go.Figure()
fig2 = go.Figure()
fig3 = go.Figure()

for day in range(30):
    for baboon in interpolated_x_days[day].columns:
        df = time_vs_dist_c[(time_vs_dist_c['day'] == day) & (time_vs_dist_c['baboon'] == baboon)]

        if len(df) == 0 or baboon != 'centroid1':
            continue

        fig.add_trace(go.Scatter(name = f"Day {day+1}, Site {df.loc[min(df.index), 'site']}", x=df['return_time'], y=df['distance'],
                        mode='lines',
                        marker = {'color' : color_discrete_map[df.loc[min(df.index), 'site']]}))
            
        fig2.add_trace(go.Scatter(name = f"Day {day+1}, Site {df.loc[min(df.index), 'site']}", x=df['return_time'], y=df['speed'],
                        mode='lines',
                        marker = {'color' : color_discrete_map[df.loc[min(df.index), 'site']]}))

        fig3.add_trace(go.Scatter(name = df.loc[min(df.index), 'site'], x=df['distance'], y=df['speed'],
                        mode='lines',
                        marker = {'color' : color_discrete_map[df.loc[min(df.index), 'site']]}))

# fig = px.line(time_vs_dist_c, x='return_time', y='distance', #opacity = 0.8,
#                markers = False, color = 'site', color_discrete_map=color_discrete_map)


# fig2 = px.line(time_vs_dist_c, x='return_time', y='speed', #opacity = 0.8,
#                color = 'day', markers = False)

# fig3 = px.line(time_vs_dist_c, x='distance', y='speed', #opacity = 0.8,
#                color = 'day', markers = False)
    

fig.update_layout(title = f"Distance from sleep site vs Time of Day (after return initiation)", xaxis_title = 'Time', yaxis_title = 'Distance')
fig.show()

fig2.update_layout(title = f"Speed vs Time of Day (after return initiation)", xaxis_title = 'Time', yaxis_title = 'Speed (meters/minute)')
fig2.show()

fig3.update_layout(title = f"Speed vs Distance to destination (after return initiation)", xaxis_title = 'Distance', yaxis_title = 'Speed (meters/minute)')
fig3.update_xaxes(autorange="reversed")
fig3.show()


# In[98]:


import random
seeds = [2.5*x for x in list(range(3*26))]


# In[99]:


fig = go.Figure()
fig2 = go.Figure()

leaders = pd.read_csv('Return Initiators.csv', index_col=0)


s = 0
for baboon in baboons_variations_return.keys():
    try:
        random.seed(seeds[s])
        s += 1
        r = random.randrange(0, 246, 1)

        random.seed(seeds[s])
        s += 1
        g = random.randrange(0, 246, 1)

        random.seed(seeds[s])
        s += 1
        b = random.randrange(0, 246, 1)

#         print(s/3-1, demo.loc[s-3, 'Sex'])
        if demo.loc[s/3-1, 'Sex'] == 'M':
            color = 'blue'
            r = 10
            g = 50
            b = 250
        else:
            color = 'pink'
            r = 250
            g = 50
            b = 100
        
        if demo.loc[s/3-1, 'Age'] == 'SA':
            sz = 18
            shape = 'triangle-up-open'
        elif demo.loc[s/3-1, 'Age'] == 'A':
            sz = 8
            shape = 'square-open'
        else:
            sz = 3
            shape = 'circle-open'
        fig.add_trace(go.Scatter(name = f"Baboon {baboon}", x = baboons_variations_return[baboon]['day'], y = baboons_variations_return[baboon]['difference'], mode = 'markers', marker_size=15, marker_symbol = shape, marker_line_width=2, marker = dict(color = f'rgb({r}, {g}, {b})')))#, marker = dict(size = sz, color = f'rgba({r}, {g}, {b}, 0.5)', line=dict(color=f'rgb({r}, {g}, {b})', width=2))))
        
    except (IndexError):
        _= True
fig.update_layout(xaxis_title = 'Days', yaxis_title = 'Return Times - Centroid Return Times')

fig.show()


leaders = leaders.fillna(0)
seed = 0
for baboon in leaders.columns:
    if baboon == 'Day':
        continue
    s = int(str(baboon).replace('V', ''))
    ind = demo[demo['ID'] == s].index.to_list()[0]
    if demo.loc[ind, 'Sex'] == 'M':
        color = 'blue'
        r = 10
        g = 50
        b = 250
    else:
        color = 'pink'
        r = 250
        g = 50
        b = 100

    random.seed(seeds[seed])
    seed += 1
    r = random.randrange(0, 246, 1)

    random.seed(seeds[seed])
    seed += 1
    g = random.randrange(0, 246, 1)

    random.seed(seeds[seed])
    seed += 1
    b = random.randrange(0, 246, 1)

    if demo.loc[ind, 'Age'] == 'SA':
        sz = 18
        shape = 'triangle-up-open'
    elif demo.loc[ind, 'Age'] == 'A':
        sz = 8
        shape = 'square-open'
    else:
        sz = 3
        shape = 'circle-open'
    
    try:
        fig2.add_trace(go.Scatter(name = f"Baboon {baboon}", x = leaders.index, y = leaders[baboon], mode = 'markers', marker_size=15, marker_symbol = shape, marker_line_width=2, marker = dict(color = f'rgb({r}, {g}, {b})')))#, marker = dict(size = sz, color = f'rgba({r}, {g}, {b}, 0.5)', line=dict(color=f'rgb({r}, {g}, {b})', width=2))))
    except KeyError:
        True
fig2.update_layout(xaxis_title = 'Days', yaxis_title = 'Number of Factions Led')

fig2.show()


# In[94]:


return_index = pd.DataFrame(columns = interpolated_x_days[0].columns)
i = 0
for day in range(30):
    for baboon in distances[day].keys():
        return_index.loc[i, baboon] = peak2[day][baboon]
    return_index.loc[i, 'min'] = min(return_index.loc[i])
    return_index.loc[i, 'day'] = day+1
    i += 1
    
f = open("Return Index.csv", 'w', newline = '')
return_index.to_csv(f, index = False)
f.close()


# In[107]:


import itertools
import plotly.graph_objects as go
from scipy.stats import pearsonr
import plotly.express as px

fig = go.Figure()

st = []
for i in range(len(starts)):
    st.append(starts[i].hour + starts[i].minute/60 + starts[i].second/3600)
    

rt = []
for i in range(len(returns)):
    rt.append(returns[i].hour + returns[i].minute/60 + returns[i].second/3600)
    
cst = []
for i in range(len(centroid_starts)):
    cst.append(centroid_starts[i].hour + centroid_starts[i].minute/60 + centroid_starts[i].second/3600)
    
crt = []
for i in range(len(centroid_returns)):
    crt.append(centroid_returns[i].hour + centroid_returns[i].minute/60 + centroid_returns[i].second/3600)

c_d = []
for i in range(len(centroid_dur)):
    c_d.append(centroid_dur[i].hour + centroid_dur[i].minute/60 + centroid_dur[i].second/3600)

d = []
for i in range(len(durations)):
    d.append(durations[i].hour + durations[i].minute/60 + durations[i].second/3600)
      
corr_st_rt, t1 = pearsonr(st, rt)
corr_st_d, t2 = pearsonr(st, d)
corr_rt_d, t3 = pearsonr(rt, d)

to_plot = to_plot.sort_values(by = ['site'])
tp = to_plot[(to_plot['day'] != 22) & (to_plot['day'] != 20)]
color_discrete_map = {"A": 'crimson', "A_": 'blue', "B": 'magenta', "C": 'turquoise', "D": 'orange', "E": 'teal', "L": 'purple'}

fig = px.scatter(to_plot, x='day', y='returns',
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Day number vs return initiation times", xaxis_title = 'Day', yaxis_title = 'Return time')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()


corr_rv_maxdist, t = pearsonr(tp['reverse'], tp['max_dists'])
fig = px.scatter(tp, y='reverse', x='max_dists', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_rv_maxdist, 3)}, {t}", yaxis_title = 'Reverse Time', xaxis_title = 'Distance Travelled Towards Return')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()


corr_rt_maxdist, t = pearsonr(tp['returns'], tp['max_dists'])

fig = px.scatter(tp, x='max_dists', y='returns', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_rt_maxdist, 3)}, {t}") 
fig.update_xaxes(title_text="Distance Travelled Towards Return")
fig.update_yaxes(title_text="Return Initiation Time")

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()

corr_rt_dist, t = pearsonr(to_plot['returns'], to_plot['dist'])
fig = px.scatter(to_plot, x='dist', y='returns', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_rt_dist, 3)}, {t}") 
fig.update_xaxes(title_text="Distance from Destination at Return Initiation Time")
fig.update_yaxes(title_text="Return Initiation Time")

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()

fig = px.scatter(to_plot, x='starts', y='returns', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_st_rt, 3)}, {t}", xaxis_title = 'Start Time', yaxis_title = 'Return Initiation Time')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()


fig = px.scatter(to_plot, x='starts', y='durations', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
# fig.add_trace(go.Scatter(name = 'Centroid', x=cst, y= c_d,
#                         mode='markers'))
fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_st_d, 3)}, {t}", xaxis_title = 'Start Time', yaxis_title = 'Durations')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()


fig = px.scatter(to_plot, x='returns', y='durations', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
# fig.add_trace(go.Scatter(name = 'Centroid', x=crt, y=c_d,
#                         mode='markers'))

fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_rt_d, 3)}, {t}", xaxis_title = 'Return Initiation Time', yaxis_title = 'Durations')
# fig.update_xaxes(categoryorder="category ascending")
# fig.update_yaxes(categoryorder="category ascending")

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()



fig = px.scatter_3d(to_plot, x='returns', y='avg_speed', z='dist',
              color='site', color_discrete_map=color_discrete_map, size = [1]*len(to_plot))
# fig.add_trace(go.Scatter(name = 'Centroid', x=crt, y=centroid_avg_speed,
#                         mode='markers'))
fig.update_layout(xaxis_title = 'Return Initiation Time', yaxis_title = 'Average Speed (m/s)')
fig.show()


fig = px.scatter(to_plot, x='avg_speed', y='returns',
                       color = 'site', color_discrete_map=color_discrete_map)
# fig.add_trace(go.Scatter(name = 'Centroid', x=crt, y=centroid_max_speed,
#                         mode='markers'))

fig.update_layout(xaxis_title = 'Average Speed (m/s)', yaxis_title = 'Return Initiation Time')
fig.show()

fig = px.scatter(to_plot, x='dist', y='avg_speed',
                       color = 'site', color_discrete_map=color_discrete_map)
# fig.add_trace(go.Scatter(name = 'Centroid', x=crt, y=centroid_max_speed,
#                         mode='markers'))

fig.update_layout(xaxis_title = 'Distance from return', yaxis_title = 'Average Speed (m/s)')
fig.show()

fig = px.scatter(to_plot, x='dist', y='returns',
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(xaxis_title = 'Distance from return', yaxis_title = 'Return Time')
fig.show()


fig = px.histogram(c_d)
fig.show()


# In[110]:


corr_rv_maxdist, t = pearsonr(tp['reverse'], tp['durations'])
fig = px.scatter(tp, y='durations', x='reverse', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_rv_maxdist, 3)}, {t}", yaxis_title = 'Reverse Time', xaxis_title = 'Distance Travelled Towards Return')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()


# In[42]:


import itertools
import plotly.graph_objects as go
from scipy.stats import pearsonr
import plotly.express as px

fig = go.Figure()

st = list(to_plot_c['starts'])

rt = list(to_plot_c['returns'])

d = list(to_plot_c['durations'])

corr_st_rt, t1 = pearsonr(st, rt)
corr_st_d, t2 = pearsonr(st, d)
corr_rt_d, t3 = pearsonr(rt, d)

to_plot_c = to_plot_c.sort_values(by = ['site'])
color_discrete_map = {"A": 'crimson', "A_": 'blue', "B": 'magenta', "C": 'turquoise', "D": 'orange', "E": 'teal', "L": 'purple'}


fig = px.scatter(to_plot_c, x='day', y='starts',
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Day number vs Start times (Group Centroid)", xaxis_title = 'Day', yaxis_title = 'Start time')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()

fig = px.scatter(to_plot_c, x='reverse', y='durations', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Direction change time vs durations", xaxis_title = 'Reverse Time', yaxis_title = 'Durations')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()


fig = px.scatter(to_plot_c, x='starts', y='returns', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_st_rt, 3)}", xaxis_title = 'Start Time', yaxis_title = 'Return Time')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()


fig = px.scatter(to_plot_c, x='starts', y='durations', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
# fig.add_trace(go.Scatter(name = 'Centroid', x=cst, y= c_d,
#                         mode='markers'))
fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_st_d, 3)}", xaxis_title = 'Start Time', yaxis_title = 'Durations')

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()


fig = px.scatter(to_plot_c, x='returns', y='durations', trendline="ols", trendline_scope="overall", trendline_color_override="black",
                       color = 'site', color_discrete_map=color_discrete_map)
# fig.add_trace(go.Scatter(name = 'Centroid', x=crt, y=c_d,
#                         mode='markers'))

fig.update_layout(title = f"Pearson's correlation coefficient: {round(corr_rt_d, 3)}", xaxis_title = 'Return Time', yaxis_title = 'Durations')
# fig.update_xaxes(categoryorder="category ascending")
# fig.update_yaxes(categoryorder="category ascending")

for trace in fig['data']:
    if trace['name'] == 'Overall Trendline':
        trace['showlegend'] = False
fig.show()



fig = px.scatter_3d(to_plot_c, x='returns', y='avg_speed', z='dist',
              color='site', color_discrete_map=color_discrete_map, size = [1]*len(to_plot_c))
# fig.add_trace(go.Scatter(name = 'Centroid', x=crt, y=centroid_avg_speed,
#                         mode='markers'))
fig.update_layout(xaxis_title = 'Return Time', yaxis_title = 'Average Speed (m/s)')
fig.show()


fig = px.scatter(to_plot_c, x='avg_speed', y='returns',
                       color = 'site', color_discrete_map=color_discrete_map)
# fig.add_trace(go.Scatter(name = 'Centroid', x=crt, y=centroid_max_speed,
#                         mode='markers'))

fig.update_layout(xaxis_title = 'Average Speed (m/s)', yaxis_title = 'Return Time')
fig.show()

fig = px.scatter(to_plot_c, x='dist', y='avg_speed',
                       color = 'site', color_discrete_map=color_discrete_map)
# fig.add_trace(go.Scatter(name = 'Centroid', x=crt, y=centroid_max_speed,
#                         mode='markers'))

fig.update_layout(xaxis_title = 'Distance from return', yaxis_title = 'Average Speed (m/s)')
fig.show()

fig = px.scatter(to_plot_c, x='dist', y='returns',
                       color = 'site', color_discrete_map=color_discrete_map)
fig.update_layout(xaxis_title = 'Distance from return', yaxis_title = 'Return Time')
fig.show()


# ## Experimental from this part onwards

# In[100]:


import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 8, 10
fig, ax = plt.subplots(2, sharey=True)
ax[0].plot(x_days_full[0]['V1'], y_days_full[0]['V1'])
ax[0].set_title('Full Data')
ax[1].plot(x_days[0]['V1'], y_days[0]['V1'])
ax[1].set_title('Temporally Discretized Data (0.1 Hz)')


# In[146]:





# In[181]:


df = pd.concat([interpolated_x_days[0]['V1'].rename('X'), interpolated_y_days[0]['V1'].rename('Y')], axis = 1)
# df = pd.concat([df, df.iloc[::-1].reset_index(drop = True).rename(columns = {'X': 'X_rev', 'Y': 'Y_rev'})], axis = 1)
m = 360
df


# In[175]:


import matrixprofile as mp
import stumpy
mps, indices = stumpy.mstump(df, m)
# vals = df.values
# profile, figures = mp.analyze(vals)


# In[176]:


motifs_idx = np.argmin(mps, axis=1)
nn_idx = indices[np.arange(len(motifs_idx)), motifs_idx]


# In[177]:


motifs_idx


# In[178]:


fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0})

for k, dim_name in enumerate(df.columns):
    axs[k].set_ylabel(dim_name, fontsize='20')
    axs[k].plot(df[dim_name])
    axs[k].set_xlabel('Time', fontsize ='20')

    axs[k + mps.shape[0]].set_ylabel(dim_name.replace('T', 'P'), fontsize='20')
    axs[k + mps.shape[0]].plot(mps[k], c='orange')
    axs[k + mps.shape[0]].set_xlabel('Time', fontsize ='20')

#     axs[k].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
#     axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
#     axs[k + mps.shape[0]].axvline(x=motifs_idx[1], linestyle="dashed", c='black')
#     axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')

    
    axs[k].plot(range(motifs_idx[k], motifs_idx[k] + m), df[dim_name].iloc[motifs_idx[k] : motifs_idx[k] + m], c='red', linewidth=4)
    axs[k].plot(range(nn_idx[k], nn_idx[k] + m), df[dim_name].iloc[nn_idx[k] : nn_idx[k] + m], c='red', linewidth=4)
    axs[k + mps.shape[0]].plot(motifs_idx[k], mps[k, motifs_idx[k]] + 1, marker="v", markersize=10, color='red')
    axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')

plt.show()


# In[216]:


from sklearn.decomposition import PCA
pca = PCA(n_components=1)

df = pd.concat([interpolated_x_days[5]['V12'].rename('X'), interpolated_y_days[5]['V12'].rename('Y')], axis = 1).dropna()

principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC 1'])
pdf = pd.concat([principalDf, principalDf.iloc[::-1].reset_index(drop = True)], axis = 0).reset_index(drop = True)


# In[218]:


pdf


# In[219]:


plt.plot(pdf['PC 1'])


# In[232]:


m = 1440
mps = {}  # Store the 1-dimensional matrix profiles
motifs_idx = {}  # Store the index locations for each pair of 1-dimensional motifs (i.e., the index location of two smallest matrix profile values within each dimension)


mps[0] = stumpy.stump(pdf['PC 1'], m)
motif_distance = np.round(mps[0].min(), 1)
# print(f"The motif pair matrix profile value in {dim_name} is {motif_distance}")
motifs_idx[0] = np.argsort(mps[0][:, 0])[:2]


# In[233]:


fig, axs = plt.subplots(len(mps), sharex=True, gridspec_kw={'hspace': 0})

for i, dim_name in enumerate(list(mps.keys())):
    axs.set_ylabel(dim_name, fontsize='20')
    axs.plot(pdf['PC 1'])
    axs.set_xlabel('Time', fontsize ='20')
    for idx in motifs_idx[dim_name]:
        axs.plot(pdf['PC 1'].iloc[idx:idx+m], c='red', linewidth=4)
        axs.axvline(x=idx, linestyle="dashed", c='black')

plt.show()


# In[ ]:




