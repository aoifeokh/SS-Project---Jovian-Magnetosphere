#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import numpy and matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from scipy.io import readsav


# In[2]:


#Tick intervals
#Annually
yearly = mdates.MonthLocator(interval = 12) # ticks with label every 1 year

#Every 6 months
sixmonthly = mdates.MonthLocator(interval = 6) # ticks with label every 6 months

#Every 2 months
twomonthly = mdates.MonthLocator(interval = 2) # ticks with label every 2 months

#Every month
monthly = mdates.MonthLocator(interval = 1) # ticks with label every 1 month

#Every 2 weeks
twoweekly = mdates.DayLocator(interval = 14) # ticks with label every 14 days

#Every week
weekly = mdates.DayLocator(interval = 7) # ticks with label every 7 days

#Every day 
daily = mdates.DayLocator(interval = 1) # ticks with label every day

#Every 12 hours
twelvehourly = mdates.HourLocator(interval = 12) # ticks every 12 hours

#Every 6 hours
sixhourly = mdates.HourLocator(interval = 6) # ticks every 6 hours

#Every 2 hours
twohourly = mdates.HourLocator(interval = 2) # ticks every 2 hours

#Every hour
hourly = mdates.HourLocator(interval = 1) # ticks every hour

#Ticks every 1/2 hour
halfhourly = mdates.MinuteLocator(interval = 30) # Ticks every 30 mins

#Ticks every 1/4 hour
quarterhourly = mdates.MinuteLocator(interval = 15) # Ticks every 15 mins

#Ticks every 5 minutes
fiveminutely = mdates.MinuteLocator(interval = 5) # Ticks every 5 mins

#Use with:
#ax.xaxis.set_major_locator(major_locator)
#ax.xaxis.set_minor_locator(minor_locator)
#To set ticks.


# ## bKOM

# In[3]:


file_bkom = readsav("bKOM_2016100-2019174_timeseries_d15_channels_0-60_zlincal_calibrated.sav")


# In[4]:


print(file_bkom.keys())


# In[5]:


timeseries = file_bkom["timeseries"]
time = file_bkom["time"]
frequencies = file_bkom["frequencies"]


# In[6]:


timeseries = file_bkom["timeseries"]


# In[7]:


from doy2016_to_yyyyddd import doy2016_to_yyyyddd
time = np.array(time)
time = doy2016_to_yyyyddd(time,2016)


# In[8]:


from doy_to_ymd import * 
t_hours_tmp = (time[:] - (np.array(time,dtype=int))[:])*24
time = doy_float_to_ymd(np.array(np.array(time,dtype = int), dtype = str), t_hours_tmp) 


# In[9]:


print(time)


# In[10]:


print(timeseries)


# ## Plot Time series - bKOM

# In[11]:


#day 181 compression - single frequency
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time, timeseries[:,42], label = "112.43 kHz") 

#Set plot/axis titles 
plt.title('Time Series - bKOM - 112.43 Hz (Day 181, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 6, 29), datetime.date(2016, 6, 30))
plt.ylim(1e-18,1e-13)

plt.legend(loc = 'upper right', prop={"size":18})
plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[12]:


print(frequencies[42])
print(frequencies)
print(len(frequencies))


# In[13]:


#3 frequencies - day 181 compression
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

#plt.plot(time, timeseries[:,42]) 
#plt.plot(time, timeseries[:,43])
#plt.plot(time, timeseries[:,44]) 
plt.plot(time, timeseries[:,60]) 

#Set plot/axis titles 
plt.title('Time Series - bKOM - 112.43 kHz (Day 181, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 6, 29), datetime.date(2016, 6, 30))
plt.ylim(1e-18,1e-12)

plt.legend(loc = 'upper right', prop={"size":18})
plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[14]:


#Plot for date without compression
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time, timeseries[:,60], label = "112.43 kHz") 

#Set plot/axis titles 
plt.title('Time Series - bKOM - 112.43 Hz (Day 181, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 7, 6), datetime.date(2016, 7, 7))
plt.ylim(0,1e-3)

plt.legend(loc = 'upper right', prop={"size":18})
#plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[15]:


#Plot all frequencies - no compression
for i in range(len(frequencies)):

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    
    plt.title('Time Series - bKOM Day 188, 2016) '+('%.3f' % frequencies[i])+' kHz')
    plt.xlabel('Time (HH:MM)', fontsize = 20)
    plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)
    
    # plotting
    
    plt.plot(time, timeseries[:,i], label = ('%.3f' % frequencies[i]) +' kHz') 
    
    plt.xlim(datetime.date(2016, 7, 6), datetime.date(2016, 7, 7))
    plt.ylim(1e-18,1e-13)

    plt.legend(loc = 'upper right', prop={"size":18})

    plt.semilogy()

    date_form_orb = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form_orb)

    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    plt.show()


# In[ ]:


#Plot all frequencies - compression
for i in range(len(frequencies)):

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    
    plt.title('Time Series - bKOM Day 181, 2016) '+('%.3f' % frequencies[i])+' kHz')
    plt.xlabel('Time (HH:MM)', fontsize = 20)
    plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)
    
    # plotting
    
    plt.plot(time, timeseries[:,i], label = ('%.3f' % frequencies[i]) +' kHz') 
    
    plt.xlim(datetime.date(2016, 6, 29), datetime.date(2016, 6, 30))
    plt.ylim(1e-18,1e-13)

    plt.legend(loc = 'upper right', prop={"size":18})

    plt.semilogy()

    date_form_orb = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form_orb)

    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    
    plt.show()


# In[16]:


#day 181 compression - 06/29 - no autoplot data
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[466560:472320], timeseries[466560:472320,20], label = "10.010 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,21], label = "11.230 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,22], label = "12.622 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,23], label = "14.160 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,24], label = "15.869 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,25], label = "17.798 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,26], label = "19.971 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,27], label = "19.958 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,28], label = "22.339 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,29], label = "25.085 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,30], label = "28.198 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,31], label = "31.677 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,32], label = "35.522 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,33], label = "39.917 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,34], label = "44.861 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,35], label = "50.171 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,36], label = "56.213 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,37], label = "63.171 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,38], label = "70.862 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,39], label = "79.468 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,40], label = "89.172 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,41], label = "100.16 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,42], label = "112.43 kHz")
 
plt.plot(time[466560:472320], timeseries[466560:472320,43], label = "126.160 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,44], label = "141.540 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,45], label = "140.140 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,46], label = "157.230 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,47], label = "177.730 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,48], label = "198.240 kHz")
                          
plt.plot(time[466560:472320], timeseries[466560:472320,49], label = "222.170 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,50], label = "249.510 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,51], label = "280.270 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,52], label = "314.450 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,53], label = "352.050 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,54], label = "396.480 kHz")
                          
                          
plt.plot(time[466560:472320], timeseries[466560:472320,55], label = "447.750 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,56], label = "502.440 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,57], label = "563.960 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,58], label = "632.320 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,59], label = "707.520 kHz")
plt.plot(time[466560:472320], timeseries[466560:472320,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 181, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 6, 29), datetime.date(2016, 6, 30))
plt.ylim(1e-19,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[17]:


#day 266 compression - 09/23
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[956160:961920], timeseries[956160:961920,20], label = "10.010 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,21], label = "11.230 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,22], label = "12.622 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,23], label = "14.160 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,24], label = "15.869 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,25], label = "17.798 kHz")

plt.plot(time[956160:961920], timeseries[956160:961920,26], label = "19.971 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,27], label = "19.958 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,28], label = "22.339 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,29], label = "25.085 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,30], label = "28.198 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,31], label = "31.677 kHz")

plt.plot(time[956160:961920], timeseries[956160:961920,32], label = "35.522 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,33], label = "39.917 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,34], label = "44.861 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,35], label = "50.171 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,36], label = "56.213 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,37], label = "63.171 kHz")

plt.plot(time[956160:961920], timeseries[956160:961920,38], label = "70.862 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,39], label = "79.468 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,40], label = "89.172 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,41], label = "100.16 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,42], label = "112.43 kHz")
 
plt.plot(time[956160:961920], timeseries[956160:961920,43], label = "126.160 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,44], label = "141.540 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,45], label = "140.140 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,46], label = "157.230 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,47], label = "177.730 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,48], label = "198.240 kHz")
                          
plt.plot(time[956160:961920], timeseries[956160:961920,49], label = "222.170 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,50], label = "249.510 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,51], label = "280.270 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,52], label = "314.450 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,53], label = "352.050 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,54], label = "396.480 kHz")
                          
                          
plt.plot(time[956160:961920], timeseries[956160:961920,55], label = "447.750 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,56], label = "502.440 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,57], label = "563.960 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,58], label = "632.320 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,59], label = "707.520 kHz")
plt.plot(time[956160:961920], timeseries[956160:961920,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 266, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  
plt.xlim(datetime.date(2016, 10, 30), datetime.date(2016, 10, 31))
#plt.ylim(1e-19,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[18]:


#day 304 compression - 10/31 - bKOM emission visible
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[1175040:1180800], timeseries[1175040:1180800,20], label = "10.010 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,21], label = "11.230 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,22], label = "12.622 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,23], label = "14.160 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,24], label = "15.869 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,25], label = "17.798 kHz")

plt.plot(time[1175040:1180800], timeseries[1175040:1180800,26], label = "19.971 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,27], label = "19.958 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,28], label = "22.339 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,29], label = "25.085 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,30], label = "28.198 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,31], label = "31.677 kHz")

plt.plot(time[1175040:1180800], timeseries[1175040:1180800,32], label = "35.522 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,33], label = "39.917 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,34], label = "44.861 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,35], label = "50.171 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,36], label = "56.213 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,37], label = "63.171 kHz")

plt.plot(time[1175040:1180800], timeseries[1175040:1180800,38], label = "70.862 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,39], label = "79.468 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,40], label = "89.172 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,41], label = "100.16 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,42], label = "112.43 kHz")
 
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,43], label = "126.160 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,44], label = "141.540 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,45], label = "140.140 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,46], label = "157.230 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,47], label = "177.730 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,48], label = "198.240 kHz")
                          
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,49], label = "222.170 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,50], label = "249.510 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,51], label = "280.270 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,52], label = "314.450 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,53], label = "352.050 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,54], label = "396.480 kHz")
                          
                          
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,55], label = "447.750 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,56], label = "502.440 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,57], label = "563.960 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,58], label = "632.320 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,59], label = "707.520 kHz")
plt.plot(time[1175040:1180800], timeseries[1175040:1180800,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 304, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 10, 30), datetime.date(2016, 10, 31))
#plt.ylim(1e-19,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[19]:


#day 206 compression - 07/25 - visible bKOM
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[610560:616320], timeseries[610560:616320,20], label = "10.010 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,21], label = "11.230 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,22], label = "12.622 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,23], label = "14.160 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,24], label = "15.869 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,25], label = "17.798 kHz")

plt.plot(time[610560:616320], timeseries[610560:616320,26], label = "19.971 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,27], label = "19.958 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,28], label = "22.339 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,29], label = "25.085 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,30], label = "28.198 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,31], label = "31.677 kHz")

plt.plot(time[610560:616320], timeseries[610560:616320,32], label = "35.522 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,33], label = "39.917 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,34], label = "44.861 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,35], label = "50.171 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,36], label = "56.213 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,37], label = "63.171 kHz")

plt.plot(time[610560:616320], timeseries[610560:616320,38], label = "70.862 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,39], label = "79.468 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,40], label = "89.172 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,41], label = "100.16 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,42], label = "112.43 kHz")
 
plt.plot(time[610560:616320], timeseries[610560:616320,43], label = "126.160 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,44], label = "141.540 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,45], label = "140.140 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,46], label = "157.230 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,47], label = "177.730 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,48], label = "198.240 kHz")
                          
plt.plot(time[610560:616320], timeseries[610560:616320,49], label = "222.170 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,50], label = "249.510 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,51], label = "280.270 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,52], label = "314.450 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,53], label = "352.050 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,54], label = "396.480 kHz")
                          
                          
plt.plot(time[610560:616320], timeseries[610560:616320,55], label = "447.750 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,56], label = "502.440 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,57], label = "563.960 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,58], label = "632.320 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,59], label = "707.520 kHz")
plt.plot(time[610560:616320], timeseries[610560:616320,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 206, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 7, 24), datetime.date(2016, 7, 25))
#plt.ylim(1e-19,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[20]:


#day 225 compression - 08/13 - no obvious bKOM
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[720000:725760], timeseries[720000:725760,20], label = "10.010 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,21], label = "11.230 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,22], label = "12.622 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,23], label = "14.160 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,24], label = "15.869 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,25], label = "17.798 kHz")

plt.plot(time[720000:725760], timeseries[720000:725760,26], label = "19.971 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,27], label = "19.958 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,28], label = "22.339 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,29], label = "25.085 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,30], label = "28.198 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,31], label = "31.677 kHz")

plt.plot(time[720000:725760], timeseries[720000:725760,32], label = "35.522 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,33], label = "39.917 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,34], label = "44.861 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,35], label = "50.171 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,36], label = "56.213 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,37], label = "63.171 kHz")

plt.plot(time[720000:725760], timeseries[720000:725760,38], label = "70.862 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,39], label = "79.468 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,40], label = "89.172 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,41], label = "100.16 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,42], label = "112.43 kHz")
 
plt.plot(time[720000:725760], timeseries[720000:725760,43], label = "126.160 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,44], label = "141.540 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,45], label = "140.140 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,46], label = "157.230 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,47], label = "177.730 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,48], label = "198.240 kHz")
                          
plt.plot(time[720000:725760], timeseries[720000:725760,49], label = "222.170 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,50], label = "249.510 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,51], label = "280.270 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,52], label = "314.450 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,53], label = "352.050 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,54], label = "396.480 kHz")
                          
                          
plt.plot(time[720000:725760], timeseries[720000:725760,55], label = "447.750 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,56], label = "502.440 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,57], label = "563.960 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,58], label = "632.320 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,59], label = "707.520 kHz")
plt.plot(time[720000:725760], timeseries[720000:725760,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 225, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 8, 12), datetime.date(2016, 8, 13))
#plt.ylim(1e-19,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[21]:


#day 322 compression - 11/18 - patch of bKOM at ~19:00
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[1278720:1284480], timeseries[1278720:1284480,20], label = "10.010 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,21], label = "11.230 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,22], label = "12.622 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,23], label = "14.160 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,24], label = "15.869 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,25], label = "17.798 kHz")

plt.plot(time[1278720:1284480], timeseries[1278720:1284480,26], label = "19.971 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,27], label = "19.958 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,28], label = "22.339 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,29], label = "25.085 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,30], label = "28.198 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,31], label = "31.677 kHz")

plt.plot(time[1278720:1284480], timeseries[1278720:1284480,32], label = "35.522 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,33], label = "39.917 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,34], label = "44.861 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,35], label = "50.171 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,36], label = "56.213 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,37], label = "63.171 kHz")

plt.plot(time[1278720:1284480], timeseries[1278720:1284480,38], label = "70.862 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,39], label = "79.468 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,40], label = "89.172 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,41], label = "100.16 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,42], label = "112.43 kHz")
 
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,43], label = "126.160 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,44], label = "141.540 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,45], label = "140.140 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,46], label = "157.230 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,47], label = "177.730 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,48], label = "198.240 kHz")
                          
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,49], label = "222.170 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,50], label = "249.510 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,51], label = "280.270 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,52], label = "314.450 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,53], label = "352.050 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,54], label = "396.480 kHz")
                          
                          
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,55], label = "447.750 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,56], label = "502.440 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,57], label = "563.960 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,58], label = "632.320 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,59], label = "707.520 kHz")
plt.plot(time[1278720:1284480], timeseries[1278720:1284480,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 322, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 11, 17), datetime.date(2016, 11, 18))
#plt.ylim(1e-19,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[22]:


#day 188 compression - 07/07 - bKOM visible but box around data so don't use. 
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[506880:512640], timeseries[506880:512640,20], label = "10.010 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,21], label = "11.230 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,22], label = "12.622 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,23], label = "14.160 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,24], label = "15.869 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,25], label = "17.798 kHz")

plt.plot(time[506880:512640], timeseries[506880:512640,26], label = "19.971 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,27], label = "19.958 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,28], label = "22.339 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,29], label = "25.085 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,30], label = "28.198 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,31], label = "31.677 kHz")

plt.plot(time[506880:512640], timeseries[506880:512640,32], label = "35.522 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,33], label = "39.917 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,34], label = "44.861 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,35], label = "50.171 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,36], label = "56.213 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,37], label = "63.171 kHz")

plt.plot(time[506880:512640], timeseries[506880:512640,38], label = "70.862 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,39], label = "79.468 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,40], label = "89.172 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,41], label = "100.16 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,42], label = "112.43 kHz")
 
plt.plot(time[506880:512640], timeseries[506880:512640,43], label = "126.160 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,44], label = "141.540 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,45], label = "140.140 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,46], label = "157.230 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,47], label = "177.730 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,48], label = "198.240 kHz")
                          
plt.plot(time[506880:512640], timeseries[506880:512640,49], label = "222.170 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,50], label = "249.510 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,51], label = "280.270 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,52], label = "314.450 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,53], label = "352.050 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,54], label = "396.480 kHz")
                          
                          
plt.plot(time[506880:512640], timeseries[506880:512640,55], label = "447.750 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,56], label = "502.440 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,57], label = "563.960 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,58], label = "632.320 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,59], label = "707.520 kHz")
plt.plot(time[506880:512640], timeseries[506880:512640,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 188, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 7, 6), datetime.date(2016, 7, 7))
plt.ylim(-0.00001,0.00001)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

#plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[23]:


print("First index:")
print(np.where(time == datetime.datetime(2016, 9, 7,0,0,0)))

print("Second index:")
print(np.where(time == datetime.datetime(2016, 9, 8,0,0,0)))


# # To include in report and appendix:

# ## Day 199

# In[24]:


#day 199 compression - 07/18 - good compression example - magnetopause crossing data to back up
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[576000:581760], timeseries[576000:581760,20], label = "10.010 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,21], label = "11.230 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,22], label = "12.622 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,23], label = "14.160 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,24], label = "15.869 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,25], label = "17.798 kHz")

plt.plot(time[576000:581760], timeseries[576000:581760,26], label = "19.971 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,27], label = "19.958 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,28], label = "22.339 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,29], label = "25.085 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,30], label = "28.198 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,31], label = "31.677 kHz")

plt.plot(time[576000:581760], timeseries[576000:581760,32], label = "35.522 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,33], label = "39.917 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,34], label = "44.861 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,35], label = "50.171 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,36], label = "56.213 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,37], label = "63.171 kHz")

plt.plot(time[576000:581760], timeseries[576000:581760,38], label = "70.862 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,39], label = "79.468 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,40], label = "89.172 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,41], label = "100.16 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,42], label = "112.43 kHz")
 
plt.plot(time[576000:581760], timeseries[576000:581760,43], label = "126.160 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,44], label = "141.540 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,45], label = "140.140 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,46], label = "157.230 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,47], label = "177.730 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,48], label = "198.240 kHz")
                          
plt.plot(time[576000:581760], timeseries[576000:581760,49], label = "222.170 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,50], label = "249.510 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,51], label = "280.270 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,52], label = "314.450 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,53], label = "352.050 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,54], label = "396.480 kHz")
                          
                          
plt.plot(time[576000:581760], timeseries[576000:581760,55], label = "447.750 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,56], label = "502.440 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,57], label = "563.960 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,58], label = "632.320 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,59], label = "707.520 kHz")
plt.plot(time[576000:581760], timeseries[576000:581760,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 199, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 7, 18), datetime.date(2016, 7, 19))
#plt.ylim(1e-19,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 15))

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[25]:


#day 199 compression - 07/18 - good compression example - magnetopause crossing data to back up
fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(20,15))

ax1.plot(time[576000:581760], timeseries[576000:581760,20], c = 'tab:blue',label = "(a) 10.010 kHz")

ax2.plot(time[576000:581760], timeseries[576000:581760,41], c = 'tab:orange',label = "(b) 100.16 kHz")
 
ax3.plot(time[576000:581760], timeseries[576000:581760,48], c = 'tab:green',label = "(c) 198.240 kHz")
                          
ax4.plot(time[576000:581760], timeseries[576000:581760,52], c = 'tab:red',label = "(d) 314.450 kHz")

ax5.plot(time[576000:581760], timeseries[576000:581760,56], c = 'tab:purple',label = "(e) 502.440 kHz")

ax6.plot(time[576000:581760], timeseries[576000:581760,60], c = 'tab:cyan',label = "(f) 796.390 kHz")

plt.tight_layout(pad=5.0, h_pad=3.0, w_pad=3.0)

#Set plot titles
fig.suptitle('Time Series - bKOM (Day 199, 2016)', fontsize = 20, weight='bold')
ax1.set_title('(a) 10.010 kHz', fontsize = 20, weight='bold')
ax2.set_title('(b) 100.16 kHz', fontsize = 20, weight='bold')
ax3.set_title('(c) 198.240 kHz', fontsize = 20, weight='bold')
ax4.set_title('(d) 314.450 kHz', fontsize = 20, weight='bold')
ax5.set_title('(e) 502.440 kHz', fontsize = 20, weight='bold')
ax6.set_title('(f) 796.390 kHz', fontsize = 20, weight='bold')


#y-labels
ax1.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)
ax4.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)

#x-labels
ax4.set_xlabel('Time (HH:MM)', fontsize = 20)
ax5.set_xlabel('Time (HH:MM)', fontsize = 20)
ax6.set_xlabel('Time (HH:MM)', fontsize = 20)

ax1.set_xlim(datetime.date(2016, 7, 18), datetime.date(2016, 7, 19))

date_form_orb = mdates.DateFormatter("%H:%M")
ax1.xaxis.set_major_formatter(date_form_orb)
ax1.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax2.xaxis.set_major_formatter(date_form_orb)
ax2.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax3.xaxis.set_major_formatter(date_form_orb)
ax3.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax3.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax4.xaxis.set_major_formatter(date_form_orb)
ax4.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax4.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax5.xaxis.set_major_formatter(date_form_orb)
ax5.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax5.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax6.xaxis.set_major_formatter(date_form_orb)
ax6.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax6.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.yaxis.offsetText.set_fontsize(20)

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.yaxis.offsetText.set_fontsize(20)

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)
ax3.yaxis.offsetText.set_fontsize(20)

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)
ax4.yaxis.offsetText.set_fontsize(20)

ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)
ax5.yaxis.offsetText.set_fontsize(20)

ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)
ax6.yaxis.offsetText.set_fontsize(20)

#plt.xticks(fontsize = 20)
#plt.yticks(fontsize = 20)

plt.show()


# In[26]:


#day 199 compression - 07/18 - 07-14 to 07/19 - match autoplot -good compression example 
#- magnetopause crossing data to back up
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[552960:587520], timeseries[552960:587520,20], label = "10.010 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,21], label = "11.230 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,22], label = "12.622 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,23], label = "14.160 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,24], label = "15.869 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,25], label = "17.798 kHz")

plt.plot(time[552960:587520], timeseries[552960:587520,26], label = "19.971 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,27], label = "19.958 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,28], label = "22.339 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,29], label = "25.085 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,30], label = "28.198 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,31], label = "31.677 kHz")

plt.plot(time[552960:587520], timeseries[552960:587520,32], label = "35.522 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,33], label = "39.917 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,34], label = "44.861 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,35], label = "50.171 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,36], label = "56.213 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,37], label = "63.171 kHz")

plt.plot(time[552960:587520], timeseries[552960:587520,38], label = "70.862 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,39], label = "79.468 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,40], label = "89.172 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,41], label = "100.16 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,42], label = "112.43 kHz")
 
plt.plot(time[552960:587520], timeseries[552960:587520,43], label = "126.160 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,44], label = "141.540 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,45], label = "140.140 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,46], label = "157.230 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,47], label = "177.730 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,48], label = "198.240 kHz")
                          
plt.plot(time[552960:587520], timeseries[552960:587520,49], label = "222.170 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,50], label = "249.510 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,51], label = "280.270 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,52], label = "314.450 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,53], label = "352.050 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,54], label = "396.480 kHz")
                          
                          
plt.plot(time[552960:587520], timeseries[552960:587520,55], label = "447.750 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,56], label = "502.440 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,57], label = "563.960 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,58], label = "632.320 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,59], label = "707.520 kHz")
plt.plot(time[552960:587520], timeseries[552960:587520,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 196-201, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (DOY 2016)', fontsize = 20)  

plt.xlim(datetime.date(2016, 7, 14), datetime.date(2016, 7, 20))
#plt.ylim(1e-19,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%-j")
ax.xaxis.set_major_formatter(date_form_orb)
ax.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[27]:


#day 199 compression - 07/18 - good compression example - magnetopause crossing data to back up
fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(20,15))

ax1.plot(time[552960:587520], timeseries[552960:587520,20], c = 'tab:blue',label = "(a) 10.010 kHz")

ax2.plot(time[552960:587520], timeseries[552960:587520,41], c = 'tab:orange',label = "(b) 100.16 kHz")
 
ax3.plot(time[552960:587520], timeseries[552960:587520,48], c = 'tab:green',label = "(c) 198.240 kHz")
                          
ax4.plot(time[552960:587520], timeseries[552960:587520,52], c = 'tab:red',label = "(d) 314.450 kHz")

ax5.plot(time[552960:587520], timeseries[552960:587520,56], c = 'tab:purple',label = "(e) 502.440 kHz")

ax6.plot(time[552960:587520], timeseries[552960:587520,60], c = 'tab:cyan',label = "(f) 796.390 kHz")

plt.tight_layout(pad=5.0, h_pad=3.0, w_pad=3.0)

#Set plot titles
fig.suptitle('Time Series - bKOM (Day 196-201, 2016)', fontsize = 20, weight='bold')
ax1.set_title('(a) 10.010 kHz', fontsize = 20, weight='bold')
ax2.set_title('(b) 100.16 kHz', fontsize = 20, weight='bold')
ax3.set_title('(c) 198.240 kHz', fontsize = 20, weight='bold')
ax4.set_title('(d) 314.450 kHz', fontsize = 20, weight='bold')
ax5.set_title('(e) 502.440 kHz', fontsize = 20, weight='bold')
ax6.set_title('(f) 796.390 kHz', fontsize = 20, weight='bold')


#y-labels
ax1.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)
ax4.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)

#x-labels
ax4.set_xlabel('Time (DOY 2016)', fontsize = 20)
ax5.set_xlabel('Time (DOY 2016)', fontsize = 20)
ax6.set_xlabel('Time (DOY 2016)', fontsize = 20)

ax1.set_xlim(datetime.date(2016, 7, 14), datetime.date(2016, 7, 20))

date_form_orb = mdates.DateFormatter("%-j")
ax1.xaxis.set_major_formatter(date_form_orb)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax2.xaxis.set_major_formatter(date_form_orb)
ax2.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax3.xaxis.set_major_formatter(date_form_orb)
ax3.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax3.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax4.xaxis.set_major_formatter(date_form_orb)
ax4.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax4.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax5.xaxis.set_major_formatter(date_form_orb)
ax5.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax5.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax6.xaxis.set_major_formatter(date_form_orb)
ax6.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax6.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.yaxis.offsetText.set_fontsize(20)

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.yaxis.offsetText.set_fontsize(20)

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)
ax3.yaxis.offsetText.set_fontsize(20)

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)
ax4.yaxis.offsetText.set_fontsize(20)

ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)
ax5.yaxis.offsetText.set_fontsize(20)

ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)
ax6.yaxis.offsetText.set_fontsize(20)

#plt.xticks(fontsize = 20)
#plt.yticks(fontsize = 20)

plt.show()


# ## Day 251

# In[28]:


#day 251 compression - 09/07 - good example of compression - backed up by MP crossings - frequencies from 10-800 kHz
time0907 = time[869760:875520]
timeseries090720 = timeseries[869760:875520,20]
timeseries090741 = timeseries[869760:875520,41]
timeseries090748 = timeseries[869760:875520,48]
timeseries090752 = timeseries[869760:875520,52]
timeseries090756 = timeseries[869760:875520,56]
timeseries090760 = timeseries[869760:875520,60]

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time0907[(np.where(timeseries090720 > 0))], timeseries090720[(np.where(timeseries090720 > 0))], label = "10.010 kHz")

plt.plot(time0907[(np.where(timeseries090741 > 0))], timeseries090741[np.where(timeseries090741 > 0)], label = "100.16 kHz")

plt.plot(time0907[np.where(timeseries090748 > 0)], timeseries090748[np.where(timeseries090748 > 0)], label = "198.240 kHz")
                        
plt.plot(time0907[np.where(timeseries090752 > 0)], timeseries090752[np.where(timeseries090752 > 0)], label = "314.450 kHz")

plt.plot(time0907[np.where(timeseries090756 > 0)], timeseries090756[np.where(timeseries090756 > 0)], label = "502.440 kHz")

plt.plot(time0907[np.where(timeseries090760 > 0)], timeseries090760[np.where(timeseries090760 > 0)], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 251, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 9, 7), datetime.date(2016, 9, 8))
plt.ylim(1e-17,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 15))


plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[29]:


#day 251 compression - 09/07 - good example of compression - backed up by MP crossings - frequencies from 10-200 kHz
time0907 = time[869760:875520]
timeseries090720 = timeseries[869760:875520,20]
timeseries090727 = timeseries[869760:875520,27]
timeseries090735 = timeseries[869760:875520,35]
timeseries090741 = timeseries[869760:875520,41]
timeseries090745 = timeseries[869760:875520,45]
timeseries090748 = timeseries[869760:875520,48]


fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time0907[(np.where(timeseries090720 > 0))], timeseries090720[(np.where(timeseries090720 > 0))], label = "10.010 kHz")

plt.plot(time0907[np.where(timeseries090727 > 0)], timeseries090727[np.where(timeseries090727 > 0)], label = "19.958 kHz")

plt.plot(time0907[np.where(timeseries090735 > 0)], timeseries090735[np.where(timeseries090735 > 0)], label = "50.171 kHz")

plt.plot(time0907[(np.where(timeseries090741 > 0))], timeseries090741[np.where(timeseries090741 > 0)], label = "100.16 kHz")

plt.plot(time0907[np.where(timeseries090745 > 0)], timeseries090745[np.where(timeseries090745 > 0)], label = "140.14 kHz")

plt.plot(time0907[np.where(timeseries090748 > 0)], timeseries090748[np.where(timeseries090748 > 0)], label = "198.240 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 251, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (HH:MM)', fontsize = 20)  

plt.xlim(datetime.date(2016, 9, 7), datetime.date(2016, 9, 8))
plt.ylim(1e-17,1e-12)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(date_form_orb)
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 15))


plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[30]:


print(min(timeseries[:, 20]))
print(min(timeseries[:, 41]))
print(min(timeseries[:, 48]))
print(min(timeseries[:, 52]))
print(min(timeseries[:, 56]))
print(min(timeseries[:, 60]))


# In[31]:


#day 251 compression - 09/07 - good compression example - magnetopause crossing data to back up
fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(20,15))

ax1.plot(time[864000:869760], timeseries[864000:869760,20], c = 'tab:blue',label = "10.010 kHz")

ax2.plot(time[864000:869760], timeseries[864000:869760,41], c = 'tab:orange',label = "100.16 kHz")
 
ax3.plot(time[864000:869760], timeseries[864000:869760,48], c = 'tab:green',label = "198.240 kHz")
                          
ax4.plot(time[864000:869760], timeseries[864000:869760,52], c = 'tab:red',label = "314.450 kHz")

ax5.plot(time[864000:869760], timeseries[864000:869760,56], c = 'tab:purple',label = "502.440 kHz")

ax6.plot(time[864000:869760], timeseries[864000:869760,60], c = 'tab:cyan',label = "796.390 kHz")

plt.tight_layout(pad=5.0, h_pad=3.0, w_pad=3.0)

#Set plot titles
fig.suptitle('Time Series - bKOM (Day 250, 2016)', fontsize = 20, weight='bold')
ax1.set_title('(a) 10.010 kHz', fontsize = 20, weight='bold')
ax2.set_title('(b) 100.16 kHz', fontsize = 20, weight='bold')
ax3.set_title('(c) 198.240 kHz', fontsize = 20, weight='bold')
ax4.set_title('(d) 314.450 kHz', fontsize = 20, weight='bold')
ax5.set_title('(e) 502.440 kHz', fontsize = 20, weight='bold')
ax6.set_title('(f) 796.390 kHz', fontsize = 20, weight='bold')


#y-labels
ax1.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)
ax4.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)

#x-labels
ax4.set_xlabel('Time (HH:MM)', fontsize = 20)
ax5.set_xlabel('Time (HH:MM)', fontsize = 20)
ax6.set_xlabel('Time (HH:MM)', fontsize = 20)

ax1.set_xlim(datetime.date(2016, 9, 6), datetime.date(2016, 9, 7))
ax2.set_xlim(datetime.date(2016, 9, 6), datetime.date(2016, 9, 7))
ax3.set_xlim(datetime.date(2016, 9, 6), datetime.date(2016, 9, 7))
ax4.set_xlim(datetime.date(2016, 9, 6), datetime.date(2016, 9, 7))
ax5.set_xlim(datetime.date(2016, 9, 6), datetime.date(2016, 9, 7))
ax6.set_xlim(datetime.date(2016, 9, 6), datetime.date(2016, 9, 7))

date_form_orb = mdates.DateFormatter("%H:%M")
ax1.xaxis.set_major_formatter(date_form_orb)
ax1.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax2.xaxis.set_major_formatter(date_form_orb)
ax2.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax3.xaxis.set_major_formatter(date_form_orb)
ax3.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax3.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax4.xaxis.set_major_formatter(date_form_orb)
ax4.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax4.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax5.xaxis.set_major_formatter(date_form_orb)
ax5.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax5.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax6.xaxis.set_major_formatter(date_form_orb)
ax6.xaxis.set_major_locator(mdates.HourLocator(interval = 6))
ax6.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 30))

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.yaxis.offsetText.set_fontsize(20)

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.yaxis.offsetText.set_fontsize(20)

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)
ax3.yaxis.offsetText.set_fontsize(20)

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)
ax4.yaxis.offsetText.set_fontsize(20)

ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)
ax5.yaxis.offsetText.set_fontsize(20)

ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)
ax6.yaxis.offsetText.set_fontsize(20)

#plt.xticks(fontsize = 20)
#plt.yticks(fontsize = 20)

plt.show()


# In[32]:


time0907 = time[852480:887040]
timeseries090720 = timeseries[852480:887040,20]
timeseries090741 = timeseries[852480:887040,41]
timeseries090748 = timeseries[852480:887040,48]
timeseries090752 = timeseries[852480:887040,52]
timeseries090756 = timeseries[852480:887040,56]
timeseries090760 = timeseries[852480:887040,60]


# In[33]:


#day 248-253 compression - 09/07 - 09/04 - 09/10 - to match autoplot data - good example of compression 
#- backed up by MP crossings
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time[852480:887040], timeseries[852480:887040,20], label = "10.010 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,21], label = "11.230 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,22], label = "12.622 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,23], label = "14.160 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,24], label = "15.869 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,25], label = "17.798 kHz")

plt.plot(time[852480:887040], timeseries[852480:887040,26], label = "19.971 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,27], label = "19.958 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,28], label = "22.339 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,29], label = "25.085 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,30], label = "28.198 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,31], label = "31.677 kHz")

plt.plot(time[852480:887040], timeseries[852480:887040,32], label = "35.522 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,33], label = "39.917 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,34], label = "44.861 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,35], label = "50.171 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,36], label = "56.213 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,37], label = "63.171 kHz")

plt.plot(time[852480:887040], timeseries[852480:887040,38], label = "70.862 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,39], label = "79.468 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,40], label = "89.172 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,41], label = "100.16 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,42], label = "112.43 kHz")
 
plt.plot(time[852480:887040], timeseries[852480:887040,43], label = "126.160 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,44], label = "141.540 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,45], label = "140.140 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,46], label = "157.230 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,47], label = "177.730 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,48], label = "198.240 kHz")
                          
plt.plot(time[852480:887040], timeseries[852480:887040,49], label = "222.170 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,50], label = "249.510 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,51], label = "280.270 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,52], label = "314.450 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,53], label = "352.050 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,54], label = "396.480 kHz")
                          
                          
plt.plot(time[852480:887040], timeseries[852480:887040,55], label = "447.750 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,56], label = "502.440 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,57], label = "563.960 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,58], label = "632.320 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,59], label = "707.520 kHz")
plt.plot(time[852480:887040], timeseries[852480:887040,60], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (Day 248-253, 2016)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (DOY 2016)', fontsize = 20)  

plt.xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
plt.ylim(1e-19,1e-10)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%-j")
ax.xaxis.set_major_formatter(date_form_orb)
ax.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))


plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[34]:


#day 248-253 compression good example of compression - backed up by MP crossings
#frequencies 10-800 kHz
time0907 = time[852480:887040]
timeseries090720 = timeseries[852480:887040,20]
timeseries090741 = timeseries[852480:887040,41]
timeseries090748 = timeseries[852480:887040,48]
timeseries090752 = timeseries[852480:887040,52]
timeseries090756 = timeseries[852480:887040,56]
timeseries090760 = timeseries[852480:887040,60]

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time0907[(np.where(timeseries090720 > 0))], timeseries090720[(np.where(timeseries090720 > 0))], label = "10.010 kHz")

plt.plot(time0907[(np.where(timeseries090741 > 0))], timeseries090741[np.where(timeseries090741 > 0)], label = "100.16 kHz")

plt.plot(time0907[np.where(timeseries090748 > 0)], timeseries090748[np.where(timeseries090748 > 0)], label = "198.240 kHz")
                        
plt.plot(time0907[np.where(timeseries090752 > 0)], timeseries090752[np.where(timeseries090752 > 0)], label = "314.450 kHz")

plt.plot(time0907[np.where(timeseries090756 > 0)], timeseries090756[np.where(timeseries090756 > 0)], label = "502.440 kHz")

plt.plot(time0907[np.where(timeseries090760 > 0)], timeseries090760[np.where(timeseries090760 > 0)], label = "796.390 kHz")


#Set plot/axis titles 
plt.title('Time Series - bKOM (DoY 2016 248-253)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (DOY 2016)', fontsize = 20)  

plt.xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
plt.ylim(1e-17,1e-11)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%-j")
ax.xaxis.set_major_formatter(date_form_orb)
ax.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))


plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[35]:


#day 248-253 compression - good example of compression - backed up by MP crossings
#frequencies 10-200 kHz to match autoplot
time0907 = time[852480:887040]
timeseries090720 = timeseries[852480:887040,20]
timeseries090727 = timeseries[852480:887040,27]
timeseries090735 = timeseries[852480:887040,35]
timeseries090741 = timeseries[852480:887040,41]
timeseries090745 = timeseries[852480:887040,45]
timeseries090748 = timeseries[852480:887040,48]


fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time0907[(np.where(timeseries090720 > 0))], timeseries090720[(np.where(timeseries090720 > 0))], label = "10.010 kHz")

plt.plot(time0907[np.where(timeseries090727 > 0)], timeseries090727[np.where(timeseries090727 > 0)], label = "19.958 kHz")

plt.plot(time0907[np.where(timeseries090735 > 0)], timeseries090735[np.where(timeseries090735 > 0)], label = "50.171 kHz")

plt.plot(time0907[(np.where(timeseries090741 > 0))], timeseries090741[np.where(timeseries090741 > 0)], label = "100.16 kHz")

plt.plot(time0907[np.where(timeseries090745 > 0)], timeseries090745[np.where(timeseries090745 > 0)], label = "140.140 kHz")

plt.plot(time0907[np.where(timeseries090748 > 0)], timeseries090748[np.where(timeseries090748 > 0)], label = "198.240 kHz")
                        


#Set plot/axis titles 
plt.title('Time Series - bKOM (DoY 2016 248-253)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (DOY 2016)', fontsize = 20)  

plt.xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
plt.ylim(1e-17,1e-12)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%-j")
ax.xaxis.set_major_formatter(date_form_orb)
ax.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))


plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[39]:


#VISIBLE FREQUENCIES ON AUTOPLOT - choose 10, 39, 79, 100, 126, 140.14
#day 248-253 compression - good example of compression - backed up by MP crossings
#frequencies 10-200 kHz to match autoplot
time0907 = time[852480:887040]
timeseries090720 = timeseries[852480:887040,20]
timeseries090733 = timeseries[852480:887040,33]
timeseries090739 = timeseries[852480:887040,39]
timeseries090741 = timeseries[852480:887040,41]
timeseries090745 = timeseries[852480:887040,45]
timeseries090743 = timeseries[852480:887040,43]


fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111) 

plt.plot(time0907[(np.where(timeseries090720 > 0))], timeseries090720[(np.where(timeseries090720 > 0))], label = "10.010 kHz")

plt.plot(time0907[np.where(timeseries090733 > 0)], timeseries090733[np.where(timeseries090733 > 0)], label = "39.917 kHz")

plt.plot(time0907[np.where(timeseries090739 > 0)], timeseries090739[np.where(timeseries090739 > 0)], label = "79.468 kHz")

plt.plot(time0907[(np.where(timeseries090741 > 0))], timeseries090741[np.where(timeseries090741 > 0)], label = "100.160 kHz")

plt.plot(time0907[np.where(timeseries090743 > 0)], timeseries090743[np.where(timeseries090743 > 0)], label = "126.160 kHz")

plt.plot(time0907[np.where(timeseries090745 > 0)], timeseries090745[np.where(timeseries090745 > 0)], label = "140.140 kHz")

                        


#Set plot/axis titles 
plt.title('Time Series - bKOM (DoY 2016 248-253)', fontsize = 20, weight='bold')
plt.ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)  
plt.xlabel('Time (DOY 2016)', fontsize = 20)  

plt.xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
plt.ylim(1e-17,1e-12)

#plt.legend(loc = 'upper right', prop={"size":18})
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Radio emission frequencies", title_fontsize = '18', prop={"size":16}, loc='upper left')

plt.semilogy()

date_form_orb = mdates.DateFormatter("%-j")
ax.xaxis.set_major_formatter(date_form_orb)
ax.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))


plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[40]:


#day 248-253 compression -- good compression example - magnetopause crossing data to back up - lin scale
fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(20,15))

ax1.plot(time0907[(np.where(timeseries090720 > 0))], timeseries090720[(np.where(timeseries090720 > 0))], c = 'tab:blue',label = "10.010 kHz")

ax2.plot(time0907[np.where(timeseries090733 > 0)], timeseries090733[np.where(timeseries090733 > 0)], c = 'tab:orange',label = "39.917 kHz")
 
ax3.plot(time0907[np.where(timeseries090739 > 0)], timeseries090739[np.where(timeseries090739 > 0)], c = 'tab:green',label = "79.468 kHz")
                          
ax4.plot(time0907[(np.where(timeseries090741 > 0))], timeseries090741[np.where(timeseries090741 > 0)], c = 'tab:red',label = "100.160 kHz")

ax5.plot(time0907[np.where(timeseries090743 > 0)], timeseries090743[np.where(timeseries090743 > 0)], c = 'tab:purple',label = "126.160 kHz")

ax6.plot(time0907[np.where(timeseries090745 > 0)], timeseries090745[np.where(timeseries090745 > 0)], c = 'tab:cyan',label = "140.140 kHz")

plt.tight_layout(pad=5.0, h_pad=3.0, w_pad=3.0)

#Set plot titles
fig.suptitle('Time Series - bKOM (DoY 2016 248-253)', fontsize = 20, weight='bold')
ax1.set_title('(a) 10.010 kHz', fontsize = 20, weight='bold')
ax2.set_title('(b) 39.917 kHz', fontsize = 20, weight='bold')
ax3.set_title('(c) 79.468 kHz', fontsize = 20, weight='bold')
ax4.set_title('(d) 100.160 kHz', fontsize = 20, weight='bold')
ax5.set_title('(e) 126.160 kHz', fontsize = 20, weight='bold')
ax6.set_title('(f) 140.140 kHz', fontsize = 20, weight='bold')


#y-labels
ax1.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)
ax4.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)

#x-labels
ax4.set_xlabel('Time (DOY 2016)', fontsize = 20)
ax5.set_xlabel('Time (DOY 2016)', fontsize = 20)
ax6.set_xlabel('Time (DOY 2016)', fontsize = 20)

ax1.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax2.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax3.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax4.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax5.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax6.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))

date_form_orb = mdates.DateFormatter("%-j")
ax1.xaxis.set_major_formatter(date_form_orb)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax2.xaxis.set_major_formatter(date_form_orb)
ax2.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax3.xaxis.set_major_formatter(date_form_orb)
ax3.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax3.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax4.xaxis.set_major_formatter(date_form_orb)
ax4.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax4.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax5.xaxis.set_major_formatter(date_form_orb)
ax5.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax5.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax6.xaxis.set_major_formatter(date_form_orb)
ax6.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax6.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.yaxis.offsetText.set_fontsize(20)

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.yaxis.offsetText.set_fontsize(20)

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)
ax3.yaxis.offsetText.set_fontsize(20)

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)
ax4.yaxis.offsetText.set_fontsize(20)

ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)
ax5.yaxis.offsetText.set_fontsize(20)

ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)
ax6.yaxis.offsetText.set_fontsize(20)

#plt.xticks(fontsize = 20)
#plt.yticks(fontsize = 20)

plt.show()


# In[43]:


#day 248-253 compression -- good compression example - magnetopause crossing data to back up - log scale
fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(20,15))

ax1.plot(time0907[(np.where(timeseries090720 > 0))], timeseries090720[(np.where(timeseries090720 > 0))], c = 'tab:blue',label = "10.010 kHz")

ax2.plot(time0907[np.where(timeseries090733 > 0)], timeseries090733[np.where(timeseries090733 > 0)], c = 'tab:orange',label = "39.917 kHz")
 
ax3.plot(time0907[np.where(timeseries090739 > 0)], timeseries090739[np.where(timeseries090739 > 0)], c = 'tab:green',label = "79.468 kHz")
                          
ax4.plot(time0907[(np.where(timeseries090741 > 0))], timeseries090741[np.where(timeseries090741 > 0)], c = 'tab:red',label = "100.160 kHz")

ax5.plot(time0907[np.where(timeseries090743 > 0)], timeseries090743[np.where(timeseries090743 > 0)], c = 'tab:purple',label = "126.160 kHz")

ax6.plot(time0907[np.where(timeseries090745 > 0)], timeseries090745[np.where(timeseries090745 > 0)], c = 'tab:cyan',label = "140.140 kHz")

plt.tight_layout(pad=5.0, h_pad=3.0, w_pad=3.0)

#Set plot titles
fig.suptitle('Time Series - bKOM (DoY 2016 248-253)', fontsize = 20, weight='bold')
ax1.set_title('(a) 10.010 kHz', fontsize = 20, weight='bold')
ax2.set_title('(b) 39.917 kHz', fontsize = 20, weight='bold')
ax3.set_title('(c) 79.468 kHz', fontsize = 20, weight='bold')
ax4.set_title('(d) 100.160 kHz', fontsize = 20, weight='bold')
ax5.set_title('(e) 126.160 kHz', fontsize = 20, weight='bold')
ax6.set_title('(f) 140.140 kHz', fontsize = 20, weight='bold')


#y-labels
ax1.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)
ax4.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)

#x-labels
ax4.set_xlabel('Time (DOY 2016)', fontsize = 20)
ax5.set_xlabel('Time (DOY 2016)', fontsize = 20)
ax6.set_xlabel('Time (DOY 2016)', fontsize = 20)

ax1.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax2.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax3.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax4.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax5.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax6.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))

date_form_orb = mdates.DateFormatter("%-j")
ax1.xaxis.set_major_formatter(date_form_orb)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax2.xaxis.set_major_formatter(date_form_orb)
ax2.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax3.xaxis.set_major_formatter(date_form_orb)
ax3.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax3.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax4.xaxis.set_major_formatter(date_form_orb)
ax4.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax4.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax5.xaxis.set_major_formatter(date_form_orb)
ax5.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax5.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax6.xaxis.set_major_formatter(date_form_orb)
ax6.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax6.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.yaxis.offsetText.set_fontsize(20)

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.yaxis.offsetText.set_fontsize(20)

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)
ax3.yaxis.offsetText.set_fontsize(20)

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)
ax4.yaxis.offsetText.set_fontsize(20)

ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)
ax5.yaxis.offsetText.set_fontsize(20)

ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)
ax6.yaxis.offsetText.set_fontsize(20)

#plt.xticks(fontsize = 20)
#plt.yticks(fontsize = 20)

ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')

plt.show()


# In[37]:


#day 248-253 compression -- good compression example - magnetopause crossing data to back up - higher freqs.
#not visible on autoplot - interesting behaviour
fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,figsize=(20,15))

ax1.plot(time[852480:887040], timeseries[852480:887040,20], c = 'tab:blue',label = "10.010 kHz")

ax2.plot(time[852480:887040], timeseries[852480:887040,41], c = 'tab:orange',label = "100.16 kHz")
 
ax3.plot(time[852480:887040], timeseries[852480:887040,48], c = 'tab:green',label = "198.240 kHz")
                          
ax4.plot(time[852480:887040], timeseries[852480:887040,52], c = 'tab:red',label = "314.450 kHz")

ax5.plot(time[852480:887040], timeseries[852480:887040,56], c = 'tab:purple',label = "502.440 kHz")

ax6.plot(time[852480:887040], timeseries[852480:887040,60], c = 'tab:cyan',label = "796.390 kHz")

plt.tight_layout(pad=5.0, h_pad=3.0, w_pad=3.0)

#Set plot titles
fig.suptitle('Time Series - bKOM (Day 248-253, 2016)', fontsize = 20, weight='bold')
ax1.set_title('(a) 10.010 kHz', fontsize = 20, weight='bold')
ax2.set_title('(b) 100.16 kHz', fontsize = 20, weight='bold')
ax3.set_title('(c) 198.240 kHz', fontsize = 20, weight='bold')
ax4.set_title('(d) 314.450 kHz', fontsize = 20, weight='bold')
ax5.set_title('(e) 502.440 kHz', fontsize = 20, weight='bold')
ax6.set_title('(f) 796.390 kHz', fontsize = 20, weight='bold')


#y-labels
ax1.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)
ax4.set_ylabel('Intensity ($v^2/m^2/Hz$)', fontsize = 20)

#x-labels
ax4.set_xlabel('Time (DOY 2016)', fontsize = 20)
ax5.set_xlabel('Time (DOY 2016)', fontsize = 20)
ax6.set_xlabel('Time (DOY 2016)', fontsize = 20)

ax1.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax2.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax3.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax4.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax5.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))
ax6.set_xlim(datetime.date(2016, 9, 4), datetime.date(2016, 9, 10))

date_form_orb = mdates.DateFormatter("%-j")
ax1.xaxis.set_major_formatter(date_form_orb)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax2.xaxis.set_major_formatter(date_form_orb)
ax2.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax3.xaxis.set_major_formatter(date_form_orb)
ax3.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax3.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax4.xaxis.set_major_formatter(date_form_orb)
ax4.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax4.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax5.xaxis.set_major_formatter(date_form_orb)
ax5.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax5.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax6.xaxis.set_major_formatter(date_form_orb)
ax6.xaxis.set_major_locator(mdates.DayLocator(interval = 1))
ax6.xaxis.set_minor_locator(mdates.HourLocator(interval = 6))

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.yaxis.offsetText.set_fontsize(20)

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.yaxis.offsetText.set_fontsize(20)

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)
ax3.yaxis.offsetText.set_fontsize(20)

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)
ax4.yaxis.offsetText.set_fontsize(20)

ax5.xaxis.set_tick_params(labelsize=20)
ax5.yaxis.set_tick_params(labelsize=20)
ax5.yaxis.offsetText.set_fontsize(20)

ax6.xaxis.set_tick_params(labelsize=20)
ax6.yaxis.set_tick_params(labelsize=20)
ax6.yaxis.offsetText.set_fontsize(20)

#plt.xticks(fontsize = 20)
#plt.yticks(fontsize = 20)

plt.show()


# In[ ]:





# In[36]:


time0907 = time[864000:869760]
timeseries090720 = timeseries[864000:869760,20]
timeseries090741 = timeseries[864000:869760,41]
timeseries090748 = timeseries[864000:869760,48]
timeseries090752 = timeseries[864000:869760,52]
timeseries090756 = timeseries[864000:869760,56]
timeseries090760 = timeseries[864000:869760,60]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




