#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import numpy and matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from matplotlib.widgets import Slider
import matplotlib.image as mpimg


# In[4]:


matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('xtick', labelsize=20) 


# In[5]:


#Tick intervals
#Annually
#yearly = mdates.YearLocator(interval = 1) # ticks with label every 1 year

#Every 6 months
sixmonthly = mdates.MonthLocator(interval = 6) # ticks with label every 6 months

#Every month
monthly = mdates.MonthLocator(interval = 1) # ticks with label every 1 month

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


# In[6]:


#import data
jupmag = np.genfromtxt('juno_fgm_b_mag_2016-2020.txt', dtype=str)
#import times and total amplitude of the magnetic field. 
amda_time = jupmag[0:, 0].astype(str)
juno_mag = jupmag[0:, 1]
juno_mag = juno_mag.astype(float)


# In[7]:


#Change data type to datetime
time_table = [datetime.datetime.strptime(i,"%Y-%m-%dT%H:%M:%S.%f") for i in amda_time]


# # MAGNETOPAUSE

# # Example of lots of magnetopause crossings over short period of time: Orbit 1

# ## 07/14 - 07/16

# In[5]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig27 = plt.figure(figsize=(20, 10),)
ax27 = fig27.add_subplot(1,1,1) 

#Plot
ax27.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax27.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax27.set_xlim(datetime.datetime(2016, 7, 14, 0, 0), datetime.datetime(2016, 7, 17, 0, 0))
ax27.set_ylim([-1,12])

fig27.tight_layout()

#Times of magnetopause crossings
ax27.axvline(datetime.datetime(2016, 7, 14, 21, 18), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax27.axvline(datetime.datetime(2016, 7, 15, 7, 16), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')
ax27.axvline(datetime.datetime(2016, 7, 15, 17, 6), color='k', linestyle='dashed', linewidth=1.5)
ax27.axvline(datetime.datetime(2016, 7, 15, 19, 11), color='k', linewidth=1.5)
ax27.axvline(datetime.datetime(2016, 7, 16, 23, 7), color='k', linestyle='dashed', linewidth=1.5)

ax27.set_title('Total Magnetic field Amplitude - 07/14/2016-07/16/2016', fontsize = 18, weight='bold')

#Legend
ax27.legend()

ax27.xaxis.set_major_locator(sixhourly)
ax27.xaxis.set_minor_locator(twohourly)

ax27.grid()

plt.show()


# ## 07/19 - 07/25

# In[6]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig28 = plt.figure(figsize=(20, 10),)
ax28 = fig28.add_subplot(1,1,1) 

#Plot
ax28.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax28.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax28.set_xlim(datetime.datetime(2016, 7, 19, 0, 0), datetime.datetime(2016, 7, 26, 0, 0))
ax28.set_ylim([-1,12])

fig28.tight_layout()

#Times of magnetopause crossings
ax28.axvline(datetime.datetime(2016, 7, 19, 17, 26), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')
ax28.axvline(datetime.datetime(2016, 7, 19, 20, 43), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')

ax28.axvline(datetime.datetime(2016, 7, 20, 9, 18), color='k', linewidth=1.5)
ax28.axvline(datetime.datetime(2016, 7, 20, 19, 16), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')

ax28.axvline(datetime.datetime(2016, 7, 21, 1, 49), color='k', linewidth=1.5)
ax28.axvline(datetime.datetime(2016, 7, 21, 2, 2), color='k', linestyle='dashed', linewidth=1.5)

ax28.axvline(datetime.datetime(2016, 7, 22, 14, 31), color='k', linewidth=1.5)
ax28.axvline(datetime.datetime(2016, 7, 23, 14, 6), color='k', linestyle='dashed', linewidth=1.5)

ax28.axvline(datetime.datetime(2016, 7, 23, 19, 54), color='k', linewidth=1.5)
ax28.axvline(datetime.datetime(2016, 7, 24, 18, 47), color='k', linestyle='dashed', linewidth=1.5)

ax28.axvline(datetime.datetime(2016, 7, 25, 1, 12), color='k', linewidth=1.5)
ax28.axvline(datetime.datetime(2016, 7, 25, 14, 33), color='k', linestyle='dashed', linewidth=1.5)



ax28.set_title('Total Magnetic field Amplitude - 07/19/2016-07/25/2016', fontsize = 18, weight='bold')

#Legend
ax28.legend()

ax28.xaxis.set_major_locator(twelvehourly)
ax28.xaxis.set_minor_locator(sixhourly)

ax28.grid()

plt.show()


# ## 08/01 - 08/02

# In[12]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig29 = plt.figure(figsize=(20, 10),)
ax29 = fig29.add_subplot(1,1,1) 

#Plot
ax29.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax29.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax29.set_xlim(datetime.datetime(2016, 8, 1, 0, 0), datetime.datetime(2016, 8, 3, 0, 0))
ax29.set_ylim([-1,12])

fig29.tight_layout()

#Times of magnetopause crossings
ax29.axvline(datetime.datetime(2016, 8, 1, 19, 36), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax29.axvline(datetime.datetime(2016, 8, 1, 20, 57), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax29.axvline(datetime.datetime(2016, 8, 1, 21, 55), color='k', linestyle='dashed', linewidth=1.5)
ax29.axvline(datetime.datetime(2016, 8, 2, 8, 13), color='k', linewidth=1.5)

ax29.axvline(datetime.datetime(2016, 8, 2, 10, 50), color='k', linestyle='dashed', linewidth=1.5)
ax29.axvline(datetime.datetime(2016, 8, 2, 13, 59), color='k', linewidth=1.5)

ax29.axvline(datetime.datetime(2016, 8, 2, 22, 12), color='k', linestyle='dashed', linewidth=1.5)
ax29.axvline(datetime.datetime(2016, 8, 2, 22, 49), color='k', linewidth=1.5)


ax29.set_title('Total Magnetic field Amplitude - 08/01/2016-08/02/2016', fontsize = 18, weight='bold')

#Legend
ax29.legend()

#ax29.xaxis.set_major_locator(twelvehourly)
#ax29.xaxis.set_minor_locator(hourly)

date_form_orb = mdates.DateFormatter("%H:%M")
ax29.xaxis.set_major_formatter(date_form_orb)
ax29.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 15))

ax29.grid()

plt.show()


# In[22]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
#08/02 / Day 215 2016
fig29 = plt.figure(figsize=(20, 10),)
ax29 = fig29.add_subplot(1,1,1) 

#Plot
ax29.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Time (HH:MM)', fontsize = 17)

#Y-Label
ax29.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax29.set_xlim(datetime.datetime(2016, 8, 2, 0, 0), datetime.datetime(2016, 8, 3, 0, 0))
ax29.set_ylim([-1,6])

fig29.tight_layout()

#Times of magnetopause crossings

ax29.axvline(datetime.datetime(2016, 8, 2, 8, 13), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax29.axvline(datetime.datetime(2016, 8, 2, 10, 50), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax29.axvline(datetime.datetime(2016, 8, 2, 13, 59), color='k', linewidth=1.5)

ax29.axvline(datetime.datetime(2016, 8, 2, 22, 12), color='k', linestyle='dashed', linewidth=1.5)
ax29.axvline(datetime.datetime(2016, 8, 2, 22, 49), color='k', linewidth=1.5)


ax29.set_title('Total Magnetic field Amplitude - Day 215,2016', fontsize = 18, weight='bold')

#Legend
ax29.legend(prop={"size":18})

#ax29.xaxis.set_major_locator(twelvehourly)
#ax29.xaxis.set_minor_locator(hourly)

date_form_orb = mdates.DateFormatter("%H:%M")
ax29.xaxis.set_major_formatter(date_form_orb)
ax29.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 15))

ax29.grid()

plt.show()


# ## 08/06 - 08/08

# In[8]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig30 = plt.figure(figsize=(20, 10),)
ax30 = fig30.add_subplot(1,1,1) 

#Plot
ax30.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax30.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax30.set_xlim(datetime.datetime(2016, 8, 6, 0, 0), datetime.datetime(2016, 8, 9, 0, 0))
ax30.set_ylim([-1,12])

fig30.tight_layout()

#Times of magnetopause crossings
ax30.axvline(datetime.datetime(2016, 8, 6, 19, 56), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax30.axvline(datetime.datetime(2016, 8, 6, 23, 50), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax30.axvline(datetime.datetime(2016, 8, 7, 1, 20), color='k', linestyle='dashed', linewidth=1.5)
ax30.axvline(datetime.datetime(2016, 8, 8, 8, 44), color='k', linewidth=1.5)

ax30.axvline(datetime.datetime(2016, 8, 8, 14, 35), color='k', linestyle='dashed', linewidth=1.5)
ax30.axvline(datetime.datetime(2016, 8, 8, 15, 26), color='k', linewidth=1.5)

ax30.axvline(datetime.datetime(2016, 8, 8, 15, 38), color='k', linestyle='dashed', linewidth=1.5)


ax30.set_title('Total Magnetic field Amplitude - 08/06/2016-08/08/2016', fontsize = 18, weight='bold')

#Legend
ax30.legend()

ax30.xaxis.set_major_locator(twelvehourly)
ax30.xaxis.set_minor_locator(hourly)

ax30.grid()

plt.show()


# ## 08/11 - 08/12

# In[9]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig31 = plt.figure(figsize=(20, 10),)
ax31 = fig31.add_subplot(1,1,1) 

#Plot
ax31.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax31.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax31.set_xlim(datetime.datetime(2016, 8, 11, 0, 0), datetime.datetime(2016, 8, 13, 0, 0))
ax31.set_ylim([-1,12])

fig31.tight_layout()

#Times of magnetopause crossings
ax31.axvline(datetime.datetime(2016, 8, 11, 17, 12), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')
ax31.axvline(datetime.datetime(2016, 8, 11, 23, 25), color='k', linestyle='dashed',linewidth=1.5, label = 'Magnetopause crossing (Outward)')

ax31.axvline(datetime.datetime(2016, 8, 11, 3, 56), color='k', linewidth=1.5)
ax31.axvline(datetime.datetime(2016, 8, 12, 10, 5), color='k', linestyle='dashed',linewidth=1.5)

ax31.axvline(datetime.datetime(2016, 8, 12, 12, 39), color='k', linewidth=1.5)
ax31.axvline(datetime.datetime(2016, 8, 12, 21, 11), color='k', linestyle='dashed')

ax31.axvline(datetime.datetime(2016, 8, 12, 23, 55), color='k', linewidth=1.5)


ax31.set_title('Total Magnetic field Amplitude - 08/11/2016-08/12/2016', fontsize = 18, weight='bold')

#Legend
ax31.legend()

ax31.xaxis.set_major_locator(twelvehourly)
ax31.xaxis.set_minor_locator(hourly)

ax31.grid()

plt.show()


# # Example of lots of magnetopause crossings over short period of time: Orbit 2

# ## 09/06 - 09/07

# In[10]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig32 = plt.figure(figsize=(20, 10),)
ax32 = fig32.add_subplot(1,1,1) 

#Plot
ax32.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax32.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax32.set_xlim(datetime.datetime(2016, 9, 6, 0, 0), datetime.datetime(2016, 9, 8, 0, 0))
ax32.set_ylim([-1,12])

fig32.tight_layout()

#Times of magnetopause crossings
ax32.axvline(datetime.datetime(2016, 9, 6, 11, 2), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax32.axvline(datetime.datetime(2016, 9, 6, 21, 47), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax32.axvline(datetime.datetime(2016, 9, 7, 3, 15), color='k', linestyle='dashed', linewidth=1.5)


ax32.set_title('Total Magnetic field Amplitude - 09/06/2016-09/07/2016', fontsize = 18, weight='bold')

#Legend
ax32.legend()

ax32.xaxis.set_major_locator(twelvehourly)
ax32.xaxis.set_minor_locator(hourly)

ax32.grid()

plt.show()


# ## 09/19 - 09/20

# In[11]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig33 = plt.figure(figsize=(20, 10),)
ax33 = fig33.add_subplot(1,1,1) 

#Plot
ax33.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax33.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax33.set_xlim(datetime.datetime(2016, 9, 19, 0, 0), datetime.datetime(2016, 9, 21, 0, 0))
ax33.set_ylim([-1,12])

fig33.tight_layout()

#Times of magnetopause crossings
ax33.axvline(datetime.datetime(2016, 9, 19, 7, 5), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax33.axvline(datetime.datetime(2016, 9, 19, 20, 41), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax33.axvline(datetime.datetime(2016, 9, 20, 7, 5), color='k', linestyle='dashed', linewidth=1.5)
ax33.axvline(datetime.datetime(2016, 9, 20, 13, 20), color='k', linewidth=1.5)

ax33.set_title('Total Magnetic field Amplitude - 09/19/2016-09/20/2016', fontsize = 18, weight='bold')

#Legend
ax33.legend()

ax33.xaxis.set_major_locator(twelvehourly)
ax33.xaxis.set_minor_locator(hourly)

ax33.grid()

plt.show()


# ## 09/22 - 09/23

# In[12]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig34 = plt.figure(figsize=(20, 10),)
ax34 = fig34.add_subplot(1,1,1) 

#Plot
ax34.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax34.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax34.set_xlim(datetime.datetime(2016, 9, 22, 0, 0), datetime.datetime(2016, 9, 24, 0, 0))
ax34.set_ylim([-1,12])

fig34.tight_layout()

#Times of magnetopause crossings
ax34.axvline(datetime.datetime(2016, 9, 22, 2, 28), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax34.axvline(datetime.datetime(2016, 9, 22, 9, 40), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax34.axvline(datetime.datetime(2016, 9, 23, 18, 9), color='k', linestyle='dashed', linewidth=1.5)
ax34.axvline(datetime.datetime(2016, 9, 23, 21, 5), color='k', linewidth=1.5)

ax34.set_title('Total Magnetic field Amplitude - 09/22/2016-09/23/2016', fontsize = 18, weight='bold')

#Legend
ax34.legend()

ax34.xaxis.set_major_locator(twelvehourly)
ax34.xaxis.set_minor_locator(hourly)

ax34.grid()

plt.show()


# ## 09/28 - 09/30

# In[13]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig35 = plt.figure(figsize=(20, 10),)
ax35 = fig35.add_subplot(1,1,1) 

#Plot
ax35.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax35.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax35.set_xlim(datetime.datetime(2016, 9, 28, 0, 0), datetime.datetime(2016, 10, 1, 0, 0))
ax35.set_ylim([-1,12])

fig35.tight_layout()

#Times of magnetopause crossings
ax35.axvline(datetime.datetime(2016, 9, 28, 1, 35), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax35.axvline(datetime.datetime(2016, 9, 29, 13, 5), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax35.axvline(datetime.datetime(2016, 9, 30, 13, 17), color='k', linestyle='dashed', linewidth=1.5)
ax35.axvline(datetime.datetime(2016, 9, 30, 16, 36), color='k', linewidth=1.5)

ax35.set_title('Total Magnetic field Amplitude - 09/28/2016-09/30/2016', fontsize = 18, weight='bold')

#Legend
ax35.legend()

ax35.xaxis.set_major_locator(twelvehourly)
ax35.xaxis.set_minor_locator(hourly)

ax35.grid()

plt.show()


# ## 09/30

# In[14]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig46 = plt.figure(figsize=(20, 10),)
ax46 = fig46.add_subplot(1,1,1) 

#Plot
ax46.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax46.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax46.set_xlim(datetime.datetime(2016, 9, 30, 0, 0), datetime.datetime(2016, 10, 1, 0, 0))
ax46.set_ylim([-1,12])

fig46.tight_layout()

#Times of magnetopause crossings
ax46.axvline(datetime.datetime(2016, 9, 30, 13, 17), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax46.axvline(datetime.datetime(2016, 9, 30, 16, 36), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax46.set_title('Total Magnetic field Amplitude - 09/30/2016', fontsize = 18, weight='bold')

#Legend
ax46.legend()

ax46.xaxis.set_major_locator(sixhourly)
ax46.xaxis.set_minor_locator(hourly)

ax46.grid()

plt.show()


# ## 10/01 - 10/02

# In[15]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig36 = plt.figure(figsize=(20, 10),)
ax36 = fig36.add_subplot(1,1,1) 

#Plot
ax36.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax36.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax36.set_xlim(datetime.datetime(2016, 10, 1, 0, 0), datetime.datetime(2016, 10, 2, 12, 0))
ax36.set_ylim([-1,12])

fig36.tight_layout()

#Times of magnetopause crossings
ax36.axvline(datetime.datetime(2016, 10, 1, 3, 54), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax36.axvline(datetime.datetime(2016, 10, 1, 5, 40), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax36.axvline(datetime.datetime(2016, 10, 1, 17, 48), color='k', linestyle='dashed', linewidth=1.5)
ax36.axvline(datetime.datetime(2016, 10, 1, 21, 44), color='k', linewidth=1.5)

ax36.axvline(datetime.datetime(2016, 10, 2, 1, 4), color='k', linestyle='dashed', linewidth=1.5)

ax36.set_title('Total Magnetic field Amplitude - 10/1/2016-10/2/2016', fontsize = 18, weight='bold')

#Legend
ax36.legend()

ax36.xaxis.set_major_locator(twelvehourly)
ax36.xaxis.set_minor_locator(hourly)

ax36.grid()

plt.show()


# # Example of lots of magnetopause crossings over short period of time: Orbit 3

# ## 11/07 - 11/09

# In[16]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig37 = plt.figure(figsize=(20, 10),)
ax37 = fig37.add_subplot(1,1,1) 

#Plot
ax37.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax37.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax37.set_xlim(datetime.datetime(2016, 11, 7, 0, 0), datetime.datetime(2016, 11, 10, 0, 0))
ax37.set_ylim([-1,12])

fig37.tight_layout()

#Times of magnetopause crossings
ax37.axvline(datetime.datetime(2016, 11, 7, 23, 46), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')
ax37.axvline(datetime.datetime(2016, 11, 8, 9, 2), color='k', linestyle='dashed',linewidth=1.5, label = 'Magnetopause crossing (Outward)')

ax37.axvline(datetime.datetime(2016, 11, 8, 11, 37), color='k', linewidth=1.5)
ax37.axvline(datetime.datetime(2016, 11, 9, 17, 32), color='k', linestyle='dashed',linewidth=1.5)

ax37.axvline(datetime.datetime(2016, 11, 9, 18, 9), color='k', linewidth=1.5)



ax37.set_title('Total Magnetic field Amplitude - 11/07/2016-11/09/2016', fontsize = 18, weight='bold')

#Legend
ax37.legend()

ax37.xaxis.set_major_locator(twelvehourly)
ax37.xaxis.set_minor_locator(hourly)

ax37.grid()

plt.show()


# ## 11/13 - 11/14

# In[17]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig38 = plt.figure(figsize=(20, 10),)
ax38 = fig38.add_subplot(1,1,1) 

#Plot
ax38.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax38.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax38.set_xlim(datetime.datetime(2016, 11, 13, 0, 0), datetime.datetime(2016, 11, 15, 0, 0))
ax38.set_ylim([-1,12])

fig38.tight_layout()

#Times of magnetopause crossings
ax38.axvline(datetime.datetime(2016, 11, 13, 8, 28), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')
ax38.axvline(datetime.datetime(2016, 11, 13, 15, 8), color='k', linestyle='dashed',linewidth=1.5, label = 'Magnetopause crossing (Outward)')

ax38.axvline(datetime.datetime(2016, 11, 13, 19, 55), color='k', linewidth=1.5)
ax38.axvline(datetime.datetime(2016, 11, 13, 21, 40), color='k', linestyle='dashed',linewidth=1.5)

ax38.axvline(datetime.datetime(2016, 11, 14, 0, 45), color='k', linewidth=1.5)
ax38.axvline(datetime.datetime(2016, 11, 14, 5, 4), color='k', linestyle='dashed',linewidth=1.5)

ax38.axvline(datetime.datetime(2016, 11, 14, 10, 55), color='k', linewidth=1.5)

ax38.set_title('Total Magnetic field Amplitude - 11/13/2016-11/14/2016', fontsize = 18, weight='bold')

#Legend
ax38.legend()

ax38.xaxis.set_major_locator(twelvehourly)
ax38.xaxis.set_minor_locator(hourly)

ax38.grid()

plt.show()


# ## 11/19 

# In[18]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig39 = plt.figure(figsize=(20, 10),)
ax39 = fig39.add_subplot(1,1,1) 

#Plot
ax39.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax39.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax39.set_xlim(datetime.datetime(2016, 11, 19, 0, 0), datetime.datetime(2016, 11, 20, 0, 0))
ax39.set_ylim([-1,12])

fig39.tight_layout()

#Times of magnetopause crossings
ax39.axvline(datetime.datetime(2016, 11, 19, 6, 12), color='k', linestyle='dashed',linewidth=1.5, label = 'Magnetopause crossing (Outward)')
ax39.axvline(datetime.datetime(2016, 11, 19, 11, 24), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')

ax39.set_title('Total Magnetic field Amplitude - 11/19/2016', fontsize = 18, weight='bold')

#Legend
ax39.legend()

ax39.xaxis.set_major_locator(sixhourly)
ax39.xaxis.set_minor_locator(hourly)

ax39.grid()

plt.show()


# ## 11/24 - 11/26

# In[19]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig39 = plt.figure(figsize=(20, 10),)
ax39 = fig39.add_subplot(1,1,1) 

#Plot
ax39.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax39.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax39.set_xlim(datetime.datetime(2016, 11, 24, 0, 0), datetime.datetime(2016, 11, 26, 12, 0))
ax39.set_ylim([-1,12])

fig39.tight_layout()

#Times of magnetopause crossings
ax39.axvline(datetime.datetime(2016, 11, 24, 4, 1), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')
ax39.axvline(datetime.datetime(2016, 11, 25, 12, 22), color='k', linestyle='dashed',linewidth=1.5, label = 'Magnetopause crossing (Outward)')

ax39.axvline(datetime.datetime(2016, 11, 25, 12, 54), color='k', linewidth=1.5)
ax39.axvline(datetime.datetime(2016, 11, 25, 13, 13), color='k', linestyle='dashed',linewidth=1.5)

ax39.axvline(datetime.datetime(2016, 11, 25, 13, 16), color='k', linewidth=1.5)
ax39.axvline(datetime.datetime(2016, 11, 25, 13, 29), color='k', linestyle='dashed',linewidth=1.5)

ax39.axvline(datetime.datetime(2016, 11, 25, 14, 40), color='k', linewidth=1.5)
ax39.axvline(datetime.datetime(2016, 11, 25, 14, 52), color='k', linestyle='dashed',linewidth=1.5)

ax39.axvline(datetime.datetime(2016, 11, 25, 17, 40), color='k', linewidth=1.5)
ax39.axvline(datetime.datetime(2016, 11, 25, 20, 44), color='k', linestyle='dashed',linewidth=1.5)

ax39.set_title('Total Magnetic field Amplitude - 11/24/2016-11/25/2016', fontsize = 18, weight='bold')

#Legend
ax39.legend()

ax39.xaxis.set_major_locator(twelvehourly)
ax39.xaxis.set_minor_locator(hourly)

ax39.grid()

plt.show()


# ## 11/24

# In[20]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig39 = plt.figure(figsize=(20, 10),)
ax39 = fig39.add_subplot(1,1,1) 

#Plot
ax39.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax39.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax39.set_xlim(datetime.datetime(2016, 11, 24, 0, 0), datetime.datetime(2016, 11, 25, 0, 0))
ax39.set_ylim([-1,12])

fig39.tight_layout()

#Times of magnetopause crossings
ax39.axvline(datetime.datetime(2016, 11, 24, 4, 1), color='k', linewidth=1.5, label = 'Magnetopause crossing (Inward)')


ax39.set_title('Total Magnetic field Amplitude - 11/24/2016', fontsize = 18, weight='bold')

#Legend
ax39.legend()

ax39.xaxis.set_major_locator(twelvehourly)
ax39.xaxis.set_minor_locator(hourly)

ax39.grid()

plt.show()


# # BOW SHOCK

# # Example of lots of bow shock crossings over short period of time: Orbit 1

# ## 07/17 - 07/18

# In[21]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig40 = plt.figure(figsize=(20, 10),)
ax40 = fig40.add_subplot(1,1,1) 

#Plot
ax40.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax40.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax40.set_xlim(datetime.datetime(2016, 7, 17, 0, 0), datetime.datetime(2016, 7, 19, 0, 0))
ax40.set_ylim([-1,12])

fig40.tight_layout()

#Times of bow shock crossings
ax40.axvline(datetime.datetime(2016, 7, 17, 15, 33), color='k', linestyle='dashed',linewidth=1.5, label = 'Bow shock crossing (Outward)')
ax40.axvline(datetime.datetime(2016, 7, 18, 1, 22), color='k', label = 'Bow shock crossing (Inward)')

ax40.axvline(datetime.datetime(2016, 7, 18, 2, 21), color='k', linestyle='dashed', linewidth=1.5)
ax40.axvline(datetime.datetime(2016, 7, 18, 21, 18), color='k',linewidth=1.5)


ax40.set_title('Total Magnetic field Amplitude - 07/17/2016-07/18/2016', fontsize = 18, weight='bold')

#Legend
ax40.legend()

ax40.xaxis.set_major_locator(twelvehourly)
ax40.xaxis.set_minor_locator(hourly)

ax40.grid()

plt.show()


# ## 07/27 - 07/28 (Orbit 1 into Orbit 2)

# In[22]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig41 = plt.figure(figsize=(20, 10),)
ax41 = fig41.add_subplot(1,1,1) 

#Plot
ax41.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax41.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax41.set_xlim(datetime.datetime(2016, 7, 27, 0, 0), datetime.datetime(2016, 7, 29, 0, 0))
ax41.set_ylim([-1,12])

fig41.tight_layout()

#Times of bow shock crossings
ax41.axvline(datetime.datetime(2016, 7, 27, 9, 34), color='k', linestyle='dashed',linewidth=1.5, label = 'Bow shock crossing (Outward)')
ax41.axvline(datetime.datetime(2016, 7, 27, 9, 37), color='k', label = 'Bow shock crossing (Inward)')

ax41.axvline(datetime.datetime(2016, 7, 27, 9, 47), color='k', linestyle='dashed', linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 27, 11, 38), color='k',linewidth=1.5)

ax41.axvline(datetime.datetime(2016, 7, 27, 11, 45), color='k', linestyle='dashed',linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 27, 11, 51), color='k')

ax41.axvline(datetime.datetime(2016, 7, 27, 23, 41), color='k', linestyle='dashed', linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 28, 10, 28), color='k',linewidth=1.5)

ax41.axvline(datetime.datetime(2016, 7, 28, 11, 11), color='k', linestyle='dashed',linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 28, 11, 36), color='k')

ax41.axvline(datetime.datetime(2016, 7, 28, 12, 3), color='k', linestyle='dashed', linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 28, 12, 10), color='k',linewidth=1.5)

ax41.axvline(datetime.datetime(2016, 7, 28, 17, 57), color='k', linestyle='dashed',linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 28, 20, 39), color='k')

ax41.axvline(datetime.datetime(2016, 7, 28, 20, 56), color='k', linestyle='dashed', linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 28, 21, 2), color='k',linewidth=1.5)

ax41.set_title('Total Magnetic field Amplitude - 07/27/2016-07/28/2016', fontsize = 18, weight='bold')

#Legend
ax41.legend()

ax41.xaxis.set_major_locator(twelvehourly)
ax41.xaxis.set_minor_locator(hourly)

ax41.grid()

plt.show()


# ## 07/27

# In[23]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig42 = plt.figure(figsize=(20, 10),)
ax42 = fig42.add_subplot(1,1,1) 

#Plot
ax42.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax42.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax42.set_xlim(datetime.datetime(2016, 7, 27, 0, 0), datetime.datetime(2016, 7, 28, 0, 0))
ax42.set_ylim([-1,12])

fig42.tight_layout()

#Times of bow shock crossings
ax42.axvline(datetime.datetime(2016, 7, 27, 9, 34), color='k', linestyle='dashed',linewidth=1.5, label = 'Bow shock crossing (Outward)')
ax42.axvline(datetime.datetime(2016, 7, 27, 9, 37), color='k', label = 'Bow shock crossing (Inward)')

ax42.axvline(datetime.datetime(2016, 7, 27, 9, 47), color='k', linestyle='dashed', linewidth=1.5)
ax42.axvline(datetime.datetime(2016, 7, 27, 11, 38), color='k',linewidth=1.5)

ax42.axvline(datetime.datetime(2016, 7, 27, 11, 45), color='k', linestyle='dashed',linewidth=1.5)
ax42.axvline(datetime.datetime(2016, 7, 27, 11, 51), color='k')

ax42.axvline(datetime.datetime(2016, 7, 27, 23, 41), color='k', linestyle='dashed', linewidth=1.5)

ax42.set_title('Total Magnetic field Amplitude - 07/27/2016', fontsize = 18, weight='bold')

#Legend
ax42.legend()

ax42.xaxis.set_major_locator(twelvehourly)
ax42.xaxis.set_minor_locator(hourly)

ax42.grid()

plt.show()


# ## 07/28

# In[21]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
#07/28 / Day 210 2016
fig41 = plt.figure(figsize=(20, 10),)
ax41 = fig41.add_subplot(1,1,1) 

#Plot
ax41.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Time (HH:MM)', fontsize = 17)

#Y-Label
ax41.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax41.set_xlim(datetime.datetime(2016, 7, 28, 0, 0), datetime.datetime(2016, 7, 29, 0, 0))
ax41.set_ylim([-1,6])

fig41.tight_layout()

#Times of bow shock crossings
ax41.axvline(datetime.datetime(2016, 7, 28, 10, 28), color='k',linewidth=1.5, label = 'Bow shock crossing (Inward)')

ax41.axvline(datetime.datetime(2016, 7, 28, 11, 11), color='k', linestyle='dashed',linewidth=1.5, label = 'Bow shock crossing (Outward)')
ax41.axvline(datetime.datetime(2016, 7, 28, 11, 36), color='k')

ax41.axvline(datetime.datetime(2016, 7, 28, 12, 3), color='k', linestyle='dashed', linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 28, 12, 10), color='k',linewidth=1.5)

ax41.axvline(datetime.datetime(2016, 7, 28, 17, 57), color='k', linestyle='dashed',linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 28, 20, 39), color='k')

ax41.axvline(datetime.datetime(2016, 7, 28, 20, 56), color='k', linestyle='dashed', linewidth=1.5)
ax41.axvline(datetime.datetime(2016, 7, 28, 21, 2), color='k',linewidth=1.5)

ax41.set_title('Total Magnetic field Amplitude - Day 210, 2016', fontsize = 18, weight='bold')

#Legend
ax41.legend(prop={"size":16})

#ax41.xaxis.set_major_locator(sixhourly)
#ax41.xaxis.set_minor_locator(hourly)

date_form_orb = mdates.DateFormatter("%H:%M")
ax41.xaxis.set_major_formatter(date_form_orb)
ax41.xaxis.set_minor_locator(mdates.MinuteLocator(interval = 15))

ax41.grid()

plt.show()


# ## 08/08 - 08/09

# In[25]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig43 = plt.figure(figsize=(20, 10),)
ax43 = fig43.add_subplot(1,1,1) 

#Plot
ax43.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax43.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax43.set_xlim(datetime.datetime(2016, 8, 8, 0, 0), datetime.datetime(2016, 8, 10, 0, 0))
ax43.set_ylim([-1,12])

fig43.tight_layout()

#Times of bow shock crossings
ax43.axvline(datetime.datetime(2016, 8, 8, 1, 39), color='k', linestyle='dashed',linewidth=1.5, label = 'Bow shock crossing (Outward)')
ax43.axvline(datetime.datetime(2016, 8, 8, 3, 39), color='k', label = 'Bow shock crossing (Inward)')

ax43.axvline(datetime.datetime(2016, 8, 8, 22, 40), color='k', linestyle='dashed', linewidth=1.5)
ax43.axvline(datetime.datetime(2016, 8, 9, 3, 22), color='k',linewidth=1.5)

ax43.axvline(datetime.datetime(2016, 8, 9, 16, 2), color='k', linestyle='dashed',linewidth=1.5)
ax43.axvline(datetime.datetime(2016, 8, 9, 20, 37), color='k')

ax43.axvline(datetime.datetime(2016, 8, 9, 23, 10), color='k', linestyle='dashed', linewidth=1.5)


ax43.set_title('Total Magnetic field Amplitude - 08/08/2016-08/09/2016', fontsize = 18, weight='bold')

#Legend
ax43.legend()

ax43.xaxis.set_major_locator(twelvehourly)
ax43.xaxis.set_minor_locator(hourly)

ax43.grid()

plt.show()


# ## 08/10 - 08/11

# In[26]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig44 = plt.figure(figsize=(20, 10),)
ax44 = fig44.add_subplot(1,1,1) 

#Plot
ax44.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax44.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax44.set_xlim(datetime.datetime(2016, 8, 10, 12, 0), datetime.datetime(2016, 8, 11, 6, 0))
ax44.set_ylim([-1,12])

fig44.tight_layout()

#Times of bow shock crossings
ax44.axvline(datetime.datetime(2016, 8, 10, 19, 45), color='k', label = 'Bow shock crossing (Inward)')
ax44.axvline(datetime.datetime(2016, 8, 10, 21, 56), color='k', linestyle='dashed',linewidth=1.5, label = 'Bow shock crossing (Outward)')

ax44.axvline(datetime.datetime(2016, 8, 10, 23, 39), color='k',linewidth=1.5)
ax44.axvline(datetime.datetime(2016, 8, 10, 23, 49), color='k', linestyle='dashed', linewidth=1.5)

ax44.axvline(datetime.datetime(2016, 8, 11, 0, 3), color='k',linewidth=1.5)
ax44.axvline(datetime.datetime(2016, 8, 11, 0, 12), color='k', linestyle='dashed', linewidth=1.5)

ax44.axvline(datetime.datetime(2016, 8, 11, 1, 38), color='k',linewidth=1.5)

ax44.set_title('Total Magnetic field Amplitude - 08/10/2016-08/11/2016', fontsize = 18, weight='bold')

#Legend
ax44.legend()

ax44.xaxis.set_major_locator(sixhourly)
ax44.xaxis.set_minor_locator(hourly)

ax44.grid()

plt.show()


# ## Orbit 3

# ## 11/11 - 00:00 - 06:00

# In[27]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig45 = plt.figure(figsize=(20, 10),)
ax45 = fig45.add_subplot(1,1,1) 

#Plot
ax45.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (MM-DD HH)', fontsize = 17)

#Y-Label
ax45.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax45.set_xlim(datetime.datetime(2016, 11, 11, 0, 0), datetime.datetime(2016, 11, 11, 6, 0))
ax45.set_ylim([-1,12])

fig45.tight_layout()

#Times of bow shock crossings
ax45.axvline(datetime.datetime(2016, 11, 11, 2, 36), color='k', linestyle='dashed',linewidth=1.5, label = 'Bow shock crossing (Outward)')
ax45.axvline(datetime.datetime(2016, 11, 11, 2, 42), color='k', label = 'Bow shock crossing (Inward)')

ax45.axvline(datetime.datetime(2016, 11, 11, 2, 58), color='k', linestyle='dashed', linewidth=1.5)
ax45.axvline(datetime.datetime(2016, 11, 11, 4, 17), color='k',linewidth=1.5)

ax45.axvline(datetime.datetime(2016, 11, 11, 4, 18), color='k', linestyle='dashed', linewidth=1.5)
ax45.axvline(datetime.datetime(2016, 11, 11, 4, 47), color='k',linewidth=1.5)

ax45.axvline(datetime.datetime(2016, 11, 11, 4, 49), color='k', linestyle='dashed', linewidth=1.5)
ax45.axvline(datetime.datetime(2016, 11, 11, 5, 1), color='k',linewidth=1.5)


ax45.set_title('Total Magnetic field Amplitude - 11/11/2016 (00:00-06:00)', fontsize = 18, weight='bold')

#Legend
ax45.legend()

ax45.xaxis.set_major_locator(hourly)
ax45.xaxis.set_minor_locator(halfhourly)

ax45.grid()

plt.show()


# ## 11/11 - 23:00-23:59

# In[28]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig45 = plt.figure(figsize=(20, 10),)
ax45 = fig45.add_subplot(1,1,1) 

#Plot
ax45.plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time (DD HH:MM)', fontsize = 17)

#Y-Label
ax45.set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim
ax45.set_xlim(datetime.datetime(2016, 11, 11, 23, 0), datetime.datetime(2016, 11, 12, 0, 0))
ax45.set_ylim([-1,12])

fig45.tight_layout()

#Times of bow shock crossings
ax45.axvline(datetime.datetime(2016, 11, 11, 23, 15), color='k', linestyle='dashed',linewidth=1.5, label = 'Bow shock crossing (Outward)')
ax45.axvline(datetime.datetime(2016, 11, 11, 23, 16), color='k', label = 'Bow shock crossing (Inward)')

ax45.axvline(datetime.datetime(2016, 11, 11, 23, 18), color='k', linestyle='dashed', linewidth=1.5)
ax45.axvline(datetime.datetime(2016, 11, 11, 23, 21), color='k',linewidth=1.5)


ax45.set_title('Total Magnetic field Amplitude - 11/11/2016 (23:00-23:59)', fontsize = 18, weight='bold')

#Legend
ax45.legend()

ax45.xaxis.set_major_locator(quarterhourly)
ax45.xaxis.set_minor_locator(fiveminutely)

ax45.grid()

plt.show()


# In[ ]:




