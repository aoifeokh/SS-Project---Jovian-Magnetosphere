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
from matplotlib.widgets import Slider
import matplotlib.image as mpimg
import math


# In[2]:


matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('xtick', labelsize=20) 


# In[3]:


conda install h5py


# In[18]:


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


# In[5]:


#import data
jupmag = np.genfromtxt('juno_fgm_b_mag_2016-2020.txt', dtype=str)
bxbybz = np.genfromtxt('juno_fgm_bxbybz_jso_2016-2020.txt', dtype=str)


# Total mag field data

# In[6]:


#import times and total amplitude of the magnetic field. 
amda_time = jupmag[0:, 0].astype(str)
juno_mag = jupmag[0:, 1]
juno_mag = juno_mag.astype(float)


# In[7]:


#Change data type to datetime
time_table = [datetime.datetime.strptime(i,"%Y-%m-%dT%H:%M:%S.%f") for i in amda_time]


# bx, by, bz components of magnetic field

# In[8]:


#import times and measured bx, by, bz components of magnetic field
amda_time_mag = bxbybz[0:, 0]
bx = bxbybz[0:, 1]
bx = bx.astype(float)
by = bxbybz[0:, 2]
by = by.astype(float)
bz = bxbybz[0:, 3]
bz = bz.astype(float)


# In[9]:


#Change data type to datetime
time_table_mag = [datetime.datetime.strptime(i,"%Y-%m-%dT%H:%M:%S.%f") for i in amda_time_mag]


# # Specifying time intervals

# plt.xlim(datetime.date(2017, 8, 1), datetime.date(2018, 10, 1))
# 
# To convert to DOY:
# 
# day_of_year = datetime.now().timetuple().tm_yday

# # Bx, By, Bz components of magnetic field

# In[10]:


#Measurement #s that we want plotted. i.e. time range.
#Time range for Bx,By,Bz and mag field amplitude

#Plot Time VS magnetic field components
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111) 

plt.plot(time_table_mag, bx, label = "$B_{x}$")
plt.plot(time_table_mag, by, label = "$B_{y}$")
plt.plot(time_table_mag, bz, label = "$B_{z}$")


#Set plot/axis titles 
plt.title('Time VS x/y/z components of magnetic field', fontsize = 20, weight='bold')
plt.ylabel('Magnetic field (nT)', fontsize = 20)  
plt.xlabel('Date-time (DOY)', fontsize = 20)  

plt.legend(prop={"size":18})

#x limits/y limits - to set date range
plt.xlim(datetime.date(2017, 8, 1), datetime.date(2018, 10, 1))
plt.ylim(0,20)

#Date format
date_form_xyz = mdates.DateFormatter("%-j")
ax.xaxis.set_major_formatter(date_form_xyz)

#Time ticks
ax.xaxis.set_major_locator(monthly)
ax.xaxis.set_minor_locator(weekly)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# # Total amplitude of magnetic field

# In[11]:


#Plot Time VS Total amplitude of magnetic field
fig5 = plt.figure(figsize=(20, 15))
ax5 = fig5.add_subplot(111) 
plt.plot(time_table, juno_mag) 

#Set plot/axis titles 
plt.title('Time VS Total amplitude of magnetic field', fontsize = 20, weight='bold')
plt.ylabel('Total magnetic field amplitude (nT)', fontsize = 20)  
plt.xlabel('Date-time (DOY)', fontsize = 20)  

#Date format
date_form_mag = mdates.DateFormatter("%-j")
ax5.xaxis.set_major_formatter(date_form_mag)

#Time ticks
ax5.xaxis.set_major_locator(twomonthly)
ax5.xaxis.set_minor_locator(monthly)

#x limits/y limits - to set date range
plt.xlim(datetime.date(2017, 8, 1), datetime.date(2018, 10, 1))
plt.ylim(0,20)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[16]:


#Jupiter orbit radius data
juno_rad = np.genfromtxt('juno_jup_r_2016-2020.txt', dtype=str)
amda_time_rad = juno_rad[0:, 0]

#change data type to datetime
time_table_rad = [datetime.datetime.strptime(i,"%Y-%m-%dT%H:%M:%S.%f") for i in amda_time_rad]


juno_jup_r = juno_rad[0:, 1]
juno_jup_r = juno_jup_r.astype(float)


# In[13]:


#change data type from datetime to matplotlib dates
#time_table_radplot = mdates.date2num(time_table_rad)


# In[14]:


#Measurement no. that we want plotted. i.e. time range. (Measurements every 5 minutes, 1 day = 288 measurements)
#Time range for Rj
time1 = 9820
time2 = 9845


# # Determining duration of one orbit 

# In[15]:


#Playing with time range to determine a full orbit. This plot shows first time juno gets close and moves away.
#(time1 = 9820, time2 = 9845)

#Plot Time VS Juno orbital radius
fig1 = plt.figure(figsize=(12, 12))
ax1 = fig1.add_subplot(111) 

plt.plot(time_table_rad[time1:time2], juno_jup_r[time1:time2]) 

#Set plot/axis titles 
plt.title('Time VS Juno Orbital Radius', fontsize = 14, weight='bold')
plt.ylabel('Orbital Radius ($R_{j}$)', fontsize = 14)  
plt.xlabel('Date-time (DOY HH:MM)', fontsize = 14)  

#Date format
date_form_rad = mdates.DateFormatter("%-j %H:%M")
ax1.xaxis.set_major_formatter(date_form_rad)

#Time ticks
ax1.xaxis.set_major_locator(halfhourly)
ax1.xaxis.set_minor_locator(fiveminutely)

plt.show()


# In[16]:


time3 = 25200
time4 = 25230


# In[17]:


#Plot Time VS Juno orbital radius - trying to find when Juno gets close again to determine length of full orbit
fig2 = plt.figure(figsize=(12, 12))
ax2 = fig2.add_subplot(111) 

plt.plot(time_table_rad[time3:time4], juno_jup_r[time3:time4]) 

#Set plot/axis titles 
plt.title('Time VS Juno Orbital Radius', fontsize = 14, weight='bold')
plt.ylabel('Orbital Radius ($R_{j}$)', fontsize = 14)  
plt.xlabel('Date-time (DOY HH:MM)', fontsize = 14)  

#Date format
ax2.xaxis.set_major_formatter(date_form_rad)

#Time ticks
ax2.xaxis.set_major_locator(halfhourly)
ax2.xaxis.set_minor_locator(fiveminutely)

plt.show()

#This plot shows end of first orbit. So orbit went from 2016-07-05T02:45.00.000 to 2016-08-27T12:50.00.000
#So first orbit was ~53 days, 10 hours, 5 minutes long (according to measurements)


# In[18]:


#Plot Time VS Juno orbital radius - showing first full orbit
fig3 = plt.figure(figsize=(16, 16))
ax3 = fig3.add_subplot(111) 

plt.plot(time_table_rad, juno_jup_r) 

#Set plot/axis titles 
plt.title('Time VS Juno Orbital Radius', fontsize = 20, weight='bold')
plt.ylabel('Orbital Radius ($R_{j}$)', fontsize = 20)  
plt.xlabel('Date-time (DOY YEAR)', fontsize = 20)  

#Date format
date_form_orb = mdates.DateFormatter("%-j %Y")
ax3.xaxis.set_major_formatter(date_form_orb)

#Time ticks
ax3.xaxis.set_major_locator(yearly)
ax3.xaxis.set_minor_locator(monthly)

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# # Plots of Total magnetic field amplitude and Bx, By, Bz in planetocentric and JSO co-ordinates.

# In[19]:


#Import Bx/y/z data in planetocentric co-ordinates
jupmag_pc = np.genfromtxt('juno_fgm_bxbybz_iau_orbits_2016_2020.txt', dtype=str)


# In[20]:


#import times and amplitude of the magnetic field. 
amda_time_pc = jupmag_pc[0:, 0].astype(str)
bx_pc = jupmag_pc[0:, 1].astype(float)
by_pc = jupmag_pc[0:, 2].astype(float)
bz_pc = jupmag_pc[0:, 3].astype(float)


# In[21]:


#Change data type to datetime
time_table_mag_pc = [datetime.datetime.strptime(i,"%Y-%m-%dT%H:%M:%S.%f") for i in amda_time_pc]


# In[22]:


#Stacked plots of magnetic field from FGM
fig4 = plt.figure(figsize=(20, 10),)
ax4 = fig4.subplots(3, sharex = True) 

ax4[0].plot(time_table_mag, bx, label = "$B_{x}$")
ax4[0].plot(time_table_mag, by, label = "$B_{y}$")
ax4[0].plot(time_table_mag, bz, label = "$B_{z}$")

ax4[1].plot(time_table_mag_pc, bx_pc, label = "$B_{x}$")
ax4[1].plot(time_table_mag_pc, by_pc, label = "$B_{y}$")
ax4[1].plot(time_table_mag_pc, bz_pc, label = "$B_{z}$")

ax4[2].plot(time_table, juno_mag, label = "|B|")

plt.xlabel('Date-time', fontsize = 20)


ax4[0].set_ylabel('Magnetic field strength (nT)', fontsize = 17)
ax4[1].set_ylabel('Magnetic field strength (nT)', fontsize = 17)
ax4[2].set_ylabel('Magnetic field amplitude (nT)', fontsize = 17)

ax4[0].legend(prop={"size":18})
ax4[1].legend(prop={"size":18})
ax4[2].legend(prop={"size":18})

#Without y-lim, is hard to distinguish any features - just spikes. - Can change this limit to find large spikes
ax4[0].set_ylim([-10,10])
ax4[1].set_ylim([-10,10])
ax4[2].set_ylim([0,20])


ax4[0].axhline(0, linestyle='--')
ax4[1].axhline(0, linestyle='--')

fig4.tight_layout()

ax4[0].set_title('Magnetic field from FGM - JSO', fontsize = 20, weight='bold')
ax4[1].set_title('Magnetic field - Planetocentric co-ordinates', fontsize = 20, weight='bold')
ax4[2].set_title('Total Magnetic Field Amplitude', fontsize = 20, weight='bold')

plt.xticks(fontsize = 20)


plt.show()


# # Plotting path of orbit using x,y  JSO co-ordinates

# In[23]:


xyz_jso = np.genfromtxt('juno_jup_xyz_jso_2016_2020.txt', dtype=str)


# In[24]:


xyzjso_time = xyz_jso[0:, 0].astype(str)
x_jso = xyz_jso[0:, 1].astype(float)
y_jso = xyz_jso[0:, 2].astype(float)
z_jso = xyz_jso[0:, 3].astype(float)

#Change time data type to datetime
time_table_jso = [datetime.datetime.strptime(i,"%Y-%m-%dT%H:%M:%S.%f") for i in xyzjso_time]


# In[25]:


print((np.where(y_jso==-115)))


# In[26]:


#Plot Juno trajectory using X and Y JSO coordinates
fig6 = plt.figure(figsize=(16, 16))
ax6 = fig6.add_subplot(111) 

plt.plot(x_jso, y_jso, label = "Juno Trajectory") 

plt.plot(0,0, 'ro', label = "Jupiter")

#Set plot/axis titles 
plt.title('Trajectory of Juno - JSO Coordinates', fontsize = 20, weight='bold')
plt.ylabel('$Y_{JSO}$ ($R_j$)', fontsize = 20)  
plt.xlabel('$X_{JSO}$ ($R_j$)', fontsize = 20)  

plt.xlim(-110,30)
plt.ylim(-115,40)

plt.legend(prop={"size":18})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[ ]:





# # 3d plot of Juno orbit showing points of MP crossings

# In[26]:


print(len(x_jso))


# In[49]:


# Boundary crossing positions

bs_crossing_coord = np.genfromtxt('BS_crossing _coordinates.csv', delimiter=',',dtype=str)
mp_crossing_coord = np.genfromtxt('MP_crossing_coordinates.csv', delimiter=',',dtype=str)

#X,Y,Z positions of bow shock crossings
bsx_x = bs_crossing_coord[1:, 1].astype(float) #X
bsx_y = bs_crossing_coord[1:, 2].astype(float) #Y
bsx_z = bs_crossing_coord[1:, 3].astype(float) #Z

#Bow Shock crossing numbers
bs_no = bs_crossing_coord[1:, 0].astype(str)

#X,Y,Z positions of magnetopause crossings
mpx_x = mp_crossing_coord[:, 1].astype(float) #X
mpx_y = mp_crossing_coord[:, 2].astype(float) #Y
mpx_z = mp_crossing_coord[:, 3].astype(float) #Z

#Magnetopause crossing numbers
mp_no = mp_crossing_coord[:, 0].astype(str)

#Need to get rid of unicode
mp_no[0]= 'MP1'


# In[57]:


fig = plt.figure(figsize=(16,16))

ax = fig.add_subplot(111, projection='3d')

ax.plot(x_jso[4000:55850], y_jso[4000:55850], z_jso[4000:55850], 'k', label = "Juno Trajectory")
ax.plot(bsx_x, bsx_y, bsx_z, '+r', markersize=15, label = 'Bow shock crossing')
ax.plot(mpx_x, mpx_y, mpx_z, '+b', markersize=15, label = 'Magnetopause crossing')


ax.scatter(0,0,0, 'ro', label = "Jupiter", s = 300, c = 'orange')


#Set plot/axis titles 
plt.title('Trajectory of Juno - JSO Coordinates', fontsize = 20, weight='bold', y =1.02)

ax.set_xlabel('$X_{JSO}$ ($R_j$)', fontsize = 20)  
ax.set_ylabel('$Y_{JSO}$ ($R_j$)', fontsize = 20)  
ax.set_zlabel('$Z_{JSO}$ ($R_j$)', fontsize = 20) 

ax.xaxis.labelpad=20
ax.yaxis.labelpad=20
ax.zaxis.labelpad=20

ax.set_xlim(-20,20)
ax.set_ylim(-100,10)
ax.set_zlim(-20,20)

plt.tight_layout(pad=5.0, h_pad=4.0, w_pad=5.0)

plt.legend(prop={"size":18})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# # Orbital radius VS Magnetic field amplitude - for comparison

# In[29]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison
fig7 = plt.figure(figsize=(20, 10),)
ax7 = fig7.subplots(2, sharex = True) 

#Plot
ax7[0].plot(time_table_rad, juno_jup_r, label = "Orbital Radius ($R_{j}$)") 
ax7[1].plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date-time', fontsize = 20)

#Y-Label
ax7[0].set_ylabel('Orbital Radius ($R_{j})$)', fontsize = 20)
ax7[1].set_ylabel('Magnetic field strength (nT)', fontsize = 20)

#Legend
ax7[0].legend(prop={"size":18})
ax7[1].legend(prop={"size":18})

#x-lim and y-lim (if want to zoom in)
ax7[1].set_ylim([0,10])

fig7.tight_layout()

ax7[0].set_title('Orbital radius of Juno ($R_{j}$)', fontsize = 20, weight='bold')
ax7[1].set_title('Total Magnetic field Amplitude', fontsize = 20, weight='bold')

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# In[30]:


#Orbit number - approximate - to plot individual orbits approximately orbits
#One orbit ~ 9700 - 25300 = 15600 points
n = 2 #Orbit number 1,2,3....


start_pt_index = 9550+(15500*(n-1))
end_pt_index = 9550+(15500*(n))


# In[31]:


print(len(time_table))


# In[32]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - zoomed in
fig8 = plt.figure(figsize=(20, 10),)
ax8 = fig8.subplots(2, sharex = True) 

#Plot
ax8[0].plot(time_table_rad, juno_jup_r, label = "Orbital Radius ($R_{j}$)") 
ax8[1].plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date (DOY 2016)', fontsize = 17)

#Y-Label
ax8[0].set_ylabel('Orbital Radius ($R_{j})$)', fontsize = 17)
ax8[1].set_ylabel('Magnetic field strength (nT)', fontsize = 17)

#Trying to add in text with orbit number to legend
#ax[0].plot([], [], ' ', label = "Orbit number: {}".format(n))

#Legend
ax8[0].legend(prop={"size":18})
ax8[1].legend(prop={"size":18})

#x-lim and y-lim (if want to zoom in)
ax8[0].set_ylim([0,130])
ax8[1].set_ylim([0,100])
ax8[1].set_xlim(time_table[66000],time_table[155000])

fig8.tight_layout()

ax8[0].set_title('Orbital radius of Juno ($R_{j}$)', fontsize = 18, weight='bold')
ax8[1].set_title('Total Magnetic field Amplitude', fontsize = 18, weight='bold')

#Date format
date_form_orb = mdates.DateFormatter("%-j")
ax8[0].xaxis.set_major_formatter(date_form_orb)
ax8[1].xaxis.set_major_formatter(date_form_orb)

#Time ticks
ax3.xaxis.set_major_locator(sixmonthly)
ax3.xaxis.set_minor_locator(monthly)

plt.show()


# In[33]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig9 = plt.figure(figsize=(20, 10),)
ax9 = fig9.subplots(2) 

#Plot
ax9[0].plot(time_table_rad[start_pt_index:end_pt_index], juno_jup_r[start_pt_index:end_pt_index], label = "Orbital Radius ($R_{j}$)") 
ax9[1].plot(time_table, juno_mag, 'r', label = "|B|")

#X-label
plt.xlabel('Date (DOY 2016)', fontsize = 17)

#Y-Label
ax9[0].set_ylabel('Orbital Radius ($R_{j})$)', fontsize = 17)
ax9[1].set_ylabel('Magnetic field strength (nT)', fontsize = 17)


#x-lim and y-lim (if want to zoom in)
ax9[1].set_ylim([-5,25])
ax9[1].set_xlim(time_table[80000],time_table[100000])

fig8.tight_layout()

ax9[0].set_title('Orbital radius of Juno ($R_{j}$)', fontsize = 18, weight='bold')
ax9[1].set_title('Total Magnetic field Amplitude', fontsize = 18, weight='bold')

#Highlight time period shown on bottom plot in top plot
ax9[0].axvspan(time_table[80000],time_table[100000], color='yellow', alpha=0.5, label = 'Time period shown in magnetic field plot')

#Legend
ax9[0].legend(loc = 'upper right', prop={"size":18})
ax9[1].legend(loc = 'upper right', prop={"size":18})

#Date format
date_form_orb = mdates.DateFormatter("%-j")
ax9[0].xaxis.set_major_formatter(date_form_orb)
ax9[1].xaxis.set_major_formatter(date_form_orb)

#Time ticks
#ax3[0].xaxis.set_major_locator(weekly)
#ax3[0].xaxis.set_minor_locator(daily)

#ax3[1].xaxis.set_major_locator(daily)
#ax3[1].xaxis.set_minor_locator(twelvehourly)

plt.show()


# # Histogram - Hospodarsky Supporting Information

# ## Grouping crossings

# In[34]:


#Plot Time VS Juno orbital radius - showing first full orbit
fig20 = plt.figure(figsize=(16, 16))
ax20 = fig20.add_subplot(111) 

#plt.plot(time_table_rad[time5:time6], juno_jup_r[time5:time6]) 
plt.plot(time_table_rad, juno_jup_r) 

#Set plot/axis titles 
plt.title('Time VS Juno Orbital Radius', fontsize = 18, weight='bold')
plt.ylabel('Orbital Radius ($R_{j}$)', fontsize = 17)  
plt.xlabel('Date (DOY 2016)', fontsize = 17)  

#Date format
date_form_orb = mdates.DateFormatter("%-j %Y")
ax20.xaxis.set_major_formatter(date_form_orb)

#Time ticks
ax20.xaxis.set_major_locator(twoweekly)
ax20.xaxis.set_minor_locator(daily)

plt.xlim(datetime.date(2016, 7, 2), datetime.date(2016, 8, 30))
plt.ylim(-5,130)

plt.show()


# In[ ]:





# In[ ]:





# ## Bow shock crossing data

# In[4]:


bs_crossings = np.genfromtxt('hosp_bs_crossings', delimiter=',',dtype=str)


# In[5]:


bs_rj = bs_crossings[0:, 5].astype(float)  #crossing distance in Rj


# # Histogram - Bow shock data

# In[6]:


print(np.min(bs_rj))
print(np.max(bs_rj))
print(bs_rj.mean())


# In[7]:


fig10 = plt.figure(figsize=(20, 10),)
ax10 = fig10.add_subplot(1,1,1) 
bins = np.linspace(90, 130, 80)

plt.title('Bow Shock Range Statistics - Inbound and Outbound', fontsize = 20, fontweight = "bold")
ax10.set_ylabel('Number of Boundary Crossings', fontsize = 20) 
ax10.set_xlabel('Range of BS Crossings ($R_{j}$)', fontsize = 20)

plt.hist(bs_rj, bins, color = 'blue', label = "Total number of crossings = 51") 

#Plot line showing mean value
plt.axvline(bs_rj.mean(), color='k', linestyle='dashed', linewidth=1, label = 'Mean = 108.33 $R_{j}$')

plt.legend(loc='upper left', frameon=True, prop={"size":18})


# ## Differentiating between "In" and "Out"

# In[8]:


bs_inout = bs_crossings[0:, 4].astype(str)  #inbound or outbound


# ## Inbound

# In[9]:


#calculate mean for inbound range
mean_inbs = bs_rj[np.where(bs_inout == "In")].mean()
print(bs_rj[np.where(bs_inout == "In")].mean())


# In[10]:


fig12 = plt.figure(figsize=(20, 10),)
ax12 = fig12.add_subplot(1,1,1) 
bins = np.linspace(90, 130, 80)

plt.title('Bow Shock Range Statistics - "In"', fontsize = 18, fontweight = "bold")
ax12.set_ylabel('Number of Boundary Crossings', fontsize = 17) 
ax12.set_xlabel('Range of BS Crossings ($R_{j}$)', fontsize = 17)

plt.ylim(0,10)

plt.hist(bs_rj[np.where(bs_inout == "In")], bins, color = 'blue', label = "Total number of inbound crossings = 26") 
plt.axvline(mean_inbs, color='k', linestyle='dashed', linewidth=1, label = 'Mean = 108.75 $R_{j}$')


plt.legend(loc='upper left', frameon=True, prop={"size":18})


# ## Outbound

# In[11]:


#calculate mean for outbound range
mean_outbs = bs_rj[np.where(bs_inout == "Out")].mean()
print(mean_outbs)


# In[ ]:





# In[12]:


fig14 = plt.figure(figsize=(20, 10),)
ax14 = fig14.add_subplot(1,1,1) 
bins = np.linspace(90, 130, 80)

plt.title('Bow Shock Range Statistics - "Out"', fontsize = 18, fontweight = "bold")
ax14.set_ylabel('Number of Boundary Crossings', fontsize = 17) 
ax14.set_xlabel('Range of BS Crossings ($R_{j}$)', fontsize = 17)

plt.hist(bs_rj[np.where(bs_inout == "Out")], bins, color = 'blue', label = "Total number of outbound crossings = 25")

plt.axvline(mean_outbs, color='k', linestyle='dashed', linewidth=1, label = 'Mean = 108.01 $R_{j}$')

plt.ylim(0,10)

plt.legend(loc='upper left', frameon=True, prop={"size":18})


# ## Stack plots

# In[13]:


bins = np.linspace(90, 130, 80)

#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison
fig18 = plt.figure(figsize=(20, 10),)
ax18 = fig18.subplots(3, sharex = True) 

#Plot
ax18[0].hist(bs_rj, bins, color = 'blue', label = "Total number of crossings = 51")
ax18[0].axvline(bs_rj.mean(), color='k', linestyle='dashed', linewidth=1, label = 'Mean = 108.33 $R_{j}$')

ax18[1].hist(bs_rj[np.where(bs_inout == "In")], bins, color = 'blue', label = "Total number of inbound crossings = 26")
ax18[1].axvline(mean_inbs, color='k', linestyle='dashed', linewidth=1, label = 'Mean = 108.75 $R_{j}$')

ax18[2].hist(bs_rj[np.where(bs_inout == "Out")], bins, color = 'blue', label = "Total number of outbound crossings = 25")
ax18[2].axvline(mean_outbs, color='k', linestyle='dashed', linewidth=1, label = 'Mean = 108.01 $R_{j}$')

#X-label
plt.xlabel('Range of Bow Shock Crossings ($R_{j}$)', fontsize = 17)
plt.xticks(np.arange(90, 135, 5))

#Y-Label
ax18[0].set_ylabel('Number of Crossings', fontsize = 15)
ax18[1].set_ylabel('Number of Crossings', fontsize = 15)
ax18[2].set_ylabel('Number of Crossings', fontsize = 15)

#Legend
ax18[0].legend(loc='upper left', prop={"size":18}, frameon=True)
ax18[1].legend(loc='upper left', prop={"size":18}, frameon=True)
ax18[2].legend(loc='upper left', prop={"size":18}, frameon=True)

#x-lim and y-lim (if want to zoom in)
ax18[0].set_ylim([0,12])
ax18[1].set_ylim([0,12])
ax18[2].set_ylim([0,12])

fig18.tight_layout()

#Titles
ax18[0].set_title('Bow Shock Statistics - Inbound and Outbound', fontsize = 20, weight='bold')
ax18[1].set_title('Bow Shock - Inbound', fontsize = 20, weight='bold')
ax18[2].set_title('Bow Shock - Outbound', fontsize = 20, weight='bold')

#Grid
ax18[0].grid()
ax18[1].grid()
ax18[2].grid()

plt.show()


# ### Colour coded histogram by orbit number

# In[19]:


# Determining orbit numbers for date range in Hospodarsky supporting data

#Plot Time VS Juno orbital radius - showing first full orbit
fig20 = plt.figure(figsize=(16, 16))
ax20 = fig20.add_subplot(111) 

#plt.plot(time_table_rad[time5:time6], juno_jup_r[time5:time6]) 
plt.plot(time_table_rad, juno_jup_r) 

#Set plot/axis titles 
plt.title('Time VS Juno Orbital Radius', fontsize = 20, weight='bold')
plt.ylabel('Orbital Radius ($R_{j}$)', fontsize = 20)  
plt.xlabel('Date (DOY 2016)', fontsize = 20)  


#Time ticks
ax20.xaxis.set_major_locator(twoweekly)
ax20.xaxis.set_minor_locator(daily)

plt.xlim(datetime.date(2016, 6, 24), datetime.date(2016, 11, 23))
plt.ylim(0,130)

#Date format
date_form_orb = mdates.DateFormatter("%-j")
ax20.xaxis.set_major_formatter(date_form_orb)

plt.grid()

plt.show()


# From the plot can see that:

# Orbit 1 - Starts 2016-07-05 Ends 2016-08-27

# Orbit 2 - Starts 2016-08-27 Ends 2016-10-19

# Orbit 3 - Starts 2016-10-19

# In[20]:


#Creating arrays that will be used for indices to indicate which orbit the data is in

bs_approach = np.array([0])
bs_o1 = np.arange(1,37,1)
#bs_o2 = NONE
bs_o3 = np.arange(37,51,1)

#In
bsin_approach = np.array([0])
bsin_o1 = np.arange(2,38,2)
#bsin_o2 = NONE
bsin_o3 = np.arange(38,52,2)

#Out
#bsout_approach = NONE
bsout_o1 = np.arange(1,37,2)
#bsout_o2 = NONE
bsout_o3 = np.arange(37,51,2)


# In[21]:


#Calculate means and standard deviations for individual orbits
#All orbits and approach
print('ALL DATA')
print('Mean position of bow shock crossing: ', bs_rj.mean())
print('\n')
print('Standard deviation of bow shock crossings: ', np.std(bs_rj))
print('\n')
print('Mean of approach = ', bs_rj[bs_approach].mean())
print('\n')
print('Standard deviation  of approach: ', np.std(bs_rj[bs_approach]))
print('\n')
print('Mean in/out orbit 1 = ', bs_rj[bs_o1].mean())
print('\n')
print('Standard deviation orbit 1: ', np.std(bs_rj[bs_o1]))
print('\n')
print('Mean in/out orbit 3 = ', bs_rj[bs_o3].mean())
print('\n')
print('Standard deviation orbit 3: ', np.std(bs_rj[bs_o3]))


# In[22]:


#Calculate means and standard deviations for individual orbits
#Inbound
print('INBOUND DATA')
print('Mean position of inbound bow shock crossing: ' ,mean_inbs )
print('\n')
print('Standard deviation of inbound bow shock crossings: ', np.std(bs_rj[np.where(bs_inout == "In")]))
print('\n')
print('Mean of approach = ', bs_rj[0].mean())
print('\n')
print('Standard deviation  of approach: ', np.std(bs_rj[0]))
print('\n')
print('Mean in orbit 1 = ', bs_rj[bsin_o1].mean())
print('\n')
print('Standard deviation  in orbit 1: ', np.std(bs_rj[bsin_o1]))
print('\n')
print('Mean in orbit 3 = ', bs_rj[bsin_o3].mean())
print('\n')
print('Standard deviation in orbit 3: ', np.std(bs_rj[bsin_o3]))


# In[23]:


#Calculate means and standard deviations for individual orbits
#Outbound
print('OUTBOUND DATA')
print('Mean position of inbound bow shock crossing: ' ,mean_inbs )
print('\n')
print('Standard deviation of outbound bow shock crossings: ', np.std(bs_rj[np.where(bs_inout == "Out")]))
print('\n')
print('Mean in orbit 1 = ', bs_rj[bsout_o1].mean())
print('\n')
print('Standard deviation  in orbit 1: ', np.std(bs_rj[bsout_o1]))
print('\n')
print('Mean in orbit 3 = ', bs_rj[bsout_o3].mean())
print('\n')
print('Standard deviation in orbit 3: ', np.std(bs_rj[bsout_o3]))


# In[26]:


bins = np.linspace(90, 130, 40)

#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - COLOUR CODED
fig21 = plt.figure(figsize=(20, 10),)
ax21 = fig21.subplots(3, sharex = True) 

#Plot 1
ax21[0].hist(bs_rj[bs_approach], bins, color = 'darkcyan', label = "Approach")
ax21[0].hist(bs_rj[bs_o1], bins, color = 'cornflowerblue', label = "Orbit 1")
ax21[0].hist(bs_rj[bs_o3], bins, color = 'mediumslateblue', label = "Orbit 3")
#Stats
ax21[0].axvline(bs_rj.mean(), color='k', linestyle='dashed', linewidth=2, label = 'Mean (All data) = 108.33 $R_{j}$, $\sigma$ = 6.00 $R_{j}$')
ax21[0].axvline(bs_rj[0].mean(), color='darkcyan', linestyle='dashed', linewidth=2, label = 'Mean (Approach) = 128.1 $R_{j}$, $\sigma$ = 0.0 $R_{j}$')
ax21[0].axvline(bs_rj[bs_o1].mean(), color='cornflowerblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 1) = 106.98 $R_{j}$, $\sigma$ = 5.92 $R_{j}$')
ax21[0].axvline(bs_rj[bs_o3].mean(), color='mediumslateblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 3) = 110.41 $R_{j}$, $\sigma$ = 2.06 $R_{j}$')



#Plot 2
ax21[1].hist(bs_rj[bsin_approach], bins, color = 'darkcyan', label = "Approach")
ax21[1].hist(bs_rj[bsin_o1], bins, color = 'cornflowerblue', label = "Orbit 1")
ax21[1].hist(bs_rj[bsin_o3], bins, color = 'mediumslateblue', label = "Orbit 3")
#Stats
ax21[1].axvline(mean_inbs, color='k', linestyle='dashed', linewidth=2, label = 'Mean = 108.75 $R_{j}$, $\sigma$ = 6.00 $R_{j}$')
ax21[1].axvline(bs_rj[0].mean(), color='darkcyan', linestyle='dashed', linewidth=2, label = 'Mean (Approach) = 128.1 $R_{j}$, $\sigma$ = 0.0 $R_{j}$')
ax21[1].axvline(bs_rj[bsin_o1].mean(), color='cornflowerblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 1) = 107.02 $R_{j}$, $\sigma$ = 5.73 $R_{j}$')
ax21[1].axvline(bs_rj[bsin_o3].mean(), color='mediumslateblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 3) = 110.41 $R_{j}$, $\sigma$ = 2.06 $R_{j}$')


#Plot 3
ax21[2].hist(bs_rj[bsout_o1], bins, color = 'cornflowerblue', label = "Orbit 1")
ax21[2].hist(bs_rj[bsout_o3], bins, color = 'mediumslateblue', label = "Orbit 3")
#Stats
ax21[2].axvline(mean_outbs, color='k', linestyle='dashed', linewidth=2, label = 'Mean = 108.01 $R_{j}$')
ax21[2].axvline(bs_rj[bsout_o1].mean(), color='cornflowerblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 1) = 106.93 $R_{j}$, $\sigma$ = 6.10 $R_{j}$')
ax21[2].axvline(bs_rj[bsout_o3].mean(), color='mediumslateblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 3) = 110.40 $R_{j}$, $\sigma$ = 2.06 $R_{j}$')

#X-label
plt.xlabel('Range of Bow Shock Crossings ($R_{j}$)', fontsize = 17)
plt.xticks(np.arange(90, 135, 5))

#Y-Label
ax21[0].set_ylabel('Number of Crossings', fontsize = 15)
ax21[1].set_ylabel('Number of Crossings', fontsize = 15)
ax21[2].set_ylabel('Number of Crossings', fontsize = 15)

#Legend
ax21[0].legend(bbox_to_anchor=(1.05, 1.0), title="Inbound and Outbound data", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')
ax21[1].legend(bbox_to_anchor=(1.05, 1.0), title="Inbound data", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')
ax21[2].legend(bbox_to_anchor=(1.05, 1.0), title="Outbound data", title_fontsize = '18', prop={"size":16}, ncol = 2, loc='upper left')

#x-lim and y-lim (if want to zoom in)
ax21[0].set_ylim([0,10])
ax21[1].set_ylim([0,10])
ax21[2].set_ylim([0,10])

fig21.tight_layout(h_pad=2.0)

#Titles
ax21[0].set_title('(a) Bow Shock Statistics - Inbound and Outbound (51 crossings)', fontsize = 20, weight='bold')
ax21[1].set_title('(b) Bow Shock - Inbound (26 crossings)', fontsize = 20, weight='bold')
ax21[2].set_title('(c) Bow Shock - Outbound (25 crossings)', fontsize = 20, weight='bold')

#Grid
ax21[0].grid()
ax21[1].grid()
ax21[2].grid()

plt.show()


# ## Line chart showing overall means for bs and mp per orbit for comparison.

# In[27]:


meaninoutbs = np.array([128.1,106.9777777777778, None, 110.40714285714286]).astype(np.double)
meaninoutmp = np.array([83.82857142857142, 103.625, 105.15916666666665, 106.92692307692307]).astype(np.double)
mask1 = np.isfinite(meaninoutbs) #To skip over 'None' value for orbit 2 bow shock mean 
orbits = np.array([0,1,2,3])


# In[44]:


fig99 = plt.figure(figsize=(15, 10),)
ax99 = fig99.add_subplot(1,1,1) 

plt.plot(orbits[mask1], meaninoutbs[mask1], linestyle='-', marker='o', label = "Bow Shock")
plt.plot(orbits, meaninoutmp, linestyle='-', marker='o', label = 'Magnetopause')
labels = ['Approach', 'Orbit 1', 'Orbit 2', 'Orbit 3']

ax99.set_xticks(orbits)
ax99.set_xticklabels(labels)

ax99.set_ylabel('Average boundary crossing distance ($R_{j}$)', fontsize = 20)
ax99.set_xlabel('Orbit', fontsize = 20)
ax99.set_title('Average boundary crossing distance for approach and orbits', fontweight = 'bold', fontsize =20)

plt.grid()

plt.legend()

plt.show()


# ## Histogram - Magnetopause data

# In[24]:


mp_crossings = np.genfromtxt('hosp_mp_crossings.csv', delimiter=',',dtype=str)


# In[25]:


mp_rj = mp_crossings[0:, 5].astype(float)  #crossing distance in Rj


# In[53]:


print(np.min(mp_rj))
print(np.max(mp_rj))
print(mp_rj.mean())


# In[54]:


fig11 = plt.figure(figsize=(20, 10),)
ax11 = fig11.add_subplot(1,1,1) 
bins = np.linspace(70, 115, 90)

plt.title('Magnetopause Range Statistics - Inbound and Outbound', fontsize = 18, fontweight = "bold")
ax11.set_ylabel('Number of Boundary Crossings', fontsize = 17) 
ax11.set_xlabel('Range of MP Crossings ($R_{j}$)', fontsize = 17)

plt.hist(mp_rj, bins, color = 'blue', label = "Total number of crossings = 97")

plt.axvline(mp_rj.mean(), color='k', linestyle='dashed', linewidth=1, label = 'Mean = 103.44 $R_{j}$')

plt.legend(loc='upper left', frameon=True, prop={"size":18})


# ## Differentiating between "In" and "Out"

# In[55]:


mp_inout = mp_crossings[0:, 4].astype(str)  #inbound or outbound


# ## Inbound

# In[56]:


#calculate mean for inbound range
mean_inmp = mp_rj[np.where(mp_inout == "In")].mean()

print(mean_inmp)


# In[57]:


fig13 = plt.figure(figsize=(20, 10),)
ax13 = fig13.add_subplot(1,1,1) 
bins = np.linspace(70, 115, 90)

plt.title('Magnetopause Range Statistics - "In"', fontsize = 18, fontweight = "bold")
ax13.set_ylabel('Number of Boundary Crossings', fontsize = 17) 
ax13.set_xlabel('Range of MP Crossings ($R_{j}$)', fontsize = 17)

plt.hist(mp_rj[np.where(mp_inout == "In")], bins, color = 'blue', label = "Total number of inbound crossings = 49")
plt.hist(mp_rj[np.where(mp_inout == "in")], bins, color = 'blue',)

plt.axvline(mean_inmp, color='k', linestyle='dashed', linewidth=1, label = 'Mean = 103.59 $R_{j}$')

plt.ylim(0,12)

plt.legend(loc='upper left', frameon=True, prop={"size":18})


# ## Outbound

# In[58]:


#calculate mean for outbound range
mean_outmp = (mp_rj[np.where(mp_inout == "Out")]).mean()

print(mean_outmp)


# In[59]:


fig15 = plt.figure(figsize=(20, 10),)
ax15 = fig15.add_subplot(1,1,1) 
bins = np.linspace(70, 115, 90)

plt.title('Magnetopause Range Statistics - "Out"', fontsize = 18, fontweight = "bold")
ax15.set_ylabel('Number of Boundary Crossings', fontsize = 17) 
ax15.set_xlabel('Range of BS Crossings ($R_{j}$)', fontsize = 17)

plt.hist(mp_rj[np.where(mp_inout == "Out")], bins, color = 'blue', label = "Total number of outbound crossings = 48")
plt.hist(mp_rj[np.where(mp_inout == "out")], bins, color = 'blue')

plt.axvline(mean_outmp, color='k', linestyle='dashed', linewidth=1, label = 'Mean = 103.28 $R_{j}$')

plt.ylim(0,12)

plt.legend(loc='upper left', frameon=True, prop={"size":18})


# ## Stack the in/out, in and out plots

# In[60]:


bins = np.linspace(70, 115, 90)

#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison
fig17 = plt.figure(figsize=(20, 10),)
ax17 = fig17.subplots(3, sharex = True) 

#Plot
ax17[0].hist(mp_rj, bins, color = 'blue', label = "Total number of crossings = 97")
ax17[0].axvline(mp_rj.mean(), color='k', linestyle='dashed', linewidth=1, label = 'Mean = 103.44 $R_{j}$')

ax17[1].hist(mp_rj[np.where(mp_inout == "In")], bins, color = 'blue', label = "Total number of inbound crossings = 49")
ax17[1].hist(mp_rj[np.where(mp_inout == "in")], bins, color = 'blue',)
ax17[1].axvline(mean_inmp, color='k', linestyle='dashed', linewidth=1, label = 'Mean = 103.59 $R_{j}$')

ax17[2].hist(mp_rj[np.where(mp_inout == "Out")], bins, color = 'blue', label = "Total number of outbound crossings = 48")
ax17[2].hist(mp_rj[np.where(mp_inout == "out")], bins, color = 'blue')
ax17[2].axvline(mean_outmp, color='k', linestyle='dashed', linewidth=1, label = 'Mean = 103.28 $R_{j}$')

#X-label
plt.xlabel('Range of Bow Shock Crossings ($R_{j}$)', fontsize = 17)
plt.xticks(np.arange(70, 115, 5))

#Y-Label
ax17[0].set_ylabel('Number of Crossings', fontsize = 15)
ax17[1].set_ylabel('Number of Crossings', fontsize = 15)
ax17[2].set_ylabel('Number of Crossings', fontsize = 15)

#Legend
ax17[0].legend(loc='upper left', frameon=True)
ax17[1].legend(loc='upper left', frameon=True)
ax17[2].legend(loc='upper left', frameon=True)

#x-lim and y-lim (if want to zoom in)
ax17[0].set_ylim([0,12])
ax17[1].set_ylim([0,12])
ax17[2].set_ylim([0,12])

fig17.tight_layout()

#Titles
ax17[0].set_title('Magnetopause Range Statistics - Inbound and Outbound', fontsize = 18, weight='bold')
ax17[1].set_title('Magnetopause - Inbound', fontsize = 18, weight='bold')
ax17[2].set_title('Magnetopause - Outbound', fontsize = 18, weight='bold')

#Grid
ax17[0].grid()
ax17[1].grid()
ax17[2].grid()

plt.show()


# ### Colour coded histogram by orbit number

# In[61]:


#Creating arrays that will be used for indices to indicate which orbit the data is in

#In and out
mp_approach = np.arange(0,7,1)
mp_o1 = np.arange(7,47,1)
mp_o2 = np.arange(47,71,1)
mp_o3 = np.arange(71,97,1)

#In
mpin_approach = np.array([0,2,4,6])
mpin_o1 = np.arange(8,48,2)
mpin_o2 = np.arange(48,72,2)
mpin_o3 = np.arange(72,98,2)

#Out
mpout_approach = np.array([1,3,5])
mpout_o1 = np.arange(7,47,2)
mpout_o2 = np.arange(47,71,2)
mpout_o3 = np.arange(71,97,2)


# In[62]:


#Calculate means and standard deviations for individual orbits
#All orbits and approach
print('ALL DATA')
print('Mean position of magnetopause crossing: ', mp_rj.mean())
print('\n')
print('Standard deviation of magnetopause crossings: ', np.std(mp_rj))
print('\n')
print('Mean of approach = ', mp_rj[mp_approach].mean())
print('\n')
print('Standard deviation  of approach: ', np.std(mp_rj[mp_approach]))
print('\n')
print('Mean in/out orbit 1 = ', mp_rj[mp_o1].mean())
print('\n')
print('Standard deviation orbit 1: ', np.std(mp_rj[mp_o1]))
print('\n')
print('Mean in/out orbit 2 = ', mp_rj[mp_o2].mean())
print('\n')
print('Standard deviation orbit 2: ', np.std(mp_rj[mp_o2]))
print('\n')
print('Mean in/out orbit 3 = ', mp_rj[mp_o3].mean())
print('\n')
print('Standard deviation orbit 3: ', np.std(mp_rj[mp_o3]))


# In[63]:


#Calculate means and standard deviations for individual orbits
#All orbits and approach
print('INBOUND DATA')
print('Mean position of magnetopause crossing: ', mean_inmp)
print('\n')
print('Standard deviation of magnetopause crossings: ', np.std(mp_rj[np.where(mp_inout == "In")]))
print('\n')
print('Mean of approach = ', mp_rj[mpin_approach].mean())
print('\n')
print('Standard deviation  of approach: ', np.std(mp_rj[mpin_approach]))
print('\n')
print('Mean in orbit 1 = ', mp_rj[mpin_o1].mean())
print('\n')
print('Standard deviation orbit 1: ', np.std(mp_rj[mpin_o1]))
print('\n')
print('Mean in orbit 2 = ', mp_rj[mpin_o2].mean())
print('\n')
print('Standard deviation orbit 2: ', np.std(mp_rj[mpin_o2]))
print('\n')
print('Mean in orbit 3 = ', mp_rj[mpin_o3].mean())
print('\n')
print('Standard deviation orbit 3: ', np.std(mp_rj[mpin_o3]))


# In[64]:


#Calculate means and standard deviations for individual orbits
#All orbits and approach
print('OUTBOUND DATA')
print('Mean position of magnetopause crossing: ', mean_outmp)
print('\n')
print('Standard deviation of magnetopause crossings: ', np.std(mp_rj[np.where(mp_inout == "Out")]))
print('\n')
print('Mean of approach = ', mp_rj[mpout_approach].mean())
print('\n')
print('Standard deviation  of approach: ', np.std(mp_rj[mpout_approach]))
print('\n')
print('Mean out orbit 1 = ', mp_rj[mpout_o1].mean())
print('\n')
print('Standard deviation orbit 1: ', np.std(mp_rj[mpout_o1]))
print('\n')
print('Mean out orbit 2 = ', mp_rj[mpin_o2].mean())
print('\n')
print('Standard deviation orbit 2: ', np.std(mp_rj[mpout_o2]))
print('\n')
print('Mean out orbit 3 = ', mp_rj[mpout_o3].mean())
print('\n')
print('Standard deviation orbit 3: ', np.std(mp_rj[mpout_o3]))


# In[65]:


bins = np.linspace(70, 115, 45)

#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - COLOUR CODED
fig22 = plt.figure(figsize=(20, 10),)
ax22 = fig22.subplots(3, sharex = True) 

#Plot
ax22[0].hist(mp_rj[mp_approach], bins, color = 'darkcyan', label = "Approach")
ax22[0].hist(mp_rj[mp_o1], bins, color = 'cornflowerblue', label = "Orbit 1")
ax22[0].hist(mp_rj[mp_o2], bins, color = 'blue', label = "Orbit 2")
ax22[0].hist(mp_rj[mp_o3], bins, color = 'mediumslateblue', label = "Orbit 3")
#Stats
ax22[0].axvline(mp_rj.mean(), color='k', linestyle='dashed', linewidth=2, label = 'Mean = 103.44 $R_{j}$, $\sigma$ = 10.09 $R_{j}$')
ax22[0].axvline(mp_rj[mp_approach].mean(), color='darkcyan', linestyle='dashed', linewidth=2, label = 'Mean (Approach) = 83.83 $R_{j}$, $\sigma$ = 12.67 $R_{j}$')
ax22[0].axvline(mp_rj[mp_o1].mean(), color='cornflowerblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 1) = 103.63 $R_{j}$, $\sigma$ = 8.65 $R_{j}$')
ax22[0].axvline(mp_rj[mp_o2].mean(), color='blue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 2) = 105.06 $R_{j}$, $\sigma$ = 9.31 $R_{j}$')
ax22[0].axvline(mp_rj[mp_o3].mean(), color='mediumslateblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 3) = 106.93 $R_{j}$, $\sigma$ = 4.79 $R_{j}$')


#Plot
ax22[1].hist(mp_rj[mpin_approach], bins, color = 'darkcyan', label = "Approach")
ax22[1].hist(mp_rj[mpin_o1], bins, color = 'cornflowerblue', label = "Orbit 1")
ax22[1].hist(mp_rj[mpin_o2], bins, color = 'blue', label = "Orbit 2")
ax22[1].hist(mp_rj[mpin_o3], bins, color = 'mediumslateblue', label = "Orbit 3")
#Stats
ax22[1].axvline(mean_inmp, color='k', linestyle='dashed', linewidth=2, label = 'Mean = 103.59 $R_{j}$, $\sigma$ = 10.07 $R_{j}$')
ax22[1].axvline(mp_rj[mpin_approach].mean(), color='darkcyan', linestyle='dashed', linewidth=2, label = 'Mean (Approach) = 80.13 $R_{j}$, $\sigma$ = 3.32 $R_{j}$')
ax22[1].axvline(mp_rj[mpin_o1].mean(), color='cornflowerblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 1) = 103.36 $R_{j}$, $\sigma$ = 8.97 $R_{j}$')
ax22[1].axvline(mp_rj[mpin_o2].mean(), color='blue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 2) = 105.16 $R_{j}$, $\sigma$ = 9.83 $R_{j}$')
ax22[1].axvline(mp_rj[mpin_o3].mean(), color='mediumslateblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 3) = 106.94 $R_{j}$, $\sigma$ = 4.59 $R_{j}$')

#Plot
ax22[2].hist(mp_rj[mpout_approach], bins, color = 'darkcyan', label = "Approach")
ax22[2].hist(mp_rj[mpout_o1], bins, color = 'cornflowerblue', label = "Orbit 1")
ax22[2].hist(mp_rj[mpout_o2], bins, color = 'blue', label = "Orbit 2")
ax22[2].hist(mp_rj[mpout_o3], bins, color = 'mediumslateblue', label = "Orbit 3")
#Stats
ax22[2].axvline(mean_outmp, color='k', linestyle='dashed', linewidth=2, label = 'Mean = 103.28 $R_{j}$, $\sigma$ = 10.09 $R_{j}$')
ax22[2].axvline(mp_rj[mpout_approach].mean(), color='darkcyan', linestyle='dashed', linewidth=2, label = 'Mean (Approach) = 86.60 $R_{j}$, $\sigma$ = 15.95 $R_{j}$')
ax22[2].axvline(mp_rj[mpout_o1].mean(), color='cornflowerblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 1) = 103.89 $R_{j}$, $\sigma$ = 8.32 $R_{j}$')
ax22[2].axvline(mp_rj[mpout_o2].mean(), color='blue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 2) = 105.16 $R_{j}$, $\sigma$ = 8.76 $R_{j}$')
ax22[2].axvline(mp_rj[mpout_o3].mean(), color='mediumslateblue', linestyle='dashed', linewidth=2, label = 'Mean (Orbit 3) = 106.92 $R_{j}$, $\sigma$ = 4.98 $R_{j}$')

#X-label
plt.xlabel('Range of Magnetopause Crossings ($R_{j}$)', fontsize = 17)
plt.xticks(np.arange(70, 120, 5))

#Y-Label
ax22[0].set_ylabel('Number of Crossings', fontsize = 15)
ax22[1].set_ylabel('Number of Crossings', fontsize = 15)
ax22[2].set_ylabel('Number of Crossings', fontsize = 15)

#Legend
ax22[0].legend(bbox_to_anchor=(1.05, 1.0), title="Inbound and Outbound data", title_fontsize = '18', ncol = 2, prop={"size":16}, loc='upper left')
ax22[1].legend(bbox_to_anchor=(1.05, 1.0), title="Inbound data", title_fontsize = '18', ncol = 2, prop={"size":16}, loc='upper left')
ax22[2].legend(bbox_to_anchor=(1.05, 1.0), title="Outbound data", title_fontsize = '18', ncol = 2, prop={"size":16}, loc='upper left')

#x-lim and y-lim (if want to zoom in)
ax22[0].set_ylim([0,10])
ax22[1].set_ylim([0,10])
ax22[2].set_ylim([0,10])

fig22.tight_layout(h_pad=2.0)

#Titles
ax22[0].set_title('(a) Magnetopause Statistics - Inbound and Outbound (97 crossings)', fontsize = 20, weight='bold')
ax22[1].set_title('(b) Magnetopause - Inbound (49 crossings)', fontsize = 20, weight='bold')
ax22[2].set_title('(c) Magnetopause - Outbound (48 crossings)', fontsize = 20, weight='bold')

#Grid
ax22[0].grid()
ax22[1].grid()
ax22[2].grid()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Determining time spent in each region; solar wind, magnetosheath, magnetosphere

# ## Time spent in solar wind

# In[66]:


#Solar wind - need to focus on time spent on out side of Bow shock. Find time intervals between out and in


# In[67]:


bs_times = bs_crossings[0:, 3].astype(str)  #crossing times 
bs_dates = bs_crossings[0:, 2].astype(str)  #crossing dates


# In[68]:


#In times and dates
time_bsin = bs_times[2::2]
dates_bsin = bs_dates[2::2]
datetime_bsin = []


# In[69]:


#To piece together dates and times
for (date, time) in zip(dates_bsin, time_bsin):
    datetime_bsin.append(date + " " + time)
    
print(datetime_bsin)


# In[70]:


#Out times and dates
time_bsout = bs_times[1::2]
dates_bsout = bs_dates[1::2]
datetime_bsout = []


# In[71]:


#To piece together dates and times
for (date, time) in zip(dates_bsout, time_bsout):
    datetime_bsout.append(date + " " + time)
    
print(datetime_bsout)


# In[72]:


#Convert from string to datetime and put in array
dt_bsin = np.array([datetime.datetime.strptime(i,"%m/%d/%Y %H:%M") for i in datetime_bsin])
dt_bsout = np.array([datetime.datetime.strptime(i,"%m/%d/%Y %H:%M") for i in datetime_bsout])


# In[73]:


#Calculate time spent in solar wind (i.e. duration between outbound and inbound bs crossings)
sw_time = dt_bsin - dt_bsout


# In[74]:


print(sw_time)


# ## Histogram of time spent in solar wind

# In[75]:


print(np.min(sw_time))
print(np.max(sw_time))


# In[76]:


print("Total time spent in solar wind:")
print(np.sum(sw_time))


# In[77]:


#fig16 = plt.figure(figsize=(20, 10),)
#ax16 = fig16.add_subplot(1,1,1) 
#bins = np.linspace(70, 115, 90)

#plt.title('Time spent in solar wind', fontsize = 18, fontweight = "bold")
#ax16.set_ylabel('Number of Time periods in this interval', fontsize = 14) 
#ax16.set_xlabel('Time spent in Solar Wind', fontsize = 14)

#plt.hist(sw_time, bins = 100, label = "Total time spent in solar wind: 3 days, 11 hrs, 8 minutes")

#plt.legend(loc='upper left', frameon=True, prop={"size":18})


# ## Time spent in magnetosphere

# In[78]:


mp_times = mp_crossings[0:, 3].astype(str)  #crossing times 
mp_dates = mp_crossings[0:, 2].astype(str)  #crossing dates


# In[79]:


#In times and dates
time_mpin = mp_times[0:-1:2] #index slicing = [start index, end index, step]
dates_mpin = mp_dates[0:-1:2]
datetime_mpin = []


# In[80]:


#To piece together dates and times
for (date, time) in zip(dates_mpin, time_mpin):
    datetime_mpin.append(date + " " + time)
    
#Remove *'s from data
datetime_mpin = [elem.replace('*', '') for elem in datetime_mpin]

print(datetime_mpin)


# In[81]:


#Out times and dates
time_mpout = mp_times[1::2]
dates_mpout = mp_dates[1::2]
datetime_mpout = []


# In[82]:


#To piece together dates and times
for (date, time) in zip(dates_mpout, time_mpout):
    datetime_mpout.append(date + " " + time)
    
#Remove *'s from data
datetime_mpout = [elem.replace('*', '') for elem in datetime_mpout]
    
print(datetime_mpout)


# In[83]:


#Convert from string to datetime and put in array
dt_mpin = np.array([datetime.datetime.strptime(i,"%m/%d/%Y %H:%M") for i in datetime_mpin])
dt_mpout = np.array([datetime.datetime.strptime(i,"%m/%d/%Y %H:%M") for i in datetime_mpout])


# In[ ]:





# In[84]:


#Calculate time spent in magnetosphere (i.e. duration between outbound and inbound mp crossings)
magsphere_time = dt_mpout - dt_mpin


# In[85]:


print("Total time spent in Magnetosphere:")
print(np.sum(magsphere_time))


# In[86]:


#Need to figure out how to calculate time in magnetosheath and what to plot for time in solar wind
#and magnetosphere


# ## Time spent in magnetosheath

# This is more difficult to calculate than time spent in solar wind or time spent in magnetosphere as I have to put the bs and mp crossings in order to use both data sets to calculate the time spent in the magnetosheath. 

# In[87]:


#import bss and mp dates and times
bss_dates = bs_crossings[0:, 2].astype(str)
bss_times = bs_crossings[0:, 3].astype(str)  
mps_dates = mp_crossings[0:, 2].astype(str)
mps_times = mp_crossings[0:, 3].astype(str)

bss_datetime = []
mps_datetime = []


# In[88]:


#To piece together dates and times
for (date, time) in zip(bss_dates, bss_times):
    bss_datetime.append(date + " " + time)
    
print(bss_datetime)


# In[89]:


#To piece together dates and times
for (date, time) in zip(mps_dates, mps_times):
    mps_datetime.append(date + " " + time)


# In[90]:


#Remove *'s from mp data
mps_datetime = [elem.replace('*', '') for elem in mps_datetime]

mps_datetime = np.array(mps_datetime)
print(mps_datetime)


# In[91]:


#Convert from string to datetime and put in array
dt_bs = np.array([datetime.datetime.strptime(i,"%m/%d/%Y %H:%M") for i in bss_datetime])
dt_mp = np.array([datetime.datetime.strptime(i,"%m/%d/%Y %H:%M") for i in mps_datetime])


# In[92]:


#Calculating total duration spent in magnetosheath by calculating individual durations
ms_time = np.array([dt_mp[0]-dt_bs[0], dt_mp[2]-dt_mp[1], dt_mp[4]-dt_mp[3],
                   dt_mp[6]-dt_mp[5], dt_mp[8]-dt_mp[7], dt_mp[10]-dt_bs[9],
                   dt_bs[1]-dt_mp[11], dt_bs[3]-dt_bs[2], dt_mp[12]-dt_bs[4],
                   dt_mp[14]-dt_mp[13], dt_mp[16]-dt_mp[15], dt_mp[18]-dt_mp[17], 
                    dt_mp[20]-dt_mp[19], dt_mp[22]-dt_mp[21], dt_bs[5]-dt_mp[23],
                   dt_bs[7]-dt_bs[6],  dt_bs[9]-dt_bs[8], dt_bs[11]-dt_bs[10],
                   dt_bs[13]-dt_bs[12], dt_bs[15]-dt_bs[14], dt_bs[17]-dt_bs[16],
                   dt_bs[19]-dt_bs[18], dt_bs[21]-dt_bs[20], dt_mp[24]-dt_bs[22],
                   dt_mp[26]-dt_mp[25], dt_mp[28]-dt_mp[27], dt_mp[30]-dt_mp[29],
                   dt_mp[32]-dt_mp[31], dt_mp[34]-dt_mp[33], dt_bs[23]-dt_mp[35],
                   dt_mp[36]-dt_bs[24], dt_mp[38]-dt_mp[37], dt_bs[25]-dt_mp[39],
                   dt_bs[27]-dt_bs[26], dt_bs[29]-dt_bs[28], dt_bs[31]-dt_bs[30],
                   dt_bs[33]-dt_bs[32], dt_bs[35]-dt_bs[34], dt_mp[40]-dt_bs[36],
                   dt_mp[42]-dt_mp[41], dt_mp[44]-dt_mp[43], dt_mp[46]-dt_mp[45],
                   dt_mp[48]-dt_mp[47], dt_mp[50]-dt_mp[49], dt_mp[52]-dt_mp[50],
                   dt_mp[54]-dt_mp[53], dt_mp[56]-dt_mp[55], dt_mp[58]-dt_mp[57],
                   dt_mp[60]-dt_mp[59], dt_mp[62]-dt_mp[61], dt_mp[64]-dt_mp[63],  
                   dt_mp[66]-dt_mp[65], dt_mp[68]-dt_mp[67], dt_mp[70]-dt_mp[69],
                   dt_mp[72]-dt_mp[71], dt_mp[74]-dt_mp[73], dt_bs[37]-dt_mp[75],
                    dt_bs[39]-dt_bs[38], dt_bs[41]-dt_bs[40], dt_bs[43]-dt_bs[42],
                    dt_bs[45]-dt_bs[44], dt_bs[47]-dt_bs[46], dt_mp[76]-dt_bs[48],
                    dt_mp[78]-dt_mp[77], dt_mp[80]-dt_mp[79], dt_mp[82]-dt_mp[81],
                    dt_mp[84]-dt_mp[83], dt_bs[49]-dt_mp[85], dt_mp[86]-dt_bs[50],
                    dt_mp[88]-dt_mp[87], dt_mp[90]-dt_mp[89], dt_mp[92]-dt_mp[91],
                    dt_mp[94]-dt_mp[93], dt_mp[96]-dt_mp[95]])


# In[93]:


#print(ms_time)
print("Total time in Magnetosheath: ")
print(np.sum(ms_time))


# In[94]:


#Could create loop to calculate duration spent in magnetosheath with all possible
#combinations of sequence. Would need to;
#-Import in/out data
#-Import bs/ms numbers
#-Put all datetime data together
#-Sort while keeping indices matching corresponding in/out data and bs/ms numbers


# # Time series - Broadband and Narrowband Kilometric Emission

# In[95]:


from scipy.io import readsav


# ## bKOM

# In[96]:


file_bkom = readsav("bKOM_2016100-2019174_timeseries_d15_channels_0-60_zlincal_calibrated.sav")


# In[97]:


print(file_bkom.keys())


# In[98]:


timeseries = file_bkom["timeseries"]
time = file_bkom["time"]
frequencies = file_bkom["frequencies"]


# In[99]:


from doy2016_to_yyyyddd import doy2016_to_yyyyddd
time = np.array(time)
time = doy2016_to_yyyyddd(time,2016)


# In[100]:


from doy_to_ymd import * 
t_hours_tmp = (time[:] - (np.array(time,dtype=int))[:])*24
time = doy_float_to_ymd(np.array(np.array(time,dtype = int), dtype = str), t_hours_tmp) 


# In[101]:


print(time)


# In[102]:


print(timeseries)


# In[103]:


#Plot Time series - bKOM
fig18 = plt.figure(figsize=(16, 12))
ax18 = fig18.add_subplot(111) 

plt.plot(time, timeseries[:,42], label = "112.43 Hz") 

#Set plot/axis titles 
plt.title('Time Series - bKOM - 112.43 Hz', fontsize = 20, weight='bold')
plt.ylabel('Time series', fontsize = 20)  
plt.xlabel('Date (DOY 2017)', fontsize = 20)  

plt.xlim(datetime.date(2017, 5, 1), datetime.date(2017, 5, 13))
plt.ylim(0,1e-14)

plt.legend(prop={"size":18})
#plt.semilogy()

date_form_orb = mdates.DateFormatter("%-j")
ax18.xaxis.set_major_formatter(date_form_orb)

plt.show()


# In[104]:


print(frequencies[42])


# ## nKOM

# In[105]:


file_nkom = readsav("nKOM_2016100-2019174_timeseries_d15_channels_0-51_zlincal_calibrated.sav")


# In[106]:


print(file_nkom.keys())


# In[107]:


ntimeseries = file_nkom["timeseries"]
ntime = file_nkom["time"]
nfrequencies = file_nkom["frequencies"]


# In[108]:


ntime = np.array(ntime)
ntime = doy2016_to_yyyyddd(ntime,2016)


# In[109]:


nt_hours_tmp = (ntime[:] - (np.array(ntime,dtype=int))[:])*24
ntime = doy_float_to_ymd(np.array(np.array(ntime,dtype = int), dtype = str), nt_hours_tmp) 


# In[110]:


print(nfrequencies)


# In[111]:


#Plot Time series - bKOM
fig19 = plt.figure(figsize=(16, 12))
ax19 = fig19.add_subplot(111) 

plt.plot(ntime, ntimeseries[:,42], label = "112.43 Hz") 

#Set plot/axis titles 
plt.title('Time Series - nKOM - 112.43 Hz', fontsize = 18, weight='bold')
plt.ylabel('Time series', fontsize = 17)  
plt.xlabel('Date (DOY 2017)', fontsize = 17)  

plt.xlim(datetime.date(2017, 5, 1), datetime.date(2017, 5, 13))
plt.ylim(0,1e-14)

plt.legend(prop={"size":18})
#plt.semilogy()

date_form_orb = mdates.DateFormatter("%-j")
ax19.xaxis.set_major_formatter(date_form_orb)

plt.show()


# # Trying to reproduce Hospodarsky figure 4

# In[112]:


#Plot Time VS Juno orbital radius - showing first full orbit 
fig23 = plt.figure(figsize=(16, 16))
ax23 = fig23.add_subplot(111) 

plt.plot(x_jso[0:56000], y_jso[0:56000], label = "Juno Trajectory") 

#plt.plot(0,0, 'ro', label = "Jupiter")

#Set plot/axis titles 
plt.title('Trajectory of Juno - JSO Co-ordinates', fontsize = 20, weight='bold')
plt.ylabel('Y co-ordinates (JSO)', fontsize = 20)  
plt.xlabel('X co-ordinates (JSO)', fontsize = 20)  

plt.xlim(-30,10)
plt.ylim(-130,-60)

plt.legend(prop={"size":18})

plt.show()


# In[113]:


print(len(x_jso))


# In[12]:


#Figure 4 Hospodarksy Paper
plt.figure(figsize = (10,10))

#Defining each different time period spent in solar wind/magnetosheath/magnetosphere
plt.plot(x_jso[0:6725], y_jso[0:6725], color = 'darkviolet', label = 'Solar Wind', linewidth = 3)
plt.plot(x_jso[6725:7167], y_jso[6725:7167], color = 'orange', label = 'Magnetosheath', linewidth = 3)
plt.plot(x_jso[7167:8044], y_jso[7167:8044], color = 'teal', label = 'Magnetosphere', linewidth = 3)
plt.plot(x_jso[8044:8167], y_jso[8044:8167], color = 'orange', linewidth = 3)
plt.plot(x_jso[8167:8176], y_jso[8167:8176], color = 'teal', linewidth = 3)
plt.plot(x_jso[8176:8196], y_jso[8176:8196], color = 'orange', linewidth = 3)
plt.plot(x_jso[8196:8270], y_jso[8196:8270], color = 'teal', linewidth = 3)
plt.plot(x_jso[8270:8347], y_jso[8270:8347], color = 'orange', linewidth = 3)
plt.plot(x_jso[8347:12639], y_jso[8347:12639], color = 'teal', linewidth = 3)
plt.plot(x_jso[12639:12758], y_jso[12639:12758], color = 'orange', linewidth = 3)
plt.plot(x_jso[12758:12876], y_jso[12758:12876], color = 'teal', linewidth = 3)
plt.plot(x_jso[12876:12901], y_jso[12876:12901], color = 'orange', linewidth = 3)
plt.plot(x_jso[12901:12948], y_jso[12901:12948], color = 'teal', linewidth = 3)
plt.plot(x_jso[12948:13434], y_jso[12948:13434], color = 'orange', linewidth = 3)
plt.plot(x_jso[13434:13551], y_jso[13434:13551], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[13551:13563], y_jso[13551:13563], color = 'orange', linewidth = 3)
plt.plot(x_jso[13563:13791], y_jso[13563:13791], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[13791:14032], y_jso[13791:14032], color = 'orange', linewidth = 3)
plt.plot(x_jso[14032:14072], y_jso[14032:14072], color = 'teal', linewidth = 3)
plt.plot(x_jso[14072:14223], y_jso[14072:14223], color = 'orange', linewidth = 3)
plt.plot(x_jso[14223:14342], y_jso[14223:14342], color = 'teal', linewidth = 3)
plt.plot(x_jso[14342:14421], y_jso[14342:14421], color = 'orange', linewidth = 3)
plt.plot(x_jso[14421:14423], y_jso[14421:14423], color = 'teal', linewidth = 3)
plt.plot(x_jso[14423:14861], y_jso[14423:14861], color = 'orange', linewidth = 3)
plt.plot(x_jso[14861:15144], y_jso[14861:15144], color = 'teal', linewidth = 3)
plt.plot(x_jso[15144:15214], y_jso[15144:15214], color = 'orange', linewidth = 3)
plt.plot(x_jso[15214:15489], y_jso[15214:15489], color = 'teal', linewidth = 3)
plt.plot(x_jso[15489:15566], y_jso[15489:15566], color = 'orange', linewidth = 3)
plt.plot(x_jso[15566:15726], y_jso[15566:15726], color = 'teal', linewidth = 3)
plt.plot(x_jso[15726:15816], y_jso[15726:15816], color = 'orange', linewidth = 3)
plt.plot(x_jso[15816:15825], y_jso[15816:15825], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[15825:16241], y_jso[15825:16241], color = 'orange', linewidth = 3)
plt.plot(x_jso[16241:16667], y_jso[16241:16667], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[16667:16995], y_jso[16667:16995], color = 'orange', linewidth = 3)
plt.plot(x_jso[16995:17802], y_jso[16995:17802], color = 'teal', linewidth = 3)
plt.plot(x_jso[17802:18129], y_jso[17802:18129], color = 'orange', linewidth = 3)
plt.plot(x_jso[18129:19246], y_jso[18129:19246], color = 'teal', linewidth = 3)
plt.plot(x_jso[19246:19293], y_jso[19246:19293], color = 'orange', linewidth = 3)
plt.plot(x_jso[19293:19311], y_jso[19293:19311], color = 'teal', linewidth = 3)
plt.plot(x_jso[19311:19627], y_jso[19311:19627], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[19627:19855], y_jso[19627:19855], color = 'orange', linewidth = 3)
plt.plot(x_jso[19855:19911], y_jso[19855:19911], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[19911:20063], y_jso[19911:20063], color = 'orange', linewidth = 3)
plt.plot(x_jso[20063:20149], y_jso[20063:20149], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[20149:20396], y_jso[20149:20396], color = 'orange', linewidth = 3)
plt.plot(x_jso[20396:20467], y_jso[20396:20467], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[20467:20653], y_jso[20467:20653], color = 'orange', linewidth = 3)
plt.plot(x_jso[20653:20728], y_jso[20653:20728], color = 'teal', linewidth = 3)
plt.plot(x_jso[20728:20887], y_jso[20728:20887], color = 'orange', linewidth = 3)
plt.plot(x_jso[20887:20989], y_jso[20887:20989], color = 'teal', linewidth = 3)
plt.plot(x_jso[20989:21022], y_jso[20989:21022], color = 'orange', linewidth = 3)
plt.plot(x_jso[21022:28066], y_jso[21022:28066], color = 'teal', linewidth = 3)
plt.plot(x_jso[28066:28196], y_jso[28066:28196], color = 'orange', linewidth = 3)
plt.plot(x_jso[28196:28261], y_jso[28196:28261], color = 'teal', linewidth = 3)
plt.plot(x_jso[28261:28797], y_jso[28261:28797], color = 'orange', linewidth = 3)
plt.plot(x_jso[28797:30224], y_jso[28797:30224], color = 'teal', linewidth = 3)
plt.plot(x_jso[30224:31076], y_jso[30224:31076], color = 'orange', linewidth = 3)
plt.plot(x_jso[31076:31763], y_jso[31076:31763], color = 'teal', linewidth = 3)
plt.plot(x_jso[31763:31926], y_jso[31763:31926], color = 'orange', linewidth = 3)
plt.plot(x_jso[31926:32051], y_jso[31926:32051], color = 'teal', linewidth = 3)
plt.plot(x_jso[32051:32126], y_jso[32051:32126], color = 'orange', linewidth = 3)
plt.plot(x_jso[32126:32572], y_jso[32126:32572], color = 'teal', linewidth = 3)
plt.plot(x_jso[32572:32658], y_jso[32572:32658], color = 'orange', linewidth = 3)
plt.plot(x_jso[32658:33048], y_jso[32658:33048], color = 'teal', linewidth = 3)
plt.plot(x_jso[33048:33083], y_jso[33048:33083], color = 'orange', linewidth = 3)
plt.plot(x_jso[33083:34289], y_jso[33083:34289], color = 'teal', linewidth = 3)
plt.plot(x_jso[34289:34715], y_jso[34289:34715], color = 'orange', linewidth = 3)
plt.plot(x_jso[34715:35005], y_jso[34715:35005], color = 'teal', linewidth = 3)
plt.plot(x_jso[35005:35045], y_jso[35005:35045], color = 'orange', linewidth = 3)
plt.plot(x_jso[35045:35181], y_jso[35045:35181], color = 'teal', linewidth = 3)
plt.plot(x_jso[35181:35202], y_jso[35181:35202], color = 'orange', linewidth = 3)
plt.plot(x_jso[35202:35347], y_jso[35202:35347], color = 'teal', linewidth = 3)
plt.plot(x_jso[35347:35395], y_jso[35347:35395], color = 'orange', linewidth = 3)
plt.plot(x_jso[35395:35435], y_jso[35395:35435], color = 'teal', linewidth = 3)
plt.plot(x_jso[35435:36053], y_jso[35435:36053], color = 'orange', linewidth = 3)
plt.plot(x_jso[36053:46075], y_jso[36053:46075], color = 'teal', linewidth = 3)
plt.plot(x_jso[46075:46186], y_jso[46075:46186], color = 'orange', linewidth = 3)
plt.plot(x_jso[46186:46216], y_jso[46186:46216], color = 'teal', linewidth = 3)
plt.plot(x_jso[46216:46576], y_jso[46216:46576], color = 'orange', linewidth = 3)
plt.plot(x_jso[46576:46584], y_jso[46576:46584], color = 'teal', linewidth = 3)
plt.plot(x_jso[46584:46973], y_jso[46584:46973], color = 'orange', linewidth = 3)
plt.plot(x_jso[46973:47222], y_jso[46973:47222], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[47222:47620], y_jso[47222:47620], color = 'orange', linewidth = 3)
plt.plot(x_jso[47620:47700], y_jso[47620:47700], color = 'teal', linewidth = 3)
plt.plot(x_jso[47700:47757], y_jso[47700:47757], color = 'orange', linewidth = 3)
plt.plot(x_jso[47757:47778], y_jso[47757:47778], color = 'teal', linewidth = 3)
plt.plot(x_jso[47778:47815], y_jso[47778:47815], color = 'orange', linewidth = 3)
plt.plot(x_jso[47815:47867], y_jso[47815:47867], color = 'teal', linewidth = 3)
plt.plot(x_jso[47867:47937], y_jso[47867:47937], color = 'orange', linewidth = 3)
plt.plot(x_jso[47937:49320], y_jso[47937:49320], color = 'teal', linewidth = 3)
plt.plot(x_jso[49320:49383], y_jso[49320:49383], color = 'orange', linewidth = 3)
plt.plot(x_jso[49383:50094], y_jso[49383:50094], color = 'teal', linewidth = 3)
plt.plot(x_jso[50094:50600], y_jso[50094:50600], color = 'orange', linewidth = 3)
plt.plot(x_jso[50600:50601], y_jso[50600:50601], color = 'darkviolet', linewidth = 3)
plt.plot(x_jso[50601:50736], y_jso[50601:50736], color = 'orange', linewidth = 3)
plt.plot(x_jso[50736:51122], y_jso[50736:51122], color = 'teal', linewidth = 3)
plt.plot(x_jso[51122:51186], y_jso[51122:51186], color = 'orange', linewidth = 3)
plt.plot(x_jso[51186:51223], y_jso[51186:51223], color = 'teal', linewidth = 3)
plt.plot(x_jso[51223:51337], y_jso[51223:51337], color = 'orange', linewidth = 3)
plt.plot(x_jso[51186:56000], y_jso[51186:56000], color = 'teal', linewidth = 3)


plt.ylim(-130, -30)
plt.xlim(-30, 20)


plt.title('Juno Trajectory: Approach and Orbits 1,2,3 (JSO Coordinates)', fontsize = 20, fontweight = 'bold')
plt.xlabel('$X_{JSO}$ $(R_j)$ \n Time in each region: \n Solar wind: 3 days, 11:08:00 \n Magnetosheath: 27 days, 14:00:00 \n Magnetosphere: 118 days, 3:04:00',fontsize = 17)
plt.ylabel('$Y_{JSO}$ $(R_j)$',fontsize = 17)

plt.grid()

plt.legend(prop={"size":18})

plt.tight_layout()

plt.show()


# In[115]:


print(time_table_jso[46973])
print(time_table_jso[47222])


# # Radio emissions case study

# ## Most compressed pass 

# In[116]:


datetime_bs = []
datetime_mp1 = []

#To piece together dates and times
for (date, time) in zip(bs_times, bs_dates):
    datetime_bs.append(date + " " + time)

for (date, time) in zip(mp_times, mp_dates):
    datetime_mp1.append(date + " " + time)

#Remove *'s from data
datetime_mp = [elem.replace('*', '') for elem in datetime_mp1] 


# In[117]:


print('Most compressed bow shock crossing:', np.min(bs_rj), 'Rj', " (", datetime_bs[np.argmin(bs_rj)], ")")
print('Most compressed magnetopause crossing:', np.min(mp_rj), 'Rj', " (", datetime_mp[np.argmin(mp_rj)], ")")
print('\n')
print('Index of most compressed bow shock crossing:', np.argmin(bs_rj))
print('Index of most compressed magnetopause crossing:', np.argmin(mp_rj))


# ## Plots showing data on day of closest bow shock crossing

# In[118]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig24 = plt.figure(figsize=(20, 10),)
ax24 = fig24.subplots(2) 

#read screenshot from autoplot data
img1 = mpimg.imread('bsmostcompressedcrossing.png')

#Plot
ax24[0].plot(time_table_rad, juno_jup_r, label = "Orbital Radius ($R_{j}$)") 
ax24[1].plot(time_table, juno_mag, 'r', label = "|B|")
ax24[1].axvline(datetime.datetime(2016, 7, 17, 15,33), color='k', linestyle='dashed', linewidth=1.5, label = 'Bow Shock crossing')
ax24[0].axvline(datetime.datetime(2016, 7, 17, 15,33), color='k', linestyle='dashed', linewidth=1.5, label = 'Bow Shock crossing')
#ax24[2].imshow(img1)

#bsmostcompressedcrossing.png

#X-label
ax24[0].set_xlabel('Date (DOY, 2016)', fontsize = 18)
ax24[1].set_xlabel('Time (HH:MM)', fontsize = 18)

#Y-Label
ax24[0].set_ylabel('Orbital Radius ($R_{j})$)', fontsize = 18)
ax24[1].set_ylabel('Magnetic field strength (nT)', fontsize = 18)


#x-lim and y-lim
ax24[0].set_xlim(datetime.datetime(2016, 7, 4, 0,0), datetime.datetime(2016, 8, 28,0,0))
ax24[0].set_ylim(-5,140)
ax24[1].set_xlim(datetime.datetime(2016, 7, 17, 0,0), datetime.datetime(2016, 7, 18,0,0))
ax24[1].set_ylim([-2,10])

fig24.tight_layout(h_pad=2.0)

ax24[0].xaxis.set_major_locator(weekly)
ax24[0].xaxis.set_minor_locator(daily)

ax24[0].set_title('(a) Orbital radius of Juno ($R_{j}$) Orbit 1', fontsize = 20, weight='bold')
ax24[1].set_title('(b) Total Magnetic field Amplitude - Day 199, 2016', fontsize = 20, weight='bold')

#Highlight time period shown on bottom plot in top plot
ax24[0].axvspan(datetime.datetime(2016, 7, 17, 0, 0), datetime.datetime(2016, 7, 18, 0, 0), color='yellow', alpha=0.5, label = 'Day 199, 2016')

#Legend
ax24[0].legend(loc = 'upper right', prop={"size":18})
ax24[1].legend(loc = 'upper right', prop={"size":18})

#Date format
date_form_orb = mdates.DateFormatter("%-j")
ax24[0].xaxis.set_major_formatter(date_form_orb)
ax24[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

ax24[1].xaxis.set_major_locator(twohourly)
ax24[1].xaxis.set_minor_locator(halfhourly)

ax24[0].grid()
ax24[1].grid()

plt.show()


# ## Plots showing data on day of closest magnetopause crossing.
# 
# ### No magnetic field data. Will plot 2nd closest magnetopause crossing.

# In[119]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig25 = plt.figure(figsize=(20, 10),)
ax25 = fig25.subplots(2) 

#Plot
ax25[0].plot(time_table_rad, juno_jup_r, label = "Orbital Radius ($R_{j}$)") 
ax25[1].plot(time_table, juno_mag, 'r', label = "|B|")
ax25[1].axvline(datetime.datetime(2016, 6, 29, 23,41), color='k', linestyle='dashed', linewidth=0.8, label = 'Bow Shock crossing')
ax25[0].axvline(datetime.datetime(2016, 6, 29, 23,41), color='k', linestyle='dashed', linewidth=0.8, label = 'Bow Shock crossing')

#X-label
ax25[0].set_xlabel('Date (DOY 2016)', fontsize = 14)
ax25[1].set_xlabel('Time (HH:MM)', fontsize = 14)

#Y-Label
ax25[0].set_ylabel('Orbital Radius ($R_{j})$)')
ax25[1].set_ylabel('Magnetic field strength (nT)')


#x-lim and y-lim
ax25[0].set_xlim(datetime.datetime(2016, 6, 4, 0,0), datetime.datetime(2016, 8, 28,0,0))
ax25[0].set_ylim(-5,140)
ax25[1].set_xlim(datetime.datetime(2016, 6, 29, 0,0), datetime.datetime(2016, 6, 30,0,0))
ax25[1].set_ylim([-2,10])

fig25.tight_layout(h_pad=2.0)


ax25[0].xaxis.set_major_locator(weekly)
ax25[0].xaxis.set_minor_locator(daily)

ax25[0].set_title('Orbital radius of Juno ($R_{j}$) (Approach - Orbit 1)', fontsize = 14, weight='bold')
ax25[1].set_title('Total Magnetic field Amplitude - Day 182 2016', fontsize = 14, weight='bold')

#Highlight time period shown on bottom plot in top plot
ax25[0].axvspan(datetime.datetime(2016, 6, 29, 0,0), datetime.datetime(2016, 6, 30,0,0), color='yellow', alpha=0.5, label = 'Day 182, 2016')

#Legend
ax25[0].legend()
ax25[1].legend()

ax25[1].xaxis.set_major_locator(twohourly)
ax25[1].xaxis.set_minor_locator(halfhourly)

ax25[0].grid()
ax25[1].grid()

#Date format
date_form_orb = mdates.DateFormatter("%-j")
ax25[0].xaxis.set_major_formatter(date_form_orb)
ax25[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

plt.show()


# ## Plots showing data on day of closest magnetopause crossing with Magnetic field data

# Opened CSV file on excel to determine closest magnetopause crossings.  Closest crossing with corresponding magnetic field data; MP8 - 07/14/2016 82 Rj

# In[120]:


#Stacked plots of magnetic field and juno trajectory and orbital radius for comparison - highlight portion of trajectory
fig26 = plt.figure(figsize=(20, 10),)
ax26 = fig26.subplots(2) 

#Plot
ax26[0].plot(time_table_rad, juno_jup_r, label = "Orbital Radius ($R_{j}$)") 
ax26[1].plot(time_table, juno_mag, 'r', label = "|B|")
ax26[1].axvline(datetime.datetime(2016, 7, 14, 21,18), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing')
ax26[0].axvline(datetime.datetime(2016, 7, 14, 21,18), color='k', linestyle='dashed', linewidth=1.5, label = 'Magnetopause crossing (Outward)')

#X-label
ax26[0].set_xlabel('Date (DOY 2016)', fontsize = 18)
ax26[1].set_xlabel('Time (HH:MM)', fontsize = 18)

#Y-Label
ax26[0].set_ylabel('Orbital Radius ($R_{j})$)', fontsize = 18)
ax26[1].set_ylabel('Magnetic field strength (nT)', fontsize = 18)


#x-lim and y-lim
ax26[0].set_xlim(datetime.datetime(2016, 7, 4, 0, 0), datetime.datetime(2016, 8, 28, 0, 0))
ax26[0].set_ylim(-5,140)
ax26[1].set_xlim(datetime.datetime(2016, 7, 14, 0, 0), datetime.datetime(2016, 7, 15, 0, 0))
ax26[1].set_ylim([-1,12])

fig26.tight_layout(h_pad=2.0)

ax26[0].xaxis.set_major_locator(weekly)
ax26[0].xaxis.set_minor_locator(daily)

ax26[0].set_title('(a) Orbital radius of Juno ($R_{j}$) (Orbit 1)', fontsize = 18, weight='bold')
ax26[1].set_title('(b) Total Magnetic field Amplitude - Day 196, 2016', fontsize = 18, weight='bold')

#Highlight time period shown on bottom plot in top plot
ax26[0].axvspan(datetime.datetime(2016, 7, 14, 0,0), datetime.datetime(2016, 7, 15,0,0), color='yellow', alpha=0.5, label = 'Day 196, 2016')

#Legend
ax26[0].legend(loc = 'upper right', prop={"size":18})
ax26[1].legend(loc = 'upper right', prop={"size":18})

ax26[1].xaxis.set_major_locator(twohourly)
ax26[1].xaxis.set_minor_locator(halfhourly)

ax26[0].grid()
ax26[1].grid()

#Date format
date_form_orb = mdates.DateFormatter("%-j")
ax26[0].xaxis.set_major_formatter(date_form_orb)
ax26[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

plt.show()


# In[ ]:





# In[121]:


print('Most rarefied bow shock crossing:', np.max(bs_rj), 'Rj', " (", datetime_bs[np.argmax(bs_rj)], ")")
print('Most rarefied magnetopause crossing:', np.max(mp_rj), 'Rj', " (", datetime_mp[np.argmax(mp_rj)], ")")
print('\n')
print('Index of most rarefied bow shock crossing:', np.argmax(bs_rj))
print('Index of most rarefied magnetopause crossing:', np.argmax(mp_rj))

#Need to look into what time period I want to use - could exclude approach.

