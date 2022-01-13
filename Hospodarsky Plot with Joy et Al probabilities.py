#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import math
import pandas as pd


# In[19]:


matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('xtick', labelsize=20) 


# # Data

# In[5]:


xyz_jso = np.genfromtxt('juno_jup_xyz_jso_2016_2020.txt', dtype=str)


# In[6]:


xyzjso_time = xyz_jso[0:, 0].astype(str)
x_jso = xyz_jso[0:, 1].astype(float)
y_jso = xyz_jso[0:, 2].astype(float)
z_jso = xyz_jso[0:, 3].astype(float)

#Change time data type to datetime
time_table_jso = [datetime.datetime.strptime(i,"%Y-%m-%dT%H:%M:%S.%f") for i in xyzjso_time]


# # Joy et Al Equations

# ## Bow Shock

# In[3]:


#Add Joy model standoffs to dataframe
def pdyn_to_bs(Pdyn=0.319, equatorial = False, noon_midnight = False, dawn_dusk = False):
    #set dynamic pressure as an entry
    deg=180.0 / np.pi
    #create grid
    xdatapts = ((np.arange(5001.0)*0.1 - 250.0)/120.0)
    xdatapts
    amagp = -1.107 + 1.591*Pdyn**(-.25)
    bmagp = -0.566 - 0.812*Pdyn**(-.25)
    cmagp =  0.048 - 0.059*Pdyn**(-.25)
    dmagp =  0.077 - 0.038*Pdyn
    emagp = -0.874 - 0.299*Pdyn
    fmagp = -0.055 + 0.124*Pdyn
    bplot = dmagp + fmagp*xdatapts
    aplot = emagp
    cplot = amagp + bmagp*xdatapts + cmagp*(xdatapts**2)
    #split to plot dawn side (0-180)
    
    yplotplus = (-1*bplot + np.sqrt((bplot**2) - 4*aplot*cplot))/(2*aplot)
    #and split to make dusk side (180-360)
    
    yplotminus = (-1*bplot - np.sqrt((bplot**2) - 4*aplot*cplot))/(2*aplot)
    #rescale x and y to jovian radii as the calculations assume R/120
    yplotplused = 120*yplotplus
    yplotminused = 120*yplotminus
    #calculate the radial distance in Rj 0-180, then 180-360 as 2 separate halves of msphere
    xdataptsed = xdatapts*120.

    rad0plus = np.sqrt(xdataptsed*xdataptsed + yplotplused*yplotplused + 0j)
    rad180plus = np.sqrt(xdataptsed*xdataptsed + yplotminused*yplotminused + 0j)
    lt0plus = 180 - (np.arccos(xdataptsed/rad0plus)*deg)
    lt180plus = 180 + (np.arccos(xdataptsed/rad180plus)*deg)
    #put together 2 sides of MP using a dataframe
    rjltdawndf = pd.DataFrame(rad0plus)
    rjltdawndf.columns = ['Rad']
    rjltdawndf['LT']=lt0plus
    rjltduskdf = pd.DataFrame(np.flip(rad180plus))
    rjltduskdf.columns = ['Rad']
    rjltduskdf['LT']=np.flip(lt180plus)
    mprjlt=pd.concat([rjltdawndf,rjltduskdf])




#   plt.plot(rjltdawndf.LT,rjltdawndf.Rad, '-k')
#   plt.plot(rjltduskdf.LT,rjltduskdf.Rad,'-k')
    
    
    return([xdataptsed, xdataptsed],[yplotplused,yplotminused])


# In[10]:


# Dynamic presssures and corresponding percentiles
(tenth_bs_x, tenth_bs_y) = pdyn_to_bs(Pdyn=0.063, equatorial = True)
(twenty5_bs_x, twenty5th_bs_y) = pdyn_to_bs(Pdyn=0.111, equatorial = True)
(fifty_bs_x, fifty_bs_y) = pdyn_to_bs(Pdyn=0.258, equatorial = True)
(seventy5_bs_x, seventy5_bs_y) = pdyn_to_bs(Pdyn=0.382, equatorial = True)
(ninety_bs_x, ninety_bs_y) = pdyn_to_bs(Pdyn=0.579, equatorial = True)


# ## Magnetopause

# In[4]:


#Add Joy model standoffs to dataframe
def pdyn_to_mp(Pdyn=0.319, equatorial = False, noon_midnight = False, dawn_dusk = False):
    #set dynamic pressure as an entry

    
    deg=180.0 / np.pi
    #create grid
    xdatapts = ((np.arange(5001.0)*0.1 - 250.0)/120.0)
    xdatapts
    amagp = -0.134 + 0.488*Pdyn**(-.25)
    bmagp = -0.581 - 0.225*Pdyn**(-.25)
    cmagp = -0.186 - 0.016*Pdyn**(-.25)
    dmagp = -0.014 + 0.096*Pdyn
    emagp = -0.814 - 0.811*Pdyn
    fmagp = -0.050 + 0.168*Pdyn

    
    if equatorial: # z = 0, x = xdatapts, y : E*y**2+(F*x+D)*y + (A + B*x) = 0 
        aplot = emagp
        bplot = dmagp + fmagp*xdatapts
        cplot = amagp + bmagp*xdatapts + cmagp*(xdatapts**2)

    if noon_midnight: # y = 0, x = xdatapts, z : -z**2 + C*x**2 + B *x+ A = 0
        aplot = -1
        bplot = 0
        cplot = bmagp*xdatapts + bmagp*xdatapts + amagp

    if dawn_dusk:# x = 0, y = xdatapts, z : -z**2 + E*y**2+ D*y + A = 0 
        aplot = -1
        bplot = 0
        cplot =  emagp*xdatapts**2 + dmagp*amagp + amagp 

        

    #split to plot dawn side (0-180) 
    yplotplus = (-1*bplot + np.sqrt((bplot**2) - 4*aplot*cplot))/(2*aplot)
    #and split to make dusk side (180-360)
    yplotminus = (-1*bplot - np.sqrt((bplot**2) - 4*aplot*cplot))/(2*aplot)
    #rescale x and y to jovian radii as the calculations assume R/120
    yplotplused = 120*yplotplus
    yplotminused = 120*yplotminus
    #calculate the radial distance in Rj 0-180, then 180-360 as 2 separate halves of msphere
    xdataptsed = xdatapts*120.

    rad0plus = np.sqrt(xdataptsed*xdataptsed + yplotplused*yplotplused + 0j)
    rad180plus = np.sqrt(xdataptsed*xdataptsed + yplotminused*yplotminused + 0j)
    lt0plus = 180 - (np.arccos(xdataptsed/rad0plus)*deg)
    lt180plus = 180 + (np.arccos(xdataptsed/rad180plus)*deg)
    #put together 2 sides of MP using a dataframe
    rjltdawndf = pd.DataFrame(rad0plus)
    rjltdawndf.columns = ['Rad']
    rjltdawndf['LT']=lt0plus
    rjltduskdf = pd.DataFrame(np.flip(rad180plus))
    rjltduskdf.columns = ['Rad']
    rjltduskdf['LT']=np.flip(lt180plus)
    mprjlt=pd.concat([rjltdawndf,rjltduskdf])
    mprjlt
    


    #plt.plot(rjltdawndf.LT,rjltdawndf.Rad, '-b')
    #plt.plot(rjltduskdf.LT,rjltduskdf.Rad, '-b')


    return([xdataptsed, xdataptsed],[yplotplused,yplotminused])
    return


# In[11]:


# Dynamic Pressures and corresponding probabilities
(tenth_mp_x, tenth_mp_y) = pdyn_to_mp(Pdyn=0.03, equatorial = True)
(twenty5_mp_x, twenty5th_mp_y) = pdyn_to_mp(Pdyn=0.048, equatorial = True)
(fifty_mp_x, fifty_mp_y) = pdyn_to_mp(Pdyn=0.209, equatorial = True)
(seventy5_mp_x, seventy5_mp_y) = pdyn_to_mp(Pdyn=0.383, equatorial = True)
(ninety_mp_x, ninety_mp_y) = pdyn_to_mp(Pdyn=0.518, equatorial = True)


# # Plot

# In[35]:


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

#Joy et Al Model positions - 50th and 90th percentiles
plt.plot(fifty_bs_x[0], fifty_bs_y[0], c = 'k', linestyle='dashed', linewidth=1.5)#, label = 'Bow Shock, 50th Percentile')
plt.plot(fifty_bs_x[1], fifty_bs_y[1], c = 'k', linestyle='dashed', linewidth=1.5)

plt.annotate(xy=[12,-110], s = 'BS 50%', fontsize = 20)

plt.plot(ninety_bs_x[0], ninety_bs_y[0], c = 'k', linestyle='dashed', linewidth=1.5)#, label = 'Bow Shock, 90th Percentile')
plt.plot(ninety_bs_x[1], ninety_bs_y[1], c = 'k', linestyle='dashed', linewidth=1.5)

plt.annotate(xy=[12,-77], s = 'BS 90%', fontsize = 20)

plt.plot(fifty_mp_x[0], fifty_mp_y[0], c = 'b', linestyle='dashed', linewidth=1.5)#, label = 'Bow Shock, 50th Percentile')
plt.plot(fifty_mp_x[1], fifty_mp_y[1], c = 'b', linestyle='dashed', linewidth=1.5)

plt.annotate(xy=[12,-90], s = 'MP 50%', fontsize = 20, c= 'b')

plt.plot(ninety_mp_x[0], ninety_mp_y[0], c = 'b', linestyle='dashed', linewidth=1.5)#, label = 'Bow Shock, 90th Percentile')
plt.plot(ninety_mp_x[1], ninety_mp_y[1], c = 'b', linestyle='dashed', linewidth=1.5)

plt.annotate(xy=[12,-55], s = 'MP 90%', fontsize = 20, c = 'b')


plt.ylim(-130, -30)
plt.xlim(-30, 20)


plt.title('Juno Trajectory: Approach and Orbits 1,2,3 (JSO Coordinates)', fontsize = 20, fontweight = 'bold')
plt.xlabel('$X_{JSO}$ $(R_j)$ \n Time in each region: \n Solar wind: 3 days, 11:08:00 \n Magnetosheath: 27 days, 14:00:00 \n Magnetosphere: 118 days, 3:04:00',fontsize = 17)
plt.ylabel('$Y_{JSO}$ $(R_j)$',fontsize = 20)

plt.grid()

plt.legend(prop={"size":18})

plt.tight_layout()

plt.show()


# In[ ]:





# In[ ]:




