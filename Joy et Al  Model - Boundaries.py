#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import numpy.polynomial.polynomial as nppol
import warnings
from matplotlib.pyplot import cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


# In[2]:


#Import Juno trajectory data
xyz_jso = np.genfromtxt('juno_jup_xyz_jso_2016_2020.txt', dtype=str)
x_jso = xyz_jso[0:, 1].astype(float)
y_jso = xyz_jso[0:, 2].astype(float)
z_jso = xyz_jso[0:, 3].astype(float)


# # Draw the BS / MP boundaries, given a Dynamic pressure (Pdyn) value

# In[3]:


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


# In[4]:


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


# # Retrieve the Pdyn value from a (x,y,z) set of coordinates (in JSE)

# In[5]:


def ms_boundaries_to_pdyn(x,y,z,magnetopause = False, bow_shock = False):
	
	#Inputs:
	#(x, y, z): JSE coordinates of the magnetopause or bow shock in planetary radius 

	#Ouput:
	# Pdyn (dynamic pressure) in nPa


	# In order to reduce numerical errors associated
	# with the least squares fitting process all lengths were scaled
	# by 120 (RJ /120).
	x/=120
	y/=120
	z/=120

	# Joy et al., 2002's equation:
	# A(Pdyn) + B(Pdyn)*x + C(Pdyn)*x**2 + D(Pdyn) * y + E(Pdyn)* y **2 + F(Pdyn)*x*y - z**2 = 0
	if bow_shock:
		#A = -1.107 + 1.591*Pdyn**(-.25)
		A_0 = -1.107
		A_1 = 1.591
		#B = -0.566 - 0.812*Pdyn**(-.25)
		B_0 = -0.566
		B_1 = - 0.812
		#C =  0.048 - 0.059*Pdyn**(-.25)
		C_0 =  0.048
		C_1 = - 0.059
		#D =  0.077 - 0.038*Pdyn
		D_0 = 0.077
		D_1 = - 0.038
		#E = -0.874 - 0.299*Pdyn
		E_0 = -0.874
		E_1 = - 0.299
		#F = -0.055 + 0.124*Pdyn
		F_0 = -0.055
		F_1 = + 0.124

	if magnetopause:
		#A = -0.134 + 0.488*Pdyn**(-.25)
		A_0 = -0.134
		A_1 = 0.488
		#B = -0.581 - 0.225*Pdyn**(-.25)
		B_0 = -0.581
		B_1 = - 0.225
		#C = -0.186 - 0.016*Pdyn**(-.25)
		C_0 = -0.186
		C_1 = - 0.016
		#D = -0.014 + 0.096*Pdyn
		D_0 = -0.014
		D_1 = 0.096
		#E = -0.814 - 0.811*Pdyn
		E_0 = -0.814
		E_1 = - 0.811
		#F = -0.050 + 0.168*Pdyn
		F_0 = -0.050
		F_1 = 0.168


	#Joy et al., 2020's equation with Pdyn being the unknown:
#		0 =
#		A_0 + B_0*x + C_0*x**2 + D_0*y +E_0 * y**2 +F_0* x * y - z**2
#		+  Pdyn**(-.25)*(A_1 + B_1*x + C_1*x**2) +
#		+ Pdyn*(F_1*x*y + D_1**y + E_1*y**2) 

# 		with P_tmp = Pdyn**(-1/4)
#		0 = 
#		(A_0 + B_0*x + C_0*x**2 + D_0*y +E_0 * y**2 +F_0* x * y - z**2)
#		+ P_tmp * (A_1 + B_1*x + C_1*x**2) +
#		+ P_tmp**(-4) (F_1*x*y + D_1**y + E_1*y**2)

#		0 = 
#		P_tmp**4 (A_0 + B_0*x + C_0*x**2 + D_0*y +E_0 * y**2 +F_0* x * y - z**2)
#		+ P_tmp**5 * (A_1 + B_1*x + C_1*x**2) +
#		+ (F_1*x*y + D_1**y + E_1*y**2)

#		a * P_tmp**5 + b * P_tmp**4 + f = 0

	b = A_0 +B_0*x + C_0*x**2 + D_0*y + E_0 * y**2 + F_0* x * y - z**2
	a = (A_1 + B_1*x + C_1*x**2)
	f = (F_1*x*y + D_1**y + E_1*y**2)

	f = nppol.Polynomial([f,0,0,0,b,a])


	roots = f.roots()

	roots = roots[(np.isreal(roots)) & (np.isreal(roots) > 0)]
	roots = roots.real
	#	P_tmp = Pdyn**(-1/4) --> Pdyn = tmp**(-4)
	roots = roots**(-4)

	return(roots)
	



#	#Joy et al., 2002's equation:
#	A + B*x + C*x**2 + D*y + E*y**2 + F*x*y - z**2 = 0
#
#	# 
#	if bow_shock:
#		A = -1.107 + 1.591*Pdyn**(-.25)
#		B = -0.566 - 0.812*Pdyn**(-.25)
#		C =  0.048 - 0.059*Pdyn**(-.25)
#		D =  0.077 - 0.038*Pdyn
#		E = -0.874 - 0.299*Pdyn
#		F = -0.055 + 0.124*Pdyn
#
#	if magnetopause:
#		A = -0.134 + 0.488*Pdyn**(-.25)
#		B = -0.581 - 0.225*Pdyn**(-.25)
#		C = -0.186 - 0.016*Pdyn**(-.25)
#		D = -0.014 + 0.096*Pdyn
#		E = -0.814 - 0.811*Pdyn
#		F = -0.050 + 0.168*Pdyn
#
#	# Pdyn is the only unkown:
#	<=> 0 = -1.107 + 1.591*Pdyn**(-.25) 
#		+ (-0.566 - 0.812*Pdyn**(-.25)) *x
#		+ (0.048 - 0.059*Pdyn**(-.25)) *x**2
#		+ (0.077 - 0.038*Pdyn) * y
#		+ (-0.874 - 0.299*Pdyn) * y**2
#		+ (-0.055 + 0.124*Pdyn) * x * y
#		- z**2
#
#
#
#		0 = 
#		- 1.107 -0.566*x + 0.048*x**2 + 0.077*y - 0.874 * y**2 - 0.055* x * y - z**2
#		+ 1.591*Pdyn**(-.25) 
#		- 0.812*Pdyn**(-.25)*x
#		- 0.059*Pdyn**(-.25) *x**2
#		- 0.038*Pdyn * y
#		- 0.299*Pdyn * y**2
#	    + 0.124*Pdyn * x * y
#		
#		
#	<=> 0 =
#		- 1.107 -0.566*x + 0.048*x**2 + 0.077*y - 0.874 * y**2 - 0.055* x * y - z**2
#		+  Pdyn**(-.25)*(1.591 - 0.812*x - 0.059*x**2) + Pdyn*(0.124*x*y - 0.038**y - 0.299*y**2) 
#	
#	then if:
#	b = - 1.107 -0.566*x + 0.048*x**2 + 0.077*y - 0.874 * y**2 - 0.055* x * y - z**2
#	a = (1.591 - 0.812*x - 0.059*x**2)
#	c = (0.124*x*y - 0.038**y - 0.299*y**2)
#	
#	<=> b + a* Pdyn**(-1/4)  + c* Pdyn = 0
#	
#	if P_tmp = Pdyn**(-1/4) <=> P_tmp = 1/Pdyn**(1/4) <=> P_tmp**4 = 1/Pdyn <=> Pdyn = P_tmp**-4 
#	
#	<=> b + a* P_tmp  + c* P_tmp**-4 = 0	
#	<=> b*P_tmp**4 + a* P_tmp*P_tmp**4  + c* P_tmp**-4 * P_tmp**4 = 0
#	<=> b*P_tmp**4 + a* P_tmp**5  + c = 0
#	
#	<=> a* P_tmp**5 + b*P_tmp**4 + c = 0
#	a = a & b = b & f = c
#	<=>
#	# Using the Tschirnhaus transformation: P = Y -b/a --> Y = P + b/a
#	Y**5 + p*Y**3 + q*Y**2 + r*Y +s = 0
#
#	with:
#	p = (- 2 b**2)/5a**2
#	q = (4*b**3)/25*a**2
#	r = (- 3b**4)/125*a**4
#	s = (3125*a**4*f + 4b**5)/3125a**5
#


# # Example

# In[6]:


import matplotlib.pyplot as plt
#from pdyn_to_ms_boundaries import *
#from ms_boundaries_to_pdyn import ms_boundaries_to_pdyn

#[x, y, z] position of a magnetopause crossing:
x_mp = 2.9
y_mp = -90
z_mp = 0

# determination of the dynamic pressure from [x, y, z] position (in JSE coordinate system ) of a magnetopause crossing
pdyn = ms_boundaries_to_pdyn(x_mp, y_mp, z_mp, magnetopause = True)

print(pdyn)

# determination of bow shock position (in equatorial plane, i.e. z = 0) for a given dynamic pressure
(x_eq_bs,y_eq_bs) = pdyn_to_bs(pdyn, equatorial = True)

# determination of magnetopause position (in equatorial plane, i.e. z = 0) for a given dynamic pressure
(x_eq_mp,y_eq_mp) = pdyn_to_mp(pdyn, equatorial = True)


# plotting
plt.ion()

plt.title('Magnetopause and Bow Shock for P$_{dyn}$ = '+('%.3f' % pdyn[0])+' nPa')
plt.xlabel('X$_{JSE}$ (R$_J$)')
plt.ylabel('Y$_{JSE}$ (R$_J$)')

# plotting magnetopause crossing
plt.plot(x_mp, y_mp, '+r')
# plotting magnetopause boundaries
plt.plot(x_eq_mp[0], y_eq_mp[0], '-k')
plt.plot(x_eq_mp[1], y_eq_mp[1], '-k')
# plotting bow shock boundaries
plt.plot(x_eq_bs[0], y_eq_bs[0], '-k')
plt.plot(x_eq_bs[1], y_eq_bs[1], '-k')


# # Routine for Hospodarsky crossings

# In[7]:


bs_crossing_coord = np.genfromtxt('BS_crossing _coordinates.csv', delimiter=',',dtype=str)
mp_crossing_coord = np.genfromtxt('MP_crossing_coordinates.csv', delimiter=',',dtype=str)


# In[8]:


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


# In[ ]:





# In[9]:


warnings.filterwarnings('ignore') #Hiding runtime warnings
#Plot all magnetpause crossings
for i in range(len(mpx_x)):
    #pressure for each x and y value
    pdyn = ms_boundaries_to_pdyn(mpx_x[i], mpx_y[i], mpx_z[i], magnetopause = True)
    
    # determination of bow shock position (in equatorial plane, i.e. z = 0) for a given dynamic pressure
    (x_eq_bs,y_eq_bs) = pdyn_to_bs(pdyn, equatorial = True)
    
    # determination of magnetopause position (in equatorial plane, i.e. z = 0) for a given dynamic pressure
    (x_eq_mp,y_eq_mp) = pdyn_to_mp(pdyn, equatorial = True)
    
    plt.title('Magnetopause and Bow Shock for P$_{dyn}$ = '+('%.3f' % pdyn[0])+' nPa')
    plt.xlabel('X$_{JSE}$ (R$_J$)')
    plt.ylabel('Y$_{JSE}$ (R$_J$)')
    
    plt.plot(mpx_x[i],mpx_y[i], '+r', label = 'Magnetopause crossing')
    
    # plotting magnetopause boundaries
    plt.plot(x_eq_mp[0], y_eq_mp[0], '-k', label = 'Magnetopause')
    plt.plot(x_eq_mp[1], y_eq_mp[1], '-k')
    
    # plotting bow shock boundaries
    plt.plot(x_eq_bs[0], y_eq_bs[0], '-k', color = 'blue', label = 'Bow Shock')
    plt.plot(x_eq_bs[1], y_eq_bs[1], '-k', color = 'blue')
    
    plt.legend()
    
    plt.show()


# ## Using magnetopause crossing co-ordinates to infer upstream Pdyn

# In[10]:


#z=0 for equatorial plane
pdyns = []
for i in range(len(mpx_x)):
    #pressure for each x and y value
    pdyn = ms_boundaries_to_pdyn(mpx_x[i], mpx_y[i], mpx_z[i], magnetopause = True)
    pdyns.append(pdyn)
    
pdyns = np.concatenate(pdyns, axis=0 )
print(pdyns)


# In[11]:


#Make X/Y/Z co-ordinates into one 3D array with crossing number and Pdyn
mp_coords = np.dstack((mp_no, mpx_x, mpx_y, mpx_z, pdyns))
#mp_coords = mp_coords.tolist()
print(mp_coords)


# In[12]:


#Make table showing co-ordinates and Pdyn
#keys = ['Crossing #', 'X','Y','Z', 'Pdyn']
#dict(zip(keys, zip(*mp_coords)))

#print(tabulate((mp_coords), headers='keys', tablefmt='fancy_grid'))


# In[13]:


print(np.min(pdyns))
print(np.max(pdyns))
print(np.mean(pdyns))
print(np.median(pdyns))
print(np.std(pdyns))


# ## Histogram - Pdyn inferred from magnetopause crossings

# In[14]:


fig = plt.figure(figsize=(20, 10),)
ax = fig.add_subplot(1,1,1) 
bins = np.linspace(0.18, 0.30, 30)

plt.title('Distribution of $P_{dyn}$ values inferred from magnetopause crossing co-ordinates ', fontsize = 18, fontweight = "bold")
ax.set_ylabel('Number of crossings', fontsize = 17) 
ax.set_xlabel('Range of $P_{dyn}$ (nPa)', fontsize = 17)

plt.hist(pdyns, bins, color = 'darkcyan', label = "Total number of crossings = 97")

plt.axvline(np.mean(pdyns), color='k', linestyle='dashed', linewidth=1, label = 'Mean = 0.21349 nPa, $\sigma$ = 0.02655 nPa')
plt.axvline(np.median(pdyns), color='k', linestyle='dashed', linewidth=2, label = 'Median = 0.20429 nPa')

plt.legend(loc='upper right', frameon=True, prop={"size":18})

ax.xaxis.set_minor_locator(MultipleLocator(0.005))

plt.grid()

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# ## Histogram grouped by orbit

# In[15]:


#Creating arrays that will be used for indices to indicate which orbit the data is in

mp_approach = np.arange(0,7,1)
mp_o1 = np.arange(7,47,1)
mp_o2 = np.arange(47,71,1)
mp_o3 = np.arange(71,97,1)


# In[16]:


#Calculate means and standard deviations for individual orbits
print('MAGNETOPAUSE Pdyn')
print('Mean of all data = ', pdyns.mean())
print('\n')
print('Standard deviation  of all data: ', np.std(pdyns))
print('\n')
print('Mean of approach = ', pdyns[mp_approach].mean())
print('\n')
print('Standard deviation  of approach: ', np.std(pdyns[mp_approach]))
print('\n')
print('Mean orbit 1 = ', pdyns[mp_o1].mean())
print('\n')
print('Standard deviation orbit 1: ', np.std(pdyns[mp_o1]))
print('\n')
print('Mean orbit 2 = ', pdyns[mp_o2].mean())
print('\n')
print('Standard deviation orbit 2: ', np.std(pdyns[mp_o2]))
print('\n')
print('Mean orbit 3 = ', pdyns[mp_o3].mean())
print('\n')
print('Standard deviation orbit 3: ', np.std(pdyns[mp_o3]))


# In[17]:


fig = plt.figure(figsize=(20, 10),)
ax = fig.add_subplot(1,1,1) 
bins = np.linspace(0.18, 0.30, 30)

plt.title('Distribution of $P_{dyn}$ values inferred from magnetopause crossing co-ordinates ', fontsize = 18, fontweight = "bold")
ax.set_ylabel('Number of crossings', fontsize = 20) 
ax.set_xlabel('Range of $P_{dyn}$ (nPa)', fontsize = 20)

#Plot
plt.hist(pdyns[mp_approach], bins, color = 'darkcyan', label = "Approach")
plt.hist(pdyns[mp_o1], bins, color = 'cornflowerblue', label = "Orbit 1")
plt.hist(pdyns[mp_o2], bins, color = 'blue', label = "Orbit 2")
plt.hist(pdyns[mp_o3], bins, color = 'mediumslateblue', label = "Orbit 3")

#Stats
plt.axvline(np.mean(pdyns), color='k', linestyle='dashed', linewidth=3, label = 'Mean (All data) = 0.21349 nPa, $\sigma$ = 0.02656 nPa')
plt.axvline(np.mean(pdyns[mp_approach]), color='darkcyan', linestyle='dashed', linewidth=3, label = 'APPROACH: Mean = 0.26668 nPa, $\sigma$ = 0.03014 nPa')
plt.axvline(np.mean(pdyns[mp_o1]), color='cornflowerblue', linestyle='dashed', linewidth=3, label = 'ORBIT 1: Mean = 0.21371 nPa, $\sigma$ = 0.02461 nPa')
plt.axvline(np.mean(pdyns[mp_o2]), color='blue', linestyle='dashed', linewidth=3, label = 'ORBIT 2: Mean = 0.20841 nPa, $\sigma$ = 0.02323 nPa')
plt.axvline(np.mean(pdyns[mp_o3]), color='mediumslateblue', linestyle='dashed', linewidth=3, label = 'ORBIT 3: Mean = 0.20353 nPa, $\sigma$ = 0.00844 nPa')

plt.legend(loc='upper right', frameon=True, prop={"size":18})

ax.xaxis.set_minor_locator(MultipleLocator(0.005))
ax.yaxis.set_minor_locator(MultipleLocator(1))

plt.grid()

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

#Legend
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Total number of crossings = 97", title_fontsize = '20', prop={'size': 18}, loc='upper left')

plt.show()


# In[18]:


#warnings.filterwarnings('ignore') #Hiding runtime warnings
#Plot all magnetpause crossings
#for i in range(len(bsx_x)):
    #pressure for each x and y value - bow shock
    #pdyn = ms_boundaries_to_pdyn(bsx_x[i], bsx_y[i], bsx_z[i], bow_shock = True)
    
   # print(pdyn)
    
    # determination of bow shock position (in equatorial plane, i.e. z = 0) for a given dynamic pressure
    #(x_eq_bs,y_eq_bs) = pdyn_to_bs(pdyn, equatorial = True)
    
    # determination of magnetopause position (in equatorial plane, i.e. z = 0) for a given dynamic pressure
    #(x_eq_mp,y_eq_mp) = pdyn_to_mp(pdyn, equatorial = True)
    
   # plt.title('Magnetopause and Bow Shock for P$_{dyn}$ = '+('%.3f' % pdyn[0])+' nPa')
   # plt.xlabel('X$_{JSE}$ (R$_J$)')
   # plt.ylabel('Y$_{JSE}$ (R$_J$)')
    
    #plt.figure()
   # plt.plot(bsx_x[i],bsx_y[i], '+r', label = 'Bow Shock Crossing')
    
    # plotting magnetopause boundaries
   # plt.plot(x_eq_mp[0], y_eq_mp[0], '-k', color = 'orange', label = 'Magnetopause')
   # plt.plot(x_eq_mp[1], y_eq_mp[1], '-k', color = 'orange')
    
    # plotting bow shock boundaries
   # plt.plot(x_eq_bs[0], y_eq_bs[0], '-k', label = 'Bow Shock')
   # plt.plot(x_eq_bs[1], y_eq_bs[1], '-k')
    
   # plt.legend()

   # plt.show()


# In[19]:


#for i in range(len(bsx_x)):
    #pressure for each x and y value - bow shock 
#   pdyn = ms_boundaries_to_pdyn(bsx_x[i], bsx_y[i], 0, bow_shock = True)
    
#   print(pdyn)

ms_boundaries_to_pdyn(5.76924,-125.969,22.6083, bow_shock = True)
ms_boundaries_to_pdyn(2.46977,-90.4384,-17.4085, bow_shock = True)
ms_boundaries_to_pdyn(-14.0082,-108.857,-16.8912, bow_shock = True)
ms_boundaries_to_pdyn(-0.85734,-105.34,-4.66505, bow_shock = True)


# # Magnetopause Crossings

# In[20]:


(x_eq_mp9,y_eq_mp9) = pdyn_to_mp(Pdyn=0.075, equatorial = True)
(x_eq_mp10,y_eq_mp10) = pdyn_to_mp(Pdyn=0.083, equatorial = True)
(x_eq_mp,y_eq_mp) = pdyn_to_mp(Pdyn=0.1, equatorial = True)
(x_eq_mp11,y_eq_mp11) = pdyn_to_mp(Pdyn=0.12, equatorial = True)
(x_eq_mp12,y_eq_mp12) = pdyn_to_mp(Pdyn=0.13, equatorial = True)
(x_eq_mp1,y_eq_mp1) = pdyn_to_mp(Pdyn=0.15, equatorial = True)
(x_eq_mp2,y_eq_mp2) = pdyn_to_mp(Pdyn=0.2, equatorial = True)
(x_eq_mp3,y_eq_mp3) = pdyn_to_mp(Pdyn=0.25, equatorial = True)
(x_eq_mp4,y_eq_mp4) = pdyn_to_mp(Pdyn=0.3, equatorial = True)
(x_eq_mp5,y_eq_mp5) = pdyn_to_mp(Pdyn=0.35, equatorial = True)
(x_eq_mp6,y_eq_mp6) = pdyn_to_mp(Pdyn=0.4, equatorial = True)
(x_eq_mp7,y_eq_mp7) = pdyn_to_mp(Pdyn=0.45, equatorial = True)
(x_eq_mp8,y_eq_mp8) = pdyn_to_mp(Pdyn=0.50, equatorial = True)


# In[46]:


plt.figure(figsize=(20, 15))

plt.title('Position of the magnetopause for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Juno Trajectory
plt.plot(x_jso[0:56000], y_jso[0:56000], c='k', label = "Juno Trajectory")

#Magnetopause locations for given Pdyn
plt.plot(x_eq_mp9[0], y_eq_mp9[0], '-k', c='r', label = ('P$_{dyn}$ = 0.075 nPa'))
plt.plot(x_eq_mp9[1], y_eq_mp9[1], '-k', c = 'r')
plt.plot(x_eq_mp10[0], y_eq_mp10[0], '-k', c='darkcyan', label = ('P$_{dyn}$ = 0.083 nPa'))
plt.plot(x_eq_mp10[1], y_eq_mp10[1], '-k', c = 'darkcyan')
plt.plot(x_eq_mp[0], y_eq_mp[0], '-k', c='tab:blue', label = ('P$_{dyn}$ = 0.100 nPa'))
plt.plot(x_eq_mp[1], y_eq_mp[1], '-k', c = 'tab:blue')
plt.plot(x_eq_mp1[0], y_eq_mp1[0], '-k', c='tab:orange', label = ('P$_{dyn}$ = 0.150 nPa'))
plt.plot(x_eq_mp1[1], y_eq_mp1[1], '-k', c='tab:orange')
plt.plot(x_eq_mp2[0], y_eq_mp2[0], '-k', c='tab:green', label = ('P$_{dyn}$ = 0.200 nPa'))
plt.plot(x_eq_mp2[1], y_eq_mp2[1], '-k', c='tab:green')
plt.plot(x_eq_mp3[0], y_eq_mp3[0], '-k', c='tab:red', label = ('P$_{dyn}$ = 0.250 nPa'))
plt.plot(x_eq_mp3[1], y_eq_mp3[1], '-k', c='tab:red')
plt.plot(x_eq_mp4[0], y_eq_mp4[0], '-k', c='tab:purple', label = ('P$_{dyn}$ = 0.300 nPa'))
plt.plot(x_eq_mp4[1], y_eq_mp4[1], '-k', c='tab:purple')
plt.plot(x_eq_mp5[0], y_eq_mp5[0], '-k', c='tab:cyan', label = ('P$_{dyn}$ = 0.350 nPa'))
plt.plot(x_eq_mp5[1], y_eq_mp5[1], '-k', c='tab:cyan')
plt.plot(x_eq_mp6[0], y_eq_mp6[0], '-k', c='tab:olive', label = ('P$_{dyn}$ = 0.400 nPa'))
plt.plot(x_eq_mp6[1], y_eq_mp6[1], '-k', c='tab:olive')
plt.plot(x_eq_mp7[0], y_eq_mp7[0], '-k', c='tab:gray', label = ('P$_{dyn}$ = 0.450 nPa'))
plt.plot(x_eq_mp7[1], y_eq_mp7[1], '-k', c='tab:gray')
plt.plot(x_eq_mp8[0], y_eq_mp8[0], '-k', c='m', label = ('P$_{dyn}$ = 0.500 nPa'))
plt.plot(x_eq_mp8[1], y_eq_mp8[1], '-k', c='m')

plt.plot(mpx_x,mpx_y, '+r', markersize = 19, label = 'Magnetopause crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '20', loc='upper left',prop={"size":18})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(-20, 100)
plt.ylim(-200, 200)

plt.grid()

plt.show()


# In[22]:


plt.figure(figsize=(20, 15))

plt.title('Position of the magnetopause for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Juno Trajectory
plt.plot(x_jso[0:56000], y_jso[0:56000], c='k', label = "Juno Trajectory")

#Magnetopause locations for given Pdyn
plt.plot(x_eq_mp9[0], y_eq_mp9[0], '-k', c='r', label = ('P$_{dyn}$ = 0.075 nPa'))
plt.plot(x_eq_mp9[1], y_eq_mp9[1], '-k', c = 'r')
plt.plot(x_eq_mp10[0], y_eq_mp10[0], '-k', c='darkcyan', label = ('P$_{dyn}$ = 0.083 nPa'))
plt.plot(x_eq_mp10[1], y_eq_mp10[1], '-k', c = 'darkcyan')
plt.plot(x_eq_mp[0], y_eq_mp[0], '-k', c='tab:blue', label = ('P$_{dyn}$ = 0.100 nPa'))
plt.plot(x_eq_mp[1], y_eq_mp[1], '-k', c = 'tab:blue')
plt.plot(x_eq_mp1[0], y_eq_mp1[0], '-k', c='tab:orange', label = ('P$_{dyn}$ = 0.150 nPa'))
plt.plot(x_eq_mp1[1], y_eq_mp1[1], '-k', c='tab:orange')
plt.plot(x_eq_mp2[0], y_eq_mp2[0], '-k', c='tab:green', label = ('P$_{dyn}$ = 0.200 nPa'))
plt.plot(x_eq_mp2[1], y_eq_mp2[1], '-k', c='tab:green')
plt.plot(x_eq_mp3[0], y_eq_mp3[0], '-k', c='tab:red', label = ('P$_{dyn}$ = 0.250 nPa'))
plt.plot(x_eq_mp3[1], y_eq_mp3[1], '-k', c='tab:red')
plt.plot(x_eq_mp4[0], y_eq_mp4[0], '-k', c='tab:purple', label = ('P$_{dyn}$ = 0.300 nPa'))
plt.plot(x_eq_mp4[1], y_eq_mp4[1], '-k', c='tab:purple')
plt.plot(x_eq_mp5[0], y_eq_mp5[0], '-k', c='tab:cyan', label = ('P$_{dyn}$ = 0.350 nPa'))
plt.plot(x_eq_mp5[1], y_eq_mp5[1], '-k', c='tab:cyan')
plt.plot(x_eq_mp6[0], y_eq_mp6[0], '-k', c='tab:olive', label = ('P$_{dyn}$ = 0.400 nPa'))
plt.plot(x_eq_mp6[1], y_eq_mp6[1], '-k', c='tab:olive')
plt.plot(x_eq_mp7[0], y_eq_mp7[0], '-k', c='tab:gray', label = ('P$_{dyn}$ = 0.450 nPa'))
plt.plot(x_eq_mp7[1], y_eq_mp7[1], '-k', c='tab:gray')
plt.plot(x_eq_mp8[0], y_eq_mp8[0], '-k', c='m', label = ('P$_{dyn}$ = 0.500 nPa'))
plt.plot(x_eq_mp8[1], y_eq_mp8[1], '-k', c='m')

plt.plot(mpx_x,mpx_y, '+b', markersize = 19, label = 'Magnetopause crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '20', loc='upper left',prop={"size":18})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(-25, 10)
plt.ylim(-120, -70)

plt.grid()

plt.show()


# In[23]:


plt.figure(figsize=(20, 15))

plt.title('Position of the magnetopause for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Magnetopause locations for given Pdyn
plt.plot(x_eq_mp9[0], y_eq_mp9[0], '-k', c='r', label = ('P$_{dyn}$ = 0.075 nPa'))
plt.plot(x_eq_mp9[1], y_eq_mp9[1], '-k', c = 'r')
plt.plot(x_eq_mp10[0], y_eq_mp10[0], '-k', c='darkcyan', label = ('P$_{dyn}$ = 0.083 nPa'))
plt.plot(x_eq_mp10[1], y_eq_mp10[1], '-k', c = 'darkcyan')
plt.plot(x_eq_mp[0], y_eq_mp[0], '-k', c='tab:blue', label = ('P$_{dyn}$ = 0.100 nPa'))
plt.plot(x_eq_mp[1], y_eq_mp[1], '-k', c = 'tab:blue')
plt.plot(x_eq_mp11[0], y_eq_mp11[0], '-k', c='palevioletred', label = ('P$_{dyn}$ = 0.120 nPa'))
plt.plot(x_eq_mp11[1], y_eq_mp11[1], '-k', c = 'palevioletred')
plt.plot(x_eq_mp12[0], y_eq_mp12[0], '-k', c='gold', label = ('P$_{dyn}$ = 0.130 nPa'))
plt.plot(x_eq_mp12[1], y_eq_mp12[1], '-k', c = 'gold')
plt.plot(x_eq_mp1[0], y_eq_mp1[0], '-k', c='tab:orange', label = ('P$_{dyn}$ = 0.150 nPa'))
plt.plot(x_eq_mp1[1], y_eq_mp1[1], '-k', c='tab:orange')
plt.plot(x_eq_mp2[0], y_eq_mp2[0], '-k', c='tab:green', label = ('P$_{dyn}$ = 0.200 nPa'))
plt.plot(x_eq_mp2[1], y_eq_mp2[1], '-k', c='tab:green')
plt.plot(x_eq_mp3[0], y_eq_mp3[0], '-k', c='tab:red', label = ('P$_{dyn}$ = 0.250 nPa'))
plt.plot(x_eq_mp3[1], y_eq_mp3[1], '-k', c='tab:red')
plt.plot(x_eq_mp4[0], y_eq_mp4[0], '-k', c='tab:purple', label = ('P$_{dyn}$ = 0.300 nPa'))
plt.plot(x_eq_mp4[1], y_eq_mp4[1], '-k', c='tab:purple')
plt.plot(x_eq_mp5[0], y_eq_mp5[0], '-k', c='tab:cyan', label = ('P$_{dyn}$ = 0.350 nPa'))
plt.plot(x_eq_mp5[1], y_eq_mp5[1], '-k', c='tab:cyan')
plt.plot(x_eq_mp6[0], y_eq_mp6[0], '-k', c='tab:olive', label = ('P$_{dyn}$ = 0.400 nPa'))
plt.plot(x_eq_mp6[1], y_eq_mp6[1], '-k', c='tab:olive')
plt.plot(x_eq_mp7[0], y_eq_mp7[0], '-k', c='tab:gray', label = ('P$_{dyn}$ = 0.450 nPa'))
plt.plot(x_eq_mp7[1], y_eq_mp7[1], '-k', c='tab:gray')
plt.plot(x_eq_mp8[0], y_eq_mp8[0], '-k', c='m', label = ('P$_{dyn}$ = 0.500 nPa'))
plt.plot(x_eq_mp8[1], y_eq_mp8[1], '-k', c='m')

plt.plot(mpx_x,mpx_y, '+b', markersize = 19, label = 'Magnetopause crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '20', loc='upper left',prop={"size":18})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(-50, 50)
plt.ylim(-120, -70)

plt.grid()

plt.show()


# # Bow Shock Crossings

# In[24]:


(x_eq_bs16,y_eq_bs16) = pdyn_to_bs(Pdyn=0.229, equatorial = True)
(x_eq_bs17,y_eq_bs17) = pdyn_to_bs(Pdyn=0.25, equatorial = True)
(x_eq_bs18,y_eq_bs18) = pdyn_to_bs(Pdyn=0.29, equatorial = True)
(x_eq_bs19,y_eq_bs19) = pdyn_to_bs(Pdyn=0.33, equatorial = True)
(x_eq_bs10,y_eq_bs10) = pdyn_to_bs(Pdyn=0.37, equatorial = True)
(x_eq_bs,y_eq_bs) = pdyn_to_bs(Pdyn=0.39, equatorial = True)
(x_eq_bs1,y_eq_bs1) = pdyn_to_bs(Pdyn=0.41, equatorial = True)
(x_eq_bs2,y_eq_bs2) = pdyn_to_bs(Pdyn=0.43, equatorial = True)
(x_eq_bs3,y_eq_bs3) = pdyn_to_bs(Pdyn=0.45, equatorial = True)
(x_eq_bs4,y_eq_bs4) = pdyn_to_bs(Pdyn=0.47, equatorial = True)
(x_eq_bs5,y_eq_bs5) = pdyn_to_bs(Pdyn=0.49, equatorial = True)
(x_eq_bs6,y_eq_bs6) = pdyn_to_bs(Pdyn=0.51, equatorial = True)
(x_eq_bs7,y_eq_bs7) = pdyn_to_bs(Pdyn=0.53, equatorial = True)
(x_eq_bs8,y_eq_bs8) = pdyn_to_bs(Pdyn=0.55, equatorial = True)
(x_eq_bs11,y_eq_bs11) = pdyn_to_bs(Pdyn=0.57, equatorial = True)
(x_eq_bs9,y_eq_bs9) = pdyn_to_bs(Pdyn=0.59, equatorial = True)
(x_eq_bs12,y_eq_bs12) = pdyn_to_bs(Pdyn=0.61, equatorial = True)
(x_eq_bs13,y_eq_bs13) = pdyn_to_bs(Pdyn=0.63, equatorial = True)
(x_eq_bs14,y_eq_bs14) = pdyn_to_bs(Pdyn=0.65, equatorial = True)
(x_eq_bs15,y_eq_bs15) = pdyn_to_bs(Pdyn=0.67, equatorial = True)


# In[25]:


plt.figure(figsize=(20, 15))

plt.title('Position of the Bow Shock for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Juno Trajectory
plt.plot(x_jso[0:56000], y_jso[0:56000], c='k', label = "Juno Trajectory")

#Bow shock locations for given Pdyn
plt.plot(x_eq_bs16[0], y_eq_bs16[0], '-k', c='blue', label = ('P$_{dyn}$ = 0.229 nPa'))
plt.plot(x_eq_bs16[1], y_eq_bs16[1], '-k', c = 'blue')
plt.plot(x_eq_bs10[0], y_eq_bs10[0], '-k', c='darkcyan', label = ('P$_{dyn}$ = 0.370 nPa'))
plt.plot(x_eq_bs10[1], y_eq_bs10[1], '-k', c = 'darkcyan')
plt.plot(x_eq_bs[0], y_eq_bs[0], '-k', c='tab:blue', label = ('P$_{dyn}$ = 0.390 nPa'))
plt.plot(x_eq_bs[1], y_eq_bs[1], '-k', c = 'tab:blue')
plt.plot(x_eq_bs1[0], y_eq_bs1[0], '-k', c='tab:orange', label = ('P$_{dyn}$ = 0.410 nPa'))
plt.plot(x_eq_bs1[1], y_eq_bs1[1], '-k', c='tab:orange')
plt.plot(x_eq_bs2[0], y_eq_bs2[0], '-k', c='tab:green', label = ('P$_{dyn}$ = 0.430 nPa'))
plt.plot(x_eq_bs2[1], y_eq_bs2[1], '-k', c='tab:green')
plt.plot(x_eq_bs3[0], y_eq_bs3[0], '-k', c='tab:red', label = ('P$_{dyn}$ = 0.450 nPa'))
plt.plot(x_eq_bs3[1], y_eq_bs3[1], '-k', c='tab:red')
plt.plot(x_eq_bs4[0], y_eq_bs4[0], '-k', c='tab:purple', label = ('P$_{dyn}$ = 0.470 nPa'))
plt.plot(x_eq_bs4[1], y_eq_bs4[1], '-k', c='tab:purple')
plt.plot(x_eq_bs5[0], y_eq_bs5[0], '-k', c='tab:cyan', label = ('P$_{dyn}$ = 0.490 nPa'))
plt.plot(x_eq_bs5[1], y_eq_bs5[1], '-k', c='tab:cyan')
plt.plot(x_eq_bs6[0], y_eq_bs6[0], '-k', c='tab:olive', label = ('P$_{dyn}$ = 0.510 nPa'))
plt.plot(x_eq_bs6[1], y_eq_bs6[1], '-k', c='tab:olive')
plt.plot(x_eq_bs7[0], y_eq_bs7[0], '-k', c='tab:gray', label = ('P$_{dyn}$ = 0.530 nPa'))
plt.plot(x_eq_bs7[1], y_eq_bs7[1], '-k', c='tab:gray')
plt.plot(x_eq_bs8[0], y_eq_bs8[0], '-k', c='m', label = ('P$_{dyn}$ = 0.550 nPa'))
plt.plot(x_eq_bs8[1], y_eq_bs8[1], '-k', c='m')
plt.plot(x_eq_bs11[0], y_eq_bs11[0], '-k', c='mediumvioletred', label = ('P$_{dyn}$ = 0.570 nPa'))
plt.plot(x_eq_bs11[1], y_eq_bs11[1], '-k', c='mediumvioletred')
plt.plot(x_eq_bs9[0], y_eq_bs9[0], '-k', c='yellowgreen', label = ('P$_{dyn}$ = 0.590 nPa'))
plt.plot(x_eq_bs9[1], y_eq_bs9[1], '-k', c = 'yellowgreen')
plt.plot(x_eq_bs12[0], y_eq_bs12[0], '-k', c='dodgerblue', label = ('P$_{dyn}$ = 0.610 nPa'))
plt.plot(x_eq_bs12[1], y_eq_bs12[1], '-k', c = 'dodgerblue')
plt.plot(x_eq_bs13[0], y_eq_bs13[0], '-k', c='lightsalmon', label = ('P$_{dyn}$ = 0.630 nPa'))
plt.plot(x_eq_bs13[1], y_eq_bs13[1], '-k', c = 'lightsalmon')
plt.plot(x_eq_bs14[0], y_eq_bs14[0], '-k', c='gold', label = ('P$_{dyn}$ = 0.650 nPa'))
plt.plot(x_eq_bs14[1], y_eq_bs14[1], '-k', c = 'gold')
plt.plot(x_eq_bs15[0], y_eq_bs15[0], '-k', c='steelblue', label = ('P$_{dyn}$ = 0.670 nPa'))
plt.plot(x_eq_bs15[1], y_eq_bs15[1], '-k', c = 'steelblue')

plt.plot(bsx_x,bsx_y, '+r', markersize = 19, label = 'Bow Shock crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '18', loc='upper left',prop={"size":17})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(-150, 100)
plt.ylim(-200, 200)

plt.grid()

plt.show()


# In[ ]:





# In[26]:


plt.figure(figsize=(20, 15))

plt.title('Position of the Bow Shock for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Juno Trajectory
plt.plot(x_jso[0:56000], y_jso[0:56000], c='k', label = "Juno Trajectory")

#Bow shock locations for given Pdyn
plt.plot(x_eq_bs16[0], y_eq_bs16[0], '-k', c='blue', label = ('P$_{dyn}$ = 0.229 nPa'))
plt.plot(x_eq_bs16[1], y_eq_bs16[1], '-k', c = 'blue')
plt.plot(x_eq_bs17[0], y_eq_bs17[0], '-k', c='chartreuse', label = ('P$_{dyn}$ = 0.250 nPa'))
plt.plot(x_eq_bs17[1], y_eq_bs17[1], '-k', c = 'chartreuse')
plt.plot(x_eq_bs18[0], y_eq_bs18[0], '-k', c='magenta', label = ('P$_{dyn}$ = 0.290 nPa'))
plt.plot(x_eq_bs18[1], y_eq_bs18[1], '-k', c = 'magenta')
plt.plot(x_eq_bs19[0], y_eq_bs19[0], '-k', c='plum', label = ('P$_{dyn}$ = 0.330 nPa'))
plt.plot(x_eq_bs19[1], y_eq_bs19[1], '-k', c = 'plum')
plt.plot(x_eq_bs10[0], y_eq_bs10[0], '-k', c='darkcyan', label = ('P$_{dyn}$ = 0.370 nPa'))
plt.plot(x_eq_bs10[1], y_eq_bs10[1], '-k', c = 'darkcyan')
plt.plot(x_eq_bs[0], y_eq_bs[0], '-k', c='tab:blue', label = ('P$_{dyn}$ = 0.390 nPa'))
plt.plot(x_eq_bs[1], y_eq_bs[1], '-k', c = 'tab:blue')
plt.plot(x_eq_bs1[0], y_eq_bs1[0], '-k', c='tab:orange', label = ('P$_{dyn}$ = 0.410 nPa'))
plt.plot(x_eq_bs1[1], y_eq_bs1[1], '-k', c='tab:orange')
plt.plot(x_eq_bs2[0], y_eq_bs2[0], '-k', c='tab:green', label = ('P$_{dyn}$ = 0.430 nPa'))
plt.plot(x_eq_bs2[1], y_eq_bs2[1], '-k', c='tab:green')
plt.plot(x_eq_bs3[0], y_eq_bs3[0], '-k', c='tab:red', label = ('P$_{dyn}$ = 0.450 nPa'))
plt.plot(x_eq_bs3[1], y_eq_bs3[1], '-k', c='tab:red')
plt.plot(x_eq_bs4[0], y_eq_bs4[0], '-k', c='tab:purple', label = ('P$_{dyn}$ = 0.470 nPa'))
plt.plot(x_eq_bs4[1], y_eq_bs4[1], '-k', c='tab:purple')
plt.plot(x_eq_bs5[0], y_eq_bs5[0], '-k', c='tab:cyan', label = ('P$_{dyn}$ = 0.490 nPa'))
plt.plot(x_eq_bs5[1], y_eq_bs5[1], '-k', c='tab:cyan')
plt.plot(x_eq_bs6[0], y_eq_bs6[0], '-k', c='tab:olive', label = ('P$_{dyn}$ = 0.510 nPa'))
plt.plot(x_eq_bs6[1], y_eq_bs6[1], '-k', c='tab:olive')
plt.plot(x_eq_bs7[0], y_eq_bs7[0], '-k', c='tab:gray', label = ('P$_{dyn}$ = 0.530 nPa'))
plt.plot(x_eq_bs7[1], y_eq_bs7[1], '-k', c='tab:gray')
plt.plot(x_eq_bs8[0], y_eq_bs8[0], '-k', c='m', label = ('P$_{dyn}$ = 0.550 nPa'))
plt.plot(x_eq_bs8[1], y_eq_bs8[1], '-k', c='m')
plt.plot(x_eq_bs11[0], y_eq_bs11[0], '-k', c='mediumvioletred', label = ('P$_{dyn}$ = 0.570 nPa'))
plt.plot(x_eq_bs11[1], y_eq_bs11[1], '-k', c='mediumvioletred')
plt.plot(x_eq_bs9[0], y_eq_bs9[0], '-k', c='yellowgreen', label = ('P$_{dyn}$ = 0.590 nPa'))
plt.plot(x_eq_bs9[1], y_eq_bs9[1], '-k', c = 'yellowgreen')
plt.plot(x_eq_bs12[0], y_eq_bs12[0], '-k', c='dodgerblue', label = ('P$_{dyn}$ = 0.610 nPa'))
plt.plot(x_eq_bs12[1], y_eq_bs12[1], '-k', c = 'dodgerblue')
plt.plot(x_eq_bs13[0], y_eq_bs13[0], '-k', c='lightsalmon', label = ('P$_{dyn}$ = 0.630 nPa'))
plt.plot(x_eq_bs13[1], y_eq_bs13[1], '-k', c = 'lightsalmon')
plt.plot(x_eq_bs14[0], y_eq_bs14[0], '-k', c='gold', label = ('P$_{dyn}$ = 0.650 nPa'))
plt.plot(x_eq_bs14[1], y_eq_bs14[1], '-k', c = 'gold')
plt.plot(x_eq_bs15[0], y_eq_bs15[0], '-k', c='steelblue', label = ('P$_{dyn}$ = 0.670 nPa'))
plt.plot(x_eq_bs15[1], y_eq_bs15[1], '-k', c = 'steelblue')

plt.plot(bsx_x,bsx_y, '+r', markersize = 19, label = 'Bow Shock crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '18', loc='upper left',prop={"size":17})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(-25, 10)
plt.ylim(-130, -90)

plt.grid()

plt.show()


# In[27]:


plt.figure(figsize=(20, 15))

plt.title('Position of the Bow Shock for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Bow shock locations for given Pdyn
plt.plot(x_eq_bs16[0], y_eq_bs16[0], '-k', c='blue', label = ('P$_{dyn}$ = 0.229 nPa'))
plt.plot(x_eq_bs16[1], y_eq_bs16[1], '-k', c = 'blue')
plt.plot(x_eq_bs10[0], y_eq_bs10[0], '-k', c='darkcyan', label = ('P$_{dyn}$ = 0.370 nPa'))
plt.plot(x_eq_bs10[1], y_eq_bs10[1], '-k', c = 'darkcyan')
plt.plot(x_eq_bs[0], y_eq_bs[0], '-k', c='tab:blue', label = ('P$_{dyn}$ = 0.390 nPa'))
plt.plot(x_eq_bs[1], y_eq_bs[1], '-k', c = 'tab:blue')
plt.plot(x_eq_bs1[0], y_eq_bs1[0], '-k', c='tab:orange', label = ('P$_{dyn}$ = 0.410 nPa'))
plt.plot(x_eq_bs1[1], y_eq_bs1[1], '-k', c='tab:orange')
plt.plot(x_eq_bs2[0], y_eq_bs2[0], '-k', c='tab:green', label = ('P$_{dyn}$ = 0.430 nPa'))
plt.plot(x_eq_bs2[1], y_eq_bs2[1], '-k', c='tab:green')
plt.plot(x_eq_bs3[0], y_eq_bs3[0], '-k', c='tab:red', label = ('P$_{dyn}$ = 0.450 nPa'))
plt.plot(x_eq_bs3[1], y_eq_bs3[1], '-k', c='tab:red')
plt.plot(x_eq_bs4[0], y_eq_bs4[0], '-k', c='tab:purple', label = ('P$_{dyn}$ = 0.470 nPa'))
plt.plot(x_eq_bs4[1], y_eq_bs4[1], '-k', c='tab:purple')
plt.plot(x_eq_bs5[0], y_eq_bs5[0], '-k', c='tab:cyan', label = ('P$_{dyn}$ = 0.490 nPa'))
plt.plot(x_eq_bs5[1], y_eq_bs5[1], '-k', c='tab:cyan')
plt.plot(x_eq_bs6[0], y_eq_bs6[0], '-k', c='tab:olive', label = ('P$_{dyn}$ = 0.510 nPa'))
plt.plot(x_eq_bs6[1], y_eq_bs6[1], '-k', c='tab:olive')
plt.plot(x_eq_bs7[0], y_eq_bs7[0], '-k', c='tab:gray', label = ('P$_{dyn}$ = 0.530 nPa'))
plt.plot(x_eq_bs7[1], y_eq_bs7[1], '-k', c='tab:gray')
plt.plot(x_eq_bs8[0], y_eq_bs8[0], '-k', c='m', label = ('P$_{dyn}$ = 0.550 nPa'))
plt.plot(x_eq_bs8[1], y_eq_bs8[1], '-k', c='m')
plt.plot(x_eq_bs11[0], y_eq_bs11[0], '-k', c='mediumvioletred', label = ('P$_{dyn}$ = 0.570 nPa'))
plt.plot(x_eq_bs11[1], y_eq_bs11[1], '-k', c='mediumvioletred')
plt.plot(x_eq_bs9[0], y_eq_bs9[0], '-k', c='yellowgreen', label = ('P$_{dyn}$ = 0.590 nPa'))
plt.plot(x_eq_bs9[1], y_eq_bs9[1], '-k', c = 'yellowgreen')
plt.plot(x_eq_bs12[0], y_eq_bs12[0], '-k', c='dodgerblue', label = ('P$_{dyn}$ = 0.610 nPa'))
plt.plot(x_eq_bs12[1], y_eq_bs12[1], '-k', c = 'dodgerblue')
plt.plot(x_eq_bs13[0], y_eq_bs13[0], '-k', c='lightsalmon', label = ('P$_{dyn}$ = 0.630 nPa'))
plt.plot(x_eq_bs13[1], y_eq_bs13[1], '-k', c = 'lightsalmon')
plt.plot(x_eq_bs14[0], y_eq_bs14[0], '-k', c='gold', label = ('P$_{dyn}$ = 0.650 nPa'))
plt.plot(x_eq_bs14[1], y_eq_bs14[1], '-k', c = 'gold')
plt.plot(x_eq_bs15[0], y_eq_bs15[0], '-k', c='steelblue', label = ('P$_{dyn}$ = 0.670 nPa'))
plt.plot(x_eq_bs15[1], y_eq_bs15[1], '-k', c = 'steelblue')

plt.plot(bsx_x,bsx_y, '+r', markersize = 19, label = 'Bow Shock crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '18', loc='upper left',prop={"size":17})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(-25, 10)
plt.ylim(-130, -90)

plt.grid()

plt.show()


# ## Determining Pdyns by eye for bow shock for approach and each orbit.

# In[28]:


#Approach - index 0
(x_eq_bs,y_eq_bs) = pdyn_to_bs(Pdyn=0.22907, equatorial = True)

plt.figure(figsize=(20, 15))

plt.title('Position of the Bow Shock for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Bow shock locations for given Pdyn
plt.plot(x_eq_bs[0], y_eq_bs[0], '-k', c='blue', label = ('P$_{dyn}$ = 0.22907 nPa'))
plt.plot(x_eq_bs[1], y_eq_bs[1], '-k', c = 'blue')

plt.plot(bsx_x[0],bsx_y[0], '+r', markersize = 19, label = 'Bow Shock crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '18', loc='upper left',prop={"size":17})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(4, 6)
plt.ylim(-127, -125)

plt.grid()

plt.show()


# In[29]:


#Orbit 1 - indices 1-35
(x_eq_bs,y_eq_bs) = pdyn_to_bs(Pdyn=0.49905, equatorial = True)

plt.figure(figsize=(20, 15))

plt.title('Position of the Bow Shock for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Bow shock locations for given Pdyn
plt.plot(x_eq_bs[0], y_eq_bs[0], '-k', c='blue', label = ('P$_{dyn}$ = 0.49905 nPa'))
plt.plot(x_eq_bs[1], y_eq_bs[1], '-k', c = 'blue')

plt.plot(bsx_x[35],bsx_y[35], '+r', markersize = 19, label = 'Bow Shock crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '18', loc='upper left',prop={"size":17})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(-2, 0)
plt.ylim(-104, -102)

plt.grid()

plt.show()


# In[30]:


#Orbit 3 - indices 37-50
(x_eq_bs,y_eq_bs) = pdyn_to_bs(Pdyn=0.66143, equatorial = True)

plt.figure(figsize=(20, 15))

plt.title('Position of the Bow Shock for different dynamic pressure values', fontsize = 25, fontweight = 'bold')
plt.xlabel('X$_{JSE}$ (R$_J$)', fontsize = 24)
plt.ylabel('Y$_{JSE}$ (R$_J$)', fontsize = 24)

#Bow shock locations for given Pdyn
plt.plot(x_eq_bs[0], y_eq_bs[0], '-k', c='blue', label = ('P$_{dyn}$ = 0.66143 nPa'))
plt.plot(x_eq_bs[1], y_eq_bs[1], '-k', c = 'blue')

plt.plot(bsx_x[50],bsx_y[50], '+r', markersize = 19, label = 'Bow Shock crossing')

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Dynamic Pressure Value", title_fontsize = '18', loc='upper left',prop={"size":17})

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
    
plt.xlim(-16, -14)
plt.ylim(-105, -103)

plt.grid()

plt.show()


# # Histogram - Pdyns inferred by eye from bow shock crossings

# In[31]:


bspdyns = np.genfromtxt('BS_Pdyn.csv', delimiter=',', dtype=str)
bspdyn = bspdyns[:, 1].astype(float)


# In[32]:


print(np.min(bspdyn))
print(np.max(bspdyn))
print(np.mean(bspdyn))
print(np.median(bspdyn))
print(np.std(bspdyn))


# In[33]:


fig = plt.figure(figsize=(20, 10),)
ax = fig.add_subplot(1,1,1) 
bins = np.linspace(0.22, 0.60, 38)

plt.title('Distribution of $P_{dyn}$ values inferred from bow shock crossing co-ordinates ', fontsize = 18, fontweight = "bold")
ax.set_ylabel('Number of crossings', fontsize = 17) 
ax.set_xlabel('Range of $P_{dyn}$ (nPa)', fontsize = 17)

plt.hist(bspdyn, bins, color = 'darkcyan', label = "Total number of crossings = 51")

plt.axvline(np.mean(bspdyn), color='k', linestyle='dashed', linewidth=2, label = 'Mean = 0.47446 nPa, $\sigma$ = 0.09497 nPa')
plt.axvline(np.median(bspdyn), color='k', linestyle='dashed', linewidth=3, label = 'Median = 0.47067 nPa')

plt.legend(loc='upper right', frameon=True, prop={"size":18})

ax.xaxis.set_minor_locator(MultipleLocator(0.005))

plt.grid()

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.show()


# ### Logarithmic x axis for comparison to Jackman and Arridge paper

# In[34]:


fig = plt.figure(figsize=(20, 10),)
ax = fig.add_subplot(1,1,1) 
bins = np.linspace(10e-3, 10, 130)

plt.title('Distribution of $P_{dyn}$ values inferred from bow shock crossing co-ordinates ', fontsize = 18, fontweight = "bold")
ax.set_ylabel('Number of crossings', fontsize = 17) 
ax.set_xlabel('Range of $P_{dyn}$ (nPa)', fontsize = 17)

plt.hist(bspdyn, bins, color = 'darkcyan', label = "Total number of crossings = 51")

plt.legend(loc='upper right', frameon=True, prop={"size":18})

ax.xaxis.set_minor_locator(MultipleLocator(0.005))

plt.grid()

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

plt.semilogx()

plt.show()


# ## Histogram grouped by orbit 

# In[35]:


#Creating arrays that will be used for indices to indicate which orbit the data is in

bs_approach = np.array([0])
bs_o1 = np.arange(1,37,1)
#bs_o2 = NONE
bs_o3 = np.arange(37,51,1)


# In[36]:


#Calculate means and standard deviations for individual orbits
#All orbits and approach
print('BOW SHOCK Pdyn')
print('Mean: ', bspdyn.mean())
print('\n')
print('Standard deviation: ', np.std(bspdyn))
print('\n')
print('Mean on approach = ', bspdyn[bs_approach].mean())
print('\n')
print('Standard deviation on approach: ', np.std(bspdyn[bs_approach]))
print('\n')
print('Mean orbit 1 = ', bspdyn[bs_o1].mean())
print('\n')
print('Standard deviation orbit 1: ', np.std(bspdyn[bs_o1]))
print('\n')
print('Mean orbit 3 = ', bspdyn[bs_o3].mean())
print('\n')
print('Standard deviation orbit 3: ', np.std(bspdyn[bs_o3]))


# In[37]:


fig = plt.figure(figsize=(20, 10),)
ax = fig.add_subplot(1,1,1) 
bins = np.linspace(0.22, 0.70, 48)

plt.title('Distribution of $P_{dyn}$ values inferred from bow shock crossing co-ordinates ', fontsize = 20, fontweight = "bold")
ax.set_ylabel('Number of crossings', fontsize = 20) 
ax.set_xlabel('Range of $P_{dyn}$ (nPa)', fontsize = 20)

#Plot
plt.hist(bspdyn[bs_approach], bins, color = 'darkcyan', label = "Approach")
plt.hist(bspdyn[bs_o1], bins, color = 'cornflowerblue', label = "Orbit 1")
plt.hist(bspdyn[bs_o3], bins, color = 'mediumslateblue', label = "Orbit 3")

#Stats
plt.axvline(np.mean(bspdyn), color='k', linestyle='dashed', linewidth=3, label = 'Mean (All data) = 0.47447 nPa, $\sigma$ = 0.09497 nPa')
plt.axvline(np.mean(bspdyn[bs_approach]), color='darkcyan', linestyle='dashed', linewidth=3, label = 'APPROACH: Mean = 0.22907 nPa, $\sigma$ = 0.0 nPa')
plt.axvline(np.mean(bspdyn[bs_o1]), color='cornflowerblue', linestyle='dashed', linewidth=3, label = 'ORBIT 1: Mean = 0.44267 nPa, $\sigma$ = 0.07588 nPa')
plt.axvline(np.mean(bspdyn[bs_o3]), color='mediumslateblue', linestyle='dashed', linewidth=3, label = 'ORBIT 3: Mean = 0.57374 nPa, $\sigma$ = 0.03591 nPa')

plt.legend(loc='upper right', frameon=True, prop={"size":18})

ax.xaxis.set_minor_locator(MultipleLocator(0.01))
ax.yaxis.set_minor_locator(MultipleLocator(1))

plt.grid()

plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)

#Legend
plt.legend(bbox_to_anchor=(1.05, 1.0), title="Total number of crossings = 51", title_fontsize = '18', prop={'size': 18}, loc='upper left')

plt.show()


# ## Corentin suggestion to verify if model is plotting correctly: Plot r vs z

# In[38]:


#Juno trajectory
x2 = x_jso**2
y2 = y_jso**2

r = np.sqrt(x2 + y2)


# In[39]:


plt.figure(figsize=(20, 15))

plt.title('Juno Trajectory - $\sqrt{{X_{JSE}}^2 + {Y_{JSE}}^2}$ VS Z ', fontsize = 25, fontweight = 'bold')
plt.xlabel('$\sqrt{{X_{JSE}}^2 + {Y_{JSE}}^2}$ (R$_J$)', fontsize = 24)
plt.ylabel('Z$_{JSE}$ (R$_J$)', fontsize = 24)

plt.plot(r[0:56000],z_jso[0:56000], c = 'k', linewidth = 1.5, label = 'Juno Trajectory')

plt.ylim(-60, 60)
plt.xlim(0,120)

plt.legend()

plt.show()


# In[40]:


#Bow shock
bsx2 = bsx_x**2
bsy2 = bsx_y**2

bsr = np.sqrt(bsx2 + bsy2)


# In[41]:


plt.figure(figsize=(20, 15))

plt.title('Bow Shock crossings -$\sqrt{{X_{JSE}}^2 + {Y_{JSE}}^2}$ VS Z ', fontsize = 25, fontweight = 'bold')
plt.xlabel('$\sqrt{{X_{JSE}}^2 + {Y_{JSE}}^2}$ (R$_J$)', fontsize = 24)
plt.ylabel('Z$_{JSE}$ (R$_J$)', fontsize = 24)

plt.plot(r[0:56000],z_jso[0:56000],c = 'k', linewidth = 1.5, label = 'Juno Trajectory')
plt.plot(bsr,bsx_z, '+r', markersize = 15, label = 'Bow Shock crossing')

plt.ylim(-20, 15)
plt.xlim(-5,120)

plt.legend()

plt.show()


# In[42]:


#Magnetopause
mpx2 = mpx_x**2
mpy2 = mpx_y**2

mpr = np.sqrt(mpx2 + mpy2)


# In[43]:


plt.figure(figsize=(20, 15))

plt.title('Magnetopause crossings - $\sqrt{{X_{JSE}}^2 + {Y_{JSE}}^2}$ VS Z ', fontsize = 25, fontweight = 'bold')
plt.xlabel('$\sqrt{{X_{JSE}}^2 + {Y_{JSE}}^2}$ (R$_J$)', fontsize = 24)
plt.ylabel('Z$_{JSE}$ (R$_J$)', fontsize = 24)

plt.plot(r[0:56000],z_jso[0:56000], c = 'k', linewidth = 1.5, label = 'Juno Trajectory')
plt.plot(mpr,mpx_z, '+r', markersize = 15, label = 'Magnetopause crossing')

plt.ylim(-20, 15)
plt.xlim(-5,120)

plt.legend()

plt.show()


# In[ ]:




