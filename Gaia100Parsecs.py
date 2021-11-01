# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:57:49 2021

@author: Anthony
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

rawdata = np.genfromtxt('rawdata100pc.txt', skip_header=3, delimiter='\t')

ufilter = rawdata[~np.isnan(rawdata[:,1])]
uvel = ufilter[:,0]

nanfilter = rawdata[~np.isnan(rawdata).any(axis=1)]

P = nanfilter[:,0]
U = nanfilter[:,1]
V = nanfilter[:,2]
W = nanfilter[:,3]

Uhist, Ubins = np.histogram(U, bins=1500)
Vhist, Vbins = np.histogram(V, bins=1500)
Whist, Wbins = np.histogram(W, bins=1500)

def gauss(x, a, x0, sigma):
    
    return a*np.exp(-(((x-x0)**2)/(2*(sigma**2))))

def dublgauss(x, a, x0, sigma, a2, x02, sigma2):
    
    return (a*np.exp(-(((x-x0)**2)/(2*(sigma**2))))) + (a2*np.exp(-(((x-x02)**2)/(2*(sigma2**2)))))

poptU, pcovU = curve_fit(gauss, Ubins[:1500], Uhist[:1500])
poptV, pcovV = curve_fit(dublgauss, Vbins[:1500], Vhist[:1500])
poptW, pcovW = curve_fit(gauss, Wbins[:1500], Whist[:1500])

pvarU = np.diag(pcovU)
sigmaU = np.sqrt(pvarU[0])

pvarV = np.diag(pcovV)
sigmaV = np.sqrt(pvarV[0])

pvarW = np.diag(pcovW)
sigmaW = np.sqrt(pvarW[0])

#for parallax---------------------------------------------------------------------------------------------------
plt.figure(0)
plt.title("Parallax Histogram")
plt.xlabel('Parallax (")')
plt.ylabel("Frequency")
plt.xlim(0, 90)
plt.hist(P, bins=1500, range=(0.99*min(P), 1.01*max(P)), color="red")

#for U---------------------------------------------------------------------------------------------------------
plt.figure(1)
plt.xlim(-200, 200)
plt.title("${\it U}$ Velocity Histogram")
plt.xlabel("${\it U}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.hist(U, bins=1500, range=(0.99*min(U), 1.01*max(U)), color="red", label="Measured")

plt.plot(Ubins[:1500], gauss(Ubins[:1500], poptU[0], poptU[1], poptU[2]), color="b", label="Best fit")
plt.legend()
plt.savefig("Gaia1.png", dpi=300)

plt.figure(2) #U velocity best fit........................
plt.title("${\it U}$ Velocity Histogram (Best Fit)")
plt.xlabel("${\it U}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.xlim(poptU[1]-(0.1*(max(Ubins)-min(Ubins))), poptU[1]+(0.1*(max(Ubins)-min(Ubins))))
plt.ylim(0, 1.1*poptU[0])
plt.plot(Ubins[:1500], gauss(Ubins[:1500], poptU[0], poptU[1], poptU[2]), color="blue")
plt.fill_between(Ubins[:1500], gauss(Ubins[:1500], poptU[0], poptU[1], poptU[2]), alpha=50, facecolor="blue")

#for V--------------------------------------------------------------------------------------------------------
plt.figure(3)
plt.xlim(-200, 200)
plt.title("${\it V}$ Velocity Histogram")
plt.xlabel("${\it V}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.hist(V, bins=1500, range=(0.99*min(V), 1.01*max(V)), color="red", label="Measured")

plt.plot(Vbins[:1500], dublgauss(Vbins[:1500], poptV[0], poptV[1], poptV[2], poptV[3], poptV[4], poptV[5]), label = "Best fit", color = "Blue")
plt.legend()
plt.savefig("Gaia2.png", dpi=300)

plt.figure(4) #V velocity combined best fits..................
plt.title("${\it V}$ Velocity Histogram (Combined Best Fits)")
plt.xlabel("${\it V}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.xlim(min(poptV[1]-(0.1*(max(Vbins)-min(Vbins))), poptV[4]-(0.1*(max(Vbins)-min(Vbins)))), max(poptV[1]+(0.1*(max(Vbins)-min(Vbins))), poptV[4]+(0.1*(max(Vbins)-min(Vbins)))))
plt.ylim([0, 1400])

plt.plot(Vbins[:1500], dublgauss(Vbins[:1500], poptV[0], poptV[1], poptV[2], poptV[3], poptV[4], poptV[5]), color = "Blue")
plt.fill_between(Vbins[:1500], dublgauss(Vbins[:1500], poptV[0], poptV[1], poptV[2], poptV[3], poptV[4], poptV[5]), alpha=50, facecolor="blue")

plt.figure(5) #V velocity separated Guass functions..................
plt.title("${\it V}$ Velocity Histogram (Separated Best Fit Functions)")
plt.xlabel("${\it V}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.xlim(min(poptV[1]-(0.1*(max(Vbins)-min(Vbins))), poptV[4]-(0.1*(max(Vbins)-min(Vbins)))), max(poptV[1]+(0.1*(max(Vbins)-min(Vbins))), poptV[4]+(0.1*(max(Vbins)-min(Vbins)))))
plt.ylim([0, 1.1*poptV[0]])

plt.plot(Vbins[:1500], gauss(Vbins[:1500], poptV[0], poptV[1], poptV[2]), color="blue")
plt.fill_between(Vbins[:1500], gauss(Vbins[:1500], poptV[0], poptV[1], poptV[2]), alpha=50, facecolor="blue")

plt.plot(Vbins[:1500], gauss(Vbins[:1500], poptV[3], poptV[4], poptV[5]), color="lime")
plt.fill_between(Vbins[:1500], gauss(Vbins[:1500], poptV[3], poptV[4], poptV[5]), alpha=50, facecolor="lime")

plt.figure(6) #V velocity best fit 1.....................
plt.title("${\it V}$ Velocity Histogram (Best Fit 1)")
plt.xlabel("${\it V}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.xlim(poptV[1]-(0.1*(max(Vbins)-min(Vbins))), poptV[1]+(0.1*(max(Vbins)-min(Vbins))))
plt.ylim([0, 1.1*poptV[0]])

plt.plot(Vbins[:1500], gauss(Vbins[:1500], poptV[0], poptV[1], poptV[2]), color="blue")
plt.fill_between(Vbins[:1500], gauss(Vbins[:1500], poptV[0], poptV[1], poptV[2]), alpha=50, facecolor="blue")

plt.figure(7) #V velocity best fit 2.....................
plt.title("${\it V}$ Velocity Histogram (Best Fit 2)")
plt.xlabel("${\it V}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.xlim(poptV[4]-(0.1*(max(Vbins)-min(Vbins))), poptV[4]+(0.1*(max(Vbins)-min(Vbins))))
plt.ylim([0, 1.1*poptV[3]])

plt.plot(Vbins[:1500], gauss(Vbins[:1500], poptV[3], poptV[4], poptV[5]), color="lime")
plt.fill_between(Vbins[:1500], gauss(Vbins[:1500], poptV[3], poptV[4], poptV[5]), alpha=50, facecolor="lime")
#for W-------------------------------------------------------------------------------------------------------
plt.figure(8)
plt.xlim(-200, 200)
plt.title("${\it W}$ Velocity Histogram")
plt.xlabel("${\it W}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.hist(W, bins=1500, range=(0.99*min(W), 1.01*max(W)), color="red", label="Measured")
plt.xlim(-100, 100)

plt.plot(Wbins[:1500], gauss(Wbins[:1500], poptW[0], poptW[1], poptW[2]), color="blue", label="Best fit")
plt.legend()
plt.savefig("Gaia3.png", dpi=300)

plt.figure(9) #W velocity best fit......................
plt.title("${\it W}$ Velocity Histogram (Best Fit)")
plt.xlabel("${\it W}$ velocity [km/s]")
plt.ylabel("Frequency")
plt.xlim(poptW[1]-(0.05*(max(Wbins)-min(Wbins))), poptW[1]+(0.05*(max(Wbins)-min(Wbins))))
plt.ylim([0, 1.1*poptW[0]])

plt.plot(Wbins[:1500], gauss(Wbins[:1500], poptW[0], poptW[1], poptW[2]), color="blue")
plt.fill_between(Wbins[:1500], gauss(Wbins[:1500], poptW[0], poptW[1], poptW[2]), alpha=50, facecolor="blue")
#--------------------------------------------------------------------------------------
#Finding the LSR
poptVpeak, pcovVpeak = curve_fit(gauss, Vbins[:1500], Vhist[:1500])

Upeak = poptU[1]
Vpeak = poptVpeak[1]
Wpeak = poptW[1]

LSR = np.array([-1*Upeak, -1*Vpeak, -1*Wpeak])
LSRMag = np.sqrt((LSR[0]**2)+(LSR[1]**2)+(LSR[2]**2))

print("The LSR is", LSR)
print("The magnitude of the LSR is", LSRMag)