#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Creating a single-compartment model with noisy current
input
'''
# Import modules for plotting and NEURON itself 
#import sys
#sys.path.append('/usr/local/Cellar/neuron/7.5/lib/python3.6/site-packages')
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import neuron
import scipy
import peakutils

# Fix seed for numpy random number generation
''' I don't really think the random seek is still applying now, but it can be resumed at any moment '''
# np.random.seed(1234)

################################################################################
# Neuron topology is defined using Sections
################################################################################
soma = neuron.h.Section(name='soma')

#print out information on the soma section to stdout (terminal)
neuron.h.psection()


################################################################################
# Set the model geometry
################################################################################
soma.L = 18.8         # section length in um
soma.diam = 18.8      # section diameter in um
soma.nseg = 1       # number of segments (compartments)


################################################################################
# Set biophysical parameters
################################################################################
soma.Ra = 123       # Axial resistivity in Ohm*m
soma.cm = 1         # membrane capacitance in uF/cm2

# insert 'passive' membrane mechanism, adjust parameters.
# None: without a leak mechanism, the neuron will be a perfect integrator
soma.insert('hh')     #use the Hodgkin-Huxley channel
#soma.gnabar_hh = 0.12  # Sodium conductance in S/cm2
#soma.gkbar_hh = 0.036  # Potassium conductance in S/cm2
''' Should be using one passive leakage conductance only for the purpose of the integrate-and-fire model '''
soma.gl_hh = 0.0003    # Leak conductance in S/cm2
#soma.el_hh = -54.3     # Reversal potential in mV     
#soma(0.5).pas.g = 0.0002    # membrane conducance in S/cm2
#soma(0.5).pas.e = -65.       # passive leak reversal potential in mV


################################################################################
# Model instrumentation
################################################################################

# Attach current clamp to the neuron
iclamp = neuron.h.IClamp(0.5, sec=soma)
iclamp.delay = 0. # switch on current after delay period in ms
iclamp.dur = 1E9   # duration of stimulus current in ms
iclamp.amp = 0.2  #these parameters set the constant input parameters

#create a white noise signal with values from a Gaussian distribution with
#expected mean of zero and standard deviation of 0.25. 
#test1=sio.loadmat("/Users/songyuan26/Desktop/Wessel_Group/Python/line_function.mat")
#time_series=test1["time_series"].transpose() #need to be in the right shape
#test1=()
#time_series=time_series-min(min(time_series)) #setting the min to 0
#time_series=time_series/max(max(time_series)) #setting the max to 1

#################

###########################################
#Creating the For Loop
###########################################

#For the constant injection current
#Iconst=np.ones(50);
#ns=10000;
#rescalingfac=0.1;
#time_series=np.random.rand(ns,1)#np.ones((ns,1),dtype=np.float32);
#time_series=Iconst;
    
'''replace this with the input function/dynamics'''

#noise = neuron.h.Vector(time_series[:,0]*rescalingfac) #skip the first one because we need 10000 only; we can change the scaling over here to get different dynamics
##noise = neuron.h.Vector(np.random.randn(10000) / 4.)
#
## Play back noise signal into clamp amplitude reference with update every dt.
#dt = 0.1
#noise.play(iclamp._ref_amp, dt)
#
## print out section information again
#neuron.h.psection()

################################################################################
# Set up recording of variables
################################################################################
t = neuron.h.Vector()   # NEURON variables can be recorded using Vector objects.
v = neuron.h.Vector()   # Here, we set up recordings of time, voltage
i = neuron.h.Vector()   # and stimulus current with the record attributes. 

t.record(neuron.h._ref_t)   # recordable variables must be preceded by '_ref_'.
v.record(soma(0.5)._ref_v)
i.record(iclamp._ref_i)


################################################################################
# Simulation control
################################################################################
neuron.h.dt = dt    # simulation time resolution
tstop = 300.        # simulation duration
v_init = -65        # membrane voltage(s) at t = 0

def initialize():
    '''
    initializing function, setting the membrane voltages to v_init and
    resetting all state variables
    '''
    neuron.h.finitialize(v_init)
    neuron.h.fcurrent()

def integrate():
    '''
    run the simulation up until the simulation duration
    '''
    while neuron.h.t < tstop:
        neuron.h.fadvance()

# run simulation
initialize()
integrate()






v2=np.array(v.to_python);
t2=np.array(t.to_python);
ndcs=peakutils.indexes(v2,thres=0.0);
ndcs2=ndcs[v2[ndcs]>0];
targt=len(ndcs2)/max(t2);  # here is the firing rate we are trying to match

print(targt)

smallesterror=np.inf;
#bestT=XXX;
#bestv=XXX;
#besti=XXX;
'''

'''


#for T in np.linspace(20,500,10):
T=30
for rescalingfac1 in np.linspace(rescalingfac/2,rescalingfac*20,100):
    
#T=2000; #by changing T value changes the frequency; probably don't go too high otherwise undersampling would occur, and the neuron will smooth everything out
    ns=10000;
    t=np.linspace(0,T,ns);
    theta=2*np.pi*t;
    time_series=scipy.sin(theta);
    time_series=time_series-min(time_series) #setting the min to 0
    time_series=time_series/max(time_series) #setting the max to 1
    
    '''replace this with the input function/dynamics'''
    noise = neuron.h.Vector(time_series[1:]*(2*rescalingfac1)) #skip the first one because we need 10000 only; we can change the scaling over here to get different dynamics
    
    # Play back noise signal into clamp amplitude reference with update every dt.
    dt = 0.1
    noise.play(iclamp._ref_amp, dt)
    
    # print out section information again
    neuron.h.psection()
    
    ################################################################################
    # Set up recording of variables
    ################################################################################
    t = neuron.h.Vector()   # NEURON variables can be recorded using Vector objects.
    v = neuron.h.Vector()   # Here, we set up recordings of time, voltage
    i = neuron.h.Vector()   # and stimulus current with the record attributes. 
    
    t.record(neuron.h._ref_t)   # recordable variables must be preceded by '_ref_'.
    v.record(soma(0.5)._ref_v)
    i.record(iclamp._ref_i)
    
    
    
    ################################################################################
    # Simulation control
    ################################################################################
    neuron.h.dt = dt    # simulation time resolution
    tstop = 300.        # simulation duration
    v_init = -65        # membrane voltage(s) at t = 0
    
    def initialize():
        '''
        initializing function, setting the membrane voltages to v_init and
        resetting all state variables
        '''
        neuron.h.finitialize(v_init)
        neuron.h.fcurrent()
    
    def integrate():
        '''
        run the simulation up until the simulation duration
        '''
        while neuron.h.t < tstop:
            neuron.h.fadvance()
    
    # run simulation
    initialize()
    integrate()
    
    
    v2=np.array(v.to_python);
    t2=np.array(t.to_python);
    ndcs=peakutils.indexes(v2,thres=0.0);
    ndcs2=ndcs[v2[ndcs]>0];
    frate=len(ndcs2)/max(t2);
    currenterror=abs(frate-targt);
    if currenterror<smallesterror:
        smallesterror=currenterror;
        bestT=T;
        bestv=v;
        besti=i;
        bestrescalingfac=rescalingfac1
            
print(bestT);
print(bestrescalingfac);
print(smallesterror);














################################################################################
# Plot simulated output
################################################################################
fig, axes = plt.subplots(3)
fig.suptitle('stimulus current and neuron response')
axes[0].plot(t, besti, 'r', lw=2)
axes[0].set_ylabel('current (nA)')

axes[1].plot(t, bestv, 'r', lw=2)
axes[1].set_ylabel('voltage (mV)')
axes[1].set_xlabel('time (ms)')

axes[2].plot( time_series, 'r', lw=1)
axes[2].set_ylabel('input time series')
axes[2].set_xlabel('time (ms)')

for ax in axes: ax.axis(ax.axis('tight'))

fig.savefig('test.pdf')
plt.show()
plt.close(fig)

#fig2,ax2=plt.subplots(1)
#ax2.plot(range(len( noise)),noise)
################################################################################
# customary cleanup of object references - the psection() function may not write
# correct information if NEURON still has object references in memory.
################################################################################
#i = None
#v = None
#t = None
#iclamp = None
#soma = None
i=list(i)

################################################################################
#save the file but change the name every time
#the name is set as: python_save_test_#.mat
################################################################################


sio.savemat("/Users/songyuan26/Desktop/Wessel_Group/python/test.mat",
            {"i":list(besti),"v":list(bestv),"t":list(t)}) #need to change the saved file every time