#!/usr/env/bin python
# -*- coding: utf-8 -*-
'''
NEURON and Python - Creating a single-compartment model with DC
current stimulus, modified by Songyuan Geng
'''
# Import modules for plotting and NEURON itself 
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import scipy
import neuron
import peakutils

##################################################################
# Neuron topology is defined using Sections
##################################################################
soma = neuron.h.Section(name='soma')

#print out information on the soma section to terminal
neuron.h.psection()


##################################################################
# Set the model geometry
##################################################################
soma.L = 30         # section length in um
soma.diam = 30      # section diameter in um
soma.nseg = 1       # number of segments (compartments)


##################################################################
# Set biophysical parameters
##################################################################
soma.Ra = 100       # Axial resistivity in Ohm*m
soma.cm = 1         # membrane capacitance in uF/cm2

# insert 'hh model' membrane mechanism, adjust parameters.
# for the integrated-and-fire model, we will only be considering the leakage 
# voltage
soma.insert('hh')          
soma.gl_hh = 0.0003   #leakage conductance in S/cm2


##################################################################
# Model instrumentation
##################################################################

# Attach current clamp to the neuron
iclamp = neuron.h.IClamp(0.5, sec=soma)
iclamp.delay = 100. # current delay period in ms
iclamp.dur = 100.   # duration of stimulus current in ms
iclamp.amp = 0.2    # amplitude of current in nA

# print out section information again
neuron.h.psection()

##################################################################
# Set up recording of variables
##################################################################
# NEURON variables can be recorded using Vector objects. Here, we
# set up recordings of time, voltage and stimulus current with the
# record attributes.
t = neuron.h.Vector()   
v = neuron.h.Vector()
i = neuron.h.Vector()
# recordable variables must be preceded by '_ref_':
t.record(neuron.h._ref_t)   
v.record(soma(0.5)._ref_v)
i.record(iclamp._ref_i)


##################################################################
# Simulation control
##################################################################
neuron.h.dt = 0.1          # simulation time resolution
tstop = 300.        # simulation duration
v_init = -65        # membrane voltage(s) at t = 0

def initialize():
    '''
    initializing function, setting the membrane voltages to v_init
    and resetting all state variables
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

##################################################################
# Find the number of spikings for constant DC current injection
##################################################################
v2=np.array(v.to_python);
t2=np.array(t.to_python);
ndcs=peakutils.indexes(v2,thres = 0.0); #somehow the peakutils is ignoring the thres
ndcs2=ndcs[v2[ndcs]>0]  #here wer are ignoring the initial blip
targt=len(ndcs2)/(max(t2)-20); #the length of t is multiplied by a factor of 10
print(targt);   #targt is the target frequency we should be aiming at

##################################################################
# Creating the 'for loop' to find the best frequency and amplitude
# of the sine input to match the constant current injection
##################################################################
output=np.inf;
idx=0;
freq=np.linspace(0,100,num=100);
outputs=[];
for idx in range(10):
    Ts=np.linspace(0, 300, num=10000);
    Amp=0.2;
    sine=Amp*scipy.sin(freq[idx]*np.pi*Ts)+Amp/2;
    sineinj=neuron.h.Vector(sine[1:]);
    
    #play the sine inject current into the amplitude reference for every dt
    dt=0.1;
    sineinj.play(iclamp._ref_amp, dt)

    # print out section information again
    neuron.h.psection()

    ##################################################################
    # Set up recording of variables
    ##################################################################
    # NEURON variables can be recorded using Vector objects. Here, we
    # set up recordings of time, voltage and stimulus current with the
    # record attributes.
    t3 = neuron.h.Vector()   
    v3 = neuron.h.Vector()
    i3 = neuron.h.Vector()
    # recordable variables must be preceded by '_ref_':
    t3.record(neuron.h._ref_t)   
    v3.record(soma(0.5)._ref_v)
    i3.record(iclamp._ref_i)


    ##################################################################
    # Simulation control
    ##################################################################
    neuron.h.dt = 0.1          # simulation time resolution
    tstop = 300.        # simulation duration
    v_init = -65        # membrane voltage(s) at t = 0

    def initialize():
        '''
        initializing function, setting the membrane voltages to v_init
        and resetting all state variables
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

    ##################################################################
    # Find the number of spikings for constant DC current injection
    ##################################################################
    v4=np.array(v3.to_python);
    t4=np.array(t3.to_python);
    ndcs3=peakutils.indexes(v4,thres = 0.0); #somehow the peakutils is ignoring the thres
    ndcs4=ndcs[v4[ndcs]>0]  #here wer are ignoring the initial blip
    output=len(ndcs4)/(max(t4)-20); #the length of t is multiplied by a factor of 10
    outputs.append(output)

##################################################################
# Plot simulated output
##################################################################
fig, axes = plt.subplots(4)
fig.suptitle('stimulus current and point-neuron response')
axes[0].plot(t, i, 'r', lw=2)
axes[0].set_ylabel('current (nA)')

axes[1].plot(t, v, 'r', lw=2)
axes[1].set_ylabel('voltage (mV)')
axes[1].set_xlabel('time (ms)')

fig.suptitle('stimulus current and point-neuron response')
axes[2].plot(t3, i3, 'r', lw=2)
axes[2].set_ylabel('current sine (nA)')

axes[3].plot(t3, v3, 'r', lw=2)
axes[3].set_ylabel('voltage sine (mV)')
axes[3].set_xlabel('time (ms)')

# tight layout
for ax in axes: ax.axis(ax.axis('tight'))

fig.savefig('example_1.pdf')
plt.show()


##################################################################
# customary cleanup of object references - the psection() function
# may not write correct information if NEURON still has object
# references in memory.
##################################################################
plt.close(fig)
i = None
v = None
t = None
iclamp = None
soma = None
