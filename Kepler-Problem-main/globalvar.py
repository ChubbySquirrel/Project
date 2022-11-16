#!/usr/bin/python
#=============================================================
# interface for global variables
#=============================================================
# function set_odepar()
def set_odepar(par):
    global odepar
    odepar = par

#=============================================================
# function get_odepar()
def get_odepar():
    global odepar
    return odepar

def set_inter(par):
    global inter
    inter = par

def get_inter():
    global inter
    return inter