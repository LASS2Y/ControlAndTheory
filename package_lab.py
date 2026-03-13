import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def LL_RT(MV,Kp,Tlag,Tlead,Ts,PV,PVInit=0,method='EBD'):
    
    """
    The function "FO_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :T: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
    
    The function "FO_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (Tlag != 0):
        K = Ts/Tlag
        if len(PV) == 0:
            PV.append(PVInit)
        
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K)) * ((1+(Tlead/(Ts)))*MV[-1] - (Tlead/(Ts))*MV[-2]))
            elif method == 'EFD':
                PV.append((1-K)*PV[-1] + K*Kp*((Tlead/(Ts)) * MV[-1] + (1 - Tlead/(Ts))*MV[-2]))
            elif method == 'TRAP':
                a = 2*Tlead/Ts
                b = 2*Tlag/Ts
                PV.append(Kp*((a+1)/(b+1))*MV[-1] + Kp*MV[-2]*((1-a)/(b+1)) - PV[-1]*((1-b)/(b+1)))            
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K)) * ((1+(Tlead/(Ts)))*MV[-1] - (Tlead/(Ts))*MV[-2]))
    else:
        PV.append(Kp*MV[-1])

def IMCTuning(Kp, Tlag1, Tlag2, theta, gamma, process="SOPDT"):
        """
        IMCTuning computes the IMC PID tuning parameters for FOPDT and SOPDT processes.
        Kp: process gain (Kp)
        Tlag1: first (main) lag time constant [s]
        Tlag2: second lag time constant [s]
        theta: delay [s]
        gamma: used to computed the desired closed loop time constant Tclp [s] (range [0.2 -> 0.9])
        process:
            FOPDT-PI: First Order Plus Dead Time for P-I control (IMC tuning case G)
            FOPDT-PID: First Order Plus Dead Time for P-I-D control (IMC tuning case H)
            SOPDT :Second Order Plus Dead Time for P-I-D control (IMC tuning case I)
            
        return : PID controller parameters Kc, Ti and Td
        """
        

        Tclp = gamma*Tlag1 
        if process=="FOPDT-PI":
            Kc = (Tlag1/(Tclp+theta))/Kp
            Ti = Tlag1
            Td = 0
        elif process=="FOPDT-PID":
            Kc= ((Tlag1 + theta/2)/(Tclp + theta/2))/Kp
            Ti = Tlag1 + theta/2
            Td = (Tlag1*theta)/(2*Tlag1+theta)
        elif process=="SOPDT": 
            Kc = ((Tlag1 + Tlag2)/(Tclp + theta))/Kp
            Ti = (Tlag1 +Tlag2)
            Td = ((Tlag1*Tlag2))/(Tlag1+Tlag2)
        else:
            Kc = (Tlag1/(Tclp+theta))/Kp
            Ti = Tlag1
            Td = 0
        return (Kc, Ti, Td)