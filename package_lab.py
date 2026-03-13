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

def PID_RT(SP, PV, Man, MVMan, MVFF, KC, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, method='EBD'):
    """
    The function "PID_RT" needs to be included in a "for or while loop".
    SP: (or SetPoint) vector
    PV: (or Process Value) vector
    Man: (or Manual controller mode) vector (True or False)
    MVMan:(or Manual value for MV) vector
    MVFF: (or Feedforward) vector
    KC: controller gain
    Ti: integral time constant [s]
    Td: derivative time constant [s]
    alpha: TFD = alpha*Td where TFD is the derivative filter time constant [s]
    Ts: sampling period [s]
    MVMin: minimum value for MV (used for saturation and anti wind-up) :MVMax: maximum value for MV (used for saturation and anti wind-up)
    MV: MV (or Manipulated Value) vector
    MVP: MVP (or Propotional part of MV) vector
    MVI: MVI (or Integral part of MV) vector
    MVD: MVD (or Derivative part of MV) vector
    E: E (or control Error) vector
    ManFF: Activated FF in manual mode (optional: default boolean value is False)
    PVInit: Init value for PV (optional: default value is 0): used if PID_RT is ran first in the squence and no value of PV available yet.
    method: discretisation method (optional: default value is 'EBD')
    EBD-EBD: EBD for integral action and EBD for derivative action 
    EBD-TRAP: EBD for integral action and TRAP for derivative action 
    TRAP-EBD: TRAP for integral action and EBD for derivative action 
    TRAP-TRAP: TRAP for integral action and TRAP for derivative action
    The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", and "MVD".
    The appended values are based on the PID algorithm, the controller mode, and feedforward.
    Note that saturation of "MV" within the limits [MVMin MVMax] is implemented with anti wind-up.
"""

# Initialisation
    TFD = Td * alpha

    # -----------------ERROR----------------
    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else:
        E.append(SP[-1] - PV[-1])

 # ----------------Proportionnal Action-----------
    if len(MVP) == 0:
        MVP.append(KC * E[-1])
    else:
        if method == 'TRAP':
            MVP.append(0.5 * KC * (E[-1] + E[-2]))
        else:
            MVP.append(KC * E[-1])

# -----------------Integral Action---------
    if len(MVI) == 0:
        MVI.append((KC * Ts / Ti) * E[-1])  # MV[k] is MV[-1] and MV[k-1] is MV[-2]
    else:
        if method == 'TRAP':
            MVI.append(MVI[-1] + (0.5 * KC * Ts / Ti) * (E[-1] + E[-2]))
        else:
            MVI.append(MVI[-1] + (KC * Ts / Ti) * E[-1])

# ------------------Derivative Action-------------
    if len(MVD) == 0:
        MVD.append(0.0)  # ⚠️ Empêche le pic initial
        #MVD.append(((KC*Td)/(TFD+Ts))*(E[-1]))
    else:
        if method == 'TRAP':
            MVD.append((((TFD - (Ts / 2)) / (TFD + (Ts / 2))) * MVD[-1] + ((KC * Td) / (TFD + (Ts / 2))) * (E[-1] - E[-2])))
        else:
            MVD.append((TFD / (TFD + Ts)) * MVD[-1] + ((KC * Td) / (TFD + Ts)) * (E[-1] - E[-2]))

    # ------------Mode manuel + anti-wind up----------
    if Man[-1] == True:
        if ManFF:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] 
        else:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]

   # -----------------Saturation----------------------
    total_MV = MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1]
    if total_MV > MVMax:
        MVI[-1] = MVMax - MVP[-1] - MVD[-1] - MVFF[-1]
    elif total_MV < MVMin:
        MVI[-1] = MVMin - MVP[-1] - MVD[-1] - MVFF[-1]

    # Ajout sur MV
    MV.append(MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1])


class PID:
    
    def __init__(self, parameters):
        self.parameters = {}
        self.parameters['Kc'] = parameters.get('Kc', 0.0)
        self.parameters['Ti'] = parameters.get('Ti', 1.0)  # éviter division par zéro
        self.parameters['Td'] = parameters.get('Td', 0.0)
        self.parameters['Tfd'] = parameters.get('Tfd', 0.0)

        # Test si la classe PID est bien disponible
try:
    test_pid = PID({'Kc': 1, 'Ti': 1, 'Td': 0.1, 'Tfd': 0.09})
    print("✅ Classe PID définie et utilisable.")
except Exception as e:
    print("❌ Erreur :", e)


#####################