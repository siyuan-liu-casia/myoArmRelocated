import matplotlib.pyplot as plt
import numpy as np

#将肌肉激活画在一张图上
def plot_and_save_muscle_excitation(muscle_excitation, plotdir):
    names = [
    "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ", 
    "PECM1", "PECM2", "PECM3", "LAT1", "LAT2", "LAT3", "CORB", "TRIlong", 
    "TRIlat", "TRImed", "ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD", 
    "ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ", "FDS5", "FDS4", 
    "FDS3", "FDS2", "FDP5", "FDP4", "FDP3", "FDP2", "EDC5", "EDC4", "EDC3", 
    "EDC2", "EDM", "EIP", "EPL", "EPB", "FPL", "APL", "OP", "RI2", "LU_RB2", 
    "UI_UB2", "RI3", "LU_RB3", "UI_UB3", "RI4", "LU_RB4", "UI_UB4", "RI5", 
    "LU_RB5", "UI_UB5"
    ]
    muscle_excitation = np.array(muscle_excitation)
    fig, axs = plt.subplots(7, 1, figsize=(20, 60))

    for i in range(7):
        for j in range(10):
            index = i * 10 + j
            if index < len(names):
                axs[i].plot(muscle_excitation[:, index], label=names[index])
                axs[i].legend()
                axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(f'{plotdir}/MuscleExcitation.png')
    plt.close()