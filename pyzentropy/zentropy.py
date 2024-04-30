import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_phonon_dos(path):
    
    original_path = os.getcwd()
    os.chdir(path)
    
    file_list = os.listdir(path)
    vdos_files = [file for file in file_list if file.startswith("vdos_")]
    volph_files = [file for file in file_list if file.startswith("volph_")]
    vdos_files.sort()
    volph_files.sort()
    
    volph_content = [float(open(file).readline().strip()) for file in volph_files]
    vdos_data = {volph_content[i]: pd.read_csv(vdos_files[i], sep="\s+", header=None, names=['Frequency (Hz)', 'DOS (1/Hz)']) 
                 for i in range(len(vdos_files))}
    
    os.chdir(original_path)
    return vdos_data


def scale_phonon_dos(path, num_atoms=5, plot=False):
    """Scales the area under the phonon DOS to 3N, where N is the number of atoms.
    YPHON normalizes the area to 3N.
    """

    vdos_data = load_phonon_dos(path)
    volumes_per_atom = list(vdos_data.keys())
    vdos_data_scaled = {}
    
    # Remove the negative frequencies
    for volume_per_atom in volumes_per_atom:
        vdos_data_scaled[volume_per_atom] = vdos_data[volume_per_atom][vdos_data[volume_per_atom]['Frequency (Hz)'] > 0]
        vdos_data_scaled[volume_per_atom] = vdos_data_scaled[volume_per_atom]
        new_row = pd.DataFrame([[0, 0]], columns=['Frequency (Hz)', 'DOS (1/Hz)'])
        vdos_data_scaled[volume_per_atom] = pd.concat([new_row, vdos_data_scaled[volume_per_atom]]).reset_index(drop=True)
    
    # Scale the area under the phonon DOS.
    num_atoms_3N = num_atoms * 3
    for volume_per_atom in volumes_per_atom:
        area = np.trapz(vdos_data_scaled[volume_per_atom]['DOS (1/Hz)'], vdos_data_scaled[volume_per_atom]['Frequency (Hz)'])
        vdos_data_scaled[volume_per_atom]['DOS (1/Hz)'] *= num_atoms_3N / area
    
    if plot == True:
        for volume_per_atom in volumes_per_atom:
            plt.plot(vdos_data[volume_per_atom]['Frequency (Hz)'], vdos_data[volume_per_atom]['DOS (1/Hz)'], label="original")
            plt.plot(vdos_data_scaled[volume_per_atom]['Frequency (Hz)'], vdos_data_scaled[volume_per_atom]['DOS (1/Hz)'], label="scaled")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("DOS (1/Hz)")
            title = f"Scaled to {round(volume_per_atom * num_atoms, 2)} Å³/{num_atoms} atoms"
            plt.title(title)
            plt.legend(edgecolor="black")
            plt.show()
    
    return vdos_data_scaled


def plot_phonon_dos(path, scale_atoms=5, save_plot=True):

    vdos_data_scaled = scale_phonon_dos(path, num_atoms=scale_atoms)
    volumes_per_atom = list(vdos_data_scaled.keys())
    
    for volume_per_atom in volumes_per_atom:
        plt.plot(vdos_data_scaled[volume_per_atom]['Frequency (Hz)']/1e12, vdos_data_scaled[volume_per_atom]['DOS (1/Hz)']*1e12, label=f"{round(volume_per_atom * scale_atoms, 2)} Å³/{scale_atoms} atoms")
    
    plt.xlabel("Frequency (THz)")
    plt.ylabel("DOS (1/THz)")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(edgecolor="black")
        
    if save_plot == True:
        plt.savefig("phonon_dos.png")
        
    plt.show()
        
        
# 2. Harmonic phonon calculations - Calculate the free energy, entropy, and cv for each structure and fit to an EOS
def harmonic():
    pass

# 3. Quasiharmonic phonon calculations -
# 4. Partition function (Zentropy)
# 5. Properties at fixed P
