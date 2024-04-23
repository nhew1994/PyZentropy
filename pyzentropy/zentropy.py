import os
import shutil
import numpy as np
import matplotlib.pyplot as plt 

def scale_phonon_dos(input_path, output_path=None, num_atoms=5, plot=False):
    
    original_path = os.getcwd()
    
    # Create a folder to store the scaled phonon DOS
    if output_path is None:
        parent_dir = os.path.dirname(input_path)
        output_path = os.path.join(parent_dir, 'scaled_phonon_dos')

    os.makedirs(output_path, exist_ok=True)
    
    # Scale the area under the phonon DOS to 3N, where N is the number of atoms.
    # YPHON normalizes the area to 3N.
    os.chdir(input_path)
    file_list = os.listdir(input_path)
    volph_files = [file for file in file_list if 'volph' in file]
    vdos_files = [file for file in file_list if 'vdos' in file]

    num_atoms_3N = num_atoms * 3

    for i, file in enumerate(vdos_files):
        data = np.loadtxt(file)

        # Remove the negative frequencies
        data_scaled = data[data[:, 0] > 0]

        # Insert a zero frequency and zero DOS at the beginning of the array
        data_scaled = np.insert(data_scaled, 0, [0, 0], axis=0)

        area = np.trapz(data_scaled[:, 1], data_scaled[:, 0])
        data_scaled[:, 1] = data_scaled[:, 1] * num_atoms_3N / area
        area = np.trapz(data_scaled[:, 1], data_scaled[:, 0])
        np.savetxt(os.path.join(output_path, file), data_scaled)

        if plot == True:
            plt.plot(data[:, 0], data[:, 1], label='original')
            plt.plot(data_scaled[:, 0], data_scaled[:, 1], label='scaled')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('DOS (1/Hz)')
            plt.legend(edgecolor='black')
            volume_per_atom = np.loadtxt(volph_files[i])
            title = f"Volume = {round(volume_per_atom.item(), 2)} Å³/atom"
            plt.title(title)
            plt.show()

    for file in volph_files:
        shutil.copy(file, output_path)

    os.chdir(original_path)
    
    def plot_phonon_dos(path):
        pass
    
    #2. Harmonic phonon calculations - Calculate the free energy, entropy, and cv for each structure and fit to an EOS
    def load_phonon_dos():
        pass
    
    def harmonic():
        pass
    
    #3. Quasiharmonic phonon calculations - 
    #4. Partition function (Zentropy)
    #5. Properties at fixed P