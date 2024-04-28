import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


def load_phonon_dos(path):
    
    original_path = os.getcwd()
    os.chdir(path)
    
    file_list = os.listdir(path)
    vdos_files = [file for file in file_list if file.startswith("vdos_")]
    volph_files = [file for file in file_list if file.startswith("volph_")]
    vdos_files.sort()
    volph_files.sort()
    vdos_data = {file: np.loadtxt(file) for file in vdos_files}
    volph_data = {file: np.loadtxt(file) for file in volph_files}

    os.chdir(original_path)
    return vdos_data, volph_data

def scale_phonon_dos(input_path, output_path=None, num_atoms=5, plot=False):
    """Scales the area under the phonon DOS to 3N, where N is the number of atoms.
    YPHON normalizes the area to 3N.
    """

    original_path = os.getcwd()

    # Create a folder to store the scaled phonon DOS
    if output_path is None:
        parent_dir = os.path.dirname(input_path)
        output_path = os.path.join(parent_dir, "scaled_phonon_dos")
    os.makedirs(output_path, exist_ok=True)

    os.chdir(input_path)
    vdos_data, volph_data = load_phonon_dos(input_path)
    vdos_files = list(vdos_data.keys())
    volph_files = list(volph_data.keys())
    
    # Scale the area under the phonon DOS.
    num_atoms_3N = num_atoms * 3
    vdos_data_scaled = {file: np.insert(vdos_data[file][vdos_data[file][:, 0] > 0], 0, [0,0], axis=0) for file in vdos_files}
    area = {file: np.trapz(vdos_data_scaled[file][:, 1], vdos_data_scaled[file][:, 0]) for file in vdos_files}
    for file in vdos_files:
        vdos_data_scaled[file][:, 1] *= num_atoms_3N / area[file]
        np.savetxt(os.path.join(output_path, file), vdos_data_scaled[file])
    
    if plot == True:
        for i, file in enumerate(vdos_files):
            plt.plot(vdos_data[file][:, 0] / 1e12, vdos_data[file][:, 1] * 1e12, label="original")
            plt.plot(vdos_data_scaled[file][:, 0] / 1e12, vdos_data_scaled[file][:, 1] * 1e12, label="scaled")
            plt.xlabel("Frequency (THz)")
            plt.ylabel("DOS (1/THz)")
            plt.legend(edgecolor="black")
            volume_per_atom = volph_data[volph_files[i]]
            title = f"Volume = {round(volume_per_atom.item(), 2)} Å³/atom"
            plt.title(title)
            plt.show()
    
    for file in volph_files:
        shutil.copy(file, output_path)
        
    os.chdir(original_path)

# Rewrite the function below!
def plot_phonon_dos(path, scale_atoms=5, save_plot=True):

    original_path = os.getcwd()
    os.chdir(path)
    file_list = os.listdir(path)
    volph_files = [file for file in file_list if "volph" in file]
    vdos_files = [file for file in file_list if "vdos" in file]

    for i, file in enumerate(vdos_files):
        
        data = np.loadtxt(file)
        area = np.trapz(data[:, 1]/1e12, data[:, 0]*1e12) 
        num_atoms = area / 3 
        scaling_factor = scale_atoms / num_atoms
        volume_per_atom = np.loadtxt(volph_files[i])
        volume = volume_per_atom * scale_atoms 
        label = f'{round(volume,1)} Å³ / {scale_atoms} atoms' 
        
        plt.plot(data[:, 0]/1e12, data[:, 1]*1e12*scaling_factor, label=label)
        ymin, ymax = plt.gca().get_ylim() 
        plt.ylim(ymin, ymax * 1.02) 
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Frequency (THz)", fontsize=14)
        plt.ylabel("DOS (1/THz)", fontsize=14)
        plt.legend(edgecolor="black", fontsize=12)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.25)
    
    new_area = round(np.trapz(data[:, 1]*1e12*scaling_factor, data[:, 0]/1e12),2)
    
    os.chdir(original_path)
    if save_plot:
        plt.savefig(os.path.join('phonon_dos.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f'The new area under the curve is {new_area} and the number of atoms is {scale_atoms}.')
        
        
# 2. Harmonic phonon calculations - Calculate the free energy, entropy, and cv for each structure and fit to an EOS
def harmonic():
    pass

# 3. Quasiharmonic phonon calculations -
# 4. Partition function (Zentropy)
# 5. Properties at fixed P
