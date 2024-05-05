import os
import numpy as np
import pandas as pd
import scipy.constants
import matplotlib.pyplot as plt


def load_phonon_dos(path):

    original_path = os.getcwd()
    os.chdir(path)

    file_list = os.listdir(path)
    vdos_files = [file for file in file_list if file.startswith("vdos_")]
    volph_files = [file for file in file_list if file.startswith("volph_")]
    vdos_files.sort()
    volph_files.sort()

    dataframes = []
    for i in range(len(vdos_files)):
        volph_content = float(open(volph_files[i]).readline().strip())
        df = pd.read_csv(
            vdos_files[i],
            sep="\s+",
            header=None,
            names=["Frequency (Hz)", "DOS (1/Hz)"],
        )
        df.insert(0, "Volume (Å³/atom)", volph_content)
        dataframes.append(df)

    vdos_data = pd.concat(dataframes)

    os.chdir(original_path)

    return vdos_data


def scale_phonon_dos(path, num_atoms=5, plot=False):
    """
    Scales the area under the phonon DOS to 3N, where N is the number of atoms.
    YPHON normalizes the area to 3N.
    """

    vdos_data = load_phonon_dos(path)

    # Remove all of the negative frequencies
    vdos_data_scaled = vdos_data[vdos_data["Frequency (Hz)"] > 0].reset_index(drop=True)

    # Add a row of zero frequency and DOS to the beginning of each volume
    volumes = np.sort(vdos_data["Volume (Å³/atom)"].unique())
    final_df = pd.DataFrame(columns=vdos_data_scaled.columns)
    for volume in volumes:
        filtered_df = vdos_data_scaled[vdos_data_scaled["Volume (Å³/atom)"] == volume]
        new_row = pd.DataFrame([[volume, 0, 0]], columns=vdos_data_scaled.columns)
        filtered_df = pd.concat([new_row, filtered_df])
        if final_df.empty:
            final_df = filtered_df
        else:
            final_df = pd.concat([final_df, filtered_df])
    vdos_data_scaled = final_df.reset_index(drop=True)

    # Scale the phonon DOS to 3N
    num_atoms_3N = num_atoms * 3
    for volume in volumes:
        frequency = vdos_data_scaled[vdos_data_scaled["Volume (Å³/atom)"] == volume][
            "Frequency (Hz)"
        ]
        dos = vdos_data_scaled[vdos_data_scaled["Volume (Å³/atom)"] == volume][
            "DOS (1/Hz)"
        ]
        area = np.trapz(dos, frequency)
        vdos_data_scaled.loc[
            vdos_data_scaled["Volume (Å³/atom)"] == volume, "DOS (1/Hz)"
        ] = (dos * num_atoms_3N / area)

    if plot == True:
        for volume in volumes:
            # Plot the original DOS
            frequency = vdos_data[vdos_data["Volume (Å³/atom)"] == volume][
                "Frequency (Hz)"
            ]
            dos = vdos_data[vdos_data["Volume (Å³/atom)"] == volume]["DOS (1/Hz)"]
            plt.plot(frequency, dos, label="Original")

            # Plot the scaled DOS
            frequency = vdos_data_scaled[
                vdos_data_scaled["Volume (Å³/atom)"] == volume
            ]["Frequency (Hz)"]
            dos = vdos_data_scaled[vdos_data_scaled["Volume (Å³/atom)"] == volume][
                "DOS (1/Hz)"
            ]
            plt.plot(frequency, dos, label=f"Scaled to {num_atoms} atoms")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("DOS (1/Hz)")
            plt.legend(edgecolor="black")
            plt.title(f"Volume: {volume} Å³/atom")
            plt.show()

    return vdos_data_scaled


# TODO: Rewrite this function
def plot_phonon_dos(path, scale_atoms=5, save_plot=True):

    vdos_data_scaled = scale_phonon_dos(path, num_atoms=scale_atoms)
    volumes_per_atom = list(vdos_data_scaled.keys())

    for volume_per_atom in volumes_per_atom:
        plt.plot(
            vdos_data_scaled[volume_per_atom]["Frequency (Hz)"] / 1e12,
            vdos_data_scaled[volume_per_atom]["DOS (1/Hz)"] * 1e12,
            label=f"{round(volume_per_atom * scale_atoms, 2)} Å³/{scale_atoms} atoms",
        )

    plt.xlabel("Frequency (THz)")
    plt.ylabel("DOS (1/THz)")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(edgecolor="black")

    if save_plot == True:
        plt.savefig("phonon_dos.png")

    plt.show()


# 2. Harmonic phonon calculations - Calculate the free energy, entropy, and cv for each structure and fit to an EOS
def harmonic(path, temp_range, plot=True):

    vdos_data_scaled = scale_phonon_dos(path)
    volumes_per_atom = list(vdos_data_scaled.keys())

    for volume_per_atom in volumes_per_atom:
        frequency_diff = vdos_data_scaled[volume_per_atom]["Frequency (Hz)"][
            1:
        ].reset_index(drop=True) - vdos_data_scaled[volume_per_atom]["Frequency (Hz)"][
            :-1
        ].reset_index(
            drop=True
        )
        frequency_diff = pd.concat([pd.Series([0]), frequency_diff]).reset_index(
            drop=True
        )
        vdos_data_scaled[volume_per_atom]["Frequency Difference (Hz)"] = frequency_diff
        frequency_mid = (
            vdos_data_scaled[volume_per_atom]["Frequency (Hz)"][1:].reset_index(
                drop=True
            )
            + vdos_data_scaled[volume_per_atom]["Frequency (Hz)"][:-1].reset_index(
                drop=True
            )
        ) * 0.5
        frequency_mid = pd.concat([pd.Series([0]), frequency_mid]).reset_index(
            drop=True
        )
        vdos_data_scaled[volume_per_atom]["Middle Frequency (Hz)"] = frequency_mid
        dos_mid = (
            vdos_data_scaled[volume_per_atom]["DOS (1/Hz)"][1:].reset_index(drop=True)
            + vdos_data_scaled[volume_per_atom]["DOS (1/Hz)"][:-1].reset_index(
                drop=True
            )
        ) * 0.5
        dos_mid = pd.concat([pd.Series([0]), dos_mid]).reset_index(drop=True)
        vdos_data_scaled[volume_per_atom]["Middle DOS (1/Hz)"] = dos_mid

    k_B = (
        scipy.constants.Boltzmann / scipy.constants.electron_volt
    )  # The Boltzmann constant in eV/K
    h = (
        scipy.constants.Planck / scipy.constants.electron_volt
    )  # The Planck's constant in eVs

    harmonic_properties = {
        volumes_per_atom[i]: pd.DataFrame(temp_range, columns=["Temperature (K)"])
        for i in range(len(volumes_per_atom))
    }
    for volume_per_atom in volumes_per_atom:
        free_energy = []
        internal_energy = []
        entropy = []
        cv = []
        for temp in temp_range:
            mid_f = vdos_data_scaled[volume_per_atom]["Middle Frequency (Hz)"][1:]
            df = vdos_data_scaled[volume_per_atom]["Frequency Difference (Hz)"][1:]
            mid_dos = vdos_data_scaled[volume_per_atom]["Middle DOS (1/Hz)"][1:]

            constant = (h * mid_f) / (2 * k_B * temp)

            A = df * mid_dos * np.log(2 * np.sinh(constant))
            free_energy.append((k_B * temp * np.sum(A)) / 5)

            A = df * mid_dos * (h * mid_f) * np.cosh(constant) / np.sinh(constant)
            internal_energy.append((0.5 * np.sum(A)) / 5)

            A = constant * np.cosh(constant) / np.sinh(constant) - np.log(
                2 * np.sinh(constant)
            )
            entropy.append((k_B * np.sum(df * mid_dos * A)) / 5)

            A = (1 / np.sinh(constant)) ** 2
            cv.append((k_B * np.sum(df * mid_dos * constant**2 * A)) / 5)

        harmonic_properties[volume_per_atom]["Free Energy (eV/atom)"] = free_energy
        harmonic_properties[volume_per_atom][
            "Internal Energy (eV/atom)"
        ] = internal_energy
        harmonic_properties[volume_per_atom]["Entropy (eV/K/atom)"] = entropy
        harmonic_properties[volume_per_atom]["Heat Capacity (eV/K/atom)"] = cv

    if plot == True:
        properties = [
            ("Free Energy (eV/atom)", "Free Energy (eV/atom)"),
            ("Entropy (eV/K/atom)", "Entropy (eV/K/atom)"),
            ("Heat Capacity (eV/K/atom)", "Heat Capacity (eV/K/atom)"),
        ]
        for property_name, ylabel in properties:
            plt.figure()
            for volume_per_atom in volumes_per_atom:
                plt.plot(
                    temp_range,
                    harmonic_properties[volume_per_atom][property_name],
                    label=f"{volume_per_atom} (Å³/atom)",
                )
            plt.legend(edgecolor="black")
            plt.xlabel("Temperature (K)")
            plt.ylabel(ylabel)

    return vdos_data_scaled, harmonic_properties


def eosfitall(volume, energy, m, n):

    volume_range = np.linspace(np.min(volume), np.max(volume), 1000)
    volume_range = volume_range[:, np.newaxis]

    if m == 1:  # mBM: modified Birch-Murnaghan
        if n == 2:
            A1 = np.hstack((np.ones(volume.shape), volume ** (-1 / 3)))
            A2 = np.hstack((np.ones(volume_range.shape), volume_range ** (-1 / 3)))
        elif n == 3:
            A1 = np.hstack(
                (np.ones(volume.shape), volume ** (-1 / 3), volume ** (-2 / 3))
            )
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    volume_range ** (-1 / 3),
                    volume_range ** (-2 / 3),
                )
            )
        elif n == 4:
            A1 = np.hstack(
                (
                    np.ones(volume.shape),
                    volume ** (-1 / 3),
                    volume ** (-2 / 3),
                    volume ** (-1),
                )
            )
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    volume_range ** (-1 / 3),
                    volume_range ** (-2 / 3),
                    volume_range ** (-1),
                )
            )
        elif n == 5:
            A1 = np.hstack(
                (
                    np.ones(volume.shape),
                    volume ** (-1 / 3),
                    volume ** (-2 / 3),
                    volume ** (-1),
                    volume ** (-4 / 3),
                )
            )
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    volume_range ** (-1 / 3),
                    volume_range ** (-2 / 3),
                    volume_range ** (-1),
                    volume_range ** (-4 / 3),
                )
            )

    elif m == 2:  # BM: Birch-Murnaghan
        if n == 2:
            A1 = np.hstack((np.ones(volume.shape), volume ** (-2 / 3)))
            A2 = np.hstack((np.ones(volume_range.shape), volume_range ** (-2 / 3)))
        elif n == 3:
            A1 = np.hstack(
                (np.ones(volume.shape), volume ** (-2 / 3), volume ** (-4 / 3))
            )
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    volume_range ** (-2 / 3),
                    volume_range ** (-4 / 3),
                )
            )
        elif n == 4:
            A1 = np.hstack(
                (
                    np.ones(volume.shape),
                    volume ** (-2 / 3),
                    volume ** (-4 / 3),
                    volume ** (-2),
                )
            )
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    volume_range ** (-2 / 3),
                    volume_range ** (-4 / 3),
                    volume_range ** (-2),
                )
            )
        elif n == 5:
            A1 = np.hstack(
                (
                    np.ones(volume.shape),
                    volume ** (-2 / 3),
                    volume ** (-4 / 3),
                    volume ** (-2),
                    volume ** (-8 / 3),
                )
            )
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    volume_range ** (-2 / 3),
                    volume_range ** (-4 / 3),
                    volume_range ** (-2),
                    volume_range ** (-8 / 3),
                )
            )

    elif m == 3:  # LOG
        if n == 2:
            A1 = np.hstack((np.ones(volume.shape), np.log(volume)))
            A2 = np.hstack((np.ones(volume_range.shape), np.log(volume_range)))
        elif n == 3:
            A1 = np.hstack((np.ones(volume.shape), np.log(volume), np.log(volume) ** 2))
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    np.log(volume_range),
                    np.log(volume_range) ** 2,
                )
            )
        elif n == 4:
            A1 = np.hstack(
                (
                    np.ones(volume.shape),
                    np.log(volume),
                    np.log(volume) ** 2,
                    np.log(volume) ** 3,
                )
            )
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    np.log(volume_range),
                    np.log(volume_range) ** 2,
                    np.log(volume_range) ** 3,
                )
            )
        elif n == 5:
            A1 = np.hstack(
                (
                    np.ones(volume.shape),
                    np.log(volume),
                    np.log(volume) ** 2,
                    np.log(volume) ** 3,
                    np.log(volume) ** 4,
                )
            )
            A2 = np.hstack(
                (
                    np.ones(volume_range.shape),
                    np.log(volume_range),
                    np.log(volume_range) ** 2,
                    np.log(volume_range) ** 3,
                    np.log(volume_range) ** 4,
                )
            )

    eos_parameters = np.linalg.pinv(A1).dot(energy)
    energy_fit = np.dot(A2, eos_parameters)

    return eos_parameters.T, energy_fit.T


# 3. Quasiharmonic phonon calculations -
# 4. Partition function (Zentropy)
# 5. Properties at fixed P
