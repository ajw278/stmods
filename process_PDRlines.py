import os
import re
from definitions import *
import csv
import pandas as pd
import subprocess
import shutil
import matplotlib.pyplot as plt

# Dictionary for molecule names (from filenames)
molecule_mapping = {
'arhp': 'ArH+', 'chp': 'CH+', 'co': 'CO', 'h2o': 'H2O', 'c_18o': 'C18O', 
'hcn': 'HCN', 'sip': 'Si+', 'si': 'Si', 'sp': 'S+', 'spp': 'S++', 'shp': 'SH+', 
's': 'S', 'hd': 'HD', 'h': 'H', 'cpp': 'C++', 'hcop': 'HCO+', 'op': 'O+', 
'opp': 'O++', 'nepp': 'Ne++', 'nep': 'Ne+', 'npp': 'N++', 'np': 'N+', 'n': 'N', 'oh': 'OH', 'ohp': 'OH+', 
'o2': 'O2', 'hcn': 'HCN', 'h3p': 'H3+', 'h2': 'H2', 'h2_18o': 'H218O', 
'cs': 'CS', 'c': 'C', 'cp': 'C++', '13c_o':'13CO', '13c_18o': '13C18O', 'o':'O'}

# Speed of light in meters per second
SPEED_OF_LIGHT = 299792458  # m/s

def ghz_to_micrometers(frequency_ghz):
	"""Convert frequency in GHz to micrometers."""
	frequency_hz = frequency_ghz * 1e9  # Convert GHz to Hz
	wavelength_m = SPEED_OF_LIGHT / frequency_hz  # Wavelength in meters
	wavelength_micrometers = wavelength_m * 1e6  # Convert to micrometers
	return wavelength_micrometers

def invcm_to_micrometers(frequency_cm):
	"""Convert frequency in inverse cm to micrometers."""
	return 1e4/frequency_cm


def wavelength_from_infcol(info_column):
	wavelength_info = info_column.split()[0]  # First entry after 'info:'
	# Check if it's in GHz or micrometers/microns
	if 'GHz' in info_column:
		frequency_ghz = float(wavelength_info)
		wavelength = ghz_to_micrometers(frequency_ghz)  # Convert GHz to micrometers
	elif 'MHz' in info_column:
		frequency_ghz = 1e-3*float(wavelength_info)
		wavelength = ghz_to_micrometers(frequency_ghz)  # Convert GHz to micrometers
	elif 'cm-1' in info_column:
		frequency_invcm = float(wavelength_info)
		wavelength = invcm_to_micrometers(frequency_invcm)  # Convert cm^-1 to micrometers
	elif 'Angstrom' in info_column:
		wavelength = 1e-4 * float(wavelength_info)
	elif 'mm' in info_column:
		wavelength = 1e3 * float(wavelength_info)
	elif 'micron' in info_column or 'micrometers' in info_column or 'micrometres' in info_column:
		# Directly extract the wavelength (in micrometers or microns)
		wavelength = float(wavelength_info)
	else:
		raise Warning('No unit given for wavelength')
	return wavelength

def extract_transitions(data_lines, molecule_name):
	"""
	General function that uses the quantum number labels from the third line of the header.
	It dynamically adapts to the structure of the file, ensuring proper formatting of quantum numbers without repetition.
	"""
	transitions = []
	wavelengths = []

	# Parse the third line (header) to determine quantum number labels
	header_line = data_lines[2].strip('#').split()

	# Quantum labels are between 'quant:' and 'info:'
	quant_start = None
	for ih, head in enumerate(header_line):
			if head in ['quant:']:
				quant_start = ih+1
				quant_start_cols = ih+1
				break
	if quant_start is None:
		for ih, head in enumerate(header_line):
			if head in ['Aij(s-1)']:
				quant_start = ih+1
				quant_start_cols = ih+2
				break

	quant_end = None
	for ih, head in enumerate(header_line):
			if head in ['info:', 'Description', 'quant:', 'Description:'] and ih>quant_start:
				quant_end = ih
				break


	if quant_end is None:
		raise Warning('No end to the quantum number labels found.')

	# Clean up 'u', 'l' and semicolons, extract the actual quantum labels
	labs_raw = []
	for iq in range(quant_start, quant_end):
		labs_raw.append(header_line[iq])
	quantum_labels = [label.replace('_up', '').replace('_lo', '').replace(';', '').replace('l', '').replace('u', '') for label in labs_raw]
	add_quantum = [0 for iq in quantum_labels]
	for iq, qn in enumerate(quantum_labels):
		if qn=='Fu+0.5':
			data_lines[quant_start+iq] = float(data_lines[quant_start+iq])+0.5
			quantum_labels[iq] = 'F'
			add_quantum[iq] =0.5
		if qn=='Fl+0.5':
			quantum_labels[iq] = 'F'
			add_quantum[iq] =0.5
		if qn=='F+0.5':
			quantum_labels[iq] = 'F'
			add_quantum[iq] =0.5

		if qn=='Ka':
			quantum_labels[iq] = 'ka'
		if qn=='Kc':
			quantum_labels[iq] = 'kc'
		if qn=='Kb':
			quantum_labels[iq] = 'kb'

	if molecule_name=='H2O':
		print(quantum_labels)



	
	# Extract upper and lower quantum numbers
	num_quantum_labels = len(quantum_labels) // 2
	

	quant_start =quant_start_cols
	# Extract quantum numbers from data lines (from the 4th line onward)
	for line in data_lines[3:]:
		if line.strip() and not line.startswith('#'):
			columns = re.split(r'\s+', line.strip())

			for icol in range(num_quantum_labels):
				if add_quantum[icol]>0:
					columns[icol+quant_start] = int(columns[quant_start + icol].replace(';', ''))+add_quantum[icol]
					columns[icol+quant_start+num_quantum_labels] = int(columns[quant_start +num_quantum_labels+ icol].replace(';', ''))+add_quantum[icol]
				else:
					columns[icol+quant_start] = columns[quant_start + icol].replace(';', '')
					columns[icol+quant_start+num_quantum_labels] = columns[quant_start +num_quantum_labels+ icol].replace(';', '')



			upper_quantum_numbers = ','.join([f"{quantum_labels[i]}={columns[quant_start + i]}"
												for i in range(num_quantum_labels)])
			lower_quantum_numbers = ','.join([f"{quantum_labels[i]}={columns[quant_start + num_quantum_labels+i]}"
												for i in range(num_quantum_labels)])

			# Extract wavelength/frequency from the 'info:' field
			info_column = line.split('info:')[-1].strip().strip(';')
			wavelength = wavelength_from_infcol(info_column)

			
			wavelengths.append(wavelength)
			
			# Format the transition string
			transition_info = f"I({molecule_name} {upper_quantum_numbers}->{lower_quantum_numbers} angle 00 deg)"
			transitions.append(transition_info)

	return transitions, wavelengths

def extract_transitions_ID(data_lines, molecule_name):
	"""
	Specific function for S++ type transitions where 'IDu' and 'IDl' are provided in the data lines.
	The format should be El=3P,J=1 -> El=3P,J=0.
	"""
	transitions = []
	wavelengths = []

	# Extract quantum numbers from data lines (from the 4th line onward)
	for line in data_lines[3:]:
		if line.strip() and not line.startswith('#'):
			columns = re.split(r'\s+', line.strip())
			
			# Get upper and lower state IDs
			upper_id = columns[6].replace(';', '').replace('o_', '_')  # IDu (e.g., 3P_J=1)
			lower_id = columns[7].replace(';', '').replace('o_', '_') # IDl (e.g., 3P_J=0)
			
			# Split upper and lower IDs by '_' and clean up formatting
			upper_el, upper_j = re.split('_|-', upper_id)
			lower_el, lower_j = re.split('_|-', lower_id)
			
			# Clean up the repeating J issue
			upper_j = upper_j.replace('J=', '')
			lower_j = lower_j.replace('J=', '')


			info_column = line.split('info:')[-1].strip()
			wavelength = wavelength_from_infcol(info_column)
			wavelengths.append(wavelength)

			# Format the transition string
			transition_info = f"I({molecule_name} El={upper_el},J={upper_j}->El={lower_el},J={lower_j} angle 00 deg)"
			transitions.append(transition_info)

	return transitions, wavelengths

def process_all_files_and_save(pdr_cdir = PDR_CDIR, line_dir=LINE_DIR):
	cwd = os.getcwd()
	os.chdir(pdr_cdir+line_dir)
	# Get all files matching lines_XXX.dat
	files = [f for f in os.listdir() if f.startswith('line_') and f.endswith('.dat')]

	intensity_strings = []
	wavelengths = []
	mols = []

	# Process each file
	for file in files:
		molecule_key = file[5:].split('.')[0]
		molecule_name = molecule_mapping.get(molecule_key, molecule_key)
		if not molecule_name in ['H3+']:
			with open(file, 'r') as f:
				lines = f.readlines()
			
			data_lines = lines #[line for line in lines if not line.startswith('#')]  # Ignore header and footer lines
			
			# Determine if it's a special case
			if ('IDu' in lines[2] and 'IDl' in lines[2]) or ('Lev_up' in lines[2] and 'Lev_lo' in lines[2]) or ('levu' in lines[2] and 'levl' in lines[2]):
				transitions, wvs = extract_transitions_ID(data_lines, molecule_name)
			else:
				transitions, wvs = extract_transitions(data_lines, molecule_name)
			
			# Save transitions to a new file
			output_file = f"{molecule_name}_transitions.txt"
			with open(output_file, 'w') as out_f:
				for itr, transition in enumerate(transitions):
					out_f.write(transition + '\n')
					intensity_strings.append(transition)
					wavelengths.append(wvs[itr])
					mols.append(molecule_name)

	rows = zip(mols, intensity_strings, wavelengths)


	with open('all_lines.dat', "w") as f:
		writer = csv.writer(f)
		writer.writerow(["Molecule", "IStr", "Wavelength_micron"])
		for row in rows:
			writer.writerow(row)
		
				
	os.chdir(cwd)
	rows = zip(mols, intensity_strings, wavelengths)

	with open('all_lines.dat', "w") as f:
		writer = csv.writer(f)
		writer.writerow(["Molecule", "IStr", "Wavelength_micron"])
		for row in rows:
			writer.writerow(row)

	return mols, intensity_strings, wavelengths

def get_all_transitions(min_wl, max_wl, reset=True):

	if not os.path.isfile('all_lines.dat') or reset:
		process_all_files_and_save(pdr_cdir = PDR_CDIR, line_dir=LINE_DIR)


	df = pd.read_csv('all_lines.dat', sep=',', header=0)

	# Filter the dataframe based on the specified range
	filtered_df = df[(df['Wavelength_micron'] >= min_wl) & (df['Wavelength_micron'] <= max_wl)]
	return filtered_df


def process_line_intensities(line_ifile):
	line_data_processed = []
	with open(line_ifile, 'r') as file:
		for line in file:
			# Skip comment lines
			if line.startswith('#'):
				continue
			# Extract index, value, unit, and quantity based on fixed-width positions
			index = line[0:8].strip()      # First 8 characters for index
			value = line[8:32].strip()     # Next 24 characters for value
			unit = line[32:58].strip()     # Next 26 characters for unit
			quantity = line[58:].strip()   # Everything after character 58 for quantity

			# Skip rows where the index is missing or not a valid integer
			if not index.isdigit():
				continue

			# Skip rows where the value is not a valid float (e.g., Not_In_HDF5)
			try:
				value = float(value)
			except ValueError:
				continue

			# Append to data list
			line_data_processed.append([int(index), value, unit, quantity])
	
	return pd.DataFrame(line_data_processed, columns=['index', 'value', 'unit', 'quantity'])



def build_final_line_table(line_ifile, outfile='PDR_lineintensities.dat', line_dfile='all_lines.dat'):
	# Load the all_lines.dat file
	all_lines_df = pd.read_csv(line_dfile, header=0)

	# Load the line_intensities.dat file with whitespace delimiter
	line_intensities = process_line_intensities(line_ifile)
	#Cleaning the unit and quantity fields
	line_intensities['quantity_cleaned'] = line_intensities['quantity'].str.strip()

	# Loading the all_lines.dat file for cross-matching
	all_lines_df = pd.read_csv(line_dfile, skiprows=1, names=['Molecule', 'IStr', 'Wavelength_micron'])
	all_lines_df['IStr_cleaned'] = all_lines_df['IStr'].str.strip()

	# Merge based on the cleaned 'quantity' and 'IStr'
	merged = pd.merge(all_lines_df, line_intensities, left_on='IStr_cleaned', right_on='quantity_cleaned', how='inner')

	#Final table with the necessary columns
	final_table = merged[['Molecule', 'IStr_cleaned', 'Wavelength_micron', 'value', 'unit']]
	final_table.columns = ['Molecule', 'Quantity (IStr)', 'Wavelength_micron', 'Value', 'Unit']
	
	final_table.to_csv(outfile)

	return outfile, final_table



def get_line_intensities(min_wl, max_wl, hdf5_file, datdir=None, search_fname='line_fetch.txt', outfile='line_intensities.dat', idat_path=IDAT_PATH, reset=False):

	cwd= os.getcwd()
	if not datdir is None:
		os.chdir(datdir)


	#cont_bands = ['I(continuum - 200 to 1000 microns)','I(continuum - 60 to 1000 microns)', \
	#'I(continuum - 60 to 200 microns)', 'I(continuum - 25 to 60 microns)', 'I(continuum - 3 to 1000 microns)']
	

	if not os.path.isfile(outfile) or reset:
		df = get_all_transitions(min_wl, max_wl)
		Ivals = df['IStr'].astype(str).replace('"', '').replace("'", "")  # Remove quotation marks

		Ivals.to_csv(search_fname, header=False, index=False,sep='\t', quoting=csv.QUOTE_NONE)
	elif not os.path.isfile('all_lines.dat'):
		df = get_all_transitions(min_wl, max_wl)
	else:
		df = pd.read_csv('all_lines.dat')

	run_idat_extraction(hdf5_file, search_fname, outfile, idat_path=idat_path)

	fintab_name, fintab_df = build_final_line_table(outfile)


	if not datdir is None:
		shutil.copy(outfile, cwd+'/'+outfile)
		shutil.copy(fintab_name, cwd+'/'+fintab_name)

	os.chdir(cwd)

	return fintab_name, fintab_df

def get_spectrum(hdf5_file, datdir=None, search_fname='spectrum_fetch.txt', outfile='PDR_spectrum.dat', idat_path=IDAT_PATH):

	cwd= os.getcwd()
	if not datdir is None:
		os.chdir(datdir)


	if not os.path.isfile(outfile):
		with open(search_fname, 'w') as f:
			f.write('Emerging intensity\nWavelength')

		run_idat_extraction(hdf5_file, search_fname, outfile, idat_path=idat_path)

	if not datdir is None:
		shutil.copy(outfile, cwd+'/'+outfile)
		os.chdir(cwd)
	
	out_df = pd.read_csv(outfile)

	os.chdir(cwd)

	return outfile, out_df

def plot_spectrum(spectrum_file, line_file=None):

	# Read the file, skipping the header lines (marked by '#')
	data = pd.read_csv(spectrum_file, delim_whitespace=True, comment='#', header=None)

	# Assign column names based on the header information
	data.columns = ['Emerging intensity (erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹)', 'Wavelength (Å)']

	# Converting Wavelength from Ångström to microns and renormalizing Emerging intensity accordingly
	data['Wavelength (µm)'] = data['Wavelength (Å)'] * 1e-4  # 1 Å = 1e-4 µm
	data['Emerging intensity (erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹)'] = data['Emerging intensity (erg cm⁻² s⁻¹ Å⁻¹ sr⁻¹)'] * 1e4  # Conversion factor from per Ångström to per micron

	# Filtering the data to include only the range between 1 and 25 microns
	filtered_data = data[(data['Wavelength (µm)'] >= 1) & (data['Wavelength (µm)'] <= 25)]


	plt.figure(figsize=(10, 6))
	plt.plot(filtered_data['Wavelength (µm)'], filtered_data['Emerging intensity (erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹)'], color='b', lw=1)


	if not line_file is None:
		line_table = pd.read_csv(line_file, header=0)
		# Extract the N most prominent lines from the final_table
		N = 100  # Number of most prominent lines
		top_N_lines = line_table.nlargest(N, 'Value')
		# Plot vertical lines for the N most prominent transitions, colored by molecule
		molecules = top_N_lines['Molecule'].unique()
		colors = plt.cm.get_cmap('tab10', len(molecules))
		for i, molecule in enumerate(molecules):
			print(molecule)
			molecule_lines = top_N_lines[top_N_lines['Molecule'] == molecule]

			ymax = 2e-3
			plt.vlines(molecule_lines['Wavelength_micron'], ymin=ymax, ymax=5.*ymax,
					color=colors(i), label=molecule)


	# Log scale for both axes
	plt.xscale('log')
	plt.yscale('log')

	# Set the y-axis limits to an appropriate range for visibility
	plt.ylim(3e-6, 5e-3)

	# Labels and title
	plt.xlabel('Wavelength [µm]')
	plt.ylabel('Emerging intensity [erg cm⁻² s⁻¹ µm⁻¹ sr⁻¹]')
	plt.title('Slab model, 1-25 µm')

	# Display the plot
	plt.grid(True, which="both", ls="--")
	plt.show()



def run_idat_extraction(hdf5_file, selection_file, output_file, idat_path=IDAT_PATH):
	"""
	Runs IDAT to extract quantities from an HDF5 file and save them as a CSV or ASCII file.
	Args:
		hdf5_file (str): Path to the input HDF5 file.
		selection_file (str): Path to the selection file listing the quantities to extract.
		output_file (str): Path to the output file where the extracted data will be saved.
		idat_path (str): Path to the IDAT executable (default is './idat').
	"""
	# Construct the command
	command = [idat_path, 'extract', hdf5_file, selection_file, output_file]
	

	try:
		# Execute the command
		result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		# Output the command results
		print(f"IDAT Extraction completed. Output saved to {output_file}")
		print(result.stdout.decode())

	except subprocess.CalledProcessError as e:
		# Handle any errors during the extraction process
		print(f"Error during IDAT extraction: {e.stderr.decode()}")



if __name__=='__main__':
	# Call the function to process all files and save transitions
	datdir = '/home/awinter/PDR1.5.4_210817_rev2095/out/ExampleDiffuse'
	hdf5_file = 'ExampleDiffuse_s_20.hdf5'
	linefile, linedf =get_line_intensities(1.0, 25.0, hdf5_file, datdir=datdir, reset=False)
	spectfile, specdf = get_spectrum(hdf5_file, datdir=datdir)
	plot_spectrum(spectfile, line_file=linefile)
