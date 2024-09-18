import os
import re
from definitions import *
import csv

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
				quant_start = ih +1
				break
	if quant_start is None:
		for ih, head in enumerate(header_line):
			if head in ['Aij(s-1)']:
				quant_start = ih +2
				break

	for ih, head in enumerate(header_line):
			if head in ['info:', 'Description']:
				quant_end = ih
				break

	# Clean up 'u', 'l' and semicolons, extract the actual quantum labels
	quantum_labels = [label.replace('u', '').replace('l', '').replace(';', '') for label in header_line[quant_start:quant_end]]
	for iq, qn in enumerate(quantum_labels):
		if qn=='Fu+0.5':
			columns[quant_start+iq] += 0.5
			quantum_labels[iq] = 'F'
		if qn=='Fl+0.5':
			quantum_labels[iq] = 'F'
			columns[quant_start+iq] += 0.5

	# Extract quantum numbers from data lines (from the 4th line onward)
	for line in data_lines[3:]:
		if line.strip() and not line.startswith('#'):
			columns = re.split(r'\s+', line.strip())
			
			# Extract upper and lower quantum numbers
			num_quantum_labels = len(quantum_labels) // 2
			
			upper_quantum_numbers = ','.join([f"{quantum_labels[i]}={columns[quant_start + i].replace(';', '')}"
												for i in range(num_quantum_labels)])
			lower_quantum_numbers = ','.join([f"{quantum_labels[i]}={columns[quant_start + num_quantum_labels + i].replace(';', '')}"
												for i in range(num_quantum_labels)])

			# Extract wavelength/frequency from the 'info:' field
			info_column = line.split('info:')[-1].strip().strip(';')
			wavelength_info = info_column.split()[0]  # First entry after 'info:'
			

			# Check if it's in GHz or micrometers/microns
			if 'GHz' in info_column:
				frequency_ghz = float(wavelength_info)
				wavelength = ghz_to_micrometers(frequency_ghz)  # Convert GHz to micrometers
			else:
				# Directly extract the wavelength (in micrometers or microns)
				wavelength = float(wavelength_info)
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
			upper_id = columns[6].replace(';', '')  # IDu (e.g., 3P_J=1)
			lower_id = columns[7].replace(';', '')  # IDl (e.g., 3P_J=0)
			
			# Split upper and lower IDs by '_' and clean up formatting
			upper_el, upper_j = re.split('_|-', upper_id)
			lower_el, lower_j = re.split('_|-', lower_id)
			
			# Clean up the repeating J issue
			upper_j = upper_j.replace('J=', '')
			lower_j = lower_j.replace('J=', '')


			info_column = line.split('info:')[-1].strip()
			wavelength_info = info_column.split()[0]  # First entry after 'info:'

			# Check if it's in GHz or micrometers/microns
			if 'GHz' in info_column:
				frequency_ghz = float(wavelength_info)
				wavelength = ghz_to_micrometers(frequency_ghz)  # Convert GHz to micrometers
			else:
				# Directly extract the wavelength (in micrometers or microns)
				wavelength = float(wavelength_info)
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
		print(molecule_key, file)
		molecule_name = molecule_mapping.get(molecule_key, molecule_key)
		
		with open(file, 'r') as f:
			lines = f.readlines()
		
		data_lines = lines #[line for line in lines if not line.startswith('#')]  # Ignore header and footer lines
		
		# Determine if it's a special case
		if 'IDu' in lines[2] and 'IDl' in lines[2]:
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

	print(rows)

	with open('all_lines.dat', "w") as f:
		writer = csv.writer(f)
		writer.writerow(["Molecule", "IStr", "Wavelength_micron"])
		for row in rows:
			print(row,'!!!')
			writer.writerow(row)
		
				
	os.chdir(cwd)
	rows = zip(mols, intensity_strings, wavelengths)

	with open('all_lines.dat', "w") as f:
		writer = csv.writer(f)
		writer.writerow(["Molecule", "IStr", "Wavelength_micron"])
		for row in rows:
			print('...')
			print(row)
			writer.writerow(row)

	return mols, intensity_strings, wavelengths

	

if __name__=='__main__':
	# Call the function to process all files and save transitions
	process_all_files_and_save()
