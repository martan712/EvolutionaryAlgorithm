from weblogo import *
import csv
import math
from pathlib import Path
    
def hamming_distance(s1, s2):
    return sum(x != y for x, y in zip(s1.strip(), s2.strip()))

def shannon_entropy(position_values):
    """
    Calculate the Shannon entropy for a list of values (frequencies of characters at a particular position)
    """
    total = sum(position_values)
    entropy = 0.0
    for count in position_values:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p, 2)  # Using base 2 for entropy (bits)
    return entropy

def create_file_dir(filename):
    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
def dist_to_csv(dist, filename):
    # Write the Counter to a CSV file
    create_file_dir(filename)
    with open(filename, 'w+', newline='') as csvfile:
        fieldnames = ['Value', 'Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write each row (key-value pair from the Counter)
        for value, count in dist:
            writer.writerow({'Value': value, 'Count': count})

def visualize_png(inname, title, outname):
    fin = open(inname)
    sequences = read_seq_data(fin)
    logodata = LogoData.from_seqs(sequences)
    
    # Create the logo options
    logooptions = LogoOptions()
    logooptions.title = title
    
    # Increase the DPI (Resolution) for better quality
    logooptions.dpi = 300  # Default is usually 72 DPI, 300 is better quality
    
    # Optionally, adjust font sizes, scale, or other settings:
    logooptions.font_size = 40  # Adjust font size (default is usually around 12)
    logooptions.size = (24, 8)  # Adjust size of the logo, if necessary
    
    # Format the logo
    logoformat = LogoFormat(logodata, logooptions)
    png = png_formatter(logodata, logoformat)
    
    # Save the image at the high resolution
    create_file_dir(outname)
    with open(outname, 'wb') as file:
        file.write(png)
        
def extend_fitness(array, max_iters):
    return array + [1 for x in range(max_iters-len(array))]

def load_output(filename):
    with open(f"{filename}.csv", "r", newline='') as f:
        reader = csv.reader(f)
        fitnesses = [list(map(float, row)) for row in reader]  # Convert values to float
    return fitnesses