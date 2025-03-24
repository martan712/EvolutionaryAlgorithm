from weblogo import *
import csv

def hamming_distance(s1, s2):
    return sum(x != y for x, y in zip(s1.strip(), s2.strip()))

def dist_to_csv(dist, filename):
    # Write the Counter to a CSV file
    with open(filename, 'w+', newline='') as csvfile:
        fieldnames = ['Value', 'Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write each row (key-value pair from the Counter)
        for value, count in dist.items():
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
    with open(outname, 'wb') as file:
        file.write(png)