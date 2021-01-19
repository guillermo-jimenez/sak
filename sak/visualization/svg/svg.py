import defusedxml.ElementTree as ET
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
import numpy as np

def aha(quantity: list, in_path: str, out_path: str, colormap: str = 'Reds', color_resolution: int = 30):
    # Sanity check
    if len(quantity) != 16:
        raise ValueError("Only intended to print values to AHA segments")

    # NOTE TO SELF: to check which is the correct "header", simply `print(root)` and will show sth like <Element '{http://www3.medical.philips.com}restingecgdata' at 0x7f5906e247c0>

    # Retrieve colormap
    cmap = get_cmap(colormap, color_resolution+10)
    
    # Load SVG as xml file
    tree = ET.parse(in_path)
    
    # Get tree's root (all appended to it)
    root = tree.getroot()

    # eps
    eps = np.finfo('float').eps

    # For each of the quantities associated to each AHA segment
    for i in range(1,17):
        # Retrieve color in RGB and in HEX
        color_index = int(color_resolution*(quantity[i-1]-min(quantity)+eps)/(max(quantity)-min(quantity)+eps))+5
        rgb_color = cmap(color_index)[:3]
        hex_color = rgb2hex(rgb_color)

        # Step 1: change block's color
        block = root.find('{{http://www.w3.org/2000/svg}}path[@id="segment-{}"]'.format(i))
        block.set('style','fill:{};fill-opacity:1;stroke:#000000;stroke-width:3;stroke-miterlimit:10'.format(hex_color)) 
        
        # Step 2: change block's 
        text_block = root.find('{{http://www.w3.org/2000/svg}}text[@id="texts{}"]{{http://www.w3.org/2000/svg}}tspan'.format(i))
        text_block.text = str(quantity[i-1])

    tree.write(out_path)

