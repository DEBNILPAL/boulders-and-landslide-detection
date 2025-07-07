# --- File: src/10_parse_metadata.py ---

import os
import xml.etree.ElementTree as ET

def parse_xml_metadata(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}

    data = {
        'product_id': None,
        'instrument': None,
        'latitudes': {},
        'longitudes': {},
        'file_path': xml_file
    }

    try:
        data['product_id'] = root.findtext('.//logical_identifier')
        data['instrument'] = root.findtext('.//Observing_System_Component/name')

        # Geometry parameters
        coords = root.find('.//isda:System_Level_Coordinates', ns)
        if coords is not None:
            for tag in coords:
                tag_name = tag.tag.split('}')[-1]
                value = float(tag.text)
                if 'latitude' in tag_name:
                    data['latitudes'][tag_name] = value
                elif 'longitude' in tag_name:
                    data['longitudes'][tag_name] = value
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")

    return data

def filter_dataset_by_region(xml_dir, lat_range, lon_range):
    relevant_files = []
    for file in os.listdir(xml_dir):
        if file.endswith('.xml'):
            path = os.path.join(xml_dir, file)
            metadata = parse_xml_metadata(path)
            lats = metadata['latitudes'].values()
            lons = metadata['longitudes'].values()

            if any(lat_range[0] <= lat <= lat_range[1] for lat in lats) and \
               any(lon_range[0] <= lon <= lon_range[1] for lon in lons):
                relevant_files.append(metadata)
    return relevant_files
