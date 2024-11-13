import os
import pathlib
import cv2
import joblib
import numpy as np
from PIL import Image
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm import tqdm

def read_hovernet_output(dat_file, slides_paths, class_names, output_path):
    print(f"\nProcessing file: {dat_file}")
    if (
            os.path.exists(output_path / str(dat_file.stem + f"_{class_names[1]}_Centroids.png")) and
            os.path.exists(output_path / str(dat_file.stem + f"_{class_names[2]}_Centroids.png")) and
            os.path.exists(output_path / str(dat_file.stem + f"_{class_names[3]}_Centroids.png")) and
            os.path.exists(output_path / str(dat_file.stem + f"_{class_names[4]}_Centroids.png")) and
            os.path.exists(output_path / str(dat_file.stem + f"_{class_names[5]}_Centroids.png"))
    ):
        print (' File exists')
        return

    loaded_objects = joblib.load(dat_file)
    nuclei = None
    resolution = None
    units = None

    try:
        nuclei = loaded_objects["elements"]
        resolution_info = loaded_objects["element-resolution"]
        resolution = resolution_info["resolution"]
        units = resolution_info["units"]
    except EOFError:
        print(f"\nFeature file {dat_file} corrupted.")

    sorted_nuclei = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }  # contains nuclei centers sorted by class
    radius = 4  # The radius of the centroid to be drawn in the mask

    for nucleus in nuclei.values():
        nucleus_type = nucleus["type"]
        temp = sorted_nuclei[nucleus_type]
        temp.append(nucleus["centroid"])
        sorted_nuclei[nucleus_type] = temp

    sorted_nuclei.pop(0)

    # find and open the slide
    slideName = os.path.basename(dat_file)
    slideName = os.path.splitext(slideName)[0]
    slide_path = ''
    for slide in slides_paths:
        if os.path.basename(slide).startswith(slideName):
            slide_path = slide
            break

    wsi = WSIReader.open(slide_path)
    width, height = wsi.slide_dimensions(resolution=resolution, units=units)

    for key in sorted_nuclei.keys():
        if os.path.exists(output_path / str(dat_file.stem + f"_{class_names[key]}_Centroids.png")):
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        centroids = np.array(sorted_nuclei[key])


        # Draw circles for each centroid point
        for centroid in centroids:
            x, y = centroid
            mask = cv2.circle(mask, (round(x), round(y)), radius, 255, thickness=-1)

        new_size = (int(width / 4), int(height / 4))
        mask2 = Image.fromarray(mask)
        resized_mask = mask2.resize(new_size, resample=Image.BICUBIC)
        resized_mask.save(output_path / str(dat_file.stem + f"_{class_names[key]}_Centroids.png"))


def create_maks(path_to_slides,path_to_dat, output_path):

    class_names = {
        0: "noLabel",
        1: "Neoplastic",
        2: "Inflammatory",
        3: "ConnectiveSoftTissue",
        4: "Necroses",
        5: "nonNeoplastic",
    }

    slidePaths = []
    for dirpath, dirnames, filenames in os.walk(path_to_slides):
        for filename in filenames:
            if filename.endswith('.svs'):
                full_path = os.path.join(dirpath, filename)
                slidePaths.append(full_path)

    path_to_dat = pathlib.Path(path_to_dat)
    output_path = pathlib.Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    dat_files = list(path_to_dat.glob("*.dat"))

    for dat_file in tqdm(dat_files):
        read_hovernet_output(
            dat_file,
            slides_paths=slidePaths,
            class_names=class_names,
            output_path=output_path,
        )
