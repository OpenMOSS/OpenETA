import os
import shutil

source_file = "/home/aigeorge/TactileACT/data/camera_cage_not_fixed/data"
desitination_file = "/home/aigeorge/TactileACT/data/camera_cage_both/data"

# loop through ever hdf5 file in the source directory, and copy it to the destination directory with a new name.

index = 101
for file in os.listdir(source_file):
    if file.endswith(".hdf5"):
        source = os.path.join(source_file, file)
        destination = os.path.join(desitination_file, f"episode_{index}.hdf5")
        shutil.copy(source, destination)
        print("Copied file: ", file)
        index += 1