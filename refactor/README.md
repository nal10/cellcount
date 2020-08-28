Dataset:

 - Cells are fluorescently labeled with red, green or both.
 - 2-channel data is collected. Ideally, only cells marked with the red fluorescent label should appear in the red channel (same for green)
 - Due to bleed-through, we notice a tendency for the cells labeled with Green fluorescent label to appear in the red channel. 
 - The degree of bleed-through varies for each animal
 
Annotations:
 - Initial ground truth consists of approximate cell centers 
 - Composite image was used for reference, and visible cells were assigned as either red-, green-, or both-labeled
 
Pixel-wise ground truth:

1. `optim_centers_script.m`: Removes inadvertent duplicate cell center annotations. Optimizes the position of the cell center. 
2. `get_labels`: Generates intermediate .mat files with output from a fast marching based wave propagation algorithm. .mat files contain maps for  visited/not-visited points `KT`, arrival time `T`, distance `D`, simple/non-simple point class `S`.
3. `???`: Uses intermediate .mat files to generate 2-channel label maps.

Models:
 - Add unet schematic
 - 