# Ray Metadata Format

[Back to README.md](../README.md)

In the `data` folder (and in `data_held_out`, which is data that can be used for testing), we have three subdirectories. `sections` includes the jpg images of wall cross sections. `stls` includes the corresponding 3D files, with a file per stone. Finally `waveforms` includes the waveform files exported by COMSOL.Each file is a tabular text file with the following structure:

1. **Header** (lines starting with `%`) includes simulation metadata.

2. **Data Block** (from line ~10 onward):
   - First column: **Time (in seconds)**
   - Next 11 columns: **Y-component of displacement for each SLV** 

For the three folders, it is important that the data follows a numerical pattern as it does by default. Skipping numbers or using a different naming convention might lead to unexpected results.