# Ray Metadata Format

Each `rays/rayXX.txt` file contains metadata about the 66 acoustic rays (6 emitters and 11 sensors) used to simulate wave propagation through the wall section depicted in `sections/XX.jpg` (XX ranges from 00 to 99). 

Each `"numerical analyses"/XX/` folder contains six files (`PL1.txt` to `PL6.txt`), each representing the raw waveform response measured at 11 fixed simulated SLV sensors (over 10,000 time steps) when a wave is emitted by one of six AHSs through wall section XX.

## Columns in `rayXX.txt`

Each line represents one ray and contains **10 tab-separated columns**:

| Column | Description                          | Units / Format      |
|--------|--------------------------------------|---------------------|
| 1      | Ray ID (1-based index)               | Integer             |
| 2–4    | Emission point coordinates (x, y, z) | Meters (floats)     |
| 5–7    | Reception point coordinates (x, y, z)| Meters (floats)     |
| 8     | Propagation velocity                 | Meters per second    |
| 9      | Amplitude of the wave                | Arbitrary units     |
| 10      | Central frequency                    | Kilohertz (kHz)    |



## File Format of `PL*.txt`

Each file is a tabular text file with the following structure:

1. **Header** (lines starting with `%`) includes simulation metadata.

2. **Data Block** (from line ~10 onward):
   - First column: **Time (in seconds)**
   - Next 11 columns: **Y-component of displacement for each SLV** 
