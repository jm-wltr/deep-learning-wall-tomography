# COMSOL Simulations

[Back to README.md](../README.md)

The data we use for the neural network is obtained via physical simulations of sending waves through different wall structures generated randomly. All the files related to COMSOL are located in the `COMSOL` folder. 

This simulation builds a three-dimensional mortar block studded with arbitrarily shaped “stone” inclusions (imported from STL files), then uses time-dependent solid-mechanics to model how a Gaussian-modulated ultrasonic pulse propagates through the composite and is recorded at discrete receiver points. We do this for 6 emission points located along a face of the block, and 11 reception points located along the opposite face. The script exports a file for each emitter (called `PL0.txt` to `PL5.txt`), each with a column for each receptor over time.

## Instructions
The main file is `Simulation.java`, which uses the COMSOL API to define a simulation model using all the STL files directly inside whichever directory is written in `basedir.txt` (which is supposed to include all the STL files corresponding to a specific wall cross section). This Java code must be compiled using `& "C:\Program Files\COMSOL\COMSOL60\Multiphysics\bin\win64\comsolcompile.exe" "COMSOL\Simulation.java"` (substitute with the directories of comsolcompile and the Java file in your computer). This will generate a `Simulation.class` file, which essentially acts as an MPH file (the normal COMSOL filetype), except it just defines the model but does not store the results. You can open this file with COMSOL Multiphysics or execute it from the terminal using the command `"C:\Program Files\COMSOL\COMSOL60\Multiphysics\bin\win64\comsolbatch.exe" -inputfile "COMSOL\Simulation.class"  -nosave` (substitute with your actual directories). Removing the `-nosave` parameter will export an MPH file with the results. Finally, the script `run_comsol.bat` iterates over a directory including the different cross section folders, and for each one updates `basedir.txt` and immediately compiles and runs the simulation. 

Less importantly, the `oldFiles` folder includes `New.mph`, which is the original model I manually designed for a specific wall cross section before automation; and `Refactored.java` which is the (non-automated) Java file exported automatically by COMSOL from `New.mph` after manually reorganizing it and adding comments.











## COMSOL Simulations

[Back to README.md](../README.md)

Our workflow uses COMSOL Multiphysics to simulate ultrasonic-wave propagation through masonry-like composites. We generate a 3D mortar block studded with arbitrarily shaped stone inclusions (imported via STL files), through which we send Gaussian-modulated ultrasonic pulses and record the response at discrete receiver points. Each simulation produces time-series data for six emitters and eleven receptors, formatted as six text files (`PL0.txt` to `PL5.txt`), each containing eleven columns of recorded signals.

### Project Structure

All COMSOL-related files reside in the `COMSOL` folder:

* **Simulation.java**: Defines the model using the COMSOL Java API. It reads STL geometries from a directory specified in `basedir.txt` and defines and runs the corresponding simulation.
* **basedir.txt**: Contains the absolute path to the folder of STL files representing a single wall cross section. `Simulation.java` reads this file at runtime.
* **run\_comsol.bat**: Automates batch processing. It iterates over subfolders of cross sections, updates `basedir.txt` with their path, compiles the Java model, and runs the simulation.
* **oldFiles/**:

  * **New\.mph**: The original COMSOL model for a single wall section, manually built in the GUI.
  * **Refactored.java**: The Java export of `New.mph`, reorganized and annotated but not automated.

### Building and Running Simulations

The Java code must be compiled using `& "C:\Program Files\COMSOL\COMSOL60\Multiphysics\bin\win64\comsolcompile.exe" "COMSOL\Simulation.java"` (substitute with the directories of comsolcompile and the Java file in your computer). This will generate a `Simulation.class` file, which essentially acts as an MPH file (the normal COMSOL filetype), except it just defines the model but does not store the results. You can open this file with COMSOL Multiphysics or execute it from the terminal using the command `"C:\Program Files\COMSOL\COMSOL60\Multiphysics\bin\win64\comsolbatch.exe" -inputfile "COMSOL\Simulation.class"  -nosave` (substitute with your actual directories). Removing the `-nosave` parameter will export an MPH file with the results. 

The script `run_comsol.bat` iterates over a directory including the different cross section folders, and for each one updates `basedir.txt` and immediately compiles and runs the simulation. This is the main file to be run. Each run outputs six `.txt` files (`PL0.txt`–`PL5.txt`). The easiest way to check for progress is looking at `basedir.txt`. You might need to modify the file to specify the directories you are working with, and the specific folders you want to iterate over.

### Extra

These sample scripts for the terminal might be useful to move files from `sections_generator/output` to where you want to read them.

```
# source and destination
$src = 'C:\Users\Jaime\Documents\deep-learning-wall-tomography\sections_generator\output\imagesCrop'
$dst = 'C:\Users\Jaime\Documents\deep-learning-wall-tomography\data\new\sections'

# start naming at 00124.jpg
$i = 124

# for each x from 0 to 39 (00000–00039)…
0..39 | ForEach-Object {
    $prefix = $_.ToString('D5')       # zero-pad to 5 digits

    # for each crop variant 0–4
    0..4 | ForEach-Object {
        $y = $_
        $inFile  = Join-Path $src  "$prefix`_crop$y.jpg"
        if (Test-Path $inFile) {
            $outName = '{0:D5}.jpg' -f $i
            Copy-Item $inFile -Destination (Join-Path $dst $outName)
            $i++
        }
    }
}
```


```
# source & destination roots
$srcBase = 'C:\Users\Jaime\Documents\deep-learning-wall-tomography\sections_generator\output\stlsCrop'
$dstBase = 'C:\Users\Jaime\Documents\deep-learning-wall-tomography\data\new\stls'

# first new folder index = 00124
$i = 124

0..39 | ForEach-Object {
    $x = $_.ToString('D5')            # 00000 … 00039
    0..4 | ForEach-Object {
        $y = $_
        $folder = "${x}_crop${y}"
        $srcDir = Join-Path $srcBase $folder
        if (Test-Path $srcDir) {
            # make new numbered folder
            $destDir = Join-Path $dstBase ('{0:D5}' -f $i)
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null

            # copy all .stl files in
            Get-ChildItem -Path $srcDir -Filter '*.stl' | 
                Copy-Item -Destination $destDir

            $i++
        }
    }
}
```

```
# source & destination roots
$srcBase = 'C:\Users\Jaime\Documents\deep-learning-wall-tomography\sections_generator\output\stlsCrop'
$dstBase = 'C:\Users\Jaime\Documents\deep-learning-wall-tomography\data\new\waveforms'

# first new folder index = 00124
$j = 124

0..39 | ForEach-Object {
    $x = $_.ToString('D5')
    0..4 | ForEach-Object {
        $y = $_
        $folder = "${x}_crop${y}"
        $srcDir = Join-Path $srcBase $folder
        if (Test-Path $srcDir) {
            # make new numbered folder
            $destDir = Join-Path $dstBase ('{0:D5}' -f $j)
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null

            # copy all .txt files in
            Get-ChildItem -Path $srcDir -Filter '*.txt' | 
                Copy-Item -Destination $destDir

            $j++
        }
    }
}
```