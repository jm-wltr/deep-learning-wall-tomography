# COMSOL Simulations

The data we use for the neural network is obtained via physical simulations of sending waves through different wall structures generated randomly. All the files related to COMSOL are located in the `COMSOL` folder.


THIS IS THE SCRIPT USEFUL TO MODIFY FOR TRANSFERING FILES:

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
