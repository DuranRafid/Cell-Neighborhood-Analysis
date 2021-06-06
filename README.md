# Cellular Neighborhood Based Analysis of Tumor iTME

This repository contains the cellular neighborhood based analysis codes for analyzing tumor iTME across different groups. A portion of the codes are used from [this github repository.](https://github.com/nolanlab/NeighborhoodCoordination.)

## Input Format 

The input should be a csv file containing each cell's ID, their X and Y coordinate, the spot (ROI) ID/name where they belong, the sample (patient) ID of the corresponding spot, and the group/type of the patient. This information should be in separate columns in the csv file. For neighborhood identification, the sample (patient) ID and the group/type are not mandatory, but they are required for performing comparative analysis between different network groups. 

## Requirements 

```
python >=3.0
numpy
scipy
sklearn
pandas 
matplolib
seaborn
```

## Identifying the cellular neighborhood
To identify and save the neighborhood information for each cell type the following command in the cloned directory.
```
python Neighborhood_Identification.py --file-name <filename> 
```
You can provide all the other necessary arguments in the command too. If you do not provide them, the default values for the arguments will be assumed. You can know about all the arguments by the following command.

```
python Neighborhood_Identification.py --help 
```

This command will generate the neighborhood vs cell type clustermap and save the neighborhood information in `<filename>_with_Neighborhood.csv`. 
## Constructing Neighborhood-Neighborhood Network 

After identifying the neighborhood, you can use the saved neighborhood file as input to create the cell-cell network graphs across neighborhoods over different patient groups using the following command.

```
python Neighborhood_Network.py --file-name <filename> 
```
You can provide all the other necessary arguments in the command too. If you do not provide them, the default values for the arguments will be assumed. You can know about all the arguments by the following command.

```
python Neighborhood_Network.py --help 
```

