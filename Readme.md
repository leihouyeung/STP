# STP: single-cell partition for subcellular spatially resolved transcriptomics.

## Installation

'''
conda create -n STP python=3.7
conda activate STP
pip install -r requirements.txt
'''

## Usage

The required files include the h5ad file of spatial transcriptomics data and its corresponding pre-aligned nuclei-stained image.

The partition of single cell through spatial transcriptomics data is mainly done by the function called `STP` which includes following parameters:
- `adata`: h5ad file spatial transcriptomics
- `img`: Pre-aligned nulcei-stained image with spatial transcriptomics 
- `save_path`: The path to save the results
- `T`: Initial temperature (the number of selected bins to extend the nuclei)
- `T_min`: Minimum temperature for ending the algorithm
- `reduction_rate`: Rate of reduction for T ( 0 < k < 1)
- `neighbor_num`: Number of neighbors for KNN 
- `alpha`: Weight for distance energy (0 < alpha < 1)
- `beta`: Weight for transcriptional similarity energy (0 < beta < 1)
- `thres`: The threshold of nuclei segmentation

The output files include:
- Nuclei-segmentation image
- Single-cell partition image
- Connected component of partitioned cells
- The expression profiles of partitioned cells (index of cells is related to the label of the connected component of cell)


## Example:
```
adata_path = './example/example1.h5ad'
img_path = './example/example.tiff'
save_path = './example_result/'
adata = sc.read_h5ad(adata_path)
img = cv2.imread(img_path)
os.makedirs(save_path,exist_ok=True)

STP(adata,img,save_path) 
```
