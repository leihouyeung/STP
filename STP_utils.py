import cv2
import os
import scipy.sparse as ss
from scipy.ndimage.morphology import binary_dilation
import math
from random import uniform
from sklearn.neighbors import KDTree
from PIL import Image
import copy
import sparse
from skimage.filters import  threshold_local
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from cellpose import models
import torch
import numpy as np
from tqdm import tqdm
from random import sample

def Hex_to_RGB(hex):
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    rgb = [r,g,b]

    return rgb

def rowname_translator(rownames):
    total_cor = []
    for obs in rownames:
        cor = list(map(lambda x: float(x) , obs.split("_")))
        total_cor.append(cor)
    return np.array(total_cor).astype(np.int64)

def get_external_boundary_bins(arr, nuclei_label):
    ### get the external boundary of current nuclei, no -1, no assigned bin, only assign the label to unassigned bins (0)
    ### k would be kernel which is               
    ### [[0,1,0],
    ###  [1,1,1],
    ###  [0,1,0]]
    arr_copy = copy.deepcopy(arr)  
    mask = copy.deepcopy(arr)
    arr_copy[(arr_copy != nuclei_label)] = 0
    mask = mask+1
    mask[mask!=1] = False
    k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1
    boundary = binary_dilation(arr_copy, k,iterations=1,mask = mask) & (arr_copy == False)
    return np.where(boundary == True)


def get_internal_boundary_bins(arr, nuclei_label):
    ### get the internal boundary of current nuclei
    ### k would be kernel which is               
    ### [[0,1,0],
    ###  [1,1,1],
    ###  [0,1,0]]
    arr_copy = copy.deepcopy(arr)  
    arr_copy[arr_copy != nuclei_label] = 0
    k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1
    boundary = binary_dilation(arr_copy==False, k,iterations=1) & (arr_copy==nuclei_label)
    return np.where(boundary == True)

def get_centriod_bin(arr,nuclei_label):
    indices = get_internal_boundary_bins(arr, nuclei_label)
    return (round(np.mean(indices[0])),round(np.mean(indices[1])))
def get_cell_coor(x,y,w,h,image_X, image_Y, range_num = 500):
    if range_num < x < image_X-range_num:
        n_x = x - range_num
        n_x_end = x + w + range_num
    elif x <= range_num:
        n_x = 0
        n_x_end = 2 * x + w 
    elif x >= image_X-range_num:
        n_x = x - (image_X - x - w) 
        n_x_end = image_X

    if range_num < y < image_Y-range_num:
        n_y = y - range_num
        n_y_end = y + h + range_num
    elif y <= range_num:
        n_y = 0
        n_y_end = 2 * y + h 
    elif y >= image_Y-range_num:
        n_y = y - (image_Y - y - h) 
        n_y_end = image_Y
    
    return n_x, n_x_end, n_y, n_y_end



def seq_to_sparse(input_seq, input_img, input_coor,save_path):
    
    dic = input_seq.todok()
    values = list(dic.values())
    keys = list(dic.keys())
    new_values = {}
    process = tqdm(keys)
    for i in process:
        #process.set_description('processing the expression profile')
        ### 此处需要align！！！
        ### X and Y both +200
        new_values[(input_coor[i[0],0],input_coor[i[0],1],i[1])] = values[i[0]]
    X_exp = sparse.DOK((input_img.shape[0],input_img.shape[1],input_seq.shape[1]),new_values, dtype=np.uint8)
    #sparse.save_npz(os.path.join(save_path, 'X_coordinates_expression.npz'),X_exp)
    X_exp = X_exp.to_coo()

    #print('finish allocating to sparse matrix!')
    return X_exp


### preprocess nucleis
def nuclei_img_preprocess(input_img, clipLimit = 2,tileGridSize=(8,8), thres = 0.8):
    clahe = cv2.createCLAHE(clipLimit =clipLimit, tileGridSize=tileGridSize)
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    cl_img = clahe.apply(img_gray)
    ret, thresh = cv2.threshold(cl_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cl_img[cl_img < thres * ret] = 0
    
    return cl_img

### 1. dilate seq 
### 2. segment nuclei by choosing method (watershed or stardist)
###
### Return:
### markers, segmentation results of nucleis
### bounding_boxes: [x1, y1, width(x2-x1), height(y2-y1)]
### centriods: [x,y]
def segment_nuclei(cl_img,input_coor, method = 'stardist'):
    
    seq_img_dil = np.ones([cl_img.shape[0],cl_img.shape[1]]) * -1

    ### 此处需要align！！！
    seq_img_dil[input_coor[:,0],input_coor[:,1]] = 0

    seq_img_dil = seq_img_dil + 1
    k = np.ones((3,3),dtype=int); k[1] = 1; k[:,1] = 1
    seq_img_dil = binary_dilation(seq_img_dil, k,iterations=2)
    seq_img_dil = seq_img_dil - 1

    seq_img_show = np.zeros([cl_img.shape[0],cl_img.shape[1]])
    ind = np.where(seq_img_dil == 0)
    seq_img_show[ind[0],ind[1]] = 1
    final_nuclei_img = seq_img_show * cl_img
    
    if method == 'stardist':
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        final_nuclei_img = normalize(final_nuclei_img,1,99.8)
        markers, details = model.predict_instances(final_nuclei_img)
        centriods = details['points']
        bounding_boxes = np.zeros((centriods.shape[0],4))

        for i in range(1,np.max(markers)+1):
            x1,y1,x2,y2 = round(min(details['coord'][i-1,1,:])),round(min(details['coord'][i-1,0,:])),round(max(details['coord'][i-1,1,:])),round(max(details['coord'][i-1,0,:]))
            bounding_boxes[i-1,:] = np.array([x1,y1,x2-x1,y2-y1])
        bounding_boxes = np.around(bounding_boxes)
        bounding_boxes = np.asarray(bounding_boxes,dtype=np.int32)

    elif method == 'cellpose':
        torch.cuda.empty_cache()
        model = models.Cellpose(gpu=True,model_type='nuclei',device=torch.device('cuda'))
        markers, flows, styles, diams = model.eval(final_nuclei_img,batch_size=1, diameter=None,channels=[0,0])
        
        centriods = np.zeros((np.max(markers),2))
        bounding_boxes = np.zeros((np.max(markers),4))
        for i in tqdm(range(1,np.max(markers)+1)):
            spots = np.ma.where(markers == i)
            centriods[i-1,0] = np.mean(spots[0])
            centriods[i-1,1] = np.mean(spots[1])
            x1,y1,x2,y2 = round(min(spots[1])),round(min(spots[0])),round(max(spots[1])),round(max(spots[0]))
            bounding_boxes[i-1,:] = np.array([x1,y1,x2-x1,y2-y1])
        bounding_boxes = np.around(bounding_boxes)
        bounding_boxes = np.asarray(bounding_boxes,dtype=np.int32)

    return markers, centriods, bounding_boxes


### Return
### cell expression matrix:  gene * cell
### distance:[y,x]
def generate_nuclei_expression(connected_components, X_exp, centriods, b_boxes, gene_num):

    num_labels = len(centriods)
    ### align the nuclei to original seq matrix
    img_Y, img_X = connected_components.shape

    distance = np.zeros((num_labels,2))
    
    ### intialize the each cell's expression matrix
    cell_expression_matrix = np.zeros((num_labels,gene_num))

    ### align the nuclei to original seq matrix
    process = tqdm(range(1,num_labels+1))
    for i in process:
        #process.set_description('Mapping the nulcei image to spatial omics')
        x_cen,y_cen = centriods[i-1,:]
        new_x, new_x_end, new_y, new_y_end = get_cell_coor(b_boxes[i-1,0],b_boxes[i-1,1],b_boxes[i-1,2],b_boxes[i-1,3],img_X, img_Y,100)

        spots = ss.find(connected_components[new_y:new_y_end,new_x:new_x_end] == i)

        cell_expression_matrix[i-1,:] = X_exp[spots[0]+new_y,spots[1]+new_x,:].sum(axis = 0).todense()
        distance[i-1,0] = y_cen
        distance[i-1,1] = x_cen
    
    return cell_expression_matrix, distance

def show_nuclei_seg_result(img, connected_components,b_boxes,color_dict, save_path = './'):
    
    num_labels = len(b_boxes)
    img_Y, img_X = connected_components.shape

    ## print the image of segmented nucleis
    seg_img = connected_components.astype(np.uint8)
    seg_img[seg_img > 0] = 255
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)
    process = tqdm(range(1,num_labels+1))
    for i in process:
        process.set_description('Generating nuclei img')
        new_x, new_x_end, new_y, new_y_end = get_cell_coor(b_boxes[i-1,0],b_boxes[i-1,1],b_boxes[i-1,2],b_boxes[i-1,3],img_X, img_Y,100)
        spots = np.where(connected_components[new_y:new_y_end,new_x:new_x_end] == i)
        img[spots[0]+new_y,spots[1]+new_x,:] = color_dict[i-1]
    img = Image.fromarray(img)
    img.save(os.path.join(save_path , 'nulcei_img.png'))
    #return seg_img




def ST_convolution_from_nuclei(connected_components,
                               cell_expression_matrix,
                               X_exp,
                               centriods,
                               b_box,
                               color_dict,
                               T = 100, # initial temperature, in this case, T means the number of selected bins to extend the nuclei
                               T_min = 10, # the minimum temperature for ending the algorithm
                               reduction_rate = 0.50, # rate of reduction for T, 0 < k < 1
                               neighbor_num = 3, # number of neighbor for KNN 
                               alpha = 1, # weight for distance energy
                               beta = 0.1, # weight for transcriptional similarity energy
                               gamma = 3,
                               save_path = './'):

    epoch = 1       # epoch for inference
    num_labels = len(centriods)
    energy_per_cell_matrix = np.ones((num_labels,1)) * 100000 #each cell's energy value for each epoch [cell_index,epoch]

    img_Y, img_X = connected_components.shape
    total_bins = len(connected_components.flatten() > 0)
    tree = KDTree(centriods, leaf_size=10)

    while T >= T_min:

        seg_img = connected_components.astype(np.uint8)
        seg_img[seg_img > 0] = 255
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR)

        energy_per_cell_matrix = np.column_stack((energy_per_cell_matrix,np.zeros((num_labels,1))))
        process = tqdm(range(1,num_labels+1))
        for i in process:
            process.set_description('Processing all nuclei in Epoch '+ str(epoch))
            ### extend new bins for each nuclei
            new_x, new_x_end, new_y, new_y_end = get_cell_coor(b_box[i-1,0],b_box[i-1,1],b_box[i-1,2],b_box[i-1,3],img_X, img_Y,100)

            external_bins = get_external_boundary_bins(connected_components[new_y:new_y_end,new_x:new_x_end], i)
            internal_boundary_bins=get_internal_boundary_bins(connected_components[new_y:new_y_end,new_x:new_x_end], i)
            ### if the adjacent bins of a nuclei are all -1, then continue, which means this nuclei has been fully expanded
            if len(external_bins[0]) == 0:
                new_x, new_x_end, new_y, new_y_end = get_cell_coor(b_box[i-1,0],b_box[i-1,1],b_box[i-1,2],b_box[i-1,3],img_X, img_Y,100)
                spots = np.where(connected_components[new_y:new_y_end,new_x:new_x_end] == i)
                seg_img[spots[0]+new_y,spots[1]+new_x,:] = color_dict[i-1]
                continue
                
            dis_exter_to_cen = np.sqrt((external_bins[0] - centriods[i-1,0]) **2 + (external_bins[1] - centriods[i-1,1]) **2)
            dis_exter_to_cen = dis_exter_to_cen + 0.0000001
            dis_exter_to_cen = dis_exter_to_cen / np.sum(dis_exter_to_cen)
            
            sub_cell_select = True
            while sub_cell_select:

                selected_bins_index = np.random.choice(range(len(external_bins[0])),size = min(len(external_bins[0]),T),replace = False,p = dis_exter_to_cen)
                selected_bins = (external_bins[0][selected_bins_index]+new_y,external_bins[1][selected_bins_index]+new_x)
                current_cell_expression = cell_expression_matrix[i-1,:] + X_exp[selected_bins[0],selected_bins[1],:].sum(axis = 0).todense()

                ### calculate the energy value (shape & transcriptional similiarity)
                distance_energy = np.sqrt((np.sum((internal_boundary_bins[0] - centriods[i-1,0]) **2)+
                                  np.sum((internal_boundary_bins[1] - centriods[i-1,1])**2))/len(internal_boundary_bins[0]))

                _, neighbor_indices = tree.query([centriods[i-1,:]], k=neighbor_num) 

                ###calculate the expression of neighbor cells
                neighbor_expression_mean = np.mean(cell_expression_matrix[neighbor_indices[0],:])

                similarity_energy = np.sqrt(np.sum((current_cell_expression - neighbor_expression_mean) **2)) / neighbor_num
                avg_size = total_bins / num_labels
                size_energy = np.abs((avg_size - len(np.where(connected_components[new_y:new_y_end,new_x:new_x_end] == i)[0]))) / avg_size
                cell_energy = alpha * distance_energy + beta * similarity_energy + gamma * size_energy

                delta = cell_energy - energy_per_cell_matrix[i-1,epoch-1]
                if delta < 0 or uniform(0,1) < math.pow(math.e,-delta / T):
                    connected_components[selected_bins] = i
                    sub_cell_select = False

            energy_per_cell_matrix[i-1,epoch] = cell_energy
            cell_expression_matrix[i-1,:] = current_cell_expression    
            new_x, new_x_end, new_y, new_y_end = get_cell_coor(b_box[i-1,0],b_box[i-1,1],b_box[i-1,2],b_box[i-1,3],img_X, img_Y,100)

            spots = np.where(connected_components[new_y:new_y_end,new_x:new_x_end] == i)
            seg_img[spots[0]+new_y,spots[1]+new_x,:] = color_dict[i-1]

        T = math.floor(reduction_rate * T)
        new_seg_img = Image.fromarray(seg_img)

#         if epoch < 10:
#             new_seg_img.save(f'{save_path}results_img_epoch_0{epoch}.png')
#         else:
#             new_seg_img.save(f'{save_path}results_img_epoch_{epoch}.png')
        
#         if epoch % 2 == 0:
#             np.save(f'{save_path}cell_expression_matrix_epoch_{epoch}.npy',cell_expression_matrix)
#             np.save(f'{save_path}assign_matrix_epoch_{epoch}.npy',connected_components)
        ### generate image each epoch
        epoch = epoch + 1
    new_seg_img.save(f'{save_path}results_img.png')
    np.save(f'{save_path}cell_expression_matrix.npy',cell_expression_matrix)
    np.save(f'{save_path}connected_components.npy',connected_components)    
        
        
def deHaze(src, min_r=7, guide_r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):

    """References: Single Image Haze Removal Using Dark Channel Prior."""
    src_type = 255 if np.max(src) > 1 else 1
    if src_type == 255:
        src = src / 255

    Y = np.zeros(src.shape)
    Mask_img, A = Defog(src, min_r, guide_r, eps, w, maxV1)

    for k in range(3):
        Y[:, :, k] = (src[:, :, k] - Mask_img) / (1 - Mask_img / A)
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))

    if src_type == 255:
        Y = Y * 255
        Y[Y > 255] = 255

    return Y



def zmMinFilterGray(src, r=7):
    """Minimum Filtering."""
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    """Guided Filtering."""
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(src, min_r, guide_r, eps, w, maxV1):
    """Calculate the atmospheric mask image V1 and the light value A, V1 = 1-t/A"""
    V1 = np.min(src, 2)
    Dark_Channel = zmMinFilterGray(V1, min_r)

    V1 = guidedfilter(V1, Dark_Channel, guide_r, eps)
    bins = 2000
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(src, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)
    return V1, A





def STP(adata,                 # h5ad file spatial transcriptomics
        img,                   # pre-aligned nulcei-stained image with spatial transcriptomics 
        save_path,             # The path to save the results
        T = 100,               # initial temperature (the number of selected bins to extend the nuclei)
        T_min = 10,            # the minimum temperature for ending the algorithm
        reduction_rate = 0.5, # rate of reduction for T, 0 < k < 1
        neighbor_num = 3,      # number of neighbor for KNN 
        alpha = 0.5,             # weight for distance energy
        beta = 0.1,            # weight for transcriptional similarity energy
        thres = 0.8            # The threshold of nulcei segmentation
        
):
    coordinates = adata.obsm['spatial']
    #coordinates = coordinates.astype(np.int64)
    print('Preprocessing omics data and image...')
    X_exp = seq_to_sparse(adata.X,img, coordinates,save_path)
    cl_img = deHaze(img,min_r = 5, guide_r = 20)
    cl_img = nuclei_img_preprocess(cl_img.astype(np.uint8),clipLimit = 7,tileGridSize=(10,10), thres = thres)
    print('Segmentation of nuclei.')
    markers, centriods, bounding_boxes = segment_nuclei(cl_img,coordinates,method='cellpose')
    color_matrix = np.array([[255,0,0],[0,255,0],[0,0,255],[128,42,42],[138,43,226],[25,25,112],[255,128,0]],np.uint8)
    color_dict = []
    for i in range(len(centriods)):
        color_dict.append(color_matrix[sample(range(0,6),1),:])
    show_nuclei_seg_result(img,markers,bounding_boxes,color_dict, save_path = save_path) 
    cell_expression_matrix, distance = generate_nuclei_expression(markers,X_exp, centriods, bounding_boxes, adata.n_vars)
    print('Expanding the boundaries of nuclei.')
    ST_convolution_from_nuclei(markers, 
                               cell_expression_matrix, 
                               X_exp, 
                               centriods, 
                               bounding_boxes, 
                               color_dict, 
                               save_path = save_path,
                               T = T,
                               T_min = T_min,
                               reduction_rate = reduction_rate,
                               neighbor_num = neighbor_num,
                               alpha = alpha,
                               beta = beta
                               )
