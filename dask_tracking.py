import copy
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import cdist
from skimage.segmentation import relabel_sequential
import tracking as tr
import identification as idf

import dask
import dask.array as da
from dask import delayed


def get_centroid(label,labeled_maps,precip_data):
    precip_locs=np.where(labeled_maps==label,precip_data,0)
    return center_of_mass(precip_locs)

def create_weights(labeled_data,label,precip_data):
    label_loc=np.where(labeled_data==label,1,0)
    label_precip_sum=np.sum(np.where(label_loc==1,precip_data,0))
    weighted_loc=np.where(label_loc==1,precip_data/label_precip_sum,0)
    
    return weighted_loc,np.sum(np.where(labeled_data==label,1,0))

def get_splits(curr_weight_locs,prev_weight_locs,max_split=None):
    curr_coor_1d=np.argwhere(curr_weight_locs)
    prev_coor_1d=np.argwhere(prev_weight_locs)

    merged=np.concatenate((curr_coor_1d,prev_coor_1d),axis=0)

    union=np.unique(merged,axis=0)

    curr_union_weights = np.zeros(len(union))
    prev_union_weights = np.zeros(len(union))

    curr_union_weights=curr_weight_locs[union[:,0],union[:,1]]
    prev_union_weights=prev_weight_locs[union[:,0],union[:,1]]

    curr_union_weights = (curr_union_weights.reshape(curr_union_weights.shape[0], 1))
    prev_union_weights = (prev_union_weights.reshape(1, prev_union_weights.shape[0]))
        
    if max_split:
        max_split=len(curr_union_weights)//max_split
    else:
        max_split=len(curr_union_weights)//2000
        
    if max_split==0:
        max_split=1

    splits_curr=(np.array_split(curr_union_weights,max_split))
    splits_prev=(np.array_split(prev_union_weights,max_split,axis=1))
    splits_union=(np.array_split(union,max_split,axis=0))

    return splits_curr,splits_prev,splits_union

def do_the_calc(combined_futures,i,j,phi):
    dist=cdist(combined_futures[2][i],combined_futures[2][j])

    cpw=np.einsum(
    'ij, jk -> ik',
    combined_futures[0][i],
    combined_futures[1][j]
    )

    return(np.sum(np.exp(-1*phi*dist)*cpw))

def calc_func_big(combined_futures,phi):
    final_sum=[]
    
    for i in range(len(combined_futures[0])):
        for j in range(len(combined_futures[0])):
            
            final_sum.append(delayed(do_the_calc)(combined_futures,i,j,phi))

    return np.sum(dask.compute(final_sum)[0])

def calc_func(combined_futures,phi):
    final_sum=0
    
    for i in range(len(combined_futures[0])):
        for j in range(len(combined_futures[0])):
        
            dist=cdist(combined_futures[2][i],combined_futures[2][j])

            cpw=np.einsum(
            'ij, jk -> ik',
            combined_futures[0][i],
            combined_futures[1][j]
            )

            element_wise_similarity=np.exp(-1*phi*dist)*cpw
            
            final_sum = final_sum + np.sum(element_wise_similarity)

    return(final_sum)

def dask_label(precip_file,THRESHOLD,struct):
    #sample data from tutorial
    in_data=precip_file[np.newaxis]
    binary_data=np.where(in_data<THRESHOLD,0,1)
    precip_data=np.where(in_data<THRESHOLD,0,in_data)

    labeled_maps=idf.identify(binary_data,struct)

    result_data=copy.deepcopy(labeled_maps)

    return(labeled_maps,result_data,precip_data)

def track(
    precip_data : np.ndarray,
    labeled_maps : np.ndarray,
    result_data : np.ndarray,
    tau : float,
    phi : float,
    km : float,
    worker_client #dask client
):
    
########################################################################################
################## LOADING IN INITIAL DATA #############################################
########################################################################################

    #input precip_data should be 3D and first dim should be time
    num_time_slices=precip_data.shape[0]
    
    for time_index in range(1,num_time_slices):
        
        max_label_so_far=max(np.max(result_data[time_index - 1]), np.max(labeled_maps[time_index]))

        # find the labels for this time index and the labeled storms in the previous time index
        current_labels = np.unique(labeled_maps[time_index])
        previous_storms = np.unique(result_data[time_index - 1])

        # and prepare the corresponding precipitation data
        curr_precip_data = precip_data[time_index]
        prev_precip_data = precip_data[time_index - 1]
        
########################################################################################
################## PRE-PROCESSING ELIGIBLE STORMS ######################################
########################################################################################
#NOTE: This entire chunk is done in memory using numpy. Calculation time was pretty quick
#without the need for paralellization here. If storm arrays > 100,000 cells then this may
#get bogged down, but typical storm size on 3.75 km continental data was still <20,000 cells.
        
        #initialize some empty dictionaries and lists
        cdict,pdict,pred_dict={},{},{}
        for sub_dict in ['size','center_of_mass']:
            cdict[sub_dict]={}
            pdict[sub_dict]={}

        all_storm_list=[]
        pred_storms=[]

        #loop through all identified storms in current timestep
        for clabel in current_labels[1:]:

            #only calculate storm characteristics if they haven't already been recorded
            if clabel not in cdict.keys():
                curr_weight_locs,cstorm_size=create_weights(clabel,labeled_maps[time_index],curr_precip_data)
                cdict[clabel]=worker_client.scatter(curr_weight_locs)
                cdict['size'][clabel]=cstorm_size
                cdict['center_of_mass'][clabel]=get_centroid(clabel,labeled_maps[time_index],curr_precip_data)

            
            storm_list=[]
            #now loop through all identified storms in the previous timestep
            for plabel in previous_storms[1:]:

                #again only calc storm characteristics if not already calculated
                if plabel not in pdict.keys():
                    prev_weight_locs,pstorm_size=create_weights(plabel,result_data[time_index-1],prev_precip_data)
                    pdict[plabel]=worker_client.scatter(prev_weight_locs)
                    pdict['size'][plabel]=pstorm_size
                    pdict['center_of_mass'][plabel]=get_centroid(plabel,result_data[time_index-1],prev_precip_data)

                #calculate displacement between selected current storm & selected previous storm
                curr_prev_displacement=tr.displacement(cdict['center_of_mass'][clabel],
                                                   pdict['center_of_mass'][plabel])
                #if displacement between centroids > defined threshold then assign this
                #comparison a value of zero. otherwise, assign a value of 1
                bool_displacement=np.where(tr.magnitude(curr_prev_displacement)>km,0,1)

                if time_index>1:
                    #if storm data for timesteps-2 isn't already recorded, do it now
                    if not len(pred_storms):
                        pred_storms=np.unique(result_data[time_index-2])
                        for pred_storm_label in pred_storms:
                            pred_dict[pred_storm_label]=get_centroid(pred_storm_label,result_data[time_index-2],
                                                                    precip_data[time_index-2])
                    #if the prev storm exists in timestep-2
                    if np.isin(plabel,pred_storms):
                        
                        #compare the vector angle between current storm & past storm against
                        #past storm and storm in timestep-2 with matching label.
                        prev_pred_displacement=tr.displacement(pdict['center_of_mass'][plabel],
                                                           pred_dict[plabel])
                        angle_value=tr.angle(curr_prev_displacement,prev_pred_displacement)

                    #if prev storm does not exist in timestep-2
                    #then assign a bogus value
                    else:
                        angle_value=np.ones(1)[0]*999
                else:
                    angle_value=np.ones(1)[0]

                #if the angle between these two displacement vectors is less than 120
                #degrees, then assign a value of 1. if not, assign a value of zero.
                bool_angle=np.where(angle_value>2.09,0,1)

                #sum the two boolean arrays for displacmenet and angle
                storm_list.append(np.sum((bool_displacement,bool_angle)))
            all_storm_list.append(storm_list)

        #reshape boolean array into an array with dims num_curr_storms x num_prev_storms.
        #values >= 1 in this array indicate potential match candidates (either displacement
        #or vector angle falls within accepted parameters).
        eligible_storms=np.array(all_storm_list)

        
########################################################################################
################## LAZILY LOADING DASK CALCULATIONS ####################################
########################################################################################
#NOTE: This step doesn't perform any actual calculations. Just sets up tasks that can 
#be handled through dask in the next section.
        list_of_calcs=[]

        for clabel,eligible_storm_row in zip(current_labels[1:],eligible_storms):
            for plabel,storm_bool in zip(previous_storms[1:],eligible_storm_row):

                #if the boolean array for this storm pair comparison does not equal
                #at least 1, then assign a zero value representing no match
                if storm_bool==0:
                    list_of_calcs.append(da.zeros(1)[0])

                #otherwise, calculate the morphological similarity between the
                #two storms
                else:
                    
                    #if the combined size of the two storms exceeds 10,000 cells
                    if cdict['size'][clabel]+pdict['size'][plabel]>10000:
                        #break the computation up into chunks of size 5,000.
                        #this works well with paralellization and resulted in quicker
                        #computation time for large storms
                        splits=delayed(get_splits)(cdict[clabel],pdict[plabel],5000)
                        list_of_calcs.append(delayed(calc_func_big)(splits,phi))

                    #otherwise if the storms are smaller, just calculate similarity
                    #on a single worker. chunks of 100 are better here, performance
                    #seemed to be faster in serial.
                    else:
                        splits=delayed(get_splits)(cdict[clabel],pdict[plabel],100)
                        list_of_calcs.append(delayed(calc_func)(splits,phi))
                        
########################################################################################
################## PERFORMING DASK CALCULATIONS ########################################
########################################################################################
#NOTE: This step handles actual calculations. Dask is required here.

        index=1
        batch=[]
        sim_list=[]
        for item in list_of_calcs:

            #for instances where the number of storms being compared are very large
            #i.e. current_storms & previous_storms = ~150 
            #this generates a LARGE number of tasks that can bog dask down
            #each task adds a few milliseconds to computation time, and can add
            #up really easily for large task workloads
            #to get around this, we batch our calcs up 2,500 at a time, then submit
            #and run them one at a time
            batch.append(item)
            if index%2500==0:
                futures=worker_client.map(dask.compute,batch)
                sim_list.extend(dask.compute(worker_client.gather(futures))[0])

                batch=[]
            index=index+1

        if batch:
            futures=worker_client.map(dask.compute,batch)
            sim_list.extend(dask.compute(worker_client.gather(futures))[0])

        #resulting array has size current_storms x previous_storms and consists
        #of resulting values from the morphological comparison
        sim_list=np.reshape(np.array(sim_list),eligible_storms.shape)


########################################################################################
################## DETERMINING BEST MATCH ##############################################
########################################################################################

        #find positions in similarity array that pass tau threshold
        wheres=np.argwhere(sim_list>tau)+1

        final_matches=np.zeros((len(current_labels[1:]),2))

        for index,label in enumerate(current_labels[1:]):
            final_matches[index][0]=label
            sim_matches=wheres[wheres[:,0]==label][:,1]
            
            #if at least one of the sim comparisons between current storm and all prev storms
            #pass the tau threshold
            if len(sim_matches):
                
                #match is determined as the storm with the largest size that passes all criteria
                match=sim_matches[np.argmax([pdict['size'][previous_storms[l]] for l in sim_matches])]
                match=previous_storms[match]
                final_matches[index][1]=match

                result_data[time_index]=np.where(labeled_maps[time_index]==label,
                                                match,
                                                result_data[time_index])
                
            #otherwise, if no storms pass similarity comparison, assign a new label
            #and state there is no match for this storm
            else:
                match=0
                final_matches[index][1]=match

                result_data[time_index]=np.where(labeled_maps[time_index]==label,
                                                max_label_so_far+1,
                                                result_data[time_index])

                max_label_so_far+=1


            print(label,' matched',match,' in time slice',time_index+1)
            
    seq_result = relabel_sequential(result_data.astype(np.int64))[0]
    return seq_result