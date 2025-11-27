__global__ void addr_convert_kernel(
    int64_t *base_addr_list,
    int *size_list,
    int *rw_list,
    int list_size,
    int *output_set_idx_list,
    int64_t *output_line_addr_list,
    int *output_rw_list,
    int total_size,
    int chunk_size,
    int set_num
){
    int set_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int window_start=chunk_size*set_idx;
    int window_end=min(window_start+chunk_size,total_size);
    int global_offset_start=0;
    for (int i=0;i< list_size;++i) {
        for(int j=0;j<size_list[i];++j){
            int global_offset=global_offset_start+j;
            if( global_offset>=window_start && global_offset<window_end){
                // do convert
                int64_t line_addr=base_addr_list[i]/128+j;
                output_line_addr_list[global_offset]=line_addr;
                output_set_idx_list[global_offset]=line_addr%set_num;
                output_rw_list[global_offset]=rw_list[i];
            }
        }
        global_offset_start+=size_list[i];
    }
}

__global__ void cache_kernel(
    int *set_idx_list,
    int64_t *line_addr_list,
    int *rw_list,
    int total_size,
    int *access_hit_flag,
    int64_t *cache_cell,
    int64_t *cache_cell_offset,
    int *read_hit_per_set,
    int *write_hit_per_set,
    int *read_miss_per_set,
    int *write_miss_per_set,
    int cache_sets,
    int cache_ways
){
    int set_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (set_idx < cache_sets) {
        cache_cell=cache_cell+set_idx*cache_ways;
        int curr_offset=cache_cell_offset[set_idx];
        int curr_read_hit=read_hit_per_set[set_idx];
        int curr_write_hit=write_hit_per_set[set_idx];
        int curr_read_miss=read_miss_per_set[set_idx];
        int curr_write_miss=write_miss_per_set[set_idx];

        for(int idx=0;idx<total_size;++idx){
            int access_cache_set_idx=set_idx_list[idx];
            if(set_idx==access_cache_set_idx){
                int64_t access_cache_line_idx=line_addr_list[idx];
                int rw=rw_list[idx];
                // int set_index = address % num_cache_sets;
                // cache_way =8 
                // access: 1,2,3,4,5,6,7,8,9,5,5
                // t8,8   : 1,2,3,4,5,6,7,8           curr_offset=7
                //                        ^
                // t9,9   : 9,2,3,4,5,6,7,8     miss, curr_offset=0
                //          ^
                // t10,5  : 9,5,2,3,4,6,7,8     hit,  curr_offset=1
                //            ^
                // t11,5  : 9,5,2,3,4,6,7,8     hit,  curr_offset=1 (不变)
                //            ^

                if (access_cache_set_idx == set_idx) {
                    // find offset_idx
                    int offset_idx = -1;

                    for(int i=0;i<cache_ways;++i){
                        if(access_cache_line_idx==cache_cell[i]){
                            offset_idx = i;
                            break;
                        }
                    }
                    if(offset_idx==-1){
                        //miss
                        // curr_offset=(curr_offset)%cache_ways;
                        if (rw==0){
                            access_hit_flag[idx]=2; // read miss
                            curr_read_miss++;
                        }else{
                            access_hit_flag[idx]=-2; // write miss
                            curr_write_miss++;
                        }
                        curr_offset++;
                        curr_offset=curr_offset%cache_ways;
                        cache_cell[curr_offset]=access_cache_line_idx;
                    }else{
                        //hit 
                        if(offset_idx==curr_offset){
                        
                        }else if(offset_idx>curr_offset){
                            // t9,9   : 9,2,3,4,5,6,7,8     miss, curr_offset=0
                            //          ^
                            // t10,5  : 9,5,2,3,4,6,7,8     hit,  curr_offset=1, offset_idx=4
                            //            ^ curr_offset
                            //                  ^ offset_idx
                            for(int i=offset_idx;i>curr_offset+1;--i){
                                cache_cell[i]=cache_cell[(i-1)%cache_ways];
                            }
                            curr_offset++;
                            curr_offset=curr_offset%cache_ways;
                            cache_cell[curr_offset]=access_cache_line_idx;
                        }else{
                            // t10,3  : 9,5,2,3,4,6,7,8
                            //                ^
                            // t11,5  : 9,2,3,5,4,6,7,8     hit,  curr_offset=3, offset_idx=1
                            //                ^ curr_offset
                            //            ^ offset_idx
                            for(int i=offset_idx;i<curr_offset;++i){
                                cache_cell[i]=cache_cell[(i+1)%cache_ways];
                            }
                            cache_cell[curr_offset]=access_cache_line_idx;
                        }

                        if (rw==0){
                            access_hit_flag[idx]=1; // read hit
                            curr_read_hit++;
                        }else{
                            access_hit_flag[idx]=-1; // write hit
                            curr_read_hit++;
                        }
                    }
                }

            }
        }
        cache_cell_offset[set_idx]=curr_offset;
        read_hit_per_set[set_idx]=curr_read_hit;
        write_hit_per_set[set_idx]=curr_write_hit;
        read_miss_per_set[set_idx]=curr_read_miss;
        write_miss_per_set[set_idx]=curr_write_miss;
    }
}
