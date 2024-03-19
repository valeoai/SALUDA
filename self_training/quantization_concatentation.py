def voxelize(self, data):
        data_pts = data['coordinates']
        data_labels = data['labels']
        data_features = data['features']

        #Quantization step
        data_pts = np.round(data_pts / self.voxel_size).astype(np.int32)
        data_pts -= data_pts.min(0, keepdims=1)   
        
        data_pts, voxel_idx, inverse_map = sparse_quantize(data_pts,return_index=True,return_inverse=True)
        data_pts = torch.tensor(data_pts, dtype=torch.int)
        voxel_idx = torch.tensor(voxel_idx)
        
        data_features = data_features[voxel_idx]
        data_labels = data_labels[voxel_idx]
        
        #Make a torch tensor
        data_features = torch.from_numpy(data_features)
        data_labels = torch.from_numpy(data_labels)
        inverse_map = torch.from_numpy(inverse_map)

        #Create the sparse Tensor
        sparse_input = SparseTensor(coords=data_pts, feats=data_features)

        

        return {'sparse_input': sparse_input,
                'sparse_input_invmap':inverse_map,
                'coordinates':data_pts, 
                'labels': data_labels,
                'features': data_features,
                'idx': voxel_idx}