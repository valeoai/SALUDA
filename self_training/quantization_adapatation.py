#Quantization step
masked_target_pts = np.round(masked_target_pts / self.voxel_size).astype(np.int32)
masked_target_pts -= masked_target_pts.min(0, keepdims=1)   
masked_target_pts, masked_target_voxel_idx, target_sparse_input_invmap = sparse_quantize(masked_target_pts,return_index=True,return_inverse=True)

masked_target_pts = torch.tensor(masked_target_pts, dtype=torch.int)


masked_target_pts_invmap = torch.tensor(target_sparse_input_invmap, dtype=torch.long)

masked_target_labels = masked_target_labels[masked_target_voxel_idx]
masked_target_features = masked_target_features[masked_target_voxel_idx]
masked_target_features = torch.tensor(masked_target_features)

batch_index = np.ones([masked_target_labels.shape[0], 1]) * b
masked_target_labels = np.concatenate([batch_index, masked_target_labels[:,None]], axis=-1)

masked_target_pts = SparseTensor(coords=masked_target_pts, feats=masked_target_features[:,1][:,None])
new_batch['masked_target_pts'].append(masked_target_pts) #That are already sparse inputs
new_batch['masked_target_labels'].append(masked_target_labels)
new_batch['masked_target_pts_invmap'].append(masked_target_pts_invmap)




#Masked source points creation
masked_source_pts = np.round(masked_source_pts / self.voxel_size).astype(np.int32)
masked_source_pts -= masked_source_pts.min(0, keepdims=1)   
masked_source_pts, masked_source_voxel_idx, source_sparse_input_invmap = sparse_quantize(masked_source_pts, return_index=True, return_inverse=True)

masked_source_pts = torch.tensor(masked_source_pts, dtype=torch.int)
masked_source_pts_invmap = torch.tensor(source_sparse_input_invmap, dtype=torch.long)

masked_source_labels = masked_source_labels[masked_source_voxel_idx]
masked_source_features = masked_source_features[masked_source_voxel_idx]
masked_source_features= torch.tensor(masked_source_features)

batch_index = np.ones([masked_source_labels.shape[0], 1]) * b
masked_source_labels = np.concatenate([batch_index, masked_source_labels[:,None]], axis=-1)

masked_source_pts = SparseTensor(coords=masked_source_pts, feats=masked_source_features[:,1][:,None])
new_batch['masked_source_pts'].append(masked_source_pts)
new_batch['masked_source_labels'].append(masked_source_labels)
new_batch['masked_source_pts_invmap'].append(masked_source_pts_invmap)