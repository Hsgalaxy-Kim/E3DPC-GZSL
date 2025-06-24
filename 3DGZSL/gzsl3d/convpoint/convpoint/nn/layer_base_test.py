import torch
import torch.nn as nn
import gzsl3d.convpoint.convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
from sklearn.neighbors import BallTree
import numpy as np

class LayerBase(nn.Module):

    def __init__(self):
        super(LayerBase, self).__init__()
    
    def mp_indices_conv_reduction(self, pts, K, npts):
        tree = BallTree(pts, leaf_size=2)
        used = np.zeros(pts.shape[0])
        current_id = 0
        indices = []
        pts_n = []
        for ptid in range(npts):

            possible_ids = np.argwhere(used==current_id).ravel().tolist()
            while(len(possible_ids)==0):
                current_id = used.min()
                possible_ids = np.argwhere(used==current_id).ravel().tolist()

            index = possible_ids[np.random.randint(len(possible_ids))]

            # pick a point
            pt = pts[index]

            # perform the search
            dist, ids = tree.query([pt], k=K)
            ids = ids[0]

            used[ids] +=1
            used[index] += 1e7

            indices.append(ids.tolist())
            pts_n.append(pt)

        pts_n = np.array(pts_n)

        return torch.LongTensor(indices).unsqueeze(0), torch.from_numpy(pts_n).float().unsqueeze(0)

    def indices_conv_reduction(self, input_pts, K, npts):
        process_indices = []
        process_pts = []
        batch_size = input_pts.shape[0]
        for i in range(batch_size):
            indices, pts = self.mp_indices_conv_reduction(input_pts[i].cpu().detach().numpy(), K, npts)
            process_indices.append(indices)
            process_pts.append(pts)
        indices = torch.cat(process_indices, dim=0).long().cuda()
        pts = torch.cat(process_pts, dim=0).float().cuda()
        return indices, pts

    def indices_conv(self, input_pts, K):
        indices = nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(), input_pts.cpu().detach().numpy(), K, omp=True)
        indices = torch.from_numpy(indices).long()
        if input_pts.is_cuda:
            indices = indices.cuda()
        return indices, input_pts

    def indices_deconv(self, input_pts, next_pts, K):
        indices = nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(), next_pts.cpu().detach().numpy(), K, omp=True)
        indices = torch.from_numpy(indices).long()
        if input_pts.is_cuda:
            indices = indices.cuda()
        return indices, next_pts