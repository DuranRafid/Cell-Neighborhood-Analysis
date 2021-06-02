import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import cKDTree

out_file = open('network_output.txt','w')

class NeighborNetwork(object):
    def __init__(self, dataframe, ROI, X_pos, Y_pos, group, num_neighborhood):
        self.cells = dataframe
        self.ROI = ROI
        self.X_pos = X_pos
        self.Y_pos = Y_pos
        self.group = group
        self.num_neighboorhood = num_neighborhood
        self.neighborhood_name = "Neighborhood" + str(self.num_neighboorhood)
        print("Done initializing")

    def _save_fig(self, matrix, p_group):
        print('Saving figure for {}'.format(p_group))
        mask_ut = np.triu(np.ones(matrix.shape), k=1).astype(np.bool)
        fig, _ = plt.subplots(dpi=120)
        np.savetxt(str(p_group) + '_NeighberhoodContactMatrix.txt',np.log2(matrix))
        sns_heat = sns.heatmap(np.log2(matrix), mask=mask_ut, annot=True, vmax=4, vmin=-4, cmap='coolwarm')
        fig = sns_heat.get_figure()
        fig.savefig(str(p_group) + 'NeighberhoodContact_Final.png', dpi=200)

    def save_network_figs(self):
        patient_Groups = self.cells[self.group].unique()
        for p_group in patient_Groups:
            if p_group=='IDC':
                continue
            print(p_group)
            print(p_group,file=out_file)
            matrix = self.create_network(p_group)
            self._save_fig(matrix, p_group)


    def create_network(self, group_name):
        groupcells = self.cells[self.cells[self.group] == group_name]
        matrix = np.zeros((self.num_neighboorhood, self.num_neighboorhood))
        count = 0
        for group in groupcells.groupby(self.ROI):
            Xlist = group[1][self.X_pos].tolist()
            Ylist = group[1][self.Y_pos].tolist()
            Neilist = group[1][self.neighborhood_name].tolist()
            Types = [i for i in range(self.num_neighboorhood)]

            points = []
            for x, y in zip(Xlist, Ylist):
                points.append([x, y])
            points = np.array(points)
            neighbors = np.array(Neilist)
            neighbor_cell = self.get_neighbors_distance(points, neighbors, 50)
            mat = self.create_matrix(neighbor_cell, Types)
            log_of_odds_matrix = self.log_likelihood(mat)
            matrix += log_of_odds_matrix
            count += 1
            print('ROI',count,file=out_file)
            print(log_of_odds_matrix,file=out_file)
        return matrix

    def get_neighbors_distance(self, points, pointtype, distance):
        point_tree = cKDTree(points)
        # print(point_tree.count_neighbors(point_tree,200))
        neighbor_cell = []
        indices_set = point_tree.query_ball_point(points, distance).flatten()
        for i in range(len(indices_set)):
            neighborpoints = np.delete(pointtype[indices_set[i]], 0)
            neighbor_cell.append((pointtype[i], neighborpoints))
        return neighbor_cell

    def create_matrix(self, neighbor_cell, Types):
        matrix = np.zeros((len(Types), len(Types)))
        for i in range(len(neighbor_cell)):
            for j in range(len(neighbor_cell[i][1])):
                matrix[Types.index(neighbor_cell[i][0])][Types.index(neighbor_cell[i][1][j])] += 0.5
                matrix[Types.index(neighbor_cell[i][1][j])][Types.index(neighbor_cell[i][0])] += 0.5
        for i in range(matrix.shape[0]):
            matrix[i][i] = matrix[i][i] / 2
        return matrix

    def log_likelihood(self, matrix):
        new_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
        msum = 0
        for i in range(matrix.shape[0]):
            for j in range(i, matrix.shape[1]):
                msum += matrix[i][j]
        hnormsum = matrix.sum(axis=0) / msum
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                new_matrix[i][j] = (matrix[i][j]) / (hnormsum[i] * hnormsum[j] * msum + np.nextafter(0, 1))
        # return np.where(new_matrix!=0, np.log2(new_matrix+np.nextafter(0,1)), np.nextafter(0,1))
        return new_matrix

if __name__ == '__main__':
    df = pd.read_csv('IDC_and_ILC_BrCA_mIHC_Imaging_Inform_Final_Data_01_31_2021Neighborhood507.csv')
    df['Patient Outcome Group'] = [x+' '+y for (x,y) in zip(df['Patient Group'],df['Outcome Group'])]
    cells = df#df[df['Outcome Group']!='Fresh']
    neighbor_network = NeighborNetwork(dataframe=cells, X_pos='Cell X Position', Y_pos='Cell Y Position', ROI='Patient', group='Patient Group', num_neighborhood=7)
    neighbor_network.save_network_figs()
