import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import cKDTree
import argparse

out_file = open('network_output.txt', 'w')


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
        np.savetxt(str(p_group) + '_NeighberhoodContactMatrix.txt', np.log2(matrix))
        sns_heat = sns.heatmap(np.log2(matrix), mask=mask_ut, annot=True, vmax=4, vmin=-4, cmap='coolwarm')
        fig = sns_heat.get_figure()
        fig.savefig(str(p_group) + 'NeighberhoodContact_Final.png', dpi=200)

    def scale_mat(self, matrix, range_max=4, range_min = -4):
        new_mat = matrix
        xmin = matrix.min()
        xmax = matrix.max()
        convert_to_range = lambda x, xmin, xmax, range_max, range_min: (range_max - range_min) * (x - xmin) / (xmax - xmin) + range_min
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                new_mat[i][j] = convert_to_range(matrix[i][j], xmin, xmax, range_max, range_min)
        return new_mat

    def save_network_figs(self,scale=True):
        patient_Groups = self.cells[self.group].unique()
        for p_group in patient_Groups:
            print(p_group)
            print(p_group, file=out_file)
            matrix = self.create_network(p_group)
            if scale==True:
                matrix = self.scale_mat(matrix)
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
            print('Group', count, file=out_file)
            print(log_of_odds_matrix, file=out_file)
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
    parser = argparse.ArgumentParser('Train and Evaluate Harmony on your dataset')
    parser.add_argument('-file', '--file-name', type=str, default='Neighborhood_saved_data.csv', help='The name of your csv file with filepath')
    parser.add_argument('-p', '--patient-column', type=str, default='Patient Group',help='The column name in your csv file listing the patient groups')
    parser.add_argument('-s', '--spot-column', type=str, default='ROI', help='The column name in your csv file listing the spot IDs/Names')
    parser.add_argument('-x', '--x-pos', type=str, default='Cell X Position', help='The column name in your csv file listing the cell X coordinates')
    parser.add_argument('-y', '--y-pos', type=str, default='Cell Y Position', help='The column name in your csv file listing the cell y coordinates')
    parser.add_argument('-n', '--num-nei', type=int, default='7', help='The number of neighborhoods you want to construct')
    args = parser.parse_args()
    file_name = args.file_name
    patient_col = args.patient_column
    spot_col = args.spot_column
    x_pos = args.x_pos
    y_pos = args.y_pos
    num_of_neighbors = args.num_nei
    df = pd.read_csv(file_name)
    neighbor_network = NeighborNetwork(dataframe=df, X_pos=x_pos, Y_pos=y_pos, ROI=spot_col, group=patient_col,num_neighborhood=num_of_neighbors)
    neighbor_network.save_network_figs()
