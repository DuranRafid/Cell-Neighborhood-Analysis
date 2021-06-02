from Neighborhood_Identification import CellNeighborhood
from Neighborhood_Network import NeighborNetwork
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


class Neighborhood_Analysis(object):
    def __init__(self, path, Xpos, Ypos, Patient, ROI, Celltype, Group=None, NeighborhoodNo=7):
        self.Xpos = Xpos
        self.Ypos = Ypos
        self.Patient = Patient
        self.ROI = ROI
        self.Celltype = Celltype
        self.Group = Group
        self.neighoborhood_nums = NeighborhoodNo
        self.cn = CellNeighborhood(path, self.Xpos, self.Ypos, self.ROI, self.Celltype)
        #self.cn.set_method('Distancecutoff')
        #self.cn.set_method_param(50)
        #self.cn.set_num_of_neighborhoods(NeighborhoodNo)
        #self.cn.identifyNeighborhoods()
        self.cells = self.cn.cells
        self.neighborhood_name = self.cn.neighborhood_name
        print("Done Forming Neighborhoods")

    def get_Groupwise_Stripplot(self):
        groupslist = self.cells[self.Group].unique()
        if len(groupslist) > 2:
            raise ValueError("More than 2 groups is not supported currently.")
        if len(groupslist) < 2:
            raise ValueError("At least 2 groups are required for group based analysis.")

        fc = self.cells.groupby([self.Patient, self.Group]).apply(
            lambda x: x[self.neighborhood_name].value_counts(sort=False, normalize=True))
        fc.columns = range(self.neighoborhood_nums)
        melt = pd.melt(fc.reset_index(), id_vars=[self.Patient, self.Group])
        melt = melt.rename(columns={'variable': 'neighborhood', 'value': 'frequency of neighborhood'})
        f, ax = plt.subplots(dpi=200, figsize=(10, 5))
        sns.stripplot(data=melt, hue=self.Group, dodge=True, alpha=.2, x='neighborhood',
                      y='frequency of neighborhood')
        sns.pointplot(data=melt, scatter_kws={'marker': 'd'}, hue=self.Group, dodge=.5, join=False,
                      x='neighborhood', y='frequency of neighborhood')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title=self.Group, handletextpad=0, columnspacing=1, loc="upper right",
                  ncol=3, frameon=True)
        plt.savefig('GroupDiffWindow' + self.neighborhood_name + '.png')

        f = open('GroupStatWindow' + self.neighborhood_name + '.txt', 'w')
        for i in range(self.neighoborhood_nums):
            n2 = melt[melt['neighborhood'] == i]
            f.write(str(i) + ' ' + str(ttest_ind(n2[n2[self.Group] == groupslist[0]]['frequency of neighborhood'],
                                                 n2[n2[self.Group] == groupslist[1]]['frequency of neighborhood'])[
                                           1]) + '\n')
        f.close()


    def get_Groupwise_NeighborContacts(self):
        neighbor_network = NeighborNetwork(dataframe = self.cells, ROI= self.ROI, X_pos = self.Xpos, Y_pos = self.Ypos,
                                            group = self.Group, num_neighborhood= self.neighoborhood_nums)
        neighbor_network.save_network_figs()


if __name__ == '__main__':
    na = Neighborhood_Analysis(path='C:\\Spatial Transcriptomics\\SayaliProjectData\\AllILCIDC.csv',
                               Xpos='Cell X Position', Ypos='Cell Y Position', ROI='Sample Name', Patient='Sample',
                               Celltype='Phenotype', Group='Patient Group', NeighborhoodNo=7)
    na.get_Groupwise_NeighborContacts()
