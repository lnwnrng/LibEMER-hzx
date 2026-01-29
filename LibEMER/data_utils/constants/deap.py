from ..preprocess import generate_adjacency_matrix, generate_rgnn_adjacency_matrix
from .channel_location import system_10_05_loc
DEAP_CHANNEL_NAME = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ', 'C4', 'T8', 'CP6', 'CP2',
                'P4', 'P8', 'PO4', 'O2']
                
DEAP_2D_GRID_LOC = {'FP1':(0,3), 'FP2':(0,5),'AF3':(1,3), 'AF4':(1,5),'F7':(2,0), 'F3':(2,2), 'FZ':(2,4), 'F4':(2,6), 'F8':(2,8),
                    'FC5':(3,1), 'FC1':(3,3), 'FC2':(3,5), 'FC6':(3,7), 'T7':(4,0), 'C3':(4,2), 'CZ':(4,4), 'C4':(4,6), 'T8':(4,8),
                    'CP5':(5,1), 'CP1':(5,3), 'CP2':(5,5), 'CP6':(5,7), 'P7':(6,0),'P3':(6,2), 'PZ':(6,4), 'P4':(6,6), 'P8':(6,8),
                    'PO3':(7,3), 'PO4':(7,5), 'O1':(8,3), 'OZ':(8,4),'O2':(8,5)}
HSLT_DEAP_Regions = {
    'PF': ['FP1', 'AF3', 'AF4', 'FP2'],
    'F': ['F7', 'F3', 'FZ', 'F4', 'F8'],
    'LT': ['FC5', 'T7', 'CP5'],
    'C': ['FC1', 'C3', 'CZ', 'C4', 'FC2'],
    'RT': ['FC6', 'T8', 'CP6'],
    'LP': ['P7', 'P3', 'PO3'],
    'P': ['CP1', 'PZ', 'CP2'],
    'RP': ['P8', 'P4', 'PO4'],
    'O': ['O1', 'OZ', 'O2']
}
DEAP_GLOBAL_CHANNEL_PAIRS = [
    ['FP1', 'FP2'],
    ['AF3', 'AF4'],
    ['FC5', 'FC6'],
    ['CP5', 'CP6'],
    ['O1', 'O2']
]

DEAP_RGNN_ADJACENCY_MATRIX = generate_rgnn_adjacency_matrix(channel_names=DEAP_CHANNEL_NAME, channel_loc=system_10_05_loc,global_channel_pair=DEAP_GLOBAL_CHANNEL_PAIRS)




