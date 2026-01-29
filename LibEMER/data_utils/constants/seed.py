from ..preprocess import generate_adjacency_matrix, generate_rgnn_adjacency_matrix
from .channel_location import system_10_05_loc

SEED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4','F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'] #FEW_REGION_INDEX = [0,2,5,7,9,11,13,23,25,27,29,31,41,43,45,47,49,58]

SEED_2D_GRID_LOC = {
    'FP1':(0,3), 'FPZ': (0,4), 'FP2': (0,5), 'AF3': (1,3), 'AF4': (1,5), 'F7': (2,0), 'F5': (2,1), 'F3': (2,2), 'F1': (2,3), 'FZ': (2,4), 'F2': (2,5), 
    'F4': (2,6), 'F6': (2,7), 'F8': (2,8), 'FT7': (3,0), 'FC5':(3,1), 'FC3': (3,2),'FC1': (3,3), 'FCZ': (3,4), 'FC2': (3,5), 'FC4': (3,6), 'FC6': (3,7), 'FT8': (3,8), 
    'T7': (4,0), 'C5': (4,1), 'C3': (4,2), 'C1': (4,3), 'CZ': (4,4),'C2': (4,5), 'C4': (4,6), 'C6': (4,7), 'T8': (4,8), 'TP7':(5,0),'CP5': (5,1), 'CP3': (5,2), 
    'CP1': (5,3), 'CPZ': (5,4), 'CP2': (5,5), 'CP4': (5,6), 'CP6': (5,7), 'TP8':(5,8),'P7': (6,0),'P5': (6,1),'P3':(6,2),'P1': (6,3),'PZ':  (6,4),
    'P2':(6,5),'P4': (6,6),'P6': (6,7),'P8': (6,8),'PO7': (7,0),'PO5': (7,1),'PO3': (7,2),'POZ': (7,4),'PO4': (7,6),'PO6': (7,7),'PO8': (7,8),
    'CB1':(8,2),'O1':(8,3),'OZ':(8,4),'O2':(8,5),'CB2':(8,6)
}

HSLT_SEED_Regions = {
    'PF': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4'],
    'F':  ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    'LT': ['FT7', 'FC5', 'FC3', 'T7', 'C5', 'C1'], # C3
    'RT': ['FT8', 'FC4', 'FC6', 'T8', 'C2', 'C6', 'CP6'], # C4
    'C':  ['FC1', 'C3', 'CZ', 'FCZ', 'FC2', 'C4'],
    'LP': ['TP7', 'CP5', 'CP3', 'P7', 'P5', 'P3', 'P1', 'PO3'],
    'P':  ['CP1', 'CP2', 'CPZ', 'PZ'],
    'RP': ['TP8', 'CP4', 'P8', 'P6', 'P2', 'P4', 'PO4'],
    'O':  ['PO7', 'PO5', 'POZ', 'PO6', 'PO8', 'CB1', 'O1', 'O2', 'OZ', 'CB2']
}


SEED_ADJACENCY_CHANNEL = {
    'FP1': ['FPZ', 'AF3'],
    'FPZ': ['FP1', 'FP2'],
    'FP2': ['FPZ', 'AF4'],
    'AF3': ['FP1', 'F5', 'F3', 'F1'],
    'AF4': ['F2', 'F4', 'F6', 'FP2'],
    'F7': ['F5', 'FT7'],
    'F5': ['F7', 'AF3', 'F3', 'FC5'],
    'F3': ['AF3', 'F5', 'FC3', 'F1'],
    'F1': ['AF3', 'F3', 'FC1', 'FZ'],
    'FZ': ['F1', 'FCZ', 'F2'],
    'F2': ['FZ', 'FC2', 'F4', 'AF4'],
    'F4': ['F2', 'FC4', 'F6', 'AF4'],
    'F6': ['AF4', 'F4', 'FC6', 'F8'],
    'F8': ['F6', 'FT8'],
    'FT7': ['F7', 'FC5', 'T7'],
    'FC5': ['F5', 'FT7', 'C5', 'FC3'],
    'FC3': ['F3', 'FC5', 'C3', 'FC1'],
    'FC1': ['F1', 'FC3', 'C1', 'FCZ'],
    'FCZ': ['FZ', 'FC1', 'CZ', 'FC2'],
    'FC2': ['F2', 'FCZ', 'C2', 'FC4'],
    'FC4': ['F4', 'FC2', 'C4', 'FC6'],
    'FC6': ['F6', 'FC4', 'C6', 'FT8'],
    'FT8': ['F8', 'FC6', 'T8'],
    'T7': ['FT7', 'C5', 'TP7'],
    'C5': ['FC5', 'T7', 'C3', 'CP5'],
    'C3': ['FC3', 'C5', 'C1', 'CP3'],
    'C1': ['FC1', 'C3', 'CP1', 'CZ'],
    'CZ': ['FCZ', 'C1', 'CPZ', 'C2'],
    'C2': ['FC2', 'CZ', 'CP2', 'C4'],
    'C4': ['FC4', 'C2', 'CP4', 'C6'],
    'C6': ['FC6', 'C4', 'CP6', 'T8'],
    'T8': ['FT8', 'C6', 'TP8'],
    'TP7': ['T7', 'CP5', 'P7'],
    'CP5': ['C5', 'TP7', 'P5', 'CP3'],
    'CP3': ['C3', 'CP5', 'P3', 'CP1'],
    'CP1': ['C1', 'CP3', 'P1', 'CPZ'],
    'CPZ': ['CZ', 'CP1', 'PZ', 'CP2'],
    'CP2': ['C2', 'CPZ', 'P2', 'CP4'],
    'CP4': ['C4', 'CP2', 'P4', 'CP6'],
    'CP6': ['C6', 'CP4', 'P6', 'TP8'],
    'TP8': ['T8', 'CP6', 'P8'],
    'P7': ['TP7', 'P5', 'PO7'],
    'P5': ['CP5', 'P7', 'PO5', 'P3'],
    'P3': ['CP3', 'P5', 'P1'],
    'P1': ['CP1', 'P3', 'PO3', 'PZ'],
    'PZ': ['CPZ', 'P1', 'POZ', 'P2'],
    'P2': ['CP2', 'PZ', 'PO4', 'P4'],
    'P4': ['CP4', 'P2', 'P6'],
    'P6': ['CP6', 'P4', 'P8'],
    'P8': ['TP8', 'P6', 'PO8'],
    'PO7': ['P7', 'PO5', 'CB1'],
    'PO5': ['P5', 'PO7', 'CB1', 'PO3'],
    'PO3': ['P1', 'PO5', 'O1', 'POZ'],
    'POZ': ['PZ', 'PO3', 'OZ', 'PO4'],
    'PO4': ['P2', 'POZ', 'O2', 'PO6'],
    'PO6': ['P6', 'PO4', 'CB2', 'PO8'],
    'PO8': ['P8', 'PO6', 'CB2'],
    'CB1': ['PO7', 'PO5', 'O1'],
    'O1': ['CB1', 'PO3', 'OZ'],
    'OZ': ['POZ', 'O1', 'O2'],
    'O2': ['PO4', 'OZ', 'CB2'],
    'CB2': ['PO6', 'O2', 'PO8']
}

SEED_GLOBAL_CHANNEL_PAIRS = [
    ['FP1', 'FP2'],
    ['AF3', 'AF4'],
    ['F5', 'F6'],
    ['FC5', 'FC6'],
    ['C5', 'C6'],
    ['CP5', 'CP6'],
    ['P5', 'P6'],
    ['PO5', 'PO6'],
    ['O1', 'O2']
]

SEED_ADJACENCY_MATRIX = generate_adjacency_matrix(SEED_CHANNEL_NAME, SEED_ADJACENCY_CHANNEL)

SEED_RGNN_ADJACENCY_MATRIX = generate_rgnn_adjacency_matrix(channel_names=SEED_CHANNEL_NAME, channel_loc=system_10_05_loc, global_channel_pair=SEED_GLOBAL_CHANNEL_PAIRS)

