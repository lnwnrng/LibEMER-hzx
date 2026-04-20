from models.Het import Het_Model
from models.DCCA_AM import DCCA_AM
from models.BimodalLSTM import BimodalLSTM
from models.CRNN import CRNN
from models.DCCA import DCCA
from models.BDDAE import BDDAE
from models.MCAF import MCAF
from models.CMCM import CMCM
from models.CFDA_CSF import CFDA_CSF
from models.HetEmotionNet import HetEmotionNet
from models.G2G import EncoderNet

Model={
    'Het_Model': Het_Model,
    'DCCA_AM': DCCA_AM,
    'BimodalLSTM': BimodalLSTM,
    'CRNN': CRNN,
    'DCCA': DCCA,
    'BDDAE': BDDAE,
    'MCAF': MCAF,
    'CMCM': CMCM,
    'CFDA_CSF': CFDA_CSF,
    'HetEmotionNet': HetEmotionNet,
    'G2G': EncoderNet
}
