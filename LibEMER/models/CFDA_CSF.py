import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转层 (GRL)
    前向传播中，是一个恒等函数, 而在反向传播中,则会使梯度反转
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_val
        return output, None
    
class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

class CMDLoss(nn.Module):
    def __init__(self, k=2):
        super(CMDLoss, self).__init__()
        self.k = k

    def forward(self, source, target):
        loss = 0
        for idk in range(1, self.k + 1):
            source_moment = self._central_moment(source, idk)
            target_moment = self._central_moment(target, idk)
            moment_diff = torch.norm(source_moment - target_moment, p=2)
            loss += moment_diff
        return loss

    def _central_moment(self, feature, k):
        mean = torch.mean(feature, dim=0, keepdim=True)
        if k == 1:
            return mean.squeeze(0)
        moment = torch.mean((feature - mean) ** k, dim=0)
        return moment
    
class CORALLoss(nn.Module):
    def __init__(self):
        super(CORALLoss, self).__init__()

    def forward(self, eeg, bio):
        n, d = eeg.shape
        eeg_centered = eeg - torch.mean(eeg, dim=0, keepdim=True)
        bio_centered = bio - torch.mean(bio, dim=0, keepdim=True)

        eeg_cov = torch.matmul(eeg_centered.T, eeg_centered) / (n - 1)
        bio_cov = torch.matmul(bio_centered.T, bio_centered) / (n - 1)

        diff = eeg_cov - bio_cov
        loss = torch.sum(diff ** 2) / (4 * d ** 2)
        return loss

class CELoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(CELoss, self).__init__()
        self.threshold = threshold

    def forward(self, predict):
        probs = torch.softmax(predict, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        confident_mask = entropy < self.threshold

        if confident_mask.sum() == 0:
            return torch.tensor(0.0).to(predict.device)

        confident_probs = probs[confident_mask]
        max_probs, _ = torch.max(confident_probs, dim=1)
        loss = -torch.mean(torch.log(max_probs + 1e-8))
        return loss 

def central_moment(feature, k=2):
    '''
    对输入数据计算k阶中心矩, 用于CMD计算
    '''
    mean = torch.mean(feature, dim=0, keepdim=True)

    if k ==1:
        return mean.squeeze(0)
    moment = torch.mean((feature - mean)**k, dim = 0)
    return moment
def cmd_loss(source, target, k=2):
    '''
    实现对每个模态的跨域细粒度对齐, 
    对经过L2标准化后的每个模态的源域目标域数据计算CMD
    '''
    loss = 0
    for idk in range(1, k+1):
        source_moment = central_moment(source, k=idk)
        target_moment = central_moment(target, k=idk)
        moment_diff = torch.norm(source_moment - target_moment, p=2)
        loss += moment_diff

    return loss
def coral_loss(eeg, bio):
    '''
    通过最小化coral_loss来优化不同模态间的相关性
    通过两个模态特征的自协方差矩阵计算两个模态间的差异性
    '''
    n, d = eeg.shape
    eeg_centered = eeg - torch.mean(eeg, dim=0, keepdim=True)
    bio_centered = bio - torch.mean(bio, dim=0, keepdim=True)

    eeg_cov = torch.matmul(eeg_centered.T, eeg_centered)/(n-1)
    bio_cov = torch.matmul(bio_centered.T, bio_centered)/(n-1)

    diff = eeg_cov - bio_cov
    loss = torch.sum(diff**2)/(4*d**2)
    return loss

def ce_loss(predict, threshold = 0.5):
    '''
    条件熵损失,仅对预测结果置信度高的目标域样本起效
    将这些样本的熵最小化,以此提高这些样本的对结果的自信程度
    '''
    probs = F.softmax(predict, dim=1)

    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    confident_mask = entropy < threshold
    
    if confident_mask.sum() == 0:
        return torch.tensor(0.0).to(predict.device)
        
    confident_probs = probs[confident_mask]

    max_probs, _ = torch.max(confident_probs, dim=1)
    
    loss = -torch.mean(torch.log(max_probs + 1e-8))
    
    return loss


class FeatureExtractor(nn.Module):
    """
    [cite_start]用于EEG和眼动信号的特征提取器子网络 [cite: 278]。
    架构遵循论文中的表 I。
    """
    def __init__(self, input_dim, dropout_rate):
        super(FeatureExtractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(200, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(150, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.extractor(x)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            GradientReversalLayer(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class CFDA_CSF(nn.Module):
    def __init__(self, eeg_input_dim, bio_input_dim, dropout_rate,num_classes):
        super(CFDA_CSF, self).__init__()
        self.eeg_input_dim = eeg_input_dim
        self.bio_input_dim = bio_input_dim

        # 定义两个模态的特征提取器
        self.feature_extractor_eeg = FeatureExtractor(eeg_input_dim, dropout_rate) # G_f1
        self.feature_extractor_eye = FeatureExtractor(bio_input_dim, dropout_rate) # G_f2
        
        # 定义分类器和判别器
        self.classifier = Classifier(num_classes) # G_h
        self.discriminator = Discriminator()    # G_d

    def forward(self, source_eeg, source_bio, target_eeg, target_bio):
        source_eeg = source_eeg.reshape(-1, self.eeg_input_dim)
        source_bio = source_bio.reshape(-1, self.bio_input_dim)
        target_eeg = target_eeg.reshape(-1, self.eeg_input_dim)
        target_bio = target_bio.reshape(-1, self.bio_input_dim)

        #特征提取
        s_eeg_extract = self.feature_extractor_eeg(source_eeg)
        s_bio_extract = self.feature_extractor_eye(source_bio)
        t_eeg_extract = self.feature_extractor_eeg(target_eeg)
        t_bio_extract = self.feature_extractor_eye(target_bio)

        #对提取的数据应用L2归一化, 这些数据用于细粒度对齐
        norm_s_eeg_extract = F.normalize(s_eeg_extract, p=2, dim=1)
        norm_s_bio_extract = F.normalize(s_bio_extract, p=2, dim=1)
        norm_t_eeg_extract = F.normalize(t_eeg_extract, p=2, dim=1)
        norm_t_bio_extract = F.normalize(t_bio_extract, p=2, dim=1)

        source_fused = s_eeg_extract + s_bio_extract
        target_fused = t_eeg_extract + t_bio_extract

        source_pred = self.classifier(source_fused)
        
        target_pred = self.classifier(target_fused)

        #域判别
        fused_concat = torch.cat((source_fused, target_fused), dim=0)
        pred_domain = self.discriminator(fused_concat)

        return (
            source_pred, target_pred, pred_domain,
            s_eeg_extract, s_bio_extract, t_eeg_extract, t_bio_extract,
            norm_s_eeg_extract, norm_s_bio_extract, norm_t_eeg_extract, norm_t_bio_extract
        )