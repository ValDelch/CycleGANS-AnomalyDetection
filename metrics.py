import torch
from torch import nn
import torchvision.models as models
from scipy import linalg
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()


        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            # x = F.interpolate(x,
            #                   size=(299, 299),
            #                   mode='bilinear',
            #                   align_corners=False)
            width, height = x.shape[2], x.shape[3]
            pad_width = int((299-width)/2)
            pad_height = int((299-height)/2)
            x = F.pad(input=x, pad=(pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

def calculate_activation_statistics(images, model, dims=2048, cuda=False):

    model.eval()
    act=np.empty((len(images), dims))

    if cuda:
        batch=images.cuda()
    else:
        batch=images

    if images.shape[1] == 1:
        pred = model(torch.cat([batch,batch,batch], 1))[0]
    else:
        pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        # pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return np.array(mu), np.array(sigma)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2


    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        # print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))


    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet(images_real, images_fake, model, cuda=False):

    mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=cuda)
    mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=cuda)
    
    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value

def conf_mat(OK, NG, threshold):
    fp = [OK[i] for i in range(len(OK)) if OK[i] >= threshold]
    tp = [NG[i] for i in range(len(NG)) if NG[i] >= threshold]
    tn = [OK[i] for i in range(len(OK)) if OK[i] < threshold]
    fn = [NG[i] for i in range(len(NG)) if NG[i] < threshold]
    return fp,tp,fn,tn

def metric_conf(fp,tp,fn,tn):
    accuracy = (len(tp) + len(tn)) / (len(tp) + len(fn) + len(fp) + len(tn))
    precision = [(len(tp)/(len(tp)+len(fp))) if (len(tp)+len(fp)) != 0 else 0][0]
    specificity = [(len(tn)/(len(tn)+len(fp))) if (len(tn)+len(fp)) != 0 else 0][0]
    fpr = [(len(fp)/(len(fp)+len(tn))) if (len(fp)+len(tn)) != 0 else 0][0]
    fnr = [(len(fn)/(len(fn)+len(tp))) if (len(fn)+len(tp)) != 0 else 0][0]
    return accuracy,precision,specificity,specificity,fpr,fnr

def auc_roc_score(OK,NG):
    from sklearn.metrics import roc_curve, auc
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(np.array(OK+NG).reshape(-1, 1))

    y = np.concatenate((np.zeros(len(OK)), np.ones(len(NG))))
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y, scores, pos_label=1)
    auc_score = auc(false_pos_rate, true_pos_rate)
    return auc_score