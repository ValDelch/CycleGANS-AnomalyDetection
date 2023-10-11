from architectures.cycle_GANs_blocks import *
from anomalib.models.patchcore.torch_model import PatchcoreModel
from anomalib.models.padim.torch_model import PadimModel
from anomalib.models.ganomaly.torch_model import GanomalyModel
from anomalib.models.ganomaly.loss import GeneratorLoss, DiscriminatorLoss
import itertools

class DecayLR:
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (self.epochs - self.decay_epochs)
    

def return_training_setup(model_name, model_config, dataset_config, device):
    """ Return the training setup for the given model """
    if model_name == 'cgan256':
        generator = get_cgan256
    elif model_name == 'cgan64':
        generator =  get_cgan64
    elif model_name == 'ganomaly':
        generator =  get_ganomaly
    elif model_name == 'patchcore':
        generator =  get_patchcore
    elif model_name == 'padim':
        generator =  get_padim
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))
    
    return generator(model_config, dataset_config, device)


def get_cgan256(model_config, dataset_config, device):

    if model_config["always_RGB"] or dataset_config["RGB"]:
        C_in = 3
    else:
        C_in = 1

    # Models
    if dataset_config['image_size'] < 128:
        n_blocks = 6
    else:
        n_blocks = 9

    netG_normal2abnormal = ResnetGenerator(C_in=C_in, C_out=C_in, n_blocks=n_blocks, norm_layer=nn.InstanceNorm2d).to(device)
    netG_normal2abnormal.apply(weights_init)
    netG_abnormal2normal = ResnetGenerator(C_in=C_in, C_out=C_in, n_blocks=n_blocks, norm_layer=nn.InstanceNorm2d).to(device)
    netG_abnormal2normal.apply(weights_init)
    netD_normal = NLayerDiscriminator(C_in=C_in, n_layers=3, norm_layer=nn.InstanceNorm2d).to(device)
    netD_normal.apply(weights_init)
    netD_abnormal = NLayerDiscriminator(C_in=C_in, n_layers=3, norm_layer=nn.InstanceNorm2d).to(device)
    netD_abnormal.apply(weights_init)

    # Losses
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()
    adversarial_loss = GANLoss('lsgan').to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_normal2abnormal.parameters(), netG_abnormal2normal.parameters()),
                                   lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_normal = torch.optim.Adam(netD_normal.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_abnormal = torch.optim.Adam(netD_abnormal.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Schedulers
    lr_lambda = DecayLR(dataset_config["epochs"], 0, dataset_config["epochs"]//10).step
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D_normal = torch.optim.lr_scheduler.LambdaLR(optimizer_D_normal, lr_lambda=lr_lambda)
    lr_scheduler_D_abnormal = torch.optim.lr_scheduler.LambdaLR(optimizer_D_abnormal, lr_lambda=lr_lambda)

    setup = {
        'models': {
            'netG_normal2abnormal': netG_normal2abnormal,
            'netG_abnormal2normal': netG_abnormal2normal,
            'netD_normal': netD_normal,
            'netD_abnormal': netD_abnormal
        },
        'losses': {
            'cycle_loss': cycle_loss,
            'identity_loss': identity_loss,
            'adversarial_loss': adversarial_loss
        },
        'optimizers': {
            'optimizer_G': optimizer_G,
            'optimizer_D_normal': optimizer_D_normal,
            'optimizer_D_abnormal': optimizer_D_abnormal
        },
        'schedulers': {
            'lr_scheduler_G': lr_scheduler_G,
            'lr_scheduler_D_normal': lr_scheduler_D_normal,
            'lr_scheduler_D_abnormal': lr_scheduler_D_abnormal
        }
    }

    return setup


def get_cgan64(model_config, dataset_config, device):

    if model_config["always_RGB"] or dataset_config["RGB"]:
        C_in = 3
    else:
        C_in = 1

    # Models
    netG_normal2abnormal = ResnetGenerator(C_in=C_in, C_out=C_in, n_blocks=6, norm_layer=nn.InstanceNorm2d).to(device)
    netG_normal2abnormal.apply(weights_init)
    netG_abnormal2normal = ResnetGenerator(C_in=C_in, C_out=C_in, n_blocks=6, norm_layer=nn.InstanceNorm2d).to(device)
    netG_abnormal2normal.apply(weights_init)
    netD_normal = NLayerDiscriminator(C_in=C_in, n_layers=3, norm_layer=nn.InstanceNorm2d).to(device)
    netD_normal.apply(weights_init)
    netD_abnormal = NLayerDiscriminator(C_in=C_in, n_layers=3, norm_layer=nn.InstanceNorm2d).to(device)
    netD_abnormal.apply(weights_init)

    # Losses
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()
    adversarial_loss = GANLoss('lsgan').to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_normal2abnormal.parameters(), netG_abnormal2normal.parameters()),
                                   lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_normal = torch.optim.Adam(netD_normal.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_abnormal = torch.optim.Adam(netD_abnormal.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Schedulers
    lr_lambda = DecayLR(dataset_config["epochs"], 0, dataset_config["epochs"]//10).step
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D_normal = torch.optim.lr_scheduler.LambdaLR(optimizer_D_normal, lr_lambda=lr_lambda)
    lr_scheduler_D_abnormal = torch.optim.lr_scheduler.LambdaLR(optimizer_D_abnormal, lr_lambda=lr_lambda)

    setup = {
        'models': {
            'netG_normal2abnormal': netG_normal2abnormal,
            'netG_abnormal2normal': netG_abnormal2normal,
            'netD_normal': netD_normal,
            'netD_abnormal': netD_abnormal
        },
        'losses': {
            'cycle_loss': cycle_loss,
            'identity_loss': identity_loss,
            'adversarial_loss': adversarial_loss
        },
        'optimizers': {
            'optimizer_G': optimizer_G,
            'optimizer_D_normal': optimizer_D_normal,
            'optimizer_D_abnormal': optimizer_D_abnormal
        },
        'schedulers': {
            'lr_scheduler_G': lr_scheduler_G,
            'lr_scheduler_D_normal': lr_scheduler_D_normal,
            'lr_scheduler_D_abnormal': lr_scheduler_D_abnormal
        }
    }

    return setup


def get_ganomaly(model_config, dataset_config, device):

    if model_config["always_RGB"] or dataset_config["RGB"]:
        C_in = 3
    else:
        C_in = 1

    im_size = min([model_config["max_image_size"], dataset_config["image_size"]])

    # Models
    model = GanomalyModel(input_size=(im_size, im_size),
                          num_input_channels=C_in,
                          n_features=32,
                          latent_vec_size=128).to(device)
    
    # Losses
    loss_Generator = GeneratorLoss()
    loss_Discriminator = DiscriminatorLoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(model.generator.parameters(),
                                   lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(),
                                   lr=0.0002, betas=(0.5, 0.999))

    # Schedulers
    lr_lambda = DecayLR(dataset_config["epochs"], 0, dataset_config["epochs"]//10).step
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda)

    setup = {
        'models': {
            'model': model
        },
        'losses': {
            'loss_Generator': loss_Generator,
            'loss_Discriminator': loss_Discriminator
        },
        'optimizers': {
            'optimizer_G': optimizer_G,
            'optimizer_D': optimizer_D
        },
        'schedulers': {
            'lr_scheduler_G': lr_scheduler_G,
            'lr_scheduler_D': lr_scheduler_D
        }
    }

    return setup


def get_patchcore(model_config, dataset_config, device):

    im_size = min([model_config["max_image_size"], dataset_config["image_size"]])

    model = PatchcoreModel(input_size=(im_size, im_size),
                           backbone='wide_resnet50_2',
                           layers=['layer2', 'layer3'],
                           num_neighbors=9).to(device)
    
    setup = {
        'models': {
            'model': model
        },
        'losses': None,
        'optimizers': None,
        'schedulers': None
    }

    return setup


def get_padim(model_config, dataset_config, device):

    im_size = min([model_config["max_image_size"], dataset_config["image_size"]])

    model = PadimModel(input_size=(im_size, im_size),
                       backbone='resnet18',
                       layers=['layer1', 'layer2', 'layer3'],
                       n_features=32).to(device)
    
    setup = {
        'models': {
            'model': model
        },
        'losses': None,
        'optimizers': None,
        'schedulers': None
    }

    return setup