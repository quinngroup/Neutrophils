import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import gan, wgan, dcgan
import reconstruction
from custom_dataloaders import ImageFolderWithCache, CompositeImageFolder, read_all_images
from utils import RandomVerticalFlip, weights_init

# parse arguments
parser = argparse.ArgumentParser()

# main
parser.add_argument('--mode', type=str, default='train', help='mode of exacution: train | eval-gen-vs-real | eval-real-vs-real')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed, default - the answer to the ultimate question')

# dataset
parser.add_argument('--dataroot', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--dataset_type', type=str, default='folder', help='Type of dataset: folder | folder-cached ')
parser.add_argument('--image_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--image_width', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--test_data', type=str, default='', help='Path to the test set, used for --mode evaluate')

#model
parser.add_argument('--model_type', type=str, required=True, help='Architecture of the model: DCGAN')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")

# training
parser.add_argument('--GAN_algorithm', type=str, default='GAN', help='GAN algorithm to train: GAN | WGAN | WGAN-GP')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--num_disc_iters', type=int, default=1, help='Number of iterations of the discriminator for one update of the generator')

# WGAN and WGAN-GP
parser.add_argument('--wgan_clamp_lower', type=float, default=-0.01, help='for WGAN')
parser.add_argument('--wgan_clamp_upper', type=float, default=0.01, help='for WGAN')
parser.add_argument('--wgangp_lambda', type=float, default=10.0, help='for WGAN-GP')

# optimization
parser.add_argument('--num_iter', type=int, default=3000, help='number of iterations to train for')
parser.add_argument('--optimizer', type=str, default='default', help='optimizer to use for training: default (depends on GAN_algorithm) | adam | rmsprop ')
parser.add_argument('--lrD', type=float, default=None, help='learning rate for Critic, default: depends on GAN_algorithm and optimizer')
parser.add_argument('--lrG', type=float, default=None, help='learning rate for Generator, default: depends on GAN_algorithm and optimizer')
parser.add_argument('--beta1', type=float, default=None, help='beta1 for adam. default: depends on GAN_algorithm and optimizer')
parser.add_argument('--beta2', type=float, default=None, help='beta2 for adam. default: depends on GAN_algorithm and optimizer')
parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')

# logging
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--save_iter', type=int, default=None, help='How often to save models')
parser.add_argument('--image_iter', type=int, default=100, help='How often to draw samples from the models')
parser.add_argument('--fixed_noise_file', type=str, default='', help='File to get shared fixed noise (to evaluate samples)')
parser.add_argument('--prefix_fake_samples', type=str, default='fake_samples', help='Fake image prefix')
parser.add_argument('--prefix_real_samples', type=str, default='real_samples', help='Fake image prefix')

# misc
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for image reading')

opt = parser.parse_args()
print(opt)
print ("Args read")

if opt.mode == 'eval-gen-vs-real':
    assert opt.netG != '', 'You need to provide trained generator to evaluate'

# create dir for experiments
if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir -p {0}'.format(opt.experiment))

# fix random seed
print("Random Seed: ", opt.random_seed)
random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

# deal with GPUs
if opt.cuda:
    cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# make some parameters case insensitive
model_type = opt.model_type.casefold()
dataset_type = opt.dataset_type.casefold()
gan_algorithm = opt.GAN_algorithm.casefold()
optimizer_name = opt.optimizer.casefold()
execution_mode = opt.mode.casefold()

# create the dataset
#opt.mean_val = 0.070972730728464037
#opt.std_val = 0.16034813943416407
opt.mean_val = 0.5
opt.std_val = 0.5

def create_dataset(data_path):
    if dataset_type == 'folder':
        # for the regular 'folder' dataset, the normalization has to have the 1 image channels explicitly
        image_normalization = transforms.Normalize((opt.mean_val,), (opt.std_val,))

        dataset = dset.ImageFolder(root=data_path,
                                   transform=transforms.Compose([
                                       transforms.Scale((opt.image_width, opt.image_height)),
                                       transforms.CenterCrop((opt.image_height, opt.image_width)),
                                       transforms.RandomHorizontalFlip(),
                                       RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       image_normalization,
                                   ]))
    elif dataset_type == 'folder-cached':
        # for the 'folder-cached' dataset and the datasets below, the same 1-channel normalization is applied to all the channels
        image_normalization = transforms.Normalize((opt.mean_val,) * 1, (opt.std_val,) * 1)
        image_cache = read_all_images(data_path, opt.num_workers)
        dataset = ImageFolderWithCache(data_path, image_cache, do_random_flips=True,
                                       normalization=image_normalization)
    else:
        raise RuntimeError("Unknown dataset type: {0}".format(opt.dataset_type))
    return dataset


def create_dataloader(dataset, shuffle=True):
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle,
                                                 num_workers=opt.num_workers)
    return dataloader


# training set
dataset_train = None
if execution_mode not in ['reconstruction']:
    dataset_train = create_dataset(opt.dataroot)
    dataloader_train = create_dataloader(dataset_train)
    opt.class_names = dataset_train.classes
    opt.n_classes = len(dataset_train.classes)


print ('Train Dataset created.')

# test set if needed
if execution_mode in ['eval-gen-vs-real', 'eval-real-vs-real', 'reconstruction']:
    assert opt.test_data
    dataset_test = create_dataset(opt.test_data)
    dataloader_test = create_dataloader(dataset_test, shuffle=False)
    if dataset_train and execution_mode in ['eval-gen-vs-real', 'reconstruction']:
        # doe not check this for opt.mode == 'eval-real-vs-real', because it is done for comparing different classes
        assert(opt.class_names == dataset_test.classes)
    else:
        opt.class_names = dataset_test.classes
        opt.n_classes = len(dataset_test.classes)

# add more options
# bcoz working with gray scale images, nc=1
opt.original_nc = 1
opt.nc = opt.original_nc
opt.n_extra_layers = int(opt.n_extra_layers)
opt.g_input_size = opt.nz

# create the models
if gan_algorithm == 'wgan-gp':
    batch_norm_in_disc = False
else:
    batch_norm_in_disc = True

if model_type == 'dcgan':
    opt.separable_gen = False
    netG = dcgan.DCGAN_G((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ngf, opt.n_extra_layers)
    netD = dcgan.DCGAN_D((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ndf, opt.n_extra_layers,
                         use_batch_norm=batch_norm_in_disc)
    print ('netG, netD initialized..')

else:
    raise RuntimeError("Unknown model type: {0}".format(opt.model_type))

# init the generator
netG.apply(weights_init)
if opt.netG != '':  # load checkpoint if needed
    print('Loading netG from', opt.netG)
    netG.load_state_dict(torch.load(opt.netG))

# init the discriminator
netD.apply(weights_init)
if opt.netD != '':
    print('Loading netD from', opt.netD)
    netD.load_state_dict(torch.load(opt.netD))

# print the models to examine them
print(netG)
print(netD)


def gan_choice(opt_dict, param_name='parameter'):
    if gan_algorithm in opt_dict:
        pick = opt_dict[gan_algorithm]
    else:
        raise RuntimeError("Unknown value of {1}: {0}".format(opt.GAN_algorithm, param_name))
    return pick

# setup optimizer
if optimizer_name == 'default':
    optimizer_name = gan_choice({ 'gan': 'adam', 'wgan': 'rmsprop', 'wgan-gp': 'adam'}, 'optimizer')

if opt.lrD is None:
    opt.lrD = gan_choice({ 'gan': 0.0002, 'wgan': 0.00005, 'wgan-gp': 0.0001}, 'lrD')

if opt.lrG is None:
    opt.lrG = gan_choice({ 'gan': 0.0002, 'wgan': 0.00005, 'wgan-gp': 0.0001}, 'lrG')

if opt.beta1 is None:
    opt.beta1 = gan_choice({ 'gan': 0.5, 'wgan': 0.0, 'wgan-gp': 0.0}, 'beta1')

if opt.beta2 is None:
    opt.beta2 = gan_choice({ 'gan': 0.999, 'wgan': 0.9, 'wgan-gp': 0.9}, 'beta2')

if optimizer_name == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2),weight_decay=opt.weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2),weight_decay=opt.weight_decay)
elif optimizer_name == 'rmsprop':
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
else:
    raise (RuntimeError("Do not recognize optimizer %s" % opt.optimizer))

# create the GAN class
gan_model = gan_choice({'gan': gan.GAN(netG, netD, optimizerD, optimizerG, opt),
                        'wgan': wgan.WGAN(netG, netD, optimizerD, optimizerG, opt),
                        'wgan-gp': wgan.WGANGP(netG, netD, optimizerD, optimizerG, opt)},
                       'GAN_algorithm')

# the main operation
if execution_mode == 'train':
    # train the model
    print ('Training begins......')
    gan_model.train(dataloader_train, opt)

elif execution_mode == 'reconstruction':
    reconstruction.run_experiment(gan_model.netG, dataloader_test, opt.dataroot, opt, optimize_red_first=False)
else:
    raise RuntimeError("Unknown mode: {0}".format(opt.mode))
