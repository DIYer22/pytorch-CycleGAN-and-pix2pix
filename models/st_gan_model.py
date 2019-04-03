import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from boxx.ylth import * 
from boxx.ylth import timegap, getpara, randfloat, pred

class StGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D',]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = [ 'fake_raw',  'fake', 'real',]

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['D', 'Stn']
        else:  # during test time, only load Gs
            self.model_names = [ 'Stn']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netStn = networks.define_stn(7, opt.crop_size, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:  # define discriminators
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(), ), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)
            
            self.optimizer_stn = torch.optim.Adam(self.netStn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_stn)

    def composition(self, input):
        self.fg = fg = input['fg'].to(self.device)
        A = input['A'].to(self.device)
        theta, self.composited = self.netStn(A, fg)
        self.theta = theta
#        if timegap(120, 'theta') and randfloat() < .05:
#            print(theta.detach().cpu().numpy().round(2))
        return self.composited
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.fake_raw = input['A'].to(self.device)
        
        self.fake = self.composition(input)#.detach()
        
        self.real = input['B'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D_A"""
        fake = self.fake_pool.query(self.fake)
        self.loss_D = self.backward_D_basic(self.netD, self.real, fake)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        self.loss_G = self.criterionGAN(self.netD(self.fake), True)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
#        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
#        self.backward_G()             # calculate gradients for G_A and G_B
#        self.optimizer_G.step()       # update G_A and G_B's weights
        
        self.optimizer_stn.zero_grad()
        self.loss_stn = self.criterionGAN(self.netD(self.composited), True)
        
        isLog = timegap(30, 'lossNorma') and randfloat() < .3
        if self.opt.l2:
            self.loss_theta_l2 = ((((self.theta - networks.theta_mean.to(self.theta.device))**2).sum(-1).sum(-1))).sum()
            
            self.loss_theta_l2 *= self.opt.l2 
            
            self.loss_theta_l2.backward(retain_graph=True)
        
            if isLog:
                pred-(self.opt.name)
                print(self.theta[0].detach().cpu().numpy().round(2))
                print("l2 grad norma:", getpara(self.netStn).grad.abs().mean())
        # all:0.0014, l2: 0.0006, stn: 0.0013
        self.loss_stn *= self.opt.stn_w
        self.loss_stn.backward()
        
        if isLog:
            print("l2 + stn grad norma:", getpara(self.netStn).grad.abs().mean())
            
        self.optimizer_stn.step()
        
        # D_A and D_B
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
