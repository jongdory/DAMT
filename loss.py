import torch
from torch.nn import functional as F
from monai.losses import DiceLoss, DiceCELoss

class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, args, weight=None):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.weight = weight
        if weight is not None:
            self.weight = self.weight.to(device)

    def forward(self, inputs, targets):
        if self.weight is not None:
            assert inputs[1].shape == self.weight.shape, "Inputs and weight must have the same shape"
            loss = ((inputs - targets)**2 * self.weight).mean()
        else:
            loss = ((inputs - targets)**2).mean()
        return loss


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss()
        self.loc_loss = torch.nn.CrossEntropyLoss()
        self.recon_loss = torch.nn.L1Loss()
        self.contrast_loss = Contrast(args, batch_size)
        self.atlas_loss = DiceCELoss(to_onehot_y=True, softmax=True) 
        self.feat_loss = torch.nn.MSELoss()
        self.texture_loss = torch.nn.MSELoss()

        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0
        self.alpha4 = 1.0
        self.alpha5 = 0.2
        self.alpha6 = 1.0
        self.alpha7 = 1.0

    def __call__(self, output_rot, target_rot, output_loc, target_loc, output_recons, target_recons, output_recons2, target_recons2, 
                    output_contrastive, target_contrastive, output_glo_atlas, target_glo_atlas, output_loc_atlas, target_loc_atlas, 
                    output_feat, target_feat, output_texture, target_texture):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        loc_loss = self.alpha2 * self.loc_loss(output_loc, target_loc)
        recon_loss = self.alpha3 * (self.recon_loss(output_recons, target_recons) + self.recon_loss(output_recons2, target_recons2))
        contrast_loss = self.alpha4 * self.contrast_loss(output_contrastive, target_contrastive)
        atlas_loss = self.alpha5 * (self.atlas_loss(output_glo_atlas, target_glo_atlas) + self.atlas_loss(output_loc_atlas, target_loc_atlas))
        feat_loss = self.alpha6 * self.feat_loss(output_feat, target_feat)
        texture_loss = self.alpha7 * self.texture_loss(output_texture, target_texture)
            
        total_loss = rot_loss + loc_loss + recon_loss + contrast_loss + atlas_loss + feat_loss + texture_loss

        return total_loss, (rot_loss, loc_loss, contrast_loss, recon_loss, atlas_loss, feat_loss, texture_loss)