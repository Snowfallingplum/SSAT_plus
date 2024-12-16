import torch
import losses
import torch.nn as nn
import pytorch_spl_loss as spl
import torch.nn.functional as F
import torchvision.transforms as transforms

import networks
from networks import init_net

class MakeupGAN(nn.Module):
    def __init__(self, opts):
        super(MakeupGAN, self).__init__()
        self.opts = opts

        # parameters
        self.lr = opts.lr
        self.batch_size = opts.batch_size

        self.gpu = torch.device('cuda:{}'.format(opts.gpu)) if opts.gpu >= 0 else torch.device('cpu')
        self.input_dim = opts.input_dim
        self.output_dim = opts.output_dim
        self.semantic_dim = opts.semantic_dim

        self.CP_weight = opts.CP_weight
        self.GP_weight = opts.GP_weight
        self.rec_weight = opts.rec_weight
        self.color_weight = opts.color_weight
        self.adv_weight = opts.adv_weight
        self.temporal_weight = opts.temporal_weight

        # for video
        self.grid_transform = transforms.RandomPerspective(distortion_scale=0.1, p=1.0)

        self._build_model(self.opts)

    def _build_model(self,opts):
        if opts.phase=='train':
            # discriminators
            self.dis_non_makeup = init_net(
                networks.MultiScaleDis(opts.input_dim, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm),
                opts.gpu, init_type='normal', gain=0.02)
            self.dis_makeup = init_net(
                networks.MultiScaleDis(opts.input_dim, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm),
                opts.gpu, init_type='normal', gain=0.02)

            # encoders
            self.enc_content = init_net(networks.E_content(opts.input_dim), opts.gpu, init_type='normal', gain=0.02)
            self.enc_makeup = init_net(networks.E_makeup(opts.input_dim), opts.gpu, init_type='normal', gain=0.02)
            self.enc_semantic = init_net(networks.E_semantic(opts.semantic_dim), opts.gpu, init_type='normal', gain=0.02)
            self.transformer = init_net(networks.Transformer(), opts.gpu, init_type='normal', gain=0.02)
            # generator
            self.gen_transfer = init_net(networks.Decoder(opts.output_dim), opts.gpu, init_type='normal', gain=0.02)
            self.gen_removal = init_net(networks.Decoder(opts.output_dim), opts.gpu, init_type='normal', gain=0.02)

            # optimizers
            self.dis_non_makeup_opt = torch.optim.Adam(self.dis_non_makeup.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                                       weight_decay=0.0001)
            self.dis_makeup_opt = torch.optim.Adam(self.dis_makeup.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                                   weight_decay=0.0001)
            self.enc_content_opt = torch.optim.Adam(self.enc_content.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                                    weight_decay=0.0001)
            self.enc_makeup_opt = torch.optim.Adam(self.enc_makeup.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                                   weight_decay=0.0001)
            self.enc_semantic_opt = torch.optim.Adam(self.enc_semantic.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                                     weight_decay=0.0001)
            self.transformer_opt = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                                    weight_decay=0.0001)
            self.gen_transfer_opt = torch.optim.Adam(self.gen_transfer.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                            weight_decay=0.0001)
            self.gen_removal_opt = torch.optim.Adam(self.gen_removal.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                            weight_decay=0.0001)
            
            self.criterionColor = losses.ColorLoss(cluster_number=32)
            self.criterionL1 = nn.L1Loss()
            self.criterionL2 = nn.MSELoss()
        else:
            # encoders
            self.enc_content = init_net(networks.E_content(opts.input_dim), opts.gpu, init_type='normal', gain=0.02)
            self.enc_makeup = init_net(networks.E_makeup(opts.input_dim), opts.gpu, init_type='normal', gain=0.02)
            self.enc_semantic = init_net(networks.E_semantic(opts.semantic_dim), opts.gpu, init_type='normal',
                                         gain=0.02)
            self.transformer = init_net(networks.Transformer(), opts.gpu, init_type='normal', gain=0.02)
            # generator
            self.gen_transfer = init_net(networks.Decoder(opts.output_dim), opts.gpu, init_type='normal', gain=0.02)
            self.gen_removal = init_net(networks.Decoder(opts.output_dim), opts.gpu, init_type='normal', gain=0.02)

    def load_data(self,data):
        self.non_makeup = data['non_makeup'].to(self.gpu).detach()
        self.makeup = data['makeup'].to(self.gpu).detach()
        self.transfer = data['transfer'].to(self.gpu).detach()
        self.removal = data['removal'].to(self.gpu).detach()
        self.non_makeup_parse = data['non_makeup_parse'].to(self.gpu).detach()
        self.makeup_parse = data['makeup_parse'].to(self.gpu).detach()

        self.non_makeup_parse_down4 = F.interpolate(self.non_makeup_parse, scale_factor=0.25, mode='nearest')
        self.makeup_parse_down4 = F.interpolate(self.makeup_parse, scale_factor=0.25, mode='nearest')

        self.non_makeup_all_mask = data['non_makeup_all_mask'].to(self.gpu).detach()
        self.non_makeup_face_mask = data['non_makeup_face_mask'].to(self.gpu).detach()
        self.non_makeup_brow_mask = data['non_makeup_brow_mask'].to(self.gpu).detach()
        self.non_makeup_eye_mask = data['non_makeup_eye_mask'].to(self.gpu).detach()
        self.non_makeup_lip_mask = data['non_makeup_lip_mask'].to(self.gpu).detach()

        self.makeup_all_mask = data['makeup_all_mask'].to(self.gpu).detach()
        self.makeup_face_mask = data['makeup_face_mask'].to(self.gpu).detach()
        self.makeup_brow_mask = data['makeup_brow_mask'].to(self.gpu).detach()
        self.makeup_eye_mask = data['makeup_eye_mask'].to(self.gpu).detach()
        self.makeup_lip_mask = data['makeup_lip_mask'].to(self.gpu).detach()

    def forward(self):
            # first transfer and removal
            self.z_non_makeup_c = self.enc_content(self.non_makeup)
            self.z_non_makeup_s = self.enc_semantic(self.non_makeup_parse)
            self.z_non_makeup_a = self.enc_makeup(self.non_makeup)

            self.z_makeup_c = self.enc_content(self.makeup)
            self.z_makeup_s = self.enc_semantic(self.makeup_parse)
            self.z_makeup_a = self.enc_makeup(self.makeup)
            # warp makeup style
            self.mapX, self.mapY, self.z_non_makeup_a_warp, self.z_makeup_a_warp = self.transformer(self.z_non_makeup_c,
                                                                                                    self.z_makeup_c,
                                                                                                    self.z_non_makeup_s,
                                                                                                    self.z_makeup_s,
                                                                                                    self.z_non_makeup_a,
                                                                                                    self.z_makeup_a,
                                                                                                    self.non_makeup_parse_down4,
                                                                                                    self.makeup_parse_down4)
            # makeup transfer and removal
            self.z_transfer = self.gen_transfer(self.z_non_makeup_c, self.z_makeup_a_warp)
            self.z_removal = self.gen_removal(self.z_makeup_c, self.z_non_makeup_a_warp)

            # rec
            self.z_rec_non_makeup = self.gen_removal(self.z_non_makeup_c, self.z_non_makeup_a)
            self.z_rec_makeup = self.gen_transfer(self.z_makeup_c, self.z_makeup_a)

            # Temporal consistency
            need_warp_data=torch.cat([self.z_transfer.detach(),self.non_makeup.detach(),self.non_makeup_parse.detach()],dim=1)
            warp_data=self.grid_transform(need_warp_data)
            self.warp_z_transfer=warp_data[:,0:3,::].detach()
            self.warp_non_makeup=warp_data[:,3:6,::].detach()
            self.warp_non_makeup_parse=warp_data[:,6:,::].detach()

            self.warp_non_makeup_parse_down4 = F.interpolate(self.warp_non_makeup_parse, scale_factor=0.25, mode='nearest')
            self.warp_z_non_makeup_c = self.enc_content(self.warp_non_makeup)
            self.warp_z_non_makeup_s = self.enc_semantic(self.warp_non_makeup_parse)
            self.warp_z_non_makeup_a = self.enc_makeup(self.warp_non_makeup)

            self.mapX2, self.mapY2, self.z_non_makeup_a_warp2, self.z_makeup_a_warp2 = self.transformer(self.warp_z_non_makeup_c,
                                                                                                    self.z_makeup_c,
                                                                                                    self.warp_z_non_makeup_s,
                                                                                                    self.z_makeup_s,
                                                                                                    self.warp_z_non_makeup_a,
                                                                                                    self.z_makeup_a,
                                                                                                    self.warp_non_makeup_parse_down4,
                                                                                                    self.makeup_parse_down4)

            self.z_transfer_video = self.gen_transfer(self.warp_z_non_makeup_c, self.z_makeup_a_warp2)
           

    def update_D(self):
        self.forward()

        # update dis_non_makeup
        self.dis_non_makeup_opt.zero_grad()
        loss_dis_non_makeup = self.backward_D(self.dis_non_makeup, self.non_makeup, self.z_removal)
        self.dis_non_makeup_loss = loss_dis_non_makeup.item()
        self.dis_non_makeup_opt.step()

        # update dis_makeup
        self.dis_makeup_opt.zero_grad()
        loss_dis_makeup = self.backward_D(self.dis_makeup, self.makeup, self.z_transfer)
        self.dis_makeup_loss = loss_dis_makeup.item()
        self.dis_makeup_opt.step()

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def update_EG(self):
        # update G, Ec, Ea
        self.enc_content_opt.zero_grad()
        self.enc_makeup_opt.zero_grad()
        self.enc_semantic_opt.zero_grad()
        self.transformer_opt.zero_grad()
        self.gen_transfer_opt.zero_grad()
        self.gen_removal_opt.zero_grad()
        self.backward_EG()
        self.enc_content_opt.step()
        self.enc_makeup_opt.step()
        self.enc_semantic_opt.step()
        self.transformer_opt.step()
        self.gen_transfer_opt.step()
        self.gen_removal_opt.step()

    def backward_EG(self):
        # Ladv for generator
        loss_G_GAN_makeup = self.backward_G_GAN(self.z_transfer, self.dis_makeup)
        loss_G_GAN_non_makeup = self.backward_G_GAN(self.z_removal, self.dis_non_makeup)
        loss_G_GAN = (loss_G_GAN_non_makeup + loss_G_GAN_makeup) * 0.5 * self.adv_weight

        # rec loss
        loss_G_rec_non_makeup = self.criterionL1(self.non_makeup, self.z_rec_non_makeup)
        loss_G_rec_makeup = self.criterionL1(self.makeup, self.z_rec_makeup)
        loss_G_rec = (loss_G_rec_non_makeup + loss_G_rec_makeup) * 0.5* self.rec_weight

        # temporal loss
        loss_G_temporal = self.criterionL1(self.z_transfer_video, self.warp_z_transfer)
        loss_G_temporal = loss_G_temporal * 0.5 * self.temporal_weight

        # makeup loss
        GPL = spl.GPLoss()
        CPL = spl.CPLoss(rgb=True, yuv=True, yuvgrad=True)
        loss_G_CP = CPL(self.z_transfer, self.transfer) + CPL(self.z_removal, self.removal)
        loss_G_GP = GPL(self.z_transfer, self.non_makeup) + GPL(self.z_removal, self.makeup)
        loss_G_SPL = loss_G_CP * self.CP_weight + loss_G_GP * self.GP_weight

        # local color
        loss_G_color_face = self.criterionColor(self.z_transfer, self.non_makeup_face_mask, self.makeup,
                                                self.makeup_face_mask)
        loss_G_color_eye = self.criterionColor(self.z_transfer, self.non_makeup_eye_mask, self.makeup,
                                               self.makeup_eye_mask)
        loss_G_color_lip = self.criterionColor(self.z_transfer, self.non_makeup_lip_mask, self.makeup,
                                               self.makeup_lip_mask)

        loss_G_color = (loss_G_color_face*0.1 + loss_G_color_eye + loss_G_color_lip) * 0.33
        loss_G_color = loss_G_color * self.color_weight

        loss_G = loss_G_GAN + loss_G_rec + loss_G_SPL + loss_G_color + loss_G_temporal

        loss_G.backward()

        self.loss_G_GAN = loss_G_GAN.item()
        self.loss_G_rec = loss_G_rec.item()
        self.loss_G_color = loss_G_color.item()
        self.loss_G_SPL = loss_G_SPL.item()
        self.loss_G_temporal = loss_G_temporal.item()

        self.G_loss = loss_G.item()

    def backward_G_GAN(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = nn.functional.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        return loss_G

    def set_scheduler(self, opts, last_ep=0):
        self.dis_non_makeup_sch = networks.get_scheduler(self.dis_non_makeup_opt, opts, last_ep)
        self.dis_makeup_sch = networks.get_scheduler(self.dis_makeup_opt, opts, last_ep)
        self.enc_content_sch = networks.get_scheduler(self.enc_content_opt, opts, last_ep)
        self.enc_makeup_sch = networks.get_scheduler(self.enc_makeup_opt, opts, last_ep)
        self.enc_semantic_sch = networks.get_scheduler(self.enc_semantic_opt, opts, last_ep)
        self.transformer_sch = networks.get_scheduler(self.transformer_opt, opts, last_ep)
        self.gen_transfer_sch = networks.get_scheduler(self.gen_transfer_opt, opts, last_ep)
        self.gen_removal_sch = networks.get_scheduler(self.gen_removal_opt, opts, last_ep)

    def update_lr(self):
        self.dis_non_makeup_sch.step()
        self.dis_makeup_sch.step()
        self.enc_content_sch.step()
        self.enc_makeup_sch.step()
        self.enc_semantic_sch.step()
        self.transformer_sch.step()
        self.gen_transfer_sch.step()
        self.gen_removal_sch.step()

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.enc_content.load_state_dict(checkpoint['enc_c'])
        self.enc_makeup.load_state_dict(checkpoint['enc_a'])
        self.enc_semantic.load_state_dict(checkpoint['enc_s'])
        self.transformer.load_state_dict(checkpoint['enc_trans'])
        self.gen_transfer.load_state_dict(checkpoint['gen_transfer'])
        self.gen_removal.load_state_dict(checkpoint['gen_removal'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'enc_c': self.enc_content.state_dict(),
            'enc_a': self.enc_makeup.state_dict(),
            'enc_s': self.enc_semantic.state_dict(),
            'enc_trans': self.transformer.state_dict(),
            'gen_transfer': self.gen_transfer.state_dict(),
            'gen_removal': self.gen_removal.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):

        non_makeup_down = self.normalize_image(F.interpolate(self.non_makeup, scale_factor=0.25, mode='nearest'))
        n, c, h, w = non_makeup_down.shape
        non_makeup_down_warp = torch.bmm(non_makeup_down.view(n, c, h * w), self.mapY)  # n*HW*1
        non_makeup_down_warp = non_makeup_down_warp.view(n, c, h, w)
        non_makeup_warp = F.interpolate(non_makeup_down_warp, scale_factor=4)

        makeup_down = self.normalize_image(F.interpolate(self.makeup, scale_factor=0.25, mode='nearest'))
        n, c, h, w = makeup_down.shape
        makeup_down_warp = torch.bmm(makeup_down.view(n, c, h * w), self.mapX)  # n*HW*1
        makeup_down_warp = makeup_down_warp.view(n, c, h, w)
        makeup_warp = F.interpolate(makeup_down_warp, scale_factor=4)

        warp_non_makeup_down = self.normalize_image(F.interpolate(self.warp_non_makeup, scale_factor=0.25, mode='nearest'))
        n, c, h, w = warp_non_makeup_down.shape
        warp_non_makeup_down_warp = torch.bmm(warp_non_makeup_down.view(n, c, h * w), self.mapY2)  # n*HW*1
        warp_non_makeup_down_warp = warp_non_makeup_down_warp.view(n, c, h, w)
        warp_non_makeup_warp = F.interpolate(warp_non_makeup_down_warp, scale_factor=4)


        images_non_makeup = self.normalize_image(self.non_makeup).detach()
        images_makeup = self.normalize_image(self.makeup).detach()
        images_warp_non_makeup=self.normalize_image(self.warp_non_makeup).detach()
        images_z_transfer = self.normalize_image(self.z_transfer).detach()
        images_z_removal = self.normalize_image(self.z_removal).detach()
        image_z_transfer_video=self.normalize_image(self.z_transfer_video).detach()
        images_transfer = self.normalize_image(self.transfer).detach()
        images_removal = self.normalize_image(self.removal).detach()
        images_warp_z_transfer=self.normalize_image(self.warp_z_transfer)
        images_rec_non_makeup = self.normalize_image(self.z_rec_non_makeup).detach()
        images_rec_makeup = self.normalize_image(self.z_rec_makeup).detach()
        images_warp_non_makeup_mask1 = self.normalize_image(self.warp_non_makeup_parse[:,0:3,::]).detach()
        images_face_non_makeup = self.normalize_image(self.non_makeup_face_mask).detach()
        images_face_makeup = self.normalize_image(self.makeup_face_mask).detach()
        images_warp_non_makeup_mask2 = self.normalize_image(self.warp_non_makeup_parse[:, 3:6, ::]).detach()

        row1 = torch.cat(
            (images_non_makeup[0:1, ::], images_z_transfer[0:1, ::], makeup_warp[0:1, ::],
             images_transfer[0:1, ::], images_rec_non_makeup[0:1, ::], images_face_non_makeup[0:1, ::]), 3)
        row2 = torch.cat(
            (images_makeup[0:1, ::], images_z_removal[0:1, ::], non_makeup_warp[0:1, ::],
             images_removal[0:1, ::], images_rec_makeup[0:1, ::], images_face_makeup[0:1, ::]), 3)
        row3 = torch.cat(
            (images_warp_non_makeup[0:1, ::], image_z_transfer_video[0:1, ::], warp_non_makeup_warp[0:1, ::],
             images_warp_z_transfer[0:1, ::], images_warp_non_makeup_mask1[0:1, ::], images_warp_non_makeup_mask2[0:1, ::]), 3)
        return torch.cat((row1, row2,row3), 2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def test_pair(self, data):

        self.non_makeup = data['non_makeup'].to(self.gpu).detach()
        self.makeup = data['makeup'].to(self.gpu).detach()
        self.non_makeup_parse = data['non_makeup_parse'].to(self.gpu).detach()
        self.makeup_parse = data['makeup_parse'].to(self.gpu).detach()
        self.non_makeup_parse_down4 = F.interpolate(self.non_makeup_parse, scale_factor=0.25, mode='nearest')
        self.makeup_parse_down4 = F.interpolate(self.makeup_parse, scale_factor=0.25, mode='nearest')

        self.non_makeup_all_mask = data['non_makeup_all_mask'].to(self.gpu).detach()
        self.non_makeup_face_mask = data['non_makeup_face_mask'].to(self.gpu).detach()
        self.non_makeup_brow_mask = data['non_makeup_brow_mask'].to(self.gpu).detach()
        self.non_makeup_eye_mask = data['non_makeup_eye_mask'].to(self.gpu).detach()
        self.non_makeup_lip_mask = data['non_makeup_lip_mask'].to(self.gpu).detach()

        self.makeup_all_mask = data['makeup_all_mask'].to(self.gpu).detach()
        self.makeup_face_mask = data['makeup_face_mask'].to(self.gpu).detach()
        self.makeup_brow_mask = data['makeup_brow_mask'].to(self.gpu).detach()
        self.makeup_eye_mask = data['makeup_eye_mask'].to(self.gpu).detach()
        self.makeup_lip_mask = data['makeup_lip_mask'].to(self.gpu).detach()

        with torch.no_grad():
            # first transfer and removal
            self.z_non_makeup_c = self.enc_content(self.non_makeup)
            self.z_non_makeup_s = self.enc_semantic(self.non_makeup_parse)
            self.z_non_makeup_a = self.enc_makeup(self.non_makeup)

            self.z_makeup_c = self.enc_content(self.makeup)
            self.z_makeup_s = self.enc_semantic(self.makeup_parse)
            self.z_makeup_a = self.enc_makeup(self.makeup)
            # warp makeup style
            self.mapX, self.mapY, self.z_non_makeup_a_warp, self.z_makeup_a_warp = self.transformer(self.z_non_makeup_c,
                                                                                                    self.z_makeup_c,
                                                                                                    self.z_non_makeup_s,
                                                                                                    self.z_makeup_s,
                                                                                                    self.z_non_makeup_a,
                                                                                                    self.z_makeup_a,
                                                                                                    self.non_makeup_parse_down4,
                                                                                                    self.makeup_parse_down4)
            # makeup transfer and removal
            self.z_transfer = self.gen_transfer(self.z_non_makeup_c, self.z_makeup_a_warp)
            self.z_removal = self.gen_removal(self.z_makeup_c, self.z_non_makeup_a_warp)

        makeup_down = self.normalize_image(F.interpolate(self.makeup*self.makeup_all_mask, scale_factor=0.25, mode='nearest'))
        n, c, h, w = makeup_down.shape
        makeup_down_warp = torch.bmm(makeup_down.view(n, c, h * w), self.mapX)  # n*HW*1
        makeup_down_warp = makeup_down_warp.view(n, c, h, w)
        makeup_warp = F.interpolate(makeup_down_warp, scale_factor=4)

        images_non_makeup = self.normalize_image(self.non_makeup).detach()
        images_makeup = self.normalize_image(self.makeup).detach()
        images_z_transfer = self.normalize_image(self.z_transfer).detach()
      
        row1 = torch.cat((images_non_makeup[0:1, ::],images_makeup[0:1, ::], makeup_warp[0:1, ::], images_z_transfer[0:1, ::]), 3)

        return row1
