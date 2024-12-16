import torch
import os
from dataset_makeup import MakeupDataset
from model import MakeupGAN
from options import MakeupOptions
from saver import Saver
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def Makeup_train():
    # parse options
    parser = MakeupOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset = MakeupDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.nThreads)
    # model
    print('\n--- load model ---')
    model = MakeupGAN(opts)
    # model.setgpu(opts.gpu)
    if opts.resume is None:
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # saver for display and output
    saver = Saver(opts)
    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):

        for it, data in enumerate(train_loader):
            model.load_data(data)

            # update model
            model.update_D()
            model.update_EG()
            # save to display file
            saver.write_display(total_it, model)

            print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_transfer_opt.param_groups[0]['lr']))
            print('G_loss: %.4f, loss_G_GAN: %.4f, loss_G_rec: %.4f, loss_G_color: %.4f, loss_G_SPL: %.4f, loss_G_temporal: %.4f ' % (
                model.G_loss, model.loss_G_GAN, model.loss_G_rec, model.loss_G_color, model.loss_G_SPL,model.loss_G_temporal))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, model)
                break

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        # save result image
        saver.write_img(ep, model)

        # Save network weights
        saver.write_model(ep, total_it, model)
    return


if __name__ == '__main__':
    Makeup_train()
    print('The training is complete')
