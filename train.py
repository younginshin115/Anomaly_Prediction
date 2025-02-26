import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from utils import *
from losses import *
import Dataset
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from models.liteFlownet import lite_flownet as lite_flow
from models.transanomaly import TransAnomaly
from config import update_config
from models.flownet2.models import FlowNet2SD
from evaluate import val
import vessl


parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--generator', default='unet', type=str, help='The name of the model that will be used as a generator')
parser.add_argument('--s_depth', default=4, type=int, help='The depth of the spatial transformer')
parser.add_argument('--t_depth', default=4, type=int, help='The depth of the temporal transformer')

parser.add_argument('--iters', default=40000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=1000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--show_flow', default=False, action='store_true',
                    help='If True, the first batch of ground truth optic flow could be visualized and saved.')
parser.add_argument('--flownet', default='lite', type=str, help='lite: LiteFlownet, 2sd: FlowNet2SD.')
parser.add_argument('--use_intensity_loss', default=1, type=int, help='Whether use the intensity loss or not, False = 0, True = 1')
parser.add_argument('--use_gradient_loss', default=1, type=int, help='Whether use the gradient loss or not, False = 0, True = 1')
parser.add_argument('--use_flow_loss', default=1, type=int, help='Whether use the flow loss or not, False = 0, True = 1')
parser.add_argument('--use_adversarial_loss', default=1, type=int, help='Whether use the adversarial loss or not, False = 0, True = 1')
parser.add_argument('--use_content_loss', default=1, type=int, help='Whether use the content loss or not, False = 0, True = 1')
parser.add_argument('--use_style_loss', default=1, type=int, help='Whether use the style loss or not, False = 0, True = 1')


args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()

if train_cfg.generator == 'transanormaly':
    generator = TransAnomaly(batch_size=train_cfg.batch_size, num_frames=4, s_depth=train_cfg.s_depth, t_depth=train_cfg.t_depth).cuda()
else:
    generator = UNet(input_channels=12, output_channel=3).cuda()
discriminator = PixelDiscriminator(input_nc=3).cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=train_cfg.d_lr)

if train_cfg.resume:
    generator.load_state_dict(torch.load(train_cfg.resume)['net_g'])
    discriminator.load_state_dict(torch.load(train_cfg.resume)['net_d'])
    optimizer_G.load_state_dict(torch.load(train_cfg.resume)['optimizer_g'])
    optimizer_D.load_state_dict(torch.load(train_cfg.resume)['optimizer_d'])
    print(f'Pre-trained generator and discriminator have been loaded.\n')
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator and discriminator are going to be trained from scratch.\n')

assert train_cfg.flownet in ('lite', '2sd'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
if train_cfg.flownet == '2sd':
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net = lite_flow.Network()
    flow_net.load_state_dict(torch.load('models/liteFlownet/network-default.pytorch'))

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()
content_loss = ContentLoss(torch.nn.MSELoss).cuda()
style_loss = StyleLoss(torch.nn.MSELoss).cuda()

train_dataset = Dataset.train_dataset(train_cfg)

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

vessl.init(tensorboard=True)
writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_bs{train_cfg.batch_size}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator = generator.train()
discriminator = discriminator.train()
best_auc = 0

try:
    step = start_iter
    while training:
        for indice, clips, flow_strs in train_dataloader:
            input_frames = clips[:, 0:12, :, :].cuda()  # (n, 12, 256, 256)
            target_frame = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)
            input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss

            # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
            for index in indice:
                train_dataset.all_seqs[index].pop()
                if len(train_dataset.all_seqs[index]) == 0:
                    train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                    random.shuffle(train_dataset.all_seqs[index])
            if train_cfg.generator == 'transanormaly':
                # print(f"Original input size: {input_frames.shape}")
                new_input_frames = input_frames.reshape(train_cfg.batch_size, 4, 3, 256, 256)
                G_frame = generator(new_input_frames)
            else:
                G_frame = generator(input_frames)

            if train_cfg.flownet == 'lite':
                gt_flow_input = torch.cat([input_last, target_frame], 1)
                pred_flow_input = torch.cat([input_last, G_frame], 1)
                # No need to train flow_net, use .detach() to cut off gradients.
                flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net).detach()
                flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net).detach()
            else:
                gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)
                pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)

                flow_gt = (flow_net(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
                flow_pred = (flow_net(pred_flow_input * 255.) / 255.).detach()

            if train_cfg.show_flow:
                flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
                for i in range(flow.shape[0]):
                    aa = flow_to_color(flow[i], convert_to_bgr=False)
                    path = train_cfg.train_data.split('/')[-3] + '_' + flow_strs[i]
                    cv2.imwrite(f'images/{path}.jpg', aa)  # e.g. images/avenue_4_574-575.jpg
                    print(f'Saved a sample optic flow image from gt frames: \'images/{path}.jpg\'.')

            inte_l = intensity_loss(G_frame, target_frame)
            grad_l = gradient_loss(G_frame, target_frame)
            fl_l = flow_loss(flow_pred, flow_gt)
            g_l = adversarial_loss(discriminator(G_frame))
            c_l = content_loss.get_loss(G_frame, target_frame)
            s_l = style_loss(G_frame, target_frame)
            G_l_t = 1. * inte_l * train_cfg.use_intensity_loss \
                + 1. * grad_l * train_cfg.use_gradient_loss \
                    + 2. * fl_l * train_cfg.use_flow_loss \
                        + 0.05 * g_l * train_cfg.use_adversarial_loss \
                            + 0.5 * c_l * train_cfg.use_content_loss \
                                + 1 * s_l * train_cfg.use_style_loss

            # When training discriminator, don't train generator, so use .detach() to cut off gradients.
            D_l = discriminate_loss(discriminator(target_frame), discriminator(G_frame.detach()))

            # https://github.com/pytorch/pytorch/issues/39141
            # torch.optim optimizer now do inplace detection for module parameters since PyTorch 1.5
            # If I do this way:
            # ----------------------------------------
            # optimizer_D.zero_grad()
            # D_l.backward()
            # optimizer_D.step()
            # optimizer_G.zero_grad()
            # G_l_t.backward()
            # optimizer_G.step()
            # ----------------------------------------
            # The optimizer_D.step() modifies the discriminator parameters inplace.
            # But these parameters are required to compute the generator gradient for the generator.

            # Thus I should make sure no parameters are modified before calling .step(), like this:
            # ----------------------------------------
            # optimizer_G.zero_grad()
            # G_l_t.backward()
            # optimizer_G.step()
            # optimizer_D.zero_grad()
            # D_l.backward()
            # optimizer_D.step()
            # ----------------------------------------

            # Or just do .step() after all the gradients have been computed, like the following way:
            optimizer_D.zero_grad()
            D_l.backward()
            optimizer_G.zero_grad()
            G_l_t.backward()
            optimizer_D.step()
            optimizer_G.step()

            torch.cuda.synchronize()
            time_end = time.time()
            if step > start_iter:  # This doesn't include the testing time during training.
                iter_t = time_end - temp
            temp = time_end

            if step != start_iter:
                if step % 20 == 0:
                    time_remain = (train_cfg.iters - step) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                    psnr = psnr_error(G_frame, target_frame)
                    lr_g = optimizer_G.param_groups[0]['lr']
                    lr_d = optimizer_D.param_groups[0]['lr']

                    print(f"[{step}]  inte_l: {inte_l:.3f} | grad_l: {grad_l:.3f} | fl_l: {fl_l:.3f} | c_l: {c_l:.3f} | s_l: {s_l:.3f} | "
                          f"g_l: {g_l:.3f} | G_l_total: {G_l_t:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | "
                          f"iter: {iter_t:.3f}s | ETA: {eta} | lr: {lr_g} {lr_d}")

                    save_G_frame = ((G_frame[0] + 1) / 2)
                    save_G_frame = save_G_frame.cpu().detach()[(2, 1, 0), ...]
                    save_target = ((target_frame[0] + 1) / 2)
                    save_target = save_target.cpu().detach()[(2, 1, 0), ...]

                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)
                    writer.add_scalar('total_loss/g_loss_total', G_l_t, global_step=step)
                    writer.add_scalar('total_loss/d_loss', D_l, global_step=step)
                    writer.add_scalar('G_loss_total/g_loss', g_l, global_step=step)
                    writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)
                    writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)
                    writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)
                    writer.add_scalar('G_loss_total/c_loss', c_l, global_step=step)
                    writer.add_scalar('G_loss_total/s_loss', s_l, global_step=step)
                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)

                if step % int(train_cfg.iters / 100) == 0:
                    writer.add_image('image/G_frame', save_G_frame, global_step=step)
                    writer.add_image('image/target', save_target, global_step=step)

                if step % train_cfg.save_interval == 0:
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                    torch.save(model_dict, f'weights/{train_cfg.dataset}_{step}.pth')
                    if not os.path.exists(f'/output/weights_{train_cfg.generator}'):
                        print(f" [*] Make directories : /output/weights_{train_cfg.generator}")
                        os.makedirs(f'/output/weights_{train_cfg.generator}')
                    torch.save(model_dict, f'/output/weights_{train_cfg.generator}/{train_cfg.dataset}_{step}.pth')
                    print(f'\nAlready saved: \'{train_cfg.dataset}_{step}.pth\'.')

                if step % train_cfg.val_interval == 0:
                    auc = val(train_cfg, model=generator)
                    writer.add_scalar('results/auc', auc, global_step=step)
                    # Check if the current AUC is the best so far
                    if auc > best_auc:
                        best_auc = auc
                        # Save the model with the best AUC
                        model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                    'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                        torch.save(model_dict, f'weights/best_{train_cfg.dataset}.pth')

                        if not os.path.exists(f'/output/weights_{train_cfg.generator}'):
                            print(f" [*] Make directories : /output/weights_{train_cfg.generator}")
                            os.makedirs(f'/output/weights_{train_cfg.generator}')
                        torch.save(model_dict, f'/output/weights_{train_cfg.generator}/best_{train_cfg.dataset}.pth')
                        print(f'\nBest model saved: \'best_{train_cfg.dataset}.pth\' ({step}).')
                    
                    generator.train()

            step += 1
            if step > train_cfg.iters:
                training = False
                writer.add_scalar('results/auc', best_auc, global_step=step) # best auc 추가될 지 확인 필요
                model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                              'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                if not os.path.exists(f'/output/weights_{train_cfg.generator}'):
                    print(f" [*] Make directories : /output/weights_{train_cfg.generator}")
                    os.makedirs(f'/output/weights_{train_cfg.generator}')
                torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
                torch.save(model_dict, f'/output/weights_{train_cfg.generator}/latest_{train_cfg.dataset}_{step}.pth')
                break

except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')

    if glob(f'weights/latest*'):
        os.remove(glob(f'weights/latest*')[0])

    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
    if not os.path.exists(f'/output/weights_{train_cfg.generator}'):
        print(f" [*] Make directories : /output/weights_{train_cfg.generator}")
        os.makedirs(f'/output/weights_{train_cfg.generator}')
    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
    torch.save(model_dict, f'/output/weights_{train_cfg.generator}/latest_{train_cfg.dataset}_{step}.pth')

