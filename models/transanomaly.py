"""
Sorce:
https://github.com/sonsalmon/TransAnomaly.git
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.ViViT.module import Attention, PreNorm, FeedForward

class TransAnomaly(nn.Module):
    def __init__(self, batch_size, num_frames, input_channels=3):
        super(TransAnomaly, self).__init__()
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.channels_1 = 64 #endcoder 기준 1번째 layer의 채널
        self.channels_2 = 128#endcoder 기준 2번째 layer의 채널
        self.channels_3 = 256#endcoder 기준 3번째 layer의 채널
        self.channels_4 = 512#endcoder 기준 4번째 layer의 채널 : encoder 최종 채널

        #input image의 shape = (b, t, c, h, w) = (b, 4, 3, 256, 256) or (b, 4, 1, 256, 256)
        self.contracting_11 = self.conv_block(in_channels=input_channels, out_channels=self.channels_1) #3 * 256 * 256 -> 64*256*256
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2) # (64,128,128)
        self.contracting_21 = self.conv_block(in_channels=self.channels_1, out_channels=self.channels_2) #(128, 128, 128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2) #(128,64,64)
        self.contracting_31 = self.conv_block(in_channels=self.channels_2, out_channels=self.channels_3) #(256,64,64)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2) #(256,32,32)
        self.contracting_41 = self.conv_block(in_channels=self.channels_3, out_channels=self.channels_4) #(512,32,32)

        #unet encoder에서 unet docoder로 잔차 연결
        #각 레이어에서 t개(num_frames)의 contracting output을 channel 차원에서 concate 한 뒤, convolution
        #따라서 in channels = (각 레이어의 채널) * (프레임 수)
        # residual_<encoder layer #><decoder larer #>
        self.residual_14 = nn.Conv2d(in_channels=self.channels_1*self.num_frames, out_channels=self.channels_1, kernel_size=3, stride=1, padding=1)
        self.residual_23 = nn.Conv2d(in_channels=self.channels_2*self.num_frames, out_channels=self.channels_2, kernel_size=3, stride=1, padding=1)
        self.residual_32 = nn.Conv2d(in_channels=self.channels_3*self.num_frames, out_channels=self.channels_3, kernel_size=3, stride=1, padding=1)
        self.residual_41 = nn.Conv2d(in_channels=self.channels_4*self.num_frames, out_channels=self.channels_4, kernel_size=3, stride=1, padding=1)

        #vivit layer
        # input : (b, 512, 32, 32)
        # output : (b, 512, 16, 16)
        self.middle = ViViT(image_size=32, patch_size=2, num_frames=self.num_frames, in_channels=512) #(b,t,c,h,w) = (b, 4, 512,32,32) -> (b,c,h,w) = (b,512,16,16)
        # input : (512,16,16)

        #decoder
        self.expansive_11 = nn.ConvTranspose2d(in_channels=self.channels_4, out_channels=self.channels_4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=self.channels_4*2, out_channels=self.channels_4) #(512,32,32)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=self.channels_4, out_channels=self.channels_3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=self.channels_3*2, out_channels=self.channels_3)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=self.channels_3, out_channels=self.channels_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=self.channels_2*2, out_channels=self.channels_2)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=self.channels_2, out_channels=self.channels_1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=self.channels_1*2, out_channels=self.channels_1)
        self.output = nn.Conv2d(in_channels=self.channels_1, out_channels=3, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block


    def forward(self, frames):
        #frames.shape = (batch, num_frames, c, h ,w) --reshape--> (batch * num_frames, c, h, w)
        ########## encoding #######
        # print('start contracting')

        #트릭 사용. conv2d를 쓰기 때문에 차원수 줄여야함 -> batch와 frames(t) 차원 합침. / 이미지 컨볼루전은 각 프레임별로 이루어지므로 가능.
        #이제 unet encoder에서 (b * t)를 배치로 취급해서 인코딩 진행.
        tmp_frames = rearrange(frames, 'b t c h w -> (b t) c h w')
        # print(tmp_frames.shape)
        contracting_11_out = self.contracting_11(tmp_frames) # [-1, 64, 256, 256]
        # print(frames.shape)
        # contracting_11_out = self.contracting_11(frames) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # pooling : [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        # print(contracting_41_out.shape)
        # print('end contracting')
        ############

        ####### vivit layer ########
        # print('start vivit')
        # print(self.batch_size)
        # 다시 t 차원 살려줌.
        vivit_input = rearrange(contracting_41_out, '(b t) c h w -> b t c h w', b=self.batch_size)
        # print('vivit input : ',vivit_input.shape)
        middle_out = self.middle(vivit_input) # [batch_size, num_frames , 512, 32, 32]
        # middle_out = vivit_input[:,0,:,:16,:16]
        # print('vivit output : ',middle_out.shape)
        # print('end vivit')
        #middle_out.shape should be (b, c, h, w) = (-1, 512, 16,16) (논문에 제시된 형태에 따르면)
        # num_frames 개의 프레임을 보고 다음 프레임 예측하는 것이므로 예측 feature map은 1개 frame 임.
        ############

        ######### residual connect #####
        #contracting_xx_out.shape = ((b*t), c, h ,w)
        #이걸 (b, t*c, h w)로 만든 다음 convolution 해야함. -> conv2d의 in_channels= t*c, out_channels = c
        #convolution 후 shape => (b, c, h ,w) 되도록 (즉, t차원 사라지게 됨.)
        #이걸 디코딩할 때 각 layer의 feature map에  concate함.
        #residual_XY -> X : endcoding layer 번호, Y : decoding layer 번호 (u자 형태이기 떄문에 1번 레이어와 4번 레이어가 연결됨.)
        # print('start risidual calculation')
        residual_14_out = rearrange(contracting_11_out, '(b t) c h w -> b (t c) h w', b=self.batch_size)
        residual_14_out = self.residual_14(residual_14_out) #shape = (b,c,h,w) : (64,256,256)
        residual_23_out = rearrange(contracting_21_out, '(b t) c h w -> b (t c) h w', b=self.batch_size)
        residual_23_out = self.residual_23(residual_23_out)#shape = (b,c,h,w) : (128,128,128)
        residual_32_out = rearrange(contracting_31_out, '(b t) c h w -> b (t c) h w', b=self.batch_size)
        residual_32_out = self.residual_32(residual_32_out)#shape = (b,c,h,w) : (256,64,64)
        residual_41_out = rearrange(contracting_41_out, '(b t) c h w -> b (t c) h w', b=self.batch_size)
        residual_41_out = self.residual_41(residual_41_out)#shape = (b,c,h,w) : (512,32,32)
        # print('end risidual')


        ####### decoding ##########
        #vivit의 output 받아옴.
        #vivit output인 middle_out.shape = (b, 512, 16, 16)
        # print('start expanding')
        expansive_11_out = self.expansive_11(middle_out) # output의 shape = [-1, 512, 32, 32] = (b, c, h, w)
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, residual_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, residual_32_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, residual_23_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, residual_14_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, 3, 256, 256]
        # print('end')
        ############
        # return output_out
        return torch.tanh(output_out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


#수정필요
#transformer에 pretrained vit 적용 필요할까?
##
# vivit에서 class token을 -> prediction token으로 사용. 각 토큰 그룹당 하니씩 predtiction token 추가
#따라서 Np(num_patches)개 만큼 token 추가됨.
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim = 512, depth = 4, heads = 3, pool = 'cls', in_channels = 512, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()

        self.image_size = image_size #input feature map 한 변 길이
        self.patch_size = patch_size #patch 한 변 길이
        self.num_frames = num_frames # sequence 길이
        self.in_channels = in_channels # input feature map channels
        self.dim = dim #hyper parameter : 각 토큰의 차원

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)' #필요 없는 코드인듯?


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (image_size // patch_size) ** 2 # feature map 나눈 패치 개수. 논문에서 Np에 해당.
        self.patch_dim = in_channels * patch_size ** 2 #각 patch의 차원 = c * p^2
        #논문에서 제시된 대로, 각 feature map x_i의 shape=(c,h,w)=(in_channels, image_size, image_size)라고 할 떄
        #이를 2D patch로 나누고 flatten 했을 때 x_(pi)의 shape=(N_p, (p^2 * c)) = (self.num_patches, self.patch_dim)

        #위에 언급한 작업을 patch embedding에서 수행
        self.to_patch_embedding = nn.Sequential(
            #Batch X Time X Channel X H X W 형태 이미지 -> Batch X Time X N X (patch_size*patch_size*Channel) (이때 N = H*W / P*P) 형태 벡터로 임베딩
            #1X16X3X(14*16)X(14*16) -> 1X16X(14*14)X(16*16*3)

            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size), #(b,t,c, N_p, p^2*c)
            nn.Linear(self.patch_dim, self.dim), #input 차원 = patch_dim, output 차원 = dim인 MLP (self.patch_dim * self.num_frames) / 파라미터 너무 많아서 줄여줘야함.
            Rearrange('b t n d -> b n (d t)') #논문 Figure 3. 참고. 여기서 n = Np임. 그림에서 작은 상자 하나가 한 프레임 나타냄. t=4일 때 4개 상자이어붙임. 이제 이 앞에 temporal_position token 붙일
        )

        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, (self.num_frames + 1)*self.dim))
        #temporal prediction token
        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.temporal_transformer = Transformer(dim * (self.num_frames + 1), depth, heads, dim_head, dim*scale_dim, dropout) # (b,num_patchs, dim * (t+1)) 차원 인풋

        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        # self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout) # (b,num_patches * dim) 차원 인풋 => temporal prediction token을 flatten해서 인풋

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool


    def forward(self, x):
        # print(x.shape)
        x = self.to_patch_embedding(x)
        # print(x.shape) #(b,n, 512*t)
        b, n, d = x.shape
        #temporal token 차원 맞춰주기
        pred_temporal_tokens = repeat(self.temporal_token, '() () d -> b n d', b=b, n=n)
        # print(pred_temporal_tokens.shape)
        x = torch.cat((pred_temporal_tokens, x), dim=2) #(b,n,512*(t+1)) (b,256,2560)
        x += self.temporal_pos_embedding[:,:,:(d+self.dim)]
        x = self.dropout(x)
        # print('temp transformer : ',x.shape)
        x = self.temporal_transformer(x)

        x = x[:,:,:self.dim] #x.shape = (b, n ,d)

        x += self.spatial_pos_embedding

        # print('space transformer : ', x.shape)
        x = self.space_transformer(x)
        # print('after transformer : ', x.shape)
        # x = self.to_reshaping(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.image_size//2,w=self.image_size//2)

        return x
