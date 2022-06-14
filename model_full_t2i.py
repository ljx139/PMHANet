import ipdb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt_model_ljx.modules.vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig,BertEmbeddings
from packaging import version
from vilt_model_ljx.modules import heads, objectives, vilt_utils


class VilT_Classification(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.module1 = ViLTransformerSS(config)
        self.module2 = classifier(config["hidden_size"])
        
    def forward(self, img,indexVecotrs):
        text_embeds, image_embeds= self.module1(img,indexVecotrs)
        feats_i, feats_t, output, out_t = self.module2(text_embeds, image_embeds)
        
        return feats_i, feats_t, output, out_t

class classifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
#         self.dense0 = nn.Linear(144, 160)#
        self.dense1 = nn.Linear(144, 1)#64 768 T ->64 768 1///145 16 40
        self.dense1_t = nn.Linear(15, 1)#64 768 T ->64 768 1///145 16 40
        self.dense2 = nn.Linear(hidden_size, 172)#64 1 768->64 1 172/81  #vireo172/nus-wide
        self.relu = nn.ReLU()        

    def forward(self, text_embeds, image_embeds):
#         ipdb.set_trace()
        
        image_embeds =image_embeds[:,1:].transpose(1,2)#1+15+1+144
#         image_embeds = self.dense0(image_embeds)#
        text_embeds =text_embeds[:,1:].transpose(1,2)#16 40
        
        feats_i = torch.squeeze(self.dense1(image_embeds))
        feats_t = torch.squeeze(self.dense1_t(text_embeds))
        feats_i = self.relu(feats_i)
        feats_t = self.relu(feats_t)
        output = self.dense2(feats_i)        
        out_t = self.dense2(feats_t)
        
        feats_i = nn.functional.normalize(feats_i, dim=-1)
        feats_t = nn.functional.normalize(feats_t, dim=-1)
        
        return feats_i, feats_t, output, out_t

#         return output

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])#建立一个长度2的字典
        self.token_type_embeddings.apply(objectives.init_weights)
       
        
        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )
#         ipdb.set_trace()
        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)
        
            
        
        # ===================== Downstream ===================== #
# load pretrain and downstream tesk head
# exp1:   image Classification
# 172 class for food
# exp2:  text classification
# 353 class for ingredients
        
    
# load pretrained vilt
#         if (
#             self.hparams.config["load_path"].split('.')[-1] == "ckpt"
#             and not self.hparams.config["test_only"]
#         ):
# #             ipdb.set_trace()
#             ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
#             state_dict = ckpt["state_dict"]
#             self.load_state_dict(state_dict, strict=False)
#             print("- - - - - - - -\n ckpt : {} has been loaded in model \n - - - - - - - -".format(self.hparams.config["load_path"].split('/')[-1]))

        hs = self.hparams.config["hidden_size"]


        vilt_utils.set_metrics(self)
        self.current_tasks = list()




    def infer(
        self,
        img,
        indexVectors,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

#文本
        cls_token = torch.zeros(indexVectors.size(0),1).cuda()
        indexVectors = torch.cat([cls_token, indexVectors],dim=1).long()
#nus-wide
        indexVectors = indexVectors[:,:40]
#vireo172
        if indexVectors.size(1)<16:#仅用于vireo172的测试数据
            pad = torch.zeros(indexVectors.size(0),16-indexVectors.size(1)).cuda()
            indexVectors = torch.cat([indexVectors,pad],dim = 1).long()
            
        text_embeds = self.text_embeddings(indexVectors)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(indexVectors))
        
#图像
        if image_embeds is None and image_masks is None:
            (
                image_embeds,#64 145 768
                image_masks,#64 145
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )
        image_embeds =  image_embeds  + self.token_type_embeddings( torch.full_like(image_masks, image_token_type_idx))#64 145 768 


    #         ----------------------------------------------original----------------------------------------------------------------


#         x = torch.cat( [text_embeds, image_embeds], dim=1)
# #         y = torch.cat( [text_embeds, image_embeds], dim=1)
# #         ipdb.set_trace()
#         for i, blk in enumerate(self.transformer.blocks):
#             if i<6:
#                 x, _attn = blk(x)
#         y = x[:,-image_embeds.size(1):]#仅图像信息
#         z = x[:,:text_embeds.size(1)]#仅文本信息
#         for i, blk in enumerate(self.transformer.blocks):
#             if i>=6:
#                 y, _attn = blk(y)
        
#         text_embeds = self.transformer.norm(z)               
#         image_embeds = self.transformer.norm(y)
        #         ipdb.set_trace()    
# 

        x = image_embeds
#         co_masks = image_masks
        
        for i, blk in enumerate(self.transformer.blocks):
#             ipdb.set_trace()
            x, _attn = blk(x)

        image_embeds = self.transformer.norm(x)
        
        return text_embeds, image_embeds

    def forward(self, img, indexVectors):
        
#         exp1：Classification using image features
        text_embeds, image_embeds = self.infer(img, indexVectors)
        return text_embeds, image_embeds
        



    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
