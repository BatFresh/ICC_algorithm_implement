from torchvision.models import vgg16
from torchvision.models import inception_v3
import torch
import torchvision.models as models
# import torch
from ptflops import get_model_complexity_info
 
model = models.alexnet()   #调用官方的模型，
# checkpoints = '自己模型的path'
# model = resnet50
model_name = 'alexnet'
flops, params = get_model_complexity_info(model, (3,244,244),as_strings=True,print_per_layer_stat=True)
print("%s |%s |%s" % (model_name,flops,params))