from torchvision.models import vgg16
from torchvision.models import resnet18
from torchvision.models import alexnet
import torch
from ptflops import get_model_complexity_info
 
# import the definition of net
alexnet_model = alexnet()   
vgg16_model = vgg16() 
resnet18_model = resnet18()

#output the flops(workload) and params size(data size)
model_name = 'alexnet'
flops, params = get_model_complexity_info(model, (3,244,244),as_strings=True,print_per_layer_stat=True)
print("%s |%s |%s" % (model_name,flops,params))

model_name = 'vgg16'
flops, params = get_model_complexity_info(model, (3,244,244),as_strings=True,print_per_layer_stat=True)
print("%s |%s |%s" % (model_name,flops,params))

model_name = 'resnet18'
flops, params = get_model_complexity_info(model, (3,244,244),as_strings=True,print_per_layer_stat=True)
print("%s |%s |%s" % (model_name,flops,params))
