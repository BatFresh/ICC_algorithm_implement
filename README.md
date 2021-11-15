# ICC_algorithm_implement
Inference Resource Reservation for Joint Distributed Training and Inference with Edge Cooperation
# Denpendency
panda  
tqdm  
torch  
torchvision  
ptflops  
# File Description
algorithm.py ---> HEFT-lookahead,NSGA-II-looklahead,PKGO algorithm implementation  
Dnn_simulation.py-->Simulation execution 
Dataset.py---> Data generation in simulation  
Taskpre.py--->Inference task workflow definition in simulation  
graph_Alexnet.csv--->The workload and data size of DNN model,Alexnet  
graph_Vgg16.csv--->The workload and data size of DNN model,Vgg16  
graph_ResNet18.tcsv--->The workload and data size of DNN model,ResNet18  
DNN_model_workload&datasize_measurement.py -->The DNN model workload and data size measurement  

# Simulation
python Dnnn_simulation.py
