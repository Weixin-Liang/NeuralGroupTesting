# https://github.com/sovrasov/flops-counter.pytorch
import torch
from ptflops import get_model_complexity_info


def calculate_example_standard_nets():
    import torchvision.models as models
    with torch.cuda.device(0):
        # net = models.densenet161() # Computational complexity:       7.82 GMac, Number of parameters:           28.68 M 
        # net = models.densenet201() # 4.37 GMac 20.01 M

        ##################################
        # Baseline resnext single forward 
        ##################################
        # net = models.resnext101_32x8d() # Computational complexity:       16.51 GMac, Number of parameters:           88.79 M 

        net = models.resnext101_32x8d() # Computational complexity:       16.51 GMac, Number of parameters:           88.79 M 

        macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return 


def calculate_group_testing_nets():

    ##################################
    # baseline: 16.5 GMac 
    # parameters: 86.75 M (for all Gx)
    # import resnet_design2 as models 

    # Group 0; 8 images: 17.35 GMac / 8 = 2.16875

    # Group 1; 2 images: 17.95 GMac / 8 
    # Group 1; 4 images: 20.85 GMac / 8 
    # Group 1; 8 images: 26.64 GMac / 8 = 3.33
    # Group 1; 16 images: 38.23 GMac / 8 


    # Group 2; 2 images: 20.16 GMac / 2 = ; base extrator: 7.3 GMac 
    # Group 2; 4 images: 27.46 GMac / 4 = ; base extrator: 14.6 GMac 
    # Group 2; 8 images: 42.06 GMac / 8 = 5.2575
    # Group 2; 16 images: 71.27 GMac / 16 = 
    # Group 2; 32 images: 129.68 GMac / 32 = 

    ##################################

    ##################################
    # import resnet_design3 as models - TREE
    
    # TREE 022: 23.05 GMac / 8 = 4.15625 ; G2 extrator: 10.2 GMac; G1 extractor 5.79 GMac 
    # TREE 024: 33.25 GMac / 8 = 4.15625 ; G2 extrator: 20.4 GMac; G1 extractor 11.59 GMac
    # TREE 042: 28.85 GMac / 8 = 3.60625
    # TREE 222: 23.54 GMac / 8 = 2.9425 (too good to be true)

    # TREE 028 53.65 GMac
    # TREE 044 44.84 GMac

    ##################################


    # import resnet_design2 as models 
    import resnet_design3 as models 

    with torch.cuda.device(0):
        ##################################
        # Group Testing
        ##################################
        net = models.resnext101_32x8d() # Computational complexity:       16.51 GMac, Number of parameters:           88.79 M 
        macs, params = get_model_complexity_info(net, (8, 3, 224, 224), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    return 


if __name__ == '__main__':
    calculate_group_testing_nets()
    # calculate_example_standard_nets()

