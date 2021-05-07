import pickle 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

ROOT_DIR = 'plotout/'
GROUP_TESTING_DATASET_PATH = '/data/weixin/data/GroupTestingDataset'

def load_validate_dump(pkl_name, pkl_dir = './prediction_cache_0.1/', verbose=False, confidence_threshold=0.5):

    with open(pkl_dir + pkl_name, "rb") as pkl_file:
        evaluate_dict = pickle.load(pkl_file)
        target_all = evaluate_dict['target_all']
        pred_score_all = evaluate_dict['pred_score_all']
        
        if verbose: 
            print("Working On:", pkl_name )
            pred_label = (pred_score_all>confidence_threshold)
            print("confusion_matrix")
            print( confusion_matrix(target_all, pred_label))

    return pred_score_all, target_all


def main_analysis():
    
    ##################################
    # Individual Testing Baseline; K=0
    ##################################

    print("##################################")
    print("[ Individual Testing K0 Baseline ]")
    K0_score, K0_target = load_validate_dump(pkl_name="ResNeXt101FullK0.pkl", verbose=True)
        
    K0_recall = 2 * np.sum( np.logical_and(K0_target, K0_score>0.5) ) 
    K0_FPR = 100 * np.sum(   np.logical_and(K0_target==0, K0_score>0.5) ) / np.sum(K0_target==0) # False Positive Rate 
    print("Recall(%): {} FPR(%): {:3f}".format(K0_recall, K0_FPR))
    
    K0_tests = len(K0_target)
    # print("Number of Tests (1st Round): ", K0_tests)
    each_K0_GigaMACs = 16.5 # 16.5 GMacs per test 
    K0_MACs = each_K0_GigaMACs /1000 * K0_tests # TMacs 10^12
    print("Number of Tests (1st Round): ", K0_tests)
    print("Total Computation: {:.1f} TMACs".format(K0_MACs))

    del K0_score, K0_target, each_K0_GigaMACs

    ##################################
    # note that  each_K0_GigaMACs and K0_score would be re-used by downstream modules 
    ##################################


    ##################################
    # Algorithm 1 Wrapper Function
    ##################################
    def algorithm_1_wrapper(
        pkl_name:str, 
        exp_title:str, 
        each_method_GigaMACs:float, # GMacs per test. M images in total. 
        group_size:int, # M value in the paper 
        confidence_threshold:float=0.5,
        pkl_dir='./prediction_cache_0.1/', # default root dir 
        ):
        print("##################################")
        print(exp_title)
        method_score, method_target = load_validate_dump(pkl_dir=pkl_dir, pkl_name=pkl_name, verbose=True, confidence_threshold=confidence_threshold)
        
        method_tests_Round_1 = len(method_target)
        method_TeraMACs_Round_1 = each_method_GigaMACs / 1000 * method_tests_Round_1 # TMacs 10^12
        print("Number of Tests (1st Round): ", method_tests_Round_1, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_1))

        method_Round_1_next = np.repeat( (method_score>confidence_threshold), group_size) # times group size 

        print("Number Of Samples After the 1st round:", np.sum(method_Round_1_next))

        K0_score, K0_target = load_validate_dump(pkl_dir=pkl_dir, pkl_name="ResNeXt101FullK0.pkl", verbose=False)

        method_recall = 100 * np.sum( np.logical_and(
            np.logical_and(K0_target, K0_score>0.5), 
            method_Round_1_next) ) / np.sum(K0_target==1) # use K0 model as the second round 
        method_FPR = 100 * np.sum(   np.logical_and(
            np.logical_and(K0_target==0, K0_score>0.5), 
            method_Round_1_next) 
            ) / np.sum(K0_target==0) # False Positive Rate 
        print("Recall(%): {} FPR(%): {:3f}".format(method_recall, method_FPR))

        method_tests_Round_2 = np.sum(method_Round_1_next) 
        each_K0_GigaMACs = 16.5 # 16.5 GMacs per test, same as the baseline model 
        method_TeraMACs_Round_2 = each_K0_GigaMACs / 1000 * method_tests_Round_2 # TMacs 10^12
        print("Number of Tests (2nd Round): ", method_tests_Round_2, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_2))

        method_TeraMACs_total = method_TeraMACs_Round_1 + method_TeraMACs_Round_2
        method_tests_total = method_tests_Round_1 + method_tests_Round_2
        print("Total Computation: {:.1f} TeraMACs".format(method_TeraMACs_total), "Total Tests:", method_tests_total, "Relative Cost", method_TeraMACs_total/805.2)

        result_dict = {
            'method_score': method_score, # raw outputs 
            'method_target': method_target, # raw outputs 
            'method_recall': method_recall, # performance metrics 
            'method_FPR': method_FPR, # performance metrics 
            'method_tests_Round_1': method_tests_Round_1, # computation cost metrics
            'method_tests_Round_2': method_tests_Round_2, # computation cost metrics
            'method_TeraMACs_Round_1': method_TeraMACs_Round_1, # computation cost metrics
            'method_TeraMACs_Round_2': method_TeraMACs_Round_2, # computation cost metrics
            'method_TeraMACs_total': method_TeraMACs_total, # computation cost metrics
        }
        return result_dict


    ##################################
    # MixupK1 Baseline + only Algorithm 1 
    ##################################
    MixupK1_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK1Mixup.pkl', 
        exp_title='Design 1 Mixup K1 + Algorithm 1 Two-Round', 
        each_method_GigaMACs=16.5, 
        group_size=2)


    ##################################
    # MixupK3 Baseline + only Algorithm 1 
    ##################################
    
    MixupK3_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK3Mixup.pkl', 
        exp_title='Design 1 Mixup K3 + Algorithm 1 Two-Round', 
        each_method_GigaMACs=16.5, 
        group_size=4,
        )


    ##################################
    # Now We Explore Design 2. And group size 8, 16. And potentially Algorithm 2. 
    # Start with Design 2 + Algorithm 1 + Group Size  + LayerGroup 1/2 
    ##################################

    ##################################
    # Try G2 K=1
    ##################################
    K1G2_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2K1.pkl', 
        exp_title='K=1 + Design 2 (G2) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=20.16, 
        group_size=2,
        )

    ##################################
    # Try G2 K=3 
    ##################################

    K3G2_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2K3.pkl', 
        exp_title='K=3 + Design 2 (G2) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=27.46, 
        group_size=4,
        )

    ##################################
    # Try G2 K=7
    ##################################

    K7G2_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2.pkl', 
        exp_title='K=7 + Design 2 (G2) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=42.06, 
        group_size=8,
        )

    ##################################
    # Try G2 K=15 
    ##################################

    K15G2_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2K15.pkl', 
        exp_title='K=15 + Design 2 (G2) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=71.27, 
        group_size=16,
        )

    ##################################
    # Try G2 K=31 
    # Very Expensive. Since there are too many false positives
    # need to be with very good confidence threshold
    ##################################

    K31G2_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2K31.pkl', 
        exp_title='K=31 + Design 2 (G2) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=129.68, 
        group_size=32,
        )


    ##################################
    # Now we switch to Group 1 
    # Try G1 K=1, smaller per forward cost, but less accurate 
    ##################################
    K1G1_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK1_imgnet_G1.pkl', 
        exp_title='K=1 + Design 2 (G1) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=17.95, 
        group_size=2,
        )

    ##################################
    # Try G2 K=3 
    ##################################

    K3G1_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK3_imgnet_G1.pkl', 
        exp_title='K=3 + Design 2 (G1) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=20.85, 
        group_size=4,
        )


    ##################################
    # Try G2 K=7
    ##################################

    K7G1_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G1.pkl', 
        exp_title='K=7 + Design 2 (G1) + Algorithm 1 Two-Round', 
        each_method_GigaMACs=26.64, 
        group_size=8,
        )


    ##################################
    # Design 3, hierarchical design 
    ##################################

    TREE022_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_TREE024_G2_022.pkl', 
        exp_title='Design 3 (Tree022) + K=3 + Algorithm 1 Two-Round', 
        each_method_GigaMACs=23.05, 
        group_size=4,
        )

    TREE024_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_TREE024_G2.pkl', 
        exp_title='Design 3 (Tree024) + K=7 + Algorithm 1 Two-Round', 
        each_method_GigaMACs=33.25, 
        group_size=8,
        )


    TREE028_result_dict = algorithm_1_wrapper(
        pkl_name='ResNeXt101FullK7_TREE024_G2_028.pkl', 
        exp_title='Design 3 (Tree028) + K=15 + Algorithm 1 Two-Round', 
        each_method_GigaMACs=53.65, 
        group_size=16,
        )

    ##################################
    # Algorithm 2 Wrapper Function
    ##################################


    ##################################
    # Algorithm 2 Wrapper Function: Multi Round Testing 
    # Itermediate Using: ResNeXt101FullK7_imgnet_G2K1, ResNeXt101FullK7_imgnet_G2K3, K0_target
    ##################################
    def algorithm_2_wrapper(
        pkl_name:str, 
        exp_title:str, 
        each_method_GigaMACs:float, # GMacs per test. M images in total. 
        group_size:int, # M value in the paper 
        confidence_threshold:float=0.5,
        pkl_dir='./prediction_cache_0.1/', # default root dir 
        ):
        print("##################################")
        print(exp_title)
        method_score, method_target = load_validate_dump(pkl_name=pkl_name, pkl_dir=pkl_dir, verbose=True, confidence_threshold=confidence_threshold)
        
        method_tests_Round_1 = len(method_target)
        method_TeraMACs_Round_1 = each_method_GigaMACs / 1000 * method_tests_Round_1 # TMacs 10^12
        print("Number of Tests (1st Round): ", method_tests_Round_1, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_1))

        method_Round_1_next = np.repeat( (method_score>confidence_threshold), group_size) # times group size 

        print("Number Of Samples After the 1st round:", np.sum(method_Round_1_next))

        ##################################
        # Algorithm 2 comes in here 
        # Insert a Round-2
        ##################################
        if group_size == 8:
            # scheme 1: M=8, 4 K1G2 (ResNeXt101FullK7_imgnet_G2K1.pkl) + 2 base (vary with positives in K1G2)
            # candidates: ResNeXt101FullK7_imgnet_G2.pkl, ResNeXt101FullK7_TREE042_G2.pkl, ResNeXt101FullK7_TREE024_G2.pkl
            # 2nd-level: use K1G2
            each_2nd_GigaMACs = 20.16 - 7.3 # could minus the base feature extraciton 
            group_size_2nd = 2 
            method_2nd_level_score, _ = load_validate_dump(pkl_dir=pkl_dir, pkl_name='ResNeXt101FullK7_imgnet_G2K1.pkl', verbose=False, confidence_threshold=0.5)

        elif group_size == 16:
            # scheme 2: M=16, 4 K3G2 (ResNeXt101FullK7_imgnet_G2K3.pkl) + 4 base (vary with positives in K1G2)
            # candidates: ResNeXt101FullK7_imgnet_G2K15.pkl, ResNeXt101FullK7_TREE024_G2_028.pkl 
            # 2nd-level: use K3G2
            each_2nd_GigaMACs = 27.46 - 14.6 # could minus the base feature extraciton 
            group_size_2nd = 4 
            method_2nd_level_score, _ = load_validate_dump(pkl_dir=pkl_dir, pkl_name='ResNeXt101FullK7_imgnet_G2K3.pkl', verbose=False, confidence_threshold=0.5)

        else:
            raise NotImplementedError() 

        method_2nd_level_score_repeat = np.repeat(method_2nd_level_score, group_size_2nd)
        method_Round_2_next = np.logical_and(method_Round_1_next, method_2nd_level_score_repeat>0.5)
        method_tests_Round_2 = np.sum(method_Round_1_next) // group_size_2nd # div group size second level 
        method_TeraMACs_Round_2 = each_2nd_GigaMACs / 1000 * method_tests_Round_2 # TMacs 10^12
        print("Number of Tests (2nd Round): ", method_tests_Round_2, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_2))

        ##################################
        # Finish Round-2. Comes Round-3. 
        ##################################

        K0_score, K0_target = load_validate_dump(pkl_dir=pkl_dir, pkl_name="ResNeXt101FullK0.pkl", verbose=False)

        method_recall = 100 * np.sum( np.logical_and(
            np.logical_and(K0_target, K0_score>0.5), 
            method_Round_2_next) ) / np.sum(K0_target==1) # use K0 model as the second round 
        method_FPR = 100 * np.sum(   np.logical_and(
            np.logical_and(K0_target==0, K0_score>0.5), 
            method_Round_2_next) 
            ) / np.sum(K0_target==0) # False Positive Rate 
        print("Recall(%): {} FPR(%): {:3f}".format(method_recall, method_FPR))


        method_tests_Round_3 = np.sum(method_Round_2_next) 
        each_K0_GigaMACs = 16.5 # 16.5 GMacs per test, same as the baseline model 
        method_TeraMACs_Round_3 = each_K0_GigaMACs / 1000 * method_tests_Round_3 # TMacs 10^12
        print("Number of Tests (3rd Round): ", method_tests_Round_3, "\t Computation: {:.1f} TMACs".format(method_TeraMACs_Round_3))

        method_TeraMACs_total = method_TeraMACs_Round_1 + method_TeraMACs_Round_2 + method_TeraMACs_Round_3
        method_tests_total = method_tests_Round_1 + method_tests_Round_2 + method_tests_Round_3
        print("Total Computation: {:.1f} TeraMACs".format(method_TeraMACs_total), "Total Tests:", method_tests_total, "Relative Cost", method_TeraMACs_total/805.2)

        result_dict = {
            'method_score': method_score, # raw outputs 
            'method_target': method_target, # raw outputs 
            'method_recall': method_recall, # performance metrics 
            'method_FPR': method_FPR, # performance metrics 
            'method_tests_Round_1': method_tests_Round_1, # computation cost metrics
            'method_tests_Round_3': method_tests_Round_3, # computation cost metrics
            'method_TeraMACs_Round_1': method_TeraMACs_Round_1, # computation cost metrics
            'method_TeraMACs_Round_3': method_TeraMACs_Round_3, # computation cost metrics
            'method_TeraMACs_total': method_TeraMACs_total, # computation cost metrics
        }
        return result_dict


    

    K7G2_A2_result_dict = algorithm_2_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2.pkl', 
        exp_title='K=7 + Design 2 (G2) + Algorithm 2 Three-Round', 
        each_method_GigaMACs=42.06, 
        group_size=8,
        )


    TREE024_result_dict = algorithm_2_wrapper(
        pkl_name='ResNeXt101FullK7_TREE024_G2.pkl', 
        exp_title='Design 3 (Tree024) + K=7 + Algorithm 2 Three-Round', 
        each_method_GigaMACs=33.25, 
        group_size=8,
        # confidence_threshold=0.8
        )

    # Group Size 16 
    K15G2_result_dict = algorithm_2_wrapper(
        pkl_name='ResNeXt101FullK7_imgnet_G2K15.pkl', 
        exp_title='K=15 + Design 2 (G2) + Algorithm 2 Three-Round', 
        each_method_GigaMACs=71.27, 
        group_size=16,
        )


    TREE028_result_dict = algorithm_2_wrapper(
        pkl_name='ResNeXt101FullK7_TREE024_G2_028.pkl', 
        exp_title='Design 3 (Tree028) + K=7 + Algorithm 2 Three-Round', 
        each_method_GigaMACs=53.65, 
        group_size=16,
        )


    ##################################
    # End Adaptive Testing 
    # confidence_threshold
    ##################################
    print("##################################")


    ##################################
    # Now: Non-Adpative Algorithm 
    # Also Calculate Performance 
    ##################################
    # get two seed array: 


    ##################################
    # Algorithm 3 Wrapper Function
    ##################################

    def algorithm_3_wrapper(
        pkl_1_name:str, 
        pkl_2_name:str, 
        exp_title:str, 
        each_method_GigaMACs:float, # GMacs per test. M images in total. 
        each_feature_saving_GigaMACs:float, # GMacs per test. M images in total. feature caching savings 
        group_size:int, # M value in the paper 
        confidence_threshold:float=0.5):


        print("##################################")
        print(exp_title)

        base_1_score, base_1_target = load_validate_dump(pkl_name=pkl_1_name, verbose=True, confidence_threshold=confidence_threshold)
        base_2_score, base_2_target = load_validate_dump(pkl_name=pkl_2_name, verbose=True, confidence_threshold=confidence_threshold)
        
        import torch 
        indices42 = torch.randperm( len(base_1_target) * group_size, generator=torch.Generator().manual_seed(42)).numpy() 
        indices43 = torch.randperm( len(base_1_target) * group_size, generator=torch.Generator().manual_seed(43)).numpy() 

        indices43_argsort = np.argsort(indices43)

        assert np.all(indices43[indices43_argsort][indices42] == indices42)


        sample_label_arr_1 = np.repeat(base_1_score>confidence_threshold, group_size) # times group size 
        sample_label_arr_2 = np.repeat(base_2_score>confidence_threshold, group_size) # times group size 

        sample_label_arr_2 = sample_label_arr_2[indices43_argsort][indices42] # need to correct the array 

        prediction_arr = np.logical_and(sample_label_arr_1, sample_label_arr_2)


        K0_score, K0_target = load_validate_dump(pkl_name="ResNeXt101FullK0.pkl", verbose=False)

        ##################################
        # Only for sanity check 
        ##################################

        _, K0_target_seed43 = load_validate_dump(pkl_name="ResNeXt101FullK0_seed43.pkl", verbose=False)
        assert np.all( K0_target_seed43[indices43_argsort][indices42] == K0_target)


        A3_recall = 2 * np.sum( np.logical_and(K0_target, prediction_arr) ) 
        A3_FPR = 100 * np.sum(   np.logical_and(K0_target==0, prediction_arr) ) / np.sum(K0_target==0) # False Positive Rate 
        print("Recall(%): {} FPR(%): {:3f}".format(A3_recall, A3_FPR))
        
        A3_tests = len(base_1_target) * 2 # twice 
        A3_MACs = (each_method_GigaMACs*2 - each_feature_saving_GigaMACs) /1000 * (A3_tests//2) # TMacs 10^12
        
        print("Number of Tests (1st Round): ", A3_tests)
        print("Total Computation: {:.1f} TMACs".format(A3_MACs), "Relative Cost", A3_MACs/805.2)

        return 

    ##################################
    # Algorithm 3 (Non-Adaptive) + Design 3 Tree Hierarchical 
    ##################################

    algorithm_3_wrapper(
        pkl_1_name='ResNeXt101FullK7_TREE024_G2_022.pkl', 
        pkl_2_name='ResNeXt101FullK7_TREE024_G2_022_seed43.pkl', 
        exp_title='Algorithm 3 (Non-Adpative) + Design 3 (Tree022)', 
        each_method_GigaMACs=23.05, # GMacs per test. M images in total. 
        each_feature_saving_GigaMACs=5.79, # GMacs per test. M images in total. 
        group_size=4, # M value in the paper 
        confidence_threshold=0.50, 
        ) 

    algorithm_3_wrapper(
        pkl_1_name='ResNeXt101FullK7_imgnet_G2.pkl', 
        pkl_2_name='ResNeXt101FullK7_imgnet_G2_seed43.pkl', 
        exp_title='Algorithm 3 (Non-Adpative) + Design 3 (Tree024)', 
        each_method_GigaMACs=33.25, # GMacs per test. M images in total. 
        each_feature_saving_GigaMACs=11.59, # GMacs per test. M images in total. 
        group_size=8, # M value in the paper 
        confidence_threshold=0.500, 
        ) 

    ##################################
    # Now analyze different prevalence  
    # Need to overwrite 
    # Itermediate Using (M4, M2): ResNeXt101FullK7_imgnet_G2K1 (4 errors), ResNeXt101FullK7_imgnet_G2K3, 
    # M1: K0_target - single testing 
    ##################################

    prevalence_pkl_dir_map = {
        'Prevalence 0.05%': './prediction_cache_0.05/',
        # default is 0.1%, so we do not need to re-run. 
        'Prevalence 0.5%': './prediction_cache_0.5/',
        'Prevalence 1.0%': './prediction_cache_1.0/',
    }
    for prevalence_str, prevalence_pkl_dir in prevalence_pkl_dir_map.items():

        ##################################
        # Try G2 K=1
        ##################################
        K1G2_result_dict = algorithm_1_wrapper(
            pkl_name='ResNeXt101FullK7_imgnet_G2K1.pkl', 
            exp_title='[{}]'.format(prevalence_str) + 'K=1 + Design 2 (G2) + Algorithm 1 Two-Round', 
            each_method_GigaMACs=20.16, 
            group_size=2,
            pkl_dir = prevalence_pkl_dir,
            )

        ##################################
        # Try G2 K=3 
        ##################################

        K3G2_result_dict = algorithm_1_wrapper(
            pkl_name='ResNeXt101FullK7_imgnet_G2K3.pkl', 
            exp_title='[{}]'.format(prevalence_str) + 'K=3 + Design 2 (G2) + Algorithm 1 Two-Round', 
            each_method_GigaMACs=27.46, 
            group_size=4,
            pkl_dir = prevalence_pkl_dir,
            )

        ##################################
        # Try G2 K=7
        ##################################

        K7G2_result_dict = algorithm_1_wrapper(
            pkl_name='ResNeXt101FullK7_imgnet_G2.pkl', 
            exp_title='[{}]'.format(prevalence_str) + 'K=7 + Design 2 (G2) + Algorithm 1 Two-Round', 
            each_method_GigaMACs=42.06, 
            group_size=8,
            pkl_dir = prevalence_pkl_dir,
            )

        ##################################
        # Try G2 K=15 
        ##################################

        K15G2_result_dict = algorithm_1_wrapper(
            pkl_name='ResNeXt101FullK7_imgnet_G2K15.pkl', 
            exp_title='[{}]'.format(prevalence_str) + 'K=15 + Design 2 (G2) + Algorithm 1 Two-Round', 
            each_method_GigaMACs=71.27, 
            group_size=16,
            pkl_dir = prevalence_pkl_dir,
            )

        TREE022_result_dict = algorithm_1_wrapper(
            pkl_name='ResNeXt101FullK7_TREE024_G2_022.pkl', 
            exp_title='[{}]'.format(prevalence_str) + 'Design 3 (Tree022) + K=3 + Algorithm 1 Two-Round', 
            each_method_GigaMACs=23.05, 
            group_size=4,
            # confidence_threshold=0.8,
            pkl_dir = prevalence_pkl_dir,
            )

        TREE024_result_dict = algorithm_1_wrapper(
            pkl_name='ResNeXt101FullK7_TREE024_G2.pkl', 
            exp_title='[{}]'.format(prevalence_str) + 'Design 3 (Tree024) + K=7 + Algorithm 1 Two-Round', 
            each_method_GigaMACs=33.25, 
            group_size=8,
            # confidence_threshold=0.8,
            pkl_dir = prevalence_pkl_dir,
            )


        TREE028_result_dict = algorithm_1_wrapper(
            pkl_name='ResNeXt101FullK7_TREE024_G2_028.pkl', 
            exp_title='[{}]'.format(prevalence_str) + 'Design 3 (Tree028) + K=15 + Algorithm 1 Two-Round', 
            each_method_GigaMACs=53.65, 
            group_size=16,
            # confidence_threshold=0.8,
            pkl_dir = prevalence_pkl_dir,
            )


        ##################################
        # Algorithm 2 
        ##################################


        K7G2_A2_result_dict = algorithm_2_wrapper(
            pkl_name='ResNeXt101FullK7_imgnet_G2.pkl', 
            exp_title='[{}]'.format(prevalence_str) + 'K=7 + Design 2 (G2) + Algorithm 2 Three-Round', 
            each_method_GigaMACs=42.06, 
            group_size=8,
            pkl_dir = prevalence_pkl_dir,
            )

        # Group Size 16 
        K15G2_result_dict = algorithm_2_wrapper(
            pkl_name='ResNeXt101FullK7_imgnet_G2K15.pkl', 
            exp_title='[{}]'.format(prevalence_str) + ' K=15 + Design 2 (G2) + Algorithm 2 Three-Round', 
            each_method_GigaMACs=71.27, 
            group_size=16,
            pkl_dir = prevalence_pkl_dir,
            )

        TREE024_result_dict = algorithm_2_wrapper(
            pkl_name='ResNeXt101FullK7_TREE024_G2.pkl', # 5 errors not good now - 0.1947263 ; 0.98 
            exp_title='[{}]'.format(prevalence_str) + ' Design 3 (Tree024) + K=7 + Algorithm 2 Three-Round', 
            each_method_GigaMACs=33.25, 
            group_size=8,
            # confidence_threshold=0.43,
            pkl_dir = prevalence_pkl_dir,
            )




        TREE028_result_dict = algorithm_2_wrapper(
            pkl_name='ResNeXt101FullK7_TREE024_G2_028.pkl', 
            exp_title='[{}]'.format(prevalence_str) + ' Design 3 (Tree028) + K=7 + Algorithm 2 Three-Round', 
            each_method_GigaMACs=53.65, 
            group_size=16,
            # confidence_threshold=0.43
            pkl_dir = prevalence_pkl_dir,
            )


    return 

if __name__ == '__main__':


    ##################################
    # Individual Testing Baseline; K=0
    ##################################

    main_analysis()

    
