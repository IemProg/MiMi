import numpy as np
import os, sys
import pandas as pd

def best_scores_expr(path_roots, model, datasets, metric = 'acc1', round_precision = 2):
    data = {}
    new_columns = {}
    for path_root in path_roots:
        scores_model = []
        for i, dset in enumerate(datasets):
            new_columns[i] = dset
            dset_path = os.path.join(path_root, str(dset), model, "default", "logs.pkl")
            info_dset= pd.read_pickle(dset_path)
            best_score = np.round(np.max(info_dset[metric]), round_precision)
            scores_model.append(best_score)
        data[path_root] = scores_model
    df = pd.DataFrame(data)
    df = df.rename(columns=new_columns)
    return df.T

def plot_metrics(path_root, model_type, datasets, metric="loss", title_text="Training Loss", xtitle="Steps",
                 ytitle="Loss"):
    # Create traces
    fig = go.Figure()
    for dset in datasets:
        loss_dset_path = os.path.join(path_root, str(dset), model_type, "default", "logs.pkl")
        info_dset = pd.read_pickle(loss_dset_path)

        X_input = np.arange(len(info_dset[metric]))
        try:
            output = np.array(info_dset[metric])
        except:
            info_dset[metric] = [item.cpu().detach().numpy() for item in info_dset[metric]]
            output = np.array(info_dset[metric])

        fig.add_trace(go.Scatter(x=X_input, y=output,
                                 mode='lines', name=str(dset)))

    fig.update_layout(
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        title={
            'text': title_text,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    fig.show()
    fig.write_image("fig1_fewShotMag.pdf")

def acc_params(path_roots, model, dataset, round_precision=2):
    data = {"Accuracy":[], "Parameters":[]}
    for path_root in path_roots:
        dset_path = os.path.join(path_root, str(dataset), model, "default", "logs.pkl")
        info_dset = pd.read_pickle(dset_path)
        data["Accuracy"].append(np.round(np.max(info_dset["acc1"]), round_precision))
        data["Parameters"].append(info_dset["params"])
    df = pd.DataFrame(data=data, index=path_roots)
    return df


# datasets = ['clipart']
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# output_folder = "BaselinesFUll_Series_FewShotOurScore_Weighted_32_05_SEED0_PATIENCE20"
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
# scores = []
#
# for dset in datasets:
#     loss_dset_path = os.path.join(current_dir, "output", output_folder, str(dset), models, "default", "logs.pkl")
#     info_dset = pd.read_pickle(loss_dset_path)
#     scores.append(max(info_dset['acc1_test']))
# print(f"\t\t Scores: {scores}")
# #

# datasets = ['cifar100']
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# output_folder = "Ablation_Series_FewShotMagnitudePrune32_05_SEED0"
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
# scores = []
#
# for dset in datasets:
#     loss_dset_path = os.path.join(current_dir, "output", output_folder, str(dset), models, "default", "logs.pkl")
#     info_dset = pd.read_pickle(loss_dset_path)
#     scores.append(max(info_dset['acc1_test']))
# print(f"\t\t Scores: {scores}")

# datasets = 'cifar100'
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# output_folder = "Ablation_Series_FewShotMagnitudePrune32_05_SEED0"
#
# for path in output_folder:
#     path_roots.append(os.path.join(current_dir, "output", path))
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
#
# loss_dset_path = os.path.join(current_dir, "output", output_folder, str(datasets), models, "default", "logs.pkl")
# info_dset = pd.read_pickle(loss_dset_path)
#
# print(len(info_dset["acc1_test"]))
# print(info_dset["pruning"])

# datasets = 'aircraft'
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# # output_folder = ["Series_32_Magn_01_SEED0", "Series_32_Magn_02_SEED0", "Series_32_Magn_03_SEED0",
# #                   "Series_32_Magn_04_SEED0", "Series_32_Magn_05_SEED0", "Series_32_Magn_06_SEED0",
# #                    "Series_32_Magn_07_SEED0", "Series_32_Magn_08_SEED0", "Series_32_Magn_09_SEED0"]
#
# output_folder = ["Series_32_StructLAMP_01_SEED0", "Series_32_StructLAMP_02_SEED0", "Series_32_StructLAMP_03_SEED0",
#                   "Series_32_StructLAMP_04_SEED0", "Series_32_StructLAMP_05_SEED0", "Series_32_StructLAMP_06_SEED0",
#                    "Series_32_StructLAMP_07_SEED0", "Series_32_StructLAMP_08_SEED0", "Series_32_StructLAMP_09_SEED0"]
#
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
# scores = []
# for folder in output_folder:
#     loss_dset_path = os.path.join(current_dir, "output", "Series_32_MagnitudeVSstructLAMP_SEED0", folder,
#                                   datasets, models, "default", "logs.pkl")
#     info_dset = pd.read_pickle(loss_dset_path)
#     scores.append(max(info_dset['acc1_test']))
#
# print(f"\t\t Scores: {scores}")


# datasets = ['clipart', 'infograph', "painting", 'sketch'] #'quickdraw', 'real',
# datasets = 'cifar100'
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# output_folder = "Ablation_Series_UnWeightedGlobalMagnitudePrune128_05_SEED0"
#
# for path in output_folder:
#     path_roots.append(os.path.join(current_dir, "output", path))
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
#
# loss_dset_path = os.path.join(current_dir, "output", output_folder, str(datasets), models, "default",
#                               "logs.pkl")
# info_dset = pd.read_pickle(loss_dset_path)
#
# #print(max(info_dset["acc1_test"][:100]))
# print("length: ", len(info_dset["acc1_test"]))
# print(max(info_dset["acc1_test"][100:200]))
# print(max(info_dset["acc1_test"][200:300]))
# print(max(info_dset["acc1_test"][300:400]))
# print(max(info_dset["acc1_test"][400:]))


# datasets = ['cifar100']
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "VTs-Drloc-master"
#
# output_folder = ["FineTuneDomainnet_ParallelAdapters_32_SEED49",
#                  "FineTuneDomainnet_ParallelAdapters_32_SEED42",
#                  "FineTuneDomainnet_SeriesAdapters_32_SEED49",
#                  "FineTuneDomainnet_SeriesAdapters_32_SEED49"
#                  ]
#
# path_roots = [os.path.join(current_dir, "output", path) for path in output_folder]
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
# df = best_scores_expr(path_roots, models, datasets, metric="acc1_test")
# print(list(df[0].values))




datasets = ['clipart', 'infograph', "painting",'quickdraw', 'real', 'sketch']
#datasets = ['flowers102']
datasets = ['cifar(num_classes=100)', 'caltech101','dtd', 'oxford_iiit_pet', 'svhn', ""] #'cifar-10', ,
# DATA2CLS = {
# 'cifar(num_classes=100)': 100,
#             'caltech101': 102,
#
#             'dtd': 47,
#             #'oxford_flowers102': 102,
#             'oxford_iiit_pet': 37,
#             #'sun397': 397,
#             'svhn': 10,
# 'patch_camelyon': 2,
#     'eurosat': 10,
#             #'resisc45': 45,
#     'diabetic_retinopathy(config="btgraham-300")': 5,
#     'clevr(task="count_all")': 8,
#     'clevr(task="closest_object_distance")': 6,
#             'dmlab': 6,
#             #'kitti(task="closest_vehicle_distance")': 4,
#     'dsprites(predicted_attribute="label_x_position",num_classes=16)': 16,
#     'dsprites(predicted_attribute="label_orientation",num_classes=16)': 16,
#             'smallnorb(predicted_attribute="label_azimuth")': 18,
#             'smallnorb(predicted_attribute="label_elevation")': 9,
#         }
# datasets = list(DATA2CLS.keys())
#datasets = ['sketch']
#datasets = ["cifar-100", 'svhn']
    # path_roots = ["Ablation_Series_MagnitudePrune128_05_SEED0_FS", "Ablation_Series_MagnitudePrune256_05_SEED0_FS",
#               "Ablation_Series_MagnitudePrune256_05_SEED0_FS",
#               "Ablation_Series_GlobalMagnitudePrune128_05_SEED0_FS", "Ablation_Series_GlobalMagnitudePrune256_05_SEED0_FS",
#               "Ablation_Series_GlobalMagnitudePrune512_05_SEED0_FS"]

# path_roots = ["SNIP_Ablate05_Starting32_SEED0", "SNIP_Ablate05_Starting64_SEED0", "SNIP_Ablate05_Starting128_SEED0",
#               "SNIP_Ablate05_Starting256_SEED0"]

path_roots = "SwinSSF_Finetune_SEED0"

current_dir = os.getcwd()
experiment_folder = "ghost_adapters"
#output_folder = "Ablation_SeriesAdapters_32_SEED0"
models = "swin_ssf_swin_tiny_patch4_window7_224_0.2"
#models = "vit_adapters_vit_base_16_224_0.1"
#models = 'swin_adapters_swin_small_patch4_window7_224_0.3'
#models = "cvt_adapters_cvt_13_224_0.1"

myscores = []
#"flowers102", "svhn", "cifar-10"
dsets = ["cifar-100", "cifar-10", "flowers102", "svhn"]
# dsets = ["flowers102"]
dsets = ['clipart', 'infograph', "painting", 'quickdraw', 'real'] #,, 'sketch'

for dset in dsets:
    loss_dset_path = os.path.join(current_dir, "output", path_roots, dset, models, "default", "logs.pkl")
    info_dset = pd.read_pickle(loss_dset_path)

    #print(max(info_dset["acc1_test"][:100]))
    print("dset: ", dset)
    print("length: ", len(info_dset["acc1_test"])) #acc1_train, acc1_test
    # print(max(info_dset["acc1_test"][-300:-200]))
    # print(max(info_dset["acc1_test"][-200:-100]))
    # print(max(info_dset["acc1_test"][-100:]))
    print(max(info_dset["acc1_test"][0:80]))
    # print(max(info_dset["acc1_test"][80:180]))
    # print(max(info_dset["acc1_test"][180:300]))
    # print(max(info_dset["acc1_test"][300:400]))
    # print(max(info_dset["acc1_test"][400:500]))
    # print(max(info_dset["acc1_test"][500:600]))
    # print("600:700:", max(info_dset["acc1_test"][600:700]))
    # print("700:800:", max(info_dset["acc1_test"][700:800]))
    # print(max(info_dset["acc1_test"][800:900]))
    # print(max(info_dset["acc1_test"][900:1000]))
    # print(max(info_dset["acc1_test"][1000:1100]))
    # print(max(info_dset["acc1_test"][1100:1200]))
    # print(max(info_dset["acc1_test"][-100:]))
    # best = round(max(info_dset["acc1_test"][400:500]), 2)
    # myscores.append(best)
    print("--------------------------------------")

print("my scores: ", myscores)

# datasets = ['aircraft']
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# output_folder = ["Ablation_Series_MagnitudePrune128_05_SEED0"
#                  #"Ablation_Series_GlobalMagnitudePrune128_05_SEED0",
#                  #"Ablation_Series_UnWeightedGlobalMagnitudePrune64_05_SEED0",
#                  #"Ablation_Series_UnWeightedGlobalMagnitudePrune128_05_SEED0",
#                  #"Ablation_Series_UnWeightedGlobalMagnitudePrune256_05_SEED0"
#                  ]
#
# path_roots = [os.path.join(current_dir, "output", path) for path in output_folder]
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
# df = best_scores_expr(path_roots, models, datasets, metric="acc1_test")
# print(list(df[0].values))

# datasets = 'aircraft'
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# output_folder = "Ablation_Series_FewShotMagnitudePrune32_05_SEED0"
#
# for path in output_folder:
#     path_roots.append(os.path.join(current_dir, "output", path))
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
# loss_dset_path = os.path.join(current_dir, "output", output_folder, str(datasets), models, "default", "logs.pkl")
# info_dset = pd.read_pickle(loss_dset_path)
# print(info_dset["acc1_test"])
#
# print(len(info_dset["acc1_test"]))
# myscores = {"32": max(info_dset["acc1_test"][100:200]), "64": max(info_dset["acc1_test"][200:300]),
#             "128": max(info_dset["acc1_test"][300:400]), "256": max(info_dset["acc1_test"][400:])}
# print(myscores.values())

# datasets = 'aircraft'
# path_roots = []
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# output_folder = ["FineTuneDomainnet_ParallelAdapters_Attn_32_CorrectLAMP200_SEED0",
#                 "FineTuneDomainnet_ParallelAdapters_Attn_32_CorrectLAMP200_SEED0"]
#
# for path in output_folder:
#     path_roots.append(os.path.join(current_dir, "output", path))
# models = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
#
# loss_dset_path = os.path.join("output", output_folder, str(datasets), models, "default", "logs.pkl")
# info_dset = pd.read_pickle(loss_dset_path)
#
# myscores = {"32": max(info_dset["acc1"][1:102]), "64": max(info_dset["acc1"][102:202]),
#             "128": max(info_dset["acc1"][202:302]),
#             "256": max(info_dset["acc1"][302:])}
# print(myscores.values())

# datasets = ['clipart', 'infograph', "painting", 'sketch']  #'quickdraw', 'real',
# #datasets = ['cifar100']
#
# current_dir = os.getcwd()
# experiment_folder = "Swin_Pruning"
# output_folder = "Baselines_Series_FewShotGlobalMagnitude_128_05_SEED0"
# path_root = os.path.join(current_dir, experiment_folder, "output", output_folder)
# model_type = "swin_adapters_swin_tiny_patch4_window7_224_0.2"
#
# path_root = os.path.join(current_dir, "output", output_folder)
#
# plot_metrics(path_root, model_type, datasets, metric="acc1_test", title_text="Validation Accuracy",
#                             xtitle="Epochs", ytitle="Accuracy")