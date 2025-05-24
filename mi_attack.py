import os
import numpy as np
import glob
import argparse
import pdb, traceback
import torch.nn.parallel
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import torch
from torch import nn, optim
# from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
from scipy.special import softmax
from src.utilities.models import load_model_defination
from src.utilities.datasets import DatasetWrapper, IncrementalDatasetWraper
from src.utilities.training_utils import load_model

torch_parallel = True
# torch_parallel = False

#model specific arguements
parser = argparse.ArgumentParser(description='Obtaining outputs from saved models.')
parser.add_argument('-m', '--model_name', type=str, default = 'efficientnet', help='Model name')
#parser.add_argument('--models_path', type=str, default='saved_models/', help='Path to saved models')
parser.add_argument('-em', '--evaluation_model', type=str, default= None, help='if provided, evaluate on this saved model')
parser.add_argument('-d', '--dataset', default='CIFAR10', type=str)
parser.add_argument('--output-type', type=str, default='confidence',
                    help='Ensembling based on averaging confidence or logit (default: confidence)')
#parser.add_argument('--outputs_path', type=str, default='outputs/resnet20/', help='Path to saved outputs')
# parser.add_argument('--attack-type', type=str, default='aggregated',
#                     help='Ensembling based on averaging aggregated or all (whitebox) (default: aggregated)')

#Options based on code of the paper of Rezaei et al "Towards the Difficulty of Membership Inference Attacks"
# sampling = "None"
# sampling = "oversampling"
sampling = "undersampling"
balanceness_ratio = 5
what_portion_of_sampels_attacker_knows = 0.8

args = parser.parse_args()

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def false_alarm_rate(y_true, y_pred):
    TP, TN, FP, FN = classification_scores(y_true, y_pred)
    if FP + TN == 0:
        return -1
    else:
        return FP / (FP + TN)

def classification_scores(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    if CM.shape[0] <= 1:
        return (0, 0, 0, 0)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return (TP, TN, FP, FN)

def main():
    #output_path = "outputs/" + args.models_path.split('/')[-2] + "/"
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    number_of_models = 1 #len(glob.glob(args.models_path + "*.pt"))

    try:
        target_dataset = DatasetWrapper(args.dataset)

    except NotImplementedError as e:
        dataset_name, index = args.dataset.split('-')
        target_dataset = IncrementalDatasetWraper(dataset_name)
        target_dataset.select_split(int(index))

    trainset, testset, num_channels, num_classes = target_dataset.trainset,target_dataset.testset,target_dataset.num_channels, target_dataset.num_classes

    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=False)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False)

    train_output_all = np.zeros((len(trainset), num_classes * number_of_models))
    test_output_all = np.zeros((len(testset), num_classes * number_of_models))

    softmax_operation = torch.nn.Softmax(dim=1)

    model_counter = 0
    #for model_path in glob.glob(args.models_path + "*.pt"):
    #print("Processing " + model_path)



    model = load_model_defination(args.model_name, num_channels, num_classes)

    load_model(model, filename =args.evaluation_model)


    #state_dict_parallel = torch.load(model_path)['state_dict']
    #checkpoint = torch.load(model_path)
    # if torch_parallel:
    #     try:
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #     except  KeyError:
    #         model.load_state_dict(checkpoint['model']) 
    #     # model = torch.nn.DataParallel(model).cuda()
    #     # model.load_state_dict(state_dict_parallel)
    # else:
    #     new_state_dict = OrderedDict()
    #     for k, v in torch.load(model_path)['model_state_dict'].items():
    #         if 'module' in k:
    #             k = k.replace('module.', '')
    #         new_state_dict[k] = v
    #     model.load_state_dict(new_state_dict)
    model.eval()

    temp_prediction = []
    temp_target = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        output = model(inputs)

        if args.output_type == "confidence":
            predictions = softmax_operation(output)
        elif args.output_type == "logit":
            predictions = output

        if torch_parallel:
            predictions = predictions.cpu().detach().numpy()
        else:
            predictions = predictions.detach().numpy()
        temp_prediction.extend(predictions)
        if model_counter == 0:
            temp_target.extend(targets)
    train_output_all[:, model_counter * num_classes:(model_counter + 1) * num_classes] = np.array(temp_prediction)
    # Store the ground truth when processing the first model
    if model_counter == 0:
        labels_train = np.array(temp_target)

    temp_prediction = []
    temp_target = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        output = model(inputs)

        if args.output_type == "confidence":
            predictions = softmax_operation(output)
        elif args.output_type == "logit":
            predictions = output

        if torch_parallel:
            predictions = predictions.cpu().detach().numpy()
        else:
            predictions = predictions.detach().numpy()
        temp_prediction.extend(predictions)
        if model_counter == 0:
            temp_target.extend(targets)
    test_output_all[:, model_counter * num_classes:(model_counter + 1) * num_classes] = np.array(temp_prediction)
    # Store the ground truth when processing the first model
    if model_counter == 0:
        labels_test = np.array(temp_target)

    #model_counter += 1



    if len(labels_train.shape) > 1:
        labels_train = labels_train.reshape((-1))
    if len(labels_test.shape) > 1:
        labels_test = labels_test.reshape((-1))
        

    
    train_conf_all = train_output_all #np.load(args.outputs_path + "/" + args.output_type + "_train.npy")
    test_conf_all = test_output_all #np.load(args.outputs_path + "/" + args.output_type + "_test.npy")

    number_of_models = int(train_conf_all.shape[-1] / num_classes)

    train_conf_sum = np.zeros((labels_train.shape[0], num_classes)) # type: ignore
    test_conf_sum = np.zeros((labels_test.shape[0], num_classes)) # type: ignore

    train_prediction_class_sum = np.zeros((labels_train.shape[0], num_classes)) # type: ignore
    test_prediction_class_sum = np.zeros((labels_test.shape[0], num_classes)) # type: ignore

    for model_index_counter in range(number_of_models):
        train_conf_sum += train_conf_all[:, model_index_counter * num_classes:(model_index_counter + 1) * num_classes]
        test_conf_sum += test_conf_all[:, model_index_counter * num_classes:(model_index_counter + 1) * num_classes]

        temp1 = np.argmax(train_conf_all[:, model_index_counter * num_classes:(model_index_counter + 1) * num_classes], axis=1)
        temp2 = np.argmax(test_conf_all[:, model_index_counter * num_classes:(model_index_counter + 1) * num_classes], axis=1)
        train_prediction_class_sum += to_categorical(temp1, num_classes)
        test_prediction_class_sum += to_categorical(temp2, num_classes)

        if args.output_type == "confidence":
            confidence_train_for_prediction = train_conf_sum / (model_index_counter + 1)
            confidence_test_for_prediction = test_conf_sum / (model_index_counter + 1)
        elif args.output_type == "logit":
            confidence_train_for_prediction = softmax(train_conf_sum / (model_index_counter + 1), axis=1)
            confidence_test_for_prediction = softmax(test_conf_sum / (model_index_counter + 1), axis=1)
        else:
            print("Output type does not exist!")
            exit()

        # if args.attack_type == "all":
        #     confidence_train_for_attack = train_conf_all[:, 0:(model_index_counter + 1) * num_classes]
        #     confidence_test_for_attack = test_conf_all[:, 0:(model_index_counter + 1) * num_classes]
        # elif args.attack_type == "aggregated":
        confidence_train_for_attack = confidence_train_for_prediction
        confidence_test_for_attack = confidence_test_for_prediction
        # else:
        #     print("Attack type is not valid!")
        #     exit()

        labels_train_by_model = np.argmax(confidence_train_for_prediction, axis=1)
        labels_test_by_model = np.argmax(confidence_test_for_prediction, axis=1)

        acc_train = np.sum(labels_train == labels_train_by_model)/labels_train.shape[0] # type: ignore
        acc_test = np.sum(labels_test == labels_test_by_model)/labels_test.shape[0] # type: ignore

        correctly_classified_indexes_train = labels_train_by_model == labels_train
        incorrectly_classified_indexes_train = labels_train_by_model != labels_train

        correctly_classified_indexes_test = labels_test_by_model == labels_test
        incorrectly_classified_indexes_test = labels_test_by_model != labels_test

        MI_x_train_all = []
        MI_y_train_all = []
        MI_x_test_all = []
        MI_y_test_all = []
        MI_cor_labeled_indexes_all = []
        MI_incor_labeled_indexes_all = []

        for j in range(num_classes):
            #Prepare the data for training and testing attack models (for all data and also correctly labeled samples)
            class_yes_x = confidence_train_for_attack[tuple([labels_train == j])]
            class_no_x = confidence_test_for_attack[tuple([labels_test == j])]

            if class_yes_x.shape[0] < 10 or class_no_x.shape[0] < 10:
                print("Class " + str(j) + " doesn't have enough sample for training an attack model (SKIPPED)!")
                continue

            class_yes_x_correctly_labeled = correctly_classified_indexes_train[tuple([labels_train == j])]
            class_no_x_correctly_labeled = correctly_classified_indexes_test[tuple([labels_test == j])]

            class_yes_x_incorrectly_labeled = incorrectly_classified_indexes_train[tuple([labels_train == j])]
            class_no_x_incorrectly_labeled = incorrectly_classified_indexes_test[tuple([labels_test == j])]

            class_yes_size = int(class_yes_x.shape[0] * what_portion_of_sampels_attacker_knows)
            class_yes_x_train = class_yes_x[:class_yes_size]
            class_yes_y_train = np.ones(class_yes_x_train.shape[0])
            class_yes_x_test = class_yes_x[class_yes_size:]
            class_yes_y_test = np.ones(class_yes_x_test.shape[0])
            class_yes_x_correctly_labeled = class_yes_x_correctly_labeled[class_yes_size:]
            class_yes_x_incorrectly_labeled = class_yes_x_incorrectly_labeled[class_yes_size:]

            class_no_size = int(class_no_x.shape[0] * what_portion_of_sampels_attacker_knows)
            class_no_x_train = class_no_x[:class_no_size]
            class_no_y_train = np.zeros(class_no_x_train.shape[0])
            class_no_x_test = class_no_x[class_no_size:]
            class_no_y_test = np.zeros(class_no_x_test.shape[0])
            class_no_x_correctly_labeled = class_no_x_correctly_labeled[class_no_size:]
            class_no_x_incorrectly_labeled = class_no_x_incorrectly_labeled[class_no_size:]

            y_size = class_yes_x_train.shape[0]
            n_size = class_no_x_train.shape[0]
            if sampling == "undersampling":
                if y_size > n_size:
                    class_yes_x_train = class_yes_x_train[:n_size]
                    class_yes_y_train = class_yes_y_train[:n_size]
                else:
                    class_no_x_train = class_no_x_train[:y_size]
                    class_no_y_train = class_no_y_train[:y_size]
            elif sampling == "oversampling":
                if y_size > n_size:
                    class_no_x_train = np.tile(class_no_x_train, (int(y_size / n_size), 1))
                    class_no_y_train = np.zeros(class_no_x_train.shape[0])
                else:
                    class_yes_x_train = np.tile(class_yes_x_train, (int(n_size / y_size), 1))
                    class_yes_y_train = np.ones(class_yes_x_train.shape[0])

            MI_x_train = np.concatenate((class_yes_x_train, class_no_x_train), axis=0)
            MI_y_train = np.concatenate((class_yes_y_train, class_no_y_train), axis=0)
            MI_x_test = np.concatenate((class_yes_x_test, class_no_x_test), axis=0)
            MI_y_test = np.concatenate((class_yes_y_test, class_no_y_test), axis=0)

            MI_x_train_all.extend(MI_x_train)
            MI_y_train_all.extend(MI_y_train)
            MI_x_test_all.extend(MI_x_test)
            MI_y_test_all.extend(MI_y_test)

            MI_cor_labeled_indexes = np.concatenate((class_yes_x_correctly_labeled, class_no_x_correctly_labeled), axis=0)
            MI_incor_labeled_indexes = np.concatenate((class_yes_x_incorrectly_labeled, class_no_x_incorrectly_labeled), axis=0)

            MI_cor_labeled_indexes_all.extend(MI_cor_labeled_indexes)
            MI_incor_labeled_indexes_all.extend(MI_incor_labeled_indexes)

        MI_x_train_all = np.array(MI_x_train_all)
        MI_y_train_all = np.array(MI_y_train_all)
        MI_x_test_all = np.array(MI_x_test_all)
        MI_y_test_all = np.array(MI_y_test_all)

        #To shuffle the training data:
        shuffle_index = np.random.permutation(MI_x_train_all.shape[0])
        MI_x_train_all = MI_x_train_all[shuffle_index]
        MI_y_train_all = MI_y_train_all[shuffle_index]

        # MI attack
        # if args.attack_type == "all":
        #     attack_model = nn.Sequential(nn.Linear(num_classes * (model_index_counter + 1), 128), nn.ReLU(),
        #                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        # elif args.attack_type == "aggregated":
        attack_model = nn.Sequential(nn.Linear(num_classes, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        # else:
        #     print("Attack type is not valid!")
        #     exit()

        attack_model = attack_model.cuda()
        criterion = nn.BCELoss().cuda()
        optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
        MI_x_train_cuda = torch.from_numpy(MI_x_train_all).float().cuda()
        MI_y_train_cuda = torch.from_numpy(MI_y_train_all).float().cuda()
        MI_x_test_cuda = torch.from_numpy(MI_x_test_all).float().cuda()
        MI_y_test_cuda = torch.from_numpy(MI_y_test_all).float().cuda()

        for ep in range(30):
            y_pred = attack_model(MI_x_train_cuda)
            y_pred = torch.squeeze(y_pred)
            train_loss = criterion(y_pred, MI_y_train_cuda)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        y_pred = attack_model(MI_x_test_cuda).cpu().detach().numpy()
        if y_pred.shape[0] > 0:
            MI_attack_auc = roc_auc_score(MI_y_test_all, y_pred)
        else:
            MI_attack_auc = -1

        #Gap attack
        MI_predicted_y_test_blind = np.zeros((MI_x_test_all.shape[0]))
        MI_predicted_y_test_blind[MI_cor_labeled_indexes_all] = 1
        y_pred = np.array(MI_predicted_y_test_blind)
        if y_pred.shape[0] > 0:
            MI_blind_attack_auc = roc_auc_score(MI_y_test_all, y_pred)
        else:
            MI_blind_attack_auc = -1

        print("---------------------")
        #print("Ensemble of", model_index_counter + 1, "models:")
        print("Train/Test accuracy:", str(np.round(acc_train*100, 2)), str(np.round(acc_test*100, 2)))
        print(args.output_type + "-based MI attack AUC:", MI_attack_auc)
        print("Gap attack AUC:", MI_blind_attack_auc)
        print("---------------------")


if __name__ == '__main__':
    main()
