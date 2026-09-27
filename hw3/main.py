# GenAI disclosure (EN.705.641 syllabus, Green designation):
# Claude (Anthropic, model claude-opus-5-5) wrote explore_mlp_activations and
# explore_mlp_learning_rates (the class template did not include them, so their fixed
# hyper-parameters follow the upstream JHU CS 601.471/671 Sp24 hw3 skeleton this assignment
# is derived from), the file-name sanitizing in explore_mlp_structures (the original names
# contain ">", which Windows rejects), and the JSON result logging. Claude also ran all
# three experiments and generated the plots. I directed the work and reviewed it before
# submitting. LLM-generated code is cited here per the syllabus.

import json
import gensim.downloader
from easydict import EasyDict
from mlp_lm import run_mlp_lm, load_data_mlp_lm, sample_from_mlp_lm, visualize_epochs
from mlp import run_mlp, load_data_mlp, visualize_configs
from typing import List, Tuple, Dict, Union

EMBEDDING_TYPES = ["glove-twitter-50", "glove-twitter-100", "glove-twitter-200", "word2vec-google-news-300"]
HIDDEN_DIMS = [[], [512], [512, 512], [512, 512, 512]]
HIDDEN_DIMS_NAMES = ["None", "512", "512 -> 512", "512 -> 512 -> 512"]
LEARING_RATES = [0.025, 0.02, 0.01, 0.001]


def safe_name(name: str) -> str:
    # "512 -> 512" -> "512-512": spaces and ">" are not safe in file names (">" is illegal on Windows)
    return name.replace(" -> ", "-").replace(" ", "_")


def save_results(results: Dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{'-' * 10} Saved results to {path} {'-' * 10}")


def single_run_mlp_lm(train_d, dev_d):
    # TODO: once you have completed the backprop.py, you can run this function to train and evaluate your model,
    #  and visualize the training process with a plot
    train_config = EasyDict({
        # model configuration
        'embed_dim': 128,  # the dimension of the word embeddings
        'hidden_dim': 512,  # the dimension of the hidden layer
        'num_blocks': 2,  # the number of transformer blocks
        'dropout_p': 0.2,  # the probability of dropout
        'local_window_size': 6,  # the size of the local window
        # training configuration
        'batch_size': 4096,  # batch size
        'lr': 2e-6,  # learning rate
        'decay': 1.0,
        'num_epochs': 5,  # the total number of times all the training data is iterated over
        'save_path': 'model.pth',  # path where to save the model
    })

    epoch_train_losses, epoch_train_ppls, epoch_dev_losses, epoch_dev_ppls = run_mlp_lm(train_config, train_d, dev_d)
    visualize_epochs(epoch_train_losses, epoch_dev_losses, "Loss", "mlp_lm_loss.png")
    visualize_epochs(epoch_train_ppls, epoch_dev_ppls, "Perplexity", "mlp_lm_ppl.png")


def sample_from_trained_mlp_lm(dev_d):
    pretrained_config = EasyDict({
        # model configuration
        'embed_dim': 256,  # the dimension of the word embeddings
        'hidden_dim': 1048,  # the dimension of the hidden layer
        'num_blocks': 4,  # the number of transformer blocks
        'dropout_p': 0.2,  # the probability of dropout
        'local_window_size': 6,  # the size of the local window
        'save_path': 'pretrained_fixed_window_lm.dat',  # path where to save the model
        # evaluation configuration
        'batch_size': 4096,  # batch size
    })
    sample_from_mlp_lm(pretrained_config, dev_d)


def explore_mlp_structures(dev_d: Dict[str, List[Union[str, int]]],
                           train_d: Dict[str, List[Union[str, int]]],
                           test_d: Dict[str, List[Union[str, int]]]):
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    print(f"{'-' * 10} Load Pre-trained Embeddings: {EMBEDDING_TYPES[0]} {'-' * 10}")
    embeddings = gensim.downloader.load(EMBEDDING_TYPES[0])

    results = {}
    for hidden_dims, hidden_dim_names, lr in zip(HIDDEN_DIMS, HIDDEN_DIMS_NAMES, LEARING_RATES):
        train_config = EasyDict({
            'batch_size': 64,  # we use batching for
            'lr': lr,  # if embedding_type != "None" else 0.01,  # learning rate
            'num_epochs': 20,  # the total number of times all the training data is iterated over
            'hidden_dims': hidden_dims,
            'save_path': f'model_hidden_{safe_name(hidden_dim_names)}.pth',  # path where to save the model
            'embeddings': EMBEDDING_TYPES[0],
            'num_classes': 2,
        })

        (epoch_train_losses, epoch_train_accs, epoch_dev_loss, epoch_dev_accs,
         test_loss, test_acc) = run_mlp(train_config, embeddings, dev_d, train_d, test_d)
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
        visualize_epochs(epoch_train_losses, epoch_dev_loss, "Loss", f"mlp_{safe_name(hidden_dim_names)}_loss.png")
        results[hidden_dim_names] = {
            'lr': lr, 'train_loss': [float(x) for x in epoch_train_losses],
            'train_acc': [float(x) for x in epoch_train_accs],
            'dev_loss': [float(x) for x in epoch_dev_loss], 'dev_acc': [float(x) for x in epoch_dev_accs],
            'test_loss': float(test_loss), 'test_acc': float(test_acc),
        }

    visualize_configs(all_emb_epoch_dev_accs, HIDDEN_DIMS_NAMES, "Accuracy", "./all_mlp_acc.png")
    visualize_configs(all_emb_epoch_dev_losses, HIDDEN_DIMS_NAMES, "Loss", "./all_mlp_loss.png")
    save_results(results, "results_mlp_structures.json")


def explore_mlp_activations(dev_d: Dict[str, List[Union[str, int]]],
                            train_d: Dict[str, List[Union[str, int]]],
                            test_d: Dict[str, List[Union[str, int]]],
                            embeddings=None):
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    if embeddings is None:
        print(f"{'-' * 10} Load Pre-trained Embeddings: {EMBEDDING_TYPES[0]} {'-' * 10}")
        embeddings = gensim.downloader.load(EMBEDDING_TYPES[0])

    # fixed hyperparameters: a single 512-dimension hidden layer, as the assignment specifies
    batch_size = 128
    num_epochs = 20
    lr = 0.02
    num_classes = 2
    hidden_dims = [512]

    # TODO: explore different activation functions
    # activation functions to explore
    activations = ['Sigmoid', 'Tanh', 'ReLU', 'GeLU', 'LeakyReLU']
    activation_names = activations  # for visualization

    results = {}
    for activation in activations:
        train_config = EasyDict({
            'batch_size': batch_size,
            'lr': lr,
            'num_epochs': num_epochs,
            'hidden_dims': hidden_dims,
            'save_path': f'model_activation_{activation}.pth',
            'embeddings': EMBEDDING_TYPES[0],
            'num_classes': num_classes,
            'activation': activation,  # the only thing that changes across runs
        })
        (epoch_train_losses, epoch_train_accs, epoch_dev_losses, epoch_dev_accs,
         test_loss, test_acc) = run_mlp(train_config, embeddings, dev_d, train_d, test_d)
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_losses)
        results[activation] = {
            'train_loss': [float(x) for x in epoch_train_losses],
            'train_acc': [float(x) for x in epoch_train_accs],
            'dev_loss': [float(x) for x in epoch_dev_losses], 'dev_acc': [float(x) for x in epoch_dev_accs],
            'test_loss': float(test_loss), 'test_acc': float(test_acc),
        }

    # dev accuracy and dev loss across activations, two separate plots
    visualize_configs(all_emb_epoch_dev_accs, activation_names, "Accuracy", "./all_mlp_activations_acc.png")
    visualize_configs(all_emb_epoch_dev_losses, activation_names, "Loss", "./all_mlp_activations_loss.png")
    save_results(results, "results_mlp_activations.json")
    # your code ends here


def explore_mlp_learning_rates(dev_d: Dict[str, List[Union[str, int]]],
                               train_d: Dict[str, List[Union[str, int]]],
                               test_d: Dict[str, List[Union[str, int]]],
                               embeddings=None):
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    if embeddings is None:
        print(f"{'-' * 10} Load Pre-trained Embeddings: {EMBEDDING_TYPES[0]} {'-' * 10}")
        embeddings = gensim.downloader.load(EMBEDDING_TYPES[0])

    # fixed hyperparameters: a single 512-dimension hidden layer, as the assignment specifies
    batch_size = 64
    num_epochs = 20
    num_classes = 2
    hidden_dims = [512]
    activation = 'Sigmoid'

    # TODO: explore different learning rates
    # 0.02 is the provided base rate. The others run from 0.1 down to 0.00002 on a log scale.
    lrs = [0.1, 0.02, 0.002, 0.0002, 0.00002]
    lrs_names = [str(lr) for lr in lrs]  # for visualization

    results = {}
    for lr, lr_name in zip(lrs, lrs_names):
        train_config = EasyDict({
            'batch_size': batch_size,
            'lr': lr,  # the only thing that changes across runs
            'num_epochs': num_epochs,
            'hidden_dims': hidden_dims,
            'save_path': f'model_lr_{lr_name}.pth',
            'embeddings': EMBEDDING_TYPES[0],
            'num_classes': num_classes,
            'activation': activation,
        })
        (epoch_train_losses, epoch_train_accs, epoch_dev_losses, epoch_dev_accs,
         test_loss, test_acc) = run_mlp(train_config, embeddings, dev_d, train_d, test_d)
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_losses)
        results[lr_name] = {
            'train_loss': [float(x) for x in epoch_train_losses],
            'train_acc': [float(x) for x in epoch_train_accs],
            'dev_loss': [float(x) for x in epoch_dev_losses], 'dev_acc': [float(x) for x in epoch_dev_accs],
            'test_loss': float(test_loss), 'test_acc': float(test_acc),
        }

    # dev accuracy and dev loss across learning rates, two separate plots
    visualize_configs(all_emb_epoch_dev_accs, lrs_names, "Accuracy", "./all_mlp_lrs_acc.png")
    visualize_configs(all_emb_epoch_dev_losses, lrs_names, "Loss", "./all_mlp_lrs_loss.png")
    save_results(results, "results_mlp_learning_rates.json")
    # your code ends here


if __name__ == '__main__':
    # Load raw data for mlp
    # uncomment the following line to run
    dev_data, train_data, test_data = load_data_mlp()

    # Explore different hidden dimensions
    # uncomment the following line to run
    explore_mlp_structures(dev_data, train_data, test_data)

    # Explore different activations (single 512-dim hidden layer)
    # uncomment the following line to run
    explore_mlp_activations(dev_data, train_data, test_data)

    # Explore different learning rates (single 512-dim hidden layer)
    # uncomment the following line to run
    explore_mlp_learning_rates(dev_data, train_data, test_data)

    # load raw data for lm
    # uncomment the following line to run
    # train_data, dev_data = load_data_mlp_lm()

    # Run a single training run
    # uncomment the following line to run
    # single_run_mlp_lm(train_data, dev_data)

    # Sample from the pretrained model
    # uncomment the following line to run
    # sample_from_trained_mlp_lm(dev_data)

