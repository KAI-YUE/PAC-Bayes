import time

# PyTorch libraries
import torch

# My libraries
from config import load_config
from config.utils import *
from deeplearning.utils import *
from deeplearning.datasets import *
from deeplearning.dp import *

def prune_and_train(config, dataset, logger):
    record = init_record()

    dst_train, dst_test = dataset.dst_train, dataset.dst_test

    indices = load_idx(config)
    subsets = fetch_subsets(dst_train, indices)

    insensitive_train_loader, test_loader = fetch_dataloader(config, subsets["insensitive"], dst_test)
    lowscore_sensitive_train_loader = fetch_dploader(config, subsets[config.subset_keyword])

    len_sens = len(lowscore_sensitive_train_loader.dataset) if len(lowscore_sensitive_train_loader) > 0 else 0
    len_insens = len(insensitive_train_loader.dataset)
    logger.info("------- N: {}, N1: {}, N2: {} -------".format(len_sens + len_insens, len_sens, len_insens))
    model, criterion, optimizer, scheduler, start_epoch = init_all(config, dataset, len_insens, logger)
    dp_model, dp_optimizer, dp_scheduler, dp_loader, privacy_engine = init_dp_setup(config, model, lowscore_sensitive_train_loader, len_sens)

    criterion = nn.CrossEntropyLoss()

    best_testacc, best_epoch = 0.0, 0
    config.include_sensitive_this_epoch = False

    for epoch in range(start_epoch, config.epochs):
        if config.use_sensitive and epoch >= config.include_sensitive_epoch:
            if epoch % config.include_sensitive_freq == 0:
                config.include_sensitive_this_epoch = True
                logger.info("Epoch {:d}/{:d} includes sensitive data".format(epoch, config.epochs))
            else:
                config.include_sensitive_this_epoch = False
                logger.info("Epoch {:d}/{:d} does not include sensitive data".format(epoch, config.epochs))

        # train for one epoch
        interchange_train(dp_loader, insensitive_train_loader, model, dp_model, 
                          criterion, dp_optimizer, optimizer, dp_scheduler, scheduler, 
                          privacy_engine, epoch, config, logger, record)
        
        # evaluate 
        if config.test_interval > 0 and (epoch + 1) % config.test_interval == 0:
            testacc = test(test_loader, model, criterion, config, logger, record)

            # remember best prec@1 and save checkpoint
            is_best = testacc > best_testacc

            if is_best:
                best_testacc = testacc
                best_epoch = epoch + 1
                save_checkpoint({"epoch": epoch + 1,
                                "state_dict": model.state_dict(),
                                "opt_dict": optimizer.state_dict(),
                                "best_epoch": best_epoch},
                                # config.output_dir + "/checkpoint_epoch{:d}.pth".format(epoch))
                                config.output_dir + "/checkpoint_epoch.pth")

    # save the record
    logger.info('Best accuracy: {:.3f} at epoch {:d}'.format(best_testacc, best_epoch-1))
    save_record(record, config.output_dir)

        
def main():
    # load the config file, logger, and initialize the output folder
    config = load_config()
    
    # qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # bounds = np.array([0.1, 0.14, 0.17, 0.2, 0.23, 0.25, 0.27, 0.28, 0.3])*4

    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    bounds = np.array([0.1, 0.13, 0.15, 0.17, 0.23, 0.25])
    bounds = np.array([0.9, 1., 1.1, 1.3, 1.4, 1.5, 1.6, 1.7])
    bounds = np.ones(len(qs))
    
    # lr = 1.e-3*np.array([1/0.1, 1/0.13, 1/0.15, 1/0.17, 1/0.27])*2

    # epochs_list = np.array([100, 200, 300, 400, 500, 600]) 

    batch_sizes = np.arange(1, len(qs)+1)*32
    # sigmas = np.array([2, 1.208, 0.947, 0.816, 0.737, 0.683, 0.644])

    offset = 0
    for i, q in enumerate(qs[offset:]):
        i += offset
        # q = qs[i]
        
        if "highscore" in config.subset_keyword:
            q = 1 - q # for high score, we have to revert the q 
        # config.std = sigmas[i]
        

        # config.idx_path = "data/resnet18/sens0.9/q{:.1f}.dat".format(q)
        
        # config.idx_path = "data/lenet5_fmnist/sens0.9/q{:.1f}.dat".format(q)
        # config.bound = bounds[i]
        
        # config.seed += 1
        # config.epochs = epochs_list[i]
        # config.lr = 3.e-2/bounds[i]

        config.dp_batch_size = int(batch_sizes[i])

        output_dir = init_outputfolder(config)
        logger = init_logger(config, output_dir, config.seed, attach=False)

        if config.device == "cuda":
            torch.backends.cudnn.benchmark = True

        torch.random.manual_seed(config.seed)
        np.random.seed(config.seed)

        start = time.time()
        dataset = fetch_dataset(config)
        prune_and_train(config, dataset, logger)
        end = time.time()

        logger.info("{:.3} mins has elapsed".format((end-start)/60))

        logger.handlers.clear()

if __name__ == "__main__":
    main()

