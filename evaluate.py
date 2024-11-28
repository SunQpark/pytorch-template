import logging
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


logger = logging.getLogger('evaluate')

@hydra.main(config_path='conf', config_name='evaluate')
def main(config):
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint, weights_only=False)

    loaded_config = OmegaConf.create(checkpoint['config'])

    # setup data_loader instances
    data_loader = instantiate(config.data_loader)

    # restore network architecture
    model = instantiate(loaded_config.arch)
    logger.info(model)

    # load trained weights
    state_dict = checkpoint['state_dict']
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # instantiate loss and metrics
    criterion = instantiate(loaded_config.loss)
    metrics = {
        met_name: instantiate(met)
        for met_name, met in loaded_config.metrics.items()
    }

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics.values()):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met_name: total_metrics[i].item() / n_samples
        for i, met_name in enumerate(metrics.keys())
    })
    logger.info(log)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
