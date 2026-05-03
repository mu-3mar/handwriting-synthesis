import argparse
import logging
import os
import sys
import warnings

# Allow `python scripts/train.py` from repo root without installing the package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src = os.path.join(_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch

import handwriting_synthesis.callbacks
import handwriting_synthesis.tasks
from handwriting_synthesis import training
from handwriting_synthesis import data, utils, models, metrics
from handwriting_synthesis.config import (
    ConfigValidationError,
    configure_logging,
    create_run_layout,
    load_json_config,
    rebind_module_file_logger,
    save_run_config,
    validate_train_config,
)
from handwriting_synthesis.sampling import UnconditionalSampler, HandwritingSynthesizer

warnings.filterwarnings(
    "ignore",
    message="No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning
)


class ConfigOptions:
    def __init__(self, batch_size, epochs, sampling_interval,
                 num_train_examples, num_val_examples, max_length,
                 model_path, charset_path, samples_dir,
                 output_clip_value, lstm_clip_value, learning_rate=5e-5, max_grad_norm=1.0,
                 sample_every_epochs=1, sample_biases=(0.0, 0.5, 1.0)):
        self.batch_size = batch_size
        self.epochs = epochs
        self.sampling_interval = sampling_interval
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.max_length = max_length
        self.model_path = model_path
        self.charset_path = charset_path
        self.samples_dir = samples_dir
        self.output_clip_value = output_clip_value
        self.lstm_clip_value = lstm_clip_value
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.sample_every_epochs = sample_every_epochs
        self.sample_biases = tuple(sample_biases)


def attach_training_loop_file_log(checkpoints_dir: str) -> None:
    """Append per-epoch lines to runs/.../logs/training_loop.log when model_dir looks like a run layout."""
    checkpoints_dir = os.path.abspath(checkpoints_dir)
    if os.path.basename(checkpoints_dir) != "checkpoints":
        return
    logs_dir = os.path.join(os.path.dirname(checkpoints_dir), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "training_loop.log")
    rebind_module_file_logger(training.logger, path, logging.INFO)


def infer_run_layout_from_checkpoints_dir(checkpoints_dir: str):
    checkpoints_dir = os.path.abspath(checkpoints_dir)
    if os.path.basename(checkpoints_dir) != "checkpoints":
        return None
    run_dir = os.path.dirname(checkpoints_dir)
    return {
        "run_dir": run_dir,
        "samples_dir": os.path.join(run_dir, "samples"),
        "logs_dir": os.path.join(run_dir, "logs"),
    }


def parse_bias_list(raw: str):
    if not raw:
        return (0.0, 0.5, 1.0)
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return tuple(values) if values else (0.0, 0.5, 1.0)


def print_info_message(training_task_verbose, config):
    logging.getLogger("handwriting_train").info(
        f'{training_task_verbose} with options: training set size {config.num_train_examples}, '
        f'validation set size {config.num_val_examples}, '
        f'batch size {config.batch_size}, '
        f'max sequence length {config.max_length},'
        f'sampling interval (in # iterations): {config.sampling_interval}'
    )


def train_model(train_set, val_set, train_task, callbacks, config, training_task_verbose, sampler):
    print_info_message(training_task_verbose, config)

    train_metrics = [metrics.MSE(), metrics.SSE()]
    val_metrics = [metrics.MSE(), metrics.SSE()]

    loop = training.TrainingLoop(train_set, val_set, batch_size=config.batch_size, training_task=train_task,
                                 train_metrics=train_metrics, val_metrics=val_metrics)

    for cb in callbacks:
        loop.add_callback(cb)

    sample_class = sampler.__class__
    _, largest_epoch = sample_class.load_latest(check_points_dir=config.model_path,
                                                device=torch.device("cpu"))

    saver = handwriting_synthesis.callbacks.EpochModelCheckpoint(
        sampler, config.model_path, save_interval=1
    )
    loop.add_callback(saver)

    loop.start(initial_epoch=largest_epoch, epochs=config.epochs)


def train_unconditional_handwriting_generator(train_set, val_set, device, config):
    sampler, epochs = UnconditionalSampler.load_latest(config.model_path, device)
    if sampler:
        model = sampler.model
    else:
        model = models.HandwritingPredictionNetwork.get_default_model(device)
        model = model.to(device)

    if not sampler:
        mu = torch.tensor(train_set.mu, dtype=torch.float32)
        sd = torch.tensor(train_set.std, dtype=torch.float32)
        tokenizer = data.Tokenizer.from_file(config.charset_path)
        sampler = UnconditionalSampler(model, mu, sd, tokenizer.charset, num_steps=config.max_length)

    if config.output_clip_value == 0 or config.lstm_clip_value == 0:
        clip_values = None
    else:
        clip_values = (config.output_clip_value, config.lstm_clip_value)

    train_task = handwriting_synthesis.tasks.HandwritingPredictionTrainingTask(
        device, model, clip_values,
        learning_rate=config.learning_rate, max_grad_norm=config.max_grad_norm,
    )

    cb = handwriting_synthesis.callbacks.HandwritingGenerationCallback(
        model, config.samples_dir, config.max_length,
        val_set, iteration_interval=config.sampling_interval,
        sample_every_epochs=config.sample_every_epochs, sample_biases=config.sample_biases,
    )

    train_model(train_set, val_set, train_task, [cb], config,
                training_task_verbose='Training (unconditional) handwriting prediction model', sampler=sampler)


def train_handwriting_synthesis_model(train_set, val_set, device, config):
    synthesizer, epochs = HandwritingSynthesizer.load_latest(config.model_path, device)

    if synthesizer:
        model = synthesizer.model
    else:
        tokenizer = data.Tokenizer.from_file(config.charset_path)
        alphabet_size = tokenizer.size

        model = models.SynthesisNetwork.get_default_model(alphabet_size, device)
        model = model.to(device)

        mu = torch.tensor(train_set.mu, dtype=torch.float32)
        sd = torch.tensor(train_set.std, dtype=torch.float32)
        synthesizer = HandwritingSynthesizer(
            model, mu, sd, tokenizer.charset, num_steps=config.max_length
        )

    if config.output_clip_value == 0 or config.lstm_clip_value == 0:
        clip_values = None
    else:
        clip_values = (config.output_clip_value, config.lstm_clip_value)

    train_task = handwriting_synthesis.tasks.HandwritingSynthesisTask(
        synthesizer.tokenizer, device, model, clip_values,
        learning_rate=config.learning_rate, max_grad_norm=config.max_grad_norm,
    )

    cb = handwriting_synthesis.callbacks.HandwritingSynthesisCallback(
        synthesizer.tokenizer,
        1,
        model, config.samples_dir, config.max_length,
        val_set, iteration_interval=config.sampling_interval,
        sample_every_epochs=config.sample_every_epochs, sample_biases=config.sample_biases,
    )

    train_model(train_set, val_set, train_task, [cb], config,
                training_task_verbose='Training handwriting synthesis model', sampler=synthesizer)


def get_device():
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            # computations on TPU are very slow for some reason
            dev = xm.xla_device()
        except ImportError:
            pass
    return dev


def get_device_with_override(override):
    if not override or override == "auto":
        return get_device()
    if override == "cpu":
        return torch.device("cpu")
    if override == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device(override)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Starts/resumes training prediction or synthesis network.'
    )

    parser.add_argument(
        "data_dir", type=str, nargs="?", default="",
        help="Directory containing training and validation data h5 files"
    )
    parser.add_argument(
        "model_dir", type=str, nargs="?", default="",
        help="Directory storing model weights"
    )
    parser.add_argument(
        "-u", "--unconditional", default=False, action="store_true",
        help="Whether or not to train synthesis network (synthesis network is trained by default)"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs to train")
    parser.add_argument("-i", "--interval", type=int, default=100, help="Iterations between sampling")
    parser.add_argument("-c", "--charset", type=str, default='', help="Path to the charset file")

    parser.add_argument("--samples_dir", type=str, default='samples',
                        help="Path to the directory that will store samples")
    parser.add_argument("--sample-every-epochs", type=int, default=1,
                        help="Generate sample images every N epochs (default: 1)")
    parser.add_argument("--sample-biases", type=str, default="0,0.5,1",
                        help="Comma-separated biases for sampling (default: 0,0.5,1)")

    parser.add_argument(
        "--clip1", type=int, default=0,
        help="Gradient clipping value for output layer. "
             "When omitted or set to zero, no clipping is done."
    )
    parser.add_argument(
        "--clip2", type=int, default=0,
        help="Gradient clipping value for lstm layers. "
             "When omitted or set to zero, no clipping is done."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Optimizer learning rate (default: 5e-5, half of original 1e-4). "
             "Overridden by JSON config when using --config.",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=None,
        help="Global L2 grad clip before optimizer step (default: 1.0; 0 disables). "
             "Overridden by JSON config when using --config.",
    )
    parser.add_argument("--config", type=str, default="", help="Path to train config JSON")
    parser.add_argument("--device", type=str, default="", help="Device override: auto|cpu|cuda|cuda:0")
    parser.add_argument("--run_id", type=str, default="", help="Run id under runs/ (e.g. run_007)")

    args = parser.parse_args()

    if args.config:
        try:
            cfg = load_json_config(args.config)
            validate_train_config(cfg)
        except ConfigValidationError as e:
            parser.error(str(e))

        runs_dir = cfg["output"].get("runs_dir", "runs")
        run_info = create_run_layout(runs_dir=runs_dir, run_id=args.run_id or None)
        logger = configure_logging(run_info["logs_dir"], run_info["run_id"])
        rebind_module_file_logger(training.logger, os.path.join(run_info["logs_dir"], "training_loop.log"), logging.INFO)
        rebind_module_file_logger(models.logger, os.path.join(run_info["logs_dir"], "model_error.log"), logging.ERROR)
        rebind_module_file_logger(utils.logger, os.path.join(run_info["logs_dir"], "utils_error.log"), logging.ERROR)

        data_dir = cfg["dataset"]["prepared_data_dir"]
        model_dir = run_info["checkpoints_dir"]
        unconditional = bool(cfg["training"]["unconditional"])
        batch_size = int(cfg["training"]["batch_size"])
        epochs = int(cfg["training"]["epochs"])
        interval = int(cfg["training"]["sampling_interval"])
        clip1 = int(cfg["training"]["clip1"])
        clip2 = int(cfg["training"]["clip2"])
        learning_rate = float(cfg["training"].get("learning_rate", 5e-5))
        max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
        sample_every_epochs = int(cfg["training"].get("sample_every_epochs", 1))
        sample_biases = parse_bias_list(str(cfg["training"].get("sample_biases", "0,0.5,1")))
        samples_dir = run_info["samples_dir"]
        charset_arg = cfg["dataset"]["charset_path"]
        device_override = args.device or cfg["training"]["device"]

        cfg_for_run = dict(cfg)
        cfg_for_run["output"] = dict(cfg.get("output", {}))
        cfg_for_run["output"]["model_dir"] = model_dir
        cfg_for_run["output"]["samples_dir"] = samples_dir
        cfg_for_run["output"]["run_id"] = run_info["run_id"]
        cfg_for_run["output"]["run_dir"] = run_info["run_dir"]
        save_run_config(cfg_for_run, run_info["config_path"])
        logger.info(f"Created run directory: {run_info['run_dir']}")
    else:
        if not args.data_dir or not args.model_dir:
            parser.error("Either pass --config or provide positional args: data_dir model_dir")
        data_dir = args.data_dir
        model_dir = args.model_dir
        unconditional = args.unconditional
        batch_size = args.batch_size
        epochs = args.epochs
        interval = args.interval
        clip1 = args.clip1
        clip2 = args.clip2
        learning_rate = args.learning_rate if args.learning_rate is not None else 5e-5
        max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else 1.0
        sample_every_epochs = int(args.sample_every_epochs)
        sample_biases = parse_bias_list(args.sample_biases)
        samples_dir = args.samples_dir
        charset_arg = args.charset
        device_override = args.device or "auto"
        logger = logging.getLogger("handwriting_train")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | legacy | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            logger = logging.getLogger("handwriting_train")
        inferred = infer_run_layout_from_checkpoints_dir(model_dir)
        if inferred:
            os.makedirs(inferred["samples_dir"], exist_ok=True)
            os.makedirs(inferred["logs_dir"], exist_ok=True)
            if args.samples_dir == "samples":
                samples_dir = inferred["samples_dir"]
            train_log = os.path.join(inferred["logs_dir"], "training.log")
            error_log = os.path.join(inferred["logs_dir"], "error.log")
            for h in list(logger.handlers):
                if isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
            fmt = logging.Formatter(
                "%(asctime)s | legacy | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            info_h = logging.FileHandler(train_log)
            info_h.setLevel(logging.INFO)
            info_h.setFormatter(fmt)
            logger.addHandler(info_h)
            err_h = logging.FileHandler(error_log)
            err_h.setLevel(logging.ERROR)
            err_h.setFormatter(fmt)
            logger.addHandler(err_h)
            rebind_module_file_logger(
                models.logger, os.path.join(inferred["logs_dir"], "model_error.log"), logging.ERROR
            )
            rebind_module_file_logger(
                utils.logger, os.path.join(inferred["logs_dir"], "utils_error.log"), logging.ERROR
            )
        attach_training_loop_file_log(model_dir)

    device = get_device_with_override(device_override)

    logger.info(f'Using device {device}')

    with data.H5Dataset(f'{data_dir}/train.h5') as dataset:
        mu = dataset.mu
        sd = dataset.std

    train_dataset_path = os.path.join(data_dir, 'train.h5')
    val_dataset_path = os.path.join(data_dir, 'val.h5')

    default_charset_path = os.path.join(data_dir, 'charset.txt')
    charset_path = utils.get_charset_path_or_raise(charset_arg, default_charset_path)

    with data.NormalizedDataset(train_dataset_path, mu, sd) as train_set, \
            data.NormalizedDataset(val_dataset_path, mu, sd) as val_set:
        num_train_examples = len(train_set)
        num_val_examples = len(val_set)
        max_length = train_set.max_length
        model_path = model_dir

        config = ConfigOptions(batch_size=batch_size, epochs=epochs,
                               sampling_interval=interval, num_train_examples=num_train_examples,
                               num_val_examples=num_val_examples, max_length=max_length,
                               model_path=model_path,
                               charset_path=charset_path,
                               samples_dir=samples_dir,
                               output_clip_value=clip1, lstm_clip_value=clip2,
                               learning_rate=learning_rate, max_grad_norm=max_grad_norm,
                               sample_every_epochs=sample_every_epochs, sample_biases=sample_biases)

        if unconditional:
            train_unconditional_handwriting_generator(train_set, val_set, device, config)
        else:
            train_handwriting_synthesis_model(train_set, val_set, device, config)
