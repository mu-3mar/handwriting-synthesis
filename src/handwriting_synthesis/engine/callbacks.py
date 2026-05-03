import os
import re
import json
import logging
import torch
from handwriting_synthesis import utils
from handwriting_synthesis.data import transcriptions_to_tensor

logger = logging.getLogger(__name__)

class Callback:
    def on_iteration(self, epoch, epoch_iteration, iteration):
        pass

    def on_epoch(self, epoch):
        pass


class EpochModelCheckpoint(Callback):
    def __init__(self, synthesizer, save_dir, save_interval):
        self._synthesizer = synthesizer
        self._save_dir = save_dir
        self._save_interval = save_interval

    def on_epoch(self, epoch):
        if (epoch + 1) % self._save_interval == 0:
            epoch_dir = os.path.join(self._save_dir, f'Epoch_{epoch + 1}')
            self._synthesizer.save(epoch_dir)


class HandwritingGenerationCallback(Callback):
    def __init__(
        self,
        model,
        samples_dir,
        max_length,
        dataset,
        iteration_interval=10,
        sample_every_epochs=1,
        sample_biases=(0.0,),
    ):
        self.model = model
        self.samples_dir = samples_dir
        self.max_length = max_length
        self.interval = iteration_interval
        self.dataset = dataset
        self.sample_every_epochs = max(1, int(sample_every_epochs))
        self.sample_biases = tuple(sample_biases) if sample_biases else (0.0,)

    def on_iteration(self, epoch, epoch_iteration, iteration):
        # Sampling is handled at epoch boundaries for cleaner outputs.
        return

    def on_epoch(self, epoch):
        if (epoch + 1) % self.sample_every_epochs != 0:
            return
        steps = self.max_length

        random_dir = os.path.join(self.samples_dir, 'random', str(epoch + 1))
        os.makedirs(random_dir, exist_ok=True)
        names_with_contexts = self.get_names_with_contexts(epoch + 1)

        for file_name, context, text in names_with_contexts:
            base_name, ext = os.path.splitext(file_name)
            can_switch_bias = True
            for bias in self.sample_biases:
                random_path = os.path.join(random_dir, f'{base_name}_b{bias:g}{ext}')
                old_bias = getattr(self.model.mixture, "bias", None) if hasattr(self.model, "mixture") else None
                old_bias_output = (
                    getattr(self.model.output_layer, "bias", None)
                    if hasattr(self.model, "output_layer")
                    else None
                )
                try:
                    if can_switch_bias and hasattr(self.model, "mixture"):
                        self.model.mixture.bias = bias
                    if can_switch_bias and hasattr(self.model, "output_layer"):
                        self.model.output_layer.bias = bias
                    with torch.no_grad():
                        self.generate_handwriting(
                            random_path, steps=steps, stochastic=True, context=context, text=text
                        )
                except RuntimeError as e:
                    # TorchScript modules may not allow runtime mutation of `bias`.
                    can_switch_bias = False
                    logger.warning(
                        "Bias switching is not supported for this loaded checkpoint type; "
                        "continuing sampling with model default bias."
                    )
                    with torch.no_grad():
                        self.generate_handwriting(
                            random_path, steps=steps, stochastic=True, context=context, text=text
                        )
                finally:
                    if can_switch_bias and hasattr(self.model, "mixture"):
                        self.model.mixture.bias = old_bias
                    if can_switch_bias and hasattr(self.model, "output_layer"):
                        self.model.output_layer.bias = old_bias_output

    def get_names_with_contexts(self, iteration):
        file_name = f'iteration_{iteration}.png'
        context = None
        text = ''
        return [(file_name, context, text)]

    def generate_handwriting(self, save_path, steps, stochastic=True, context=None, text=''):
        mu, std = self.dataset.mu, self.dataset.std
        mu = torch.tensor(mu)
        std = torch.tensor(std)
        synthesizer = utils.HandwritingSynthesizer(self.model, mu, std, num_steps=steps, stochastic=stochastic)
        synthesizer.synthesize(c=context, output_path=save_path, show_attention=False)


class HandwritingSynthesisCallback(HandwritingGenerationCallback):
    def __init__(self, tokenizer, images_per_iterations=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_per_iteration = images_per_iterations
        self.tokenizer = tokenizer

        self.mu = torch.tensor(self.dataset.mu)
        self.std = torch.tensor(self.dataset.std)

    def get_names_with_contexts(self, iteration):
        images_per_iteration = min(len(self.dataset), self.images_per_iteration)
        res = []
        sentinel = '\n'
        for i in range(images_per_iteration):
            _, transcription = self.dataset[i]

            text = transcription + sentinel

            transcription_batch = [text]

            name = re.sub('[^0-9a-zA-Z]+', '_', transcription)
            file_name = f'{name}.png'
            context = transcriptions_to_tensor(self.tokenizer, transcription_batch)
            res.append((file_name, context, text))

        return res

    def generate_handwriting(self, save_path, steps, stochastic=True, context=None, text=''):
        super().generate_handwriting(save_path, steps, stochastic=stochastic, context=context)

        path, ext = os.path.splitext(save_path)
        save_path = f'{path}_attention{ext}'

        synthesizer = utils.HandwritingSynthesizer(
            self.model, self.mu, self.std, num_steps=steps, stochastic=stochastic
        )
        synthesizer.synthesize(c=context, output_path=save_path, show_attention=True, text=text)
