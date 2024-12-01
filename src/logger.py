import os
import logging
from torch import nn
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


abs_path = Path(os.path.abspath(__file__)).parent

def set_logging():
    # set logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
    # Disable c10d logging
    logging.getLogger('c10d').setLevel(logging.ERROR)
    
    
class TensorLogger:
    def __init__(self, logdir='./logs'):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(logdir, current_datetime)
        # Ensure the log directory exists
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        
    def log_metric(self, value, step, metric='loss', mode='Train'):
        self.writer.add_scalar(f'{mode}/{metric}', value, step)
        
    def log_report(self, report, step, mode='Val'):
        # Log metrics from the classification report
        for label, metrics in report.items():
            # Check if metrics is a dictionary
            if isinstance(metrics, dict):  # This will be True for class labels and averages
                for metric_name, value in metrics.items():
                    # Log class-specific metrics and averages except 'support'
                    if metric_name != 'support':
                        self.writer.add_scalar(f'{mode}_class_{label}/{metric_name}', value, step)
            else:
                # This handles the overall 'accuracy', which is a single float value
                # Log overall accuracy
                if label == 'accuracy':
                    self.writer.add_scalar(f'{mode}/{label}', metrics, step)
    
    def log_text(self, txt, step):
        # Log some text
        self.writer.add_text('Version Description', txt, step)
        self.writer.flush()
        
    def log_ud(self, model, step, lr=None):
        # Get the current learning rate
        if lr is None:
            lr = model.optimizers().param_groups[0]['lr']

        # Iterate through named parameters to calculate and log metrics
        for name, p in model.named_parameters():
            if p.ndim == 2 and p.grad is not None:
                # Calculate the standard deviation of the gradients adjusted by the learning rate
                grad_std = (lr * p.grad).std()
                # Calculate the standard deviation of the parameter values
                param_std = p.data.std()
                # Calculate the Update Discrepancy (ud) metric and take the log10
                metric = (grad_std / param_std).log10().item()
                # Create a formatted name that corresponds to the naming convention in the TensorBoard layout
                formatted_name = 'z_ud_' + name.replace('.', '_')
                # Log the metric using the formatted name
                self.log_metric(metric, step, formatted_name)
                
    def log_model_arch(self, model):
        # Log the model architecture at the start of training
        model_info = "<br>".join([f"{name}: {str(el)}" for name, el in model.get_elements().items()])
        self.writer.add_text('Model Architecture', model_info, 0)


class Logger(TensorBoardLogger):
    def __init__(self, model, logdir, *args, **kwargs):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(logdir, current_datetime)
        # Ensure the log directory exists
        os.makedirs(logdir, exist_ok=True)
        super().__init__(save_dir=logdir, *args, **kwargs)
        self._log_graph = True
        self.model = model
        # self.model.set_custom_logger(self)
        self.experiment.add_custom_scalars(self._layout())

    def _layout(self):
        activation_params = []
        discrepancy_params = []
        # for i, item in enumerate(self.model.get_activations().items()):
        #     name, layer = item
        #     if isinstance(layer, act_possible):
        #         # Adding activation stats layout
        #         # activation_params.append(f'act_{name}_out')
        #         activation_params.append(f'z_act_{name.replace(".", "_")}_out_sat')

        # for name, p in self.model.named_parameters():
        #     if p.ndim == 2:
        #         # Assuming you only log update discrepancies for 2D params
        #         formatted_name = 'z_ud_' + name.replace('.', '_')
        #         discrepancy_params.append(formatted_name)

        layout = {
            "Layer Metrics": {
                "Activation Out Saturation": ["Multiline", activation_params],
                "Update Discrepancy by Layer": ["Multiline", discrepancy_params]
            }
        }
        return layout

    # logging
    def log_ud(self, step, lr=None):
        # Get the current learning rate
        if lr is None:
            lr = self.model.optimizers().param_groups[0]['lr']

        # Iterate through named parameters to calculate and log metrics
        for name, p in self.model.named_parameters():
            if p.ndim == 2 and p.grad is not None:
                # Calculate the standard deviation of the gradients adjusted by the learning rate
                grad_std = (lr * p.grad).std()
                # Calculate the standard deviation of the parameter values
                param_std = p.data.std()
                # Calculate the Update Discrepancy (ud) metric and take the log10
                metric = (grad_std / param_std).log10().item()
                # Create a formatted name that corresponds to the naming convention in the TensorBoard layout
                formatted_name = 'z_ud_' + name.replace('.', '_')
                # Log the metric using the formatted name
                self._experiment.add_scalar(formatted_name, metric, global_step=step)

    def log_activation_out_sat(self):
        for i, item in enumerate(self.model.get_activations().items()):
            name, layer = item
            if isinstance(layer, nn.ReLU):
                t = layer.out.detach()  # Make sure outputs are stored during forward pass
                saturation = (t < 0.05).float().mean() * 100
                self.model.log(f'z_act_{name.replace(".", "_")}_out_sat', saturation, on_step=False, on_epoch=True, sync_dist=True)
            if isinstance(layer, nn.Tanh):
                t = layer.out.detach()  # Make sure outputs are stored during forward pass
                saturation = (t.abs() > 0.025).float().mean() * 100
                self.model.log(f'z_act_{name.replace(".", "_")}_out_sat', saturation, on_step=False, on_epoch=True, sync_dist=True)
            elif isinstance(layer, nn.ELU):
                t = layer.out.detach()  # Make sure outputs are stored during forward pass
                saturation = (t < 0).float().mean() * 100
                self.model.log(f'z_act_{name.replace(".", "_")}_out_sat', saturation, on_step=False, on_epoch=True, sync_dist=True)

    def log_out_on_epoch(self):
        for i, item in enumerate(self.model.get_activations().items()):
            name, layer = item
            self.experiment.add_histogram(f'z_act_{name.replace(".", "_")}_out_grad', layer.out.grad, self.model.global_step)
            t = layer.out.detach()
            self.experiment.add_histogram(f'z_act_{name.replace(".", "_")}_out', t, self.model.global_step)
        for i, item in enumerate(self.model.get_trainable_layers().items()):
            name, layer = item
            self.experiment.add_histogram(f'z_{name.replace(".", "_")}_out_grad', layer.out.grad,
                                          self.model.global_step)
            t = layer.out.detach()
            self.experiment.add_histogram(f'z_{name.replace(".", "_")}_out', t, self.model.global_step)

    def log_model_arch(self):
        # Log the model architecture at the start of training
        model_info = "<br>".join([f"{name}: {str(el)}" for name, el in self.model.get_elements().items()])
        self.experiment.add_text('Model Architecture', model_info, 0)

    def log_params(self):
        # Log hyperparameters as text
        hp_text = "<br>".join([f"{key}: {value}" for key, value in vars(self.model.params).items()])
        self.experiment.add_text('Hyperparameters', hp_text, 0)
        
