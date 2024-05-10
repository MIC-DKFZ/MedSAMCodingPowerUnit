from time import time
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger2HEAD
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast, nn
from torch import distributed as dist

from nnunetv2.training.dataloading.data_loader_2d_bbox import nnUNetDataLoader2DBBOX, nnUNetDataLoader2DBBOX2HEAD
from nnunetv2.training.loss.compound_losses import DC_and_Focal_loss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler, PolyLRSchedulerWarmUp
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetFineTuiningDataset

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
import pydoc
from nnunetv2.architecture.promptable_unet import PromptableUNet, PromptableResEncUNet, PromptableResEncUNetMultiHead


class nnUNetTrainerCPU(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels+1,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                        if ignore_label is not None else regions,
                                                                        'target', 'target'))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DBBOX(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None,
                                       footprint=5, bbox_dilation=20)
            dl_val = nnUNetDataLoader2DBBOX(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None,
                                        footprint=5, bbox_dilation=20)
        else:
            raise NotImplementedError("3D dataloader not implemented")
        return dl_tr, dl_val

    def modify_output(self, output, bbox_mask):
        return output

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            bbox_mask = target[0][:,-1:].to(self.device, non_blocking=True)
            target = [i[:,:-1].to(self.device, non_blocking=True) for i in target]
        else:
            bbox_mask = target[:,-1:].to(self.device, non_blocking=True)
            target = target[:,:-1].to(self.device, non_blocking=True)

        data = torch.cat([data, bbox_mask], dim=1)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data

            output = self.modify_output(output, bbox_mask)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            bbox_mask = target[0][:,-1:].to(self.device, non_blocking=True)
            target = [i[:,:-1].to(self.device, non_blocking=True) for i in target]
        else:
            bbox_mask = target[:,-1:].to(self.device, non_blocking=True)
            target = target[:,:-1].to(self.device, non_blocking=True)

        data = torch.cat([data, bbox_mask], dim=1)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            output = self.modify_output(output, bbox_mask)
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


class nnUNetTrainerCPU_ft_SAM(nnUNetTrainerCPU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3


class nnUNetTrainerCPU_Oversample(nnUNetTrainerCPU):
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        modalities = {"CT": 0, "Dermoscopy": 0, "Endoscopy": 0, "Fundus": 0, "MR": 0, "Mammo": 0, "Microscopy": 0, "OCT": 0, "PET": 0, "US": 0, "XRay": 0}
        for k in dataset_tr.keys():
            modalities[k.split("_")[0]] += 1
        assert len(dataset_tr.keys()) == sum(modalities.values()), "Some modalities are not accounted for"
        sampling_probabilities = np.array([1.0 / np.sqrt(modalities[k.split("_")[0]]) for k in dataset_tr.keys()])
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DBBOX(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=sampling_probabilities, pad_sides=None,
                                       footprint=5, bbox_dilation=20)
            dl_val = nnUNetDataLoader2DBBOX(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None,
                                        footprint=5, bbox_dilation=20)
        else:
            raise NotImplementedError("3D dataloader not implemented")
        return dl_tr, dl_val


class nnUNetTrainerCPUDA5(nnUNetTrainerCPU, nnUNetTrainerDA5):
    pass


class nnUNetTrainerCPU_BBoxLimitedLoss(nnUNetTrainerCPU):
    def modify_output(self, output, bbox_mask):
        downsampler_for_ds = [ds for ds in self.dataloader_train.transform.transforms if isinstance(ds, DownsampleSegForDSTransform2)][0]
        ds_bbox_mask = downsampler_for_ds(**{downsampler_for_ds.input_key: bbox_mask.cpu().numpy()})[downsampler_for_ds.output_key]
        return [o * torch.from_numpy(b).to(torch.bool).to(o.device).expand_as(o) for o, b in zip(output, ds_bbox_mask)]


class nnUNetTrainerCPU_FocalLoss(nnUNetTrainerCPU):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("Focal loss not implemented for region based training")
        else:
            loss = DC_and_Focal_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerCPU_ADAMW(nnUNetTrainerCPU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = ...):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.betas = (0.9, 0.999)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            betas=self.betas,
            eps=1e-04,  # default of 1e-8 may cause nan's for fp16
            weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerCPUDA5_ADAMW(nnUNetTrainerCPU_ADAMW, nnUNetTrainerCPUDA5):
    pass


class nnUNetTrainerCPU_LatePrompt(nnUNetTrainerCPU):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = ...):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        if enable_deep_supervision is not None and 'deep_supervision' not in arch_init_kwargs.keys():
            arch_init_kwargs['deep_supervision'] = enable_deep_supervision

        network = PromptableUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class nnUNetTrainerCPU_LatePromptResEnc(nnUNetTrainerCPU_LatePrompt):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        if enable_deep_supervision is not None and 'deep_supervision' not in arch_init_kwargs.keys():
            arch_init_kwargs['deep_supervision'] = enable_deep_supervision

        network = PromptableResEncUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network


class nnUNetTrainerCPU_LatePromptResEnc2HEAD(nnUNetTrainerCPU_LatePromptResEnc):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = ...):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.logger = nnUNetLogger2HEAD()

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        if enable_deep_supervision is not None and 'deep_supervision' not in arch_init_kwargs.keys():
            arch_init_kwargs['deep_supervision'] = enable_deep_supervision

        network = PromptableResEncUNetMultiHead(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            return_unet_head=True,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize'):
            network.apply(network.initialize)

        return network

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2DBBOX2HEAD(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None,
                                       footprint=5, bbox_dilation=20)
            dl_val = nnUNetDataLoader2DBBOX2HEAD(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None,
                                        footprint=5, bbox_dilation=20)
        else:
            raise NotImplementedError("3D dataloader not implemented")
        return dl_tr, dl_val

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            bbox_mask = target[0][:,-1:].to(self.device, non_blocking=True)
            target_prompt = [i[:,-2:-1].to(self.device, non_blocking=True) for i in target]
            target_full = [i[:,:-2].to(self.device, non_blocking=True) for i in target]
        else:
            bbox_mask = target[:,-1:].to(self.device, non_blocking=True)
            target_prompt = target[:,-2:-1].to(self.device, non_blocking=True)
            target_full = target[:,:-2].to(self.device, non_blocking=True)

        data = torch.cat([data, bbox_mask], dim=1)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_promt, output_full = self.network(data)
            # del data

            l_prompt = self.loss(output_promt, target_prompt)
            l_full = self.loss(output_full, target_full)
            l = l_prompt + l_full

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss_prompt': l_prompt.detach().cpu().numpy(), 'loss_full': l_full.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            bbox_mask = target[0][:,-1:].to(self.device, non_blocking=True)
            target_prompt = [i[:,-2:-1].to(self.device, non_blocking=True) for i in target]
            target_full = [i[:,:-2].to(self.device, non_blocking=True) for i in target]
        else:
            bbox_mask = target[:,-1:].to(self.device, non_blocking=True)
            target_prompt = target[:,-2:-1].to(self.device, non_blocking=True)
            target_full = target[:,:-2].to(self.device, non_blocking=True)

        data = torch.cat([data, bbox_mask], dim=1)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output_prompt, output_full = self.network(data)
            del data

            l_prompt = self.loss(output_prompt, target_prompt)
            l_full = self.loss(output_full, target_full)
            l = l_prompt + l_full

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output_prompt[0]
            target = target_prompt[0]
        else:
            output = output_prompt
            target = target_prompt

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss_prompt': l_prompt.detach().cpu().numpy(), 'loss_full': l_full.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr_prompt = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr_prompt, outputs['loss_prompt'])
            loss_here_prompt = np.vstack(losses_tr_prompt).mean()
            losses_tr_full = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr_full, outputs['loss_full'])
            loss_here_full = np.vstack(losses_tr_full).mean()
        else:
            loss_here_prompt = np.mean(outputs['loss_prompt'])
            loss_here_full = np.mean(outputs['loss_full'])

        self.logger.log('train_losses_prompt', loss_here_prompt, self.current_epoch)
        self.logger.log('train_losses_full', loss_here_full, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val_prompt = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val_prompt, outputs_collated['loss_prompt'])
            loss_here_prompt = np.vstack(losses_val_prompt).mean()
            losses_val_full = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val_full, outputs_collated['loss_full'])
            loss_here_full = np.vstack(losses_val_full).mean()
        else:
            loss_here_prompt = np.mean(outputs_collated['loss_prompt'])
            loss_here_full = np.mean(outputs_collated['loss_full'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses_prompt', loss_here_prompt, self.current_epoch)
        self.logger.log('val_losses_full', loss_here_full, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss_prompt', np.round(self.logger.my_fantastic_logging['train_losses_prompt'][-1], decimals=4))
        self.print_to_log_file('train_loss_full', np.round(self.logger.my_fantastic_logging['train_losses_full'][-1], decimals=4))
        self.print_to_log_file('val_loss_prompt', np.round(self.logger.my_fantastic_logging['val_losses_prompt'][-1], decimals=4))
        self.print_to_log_file('val_loss_full', np.round(self.logger.my_fantastic_logging['val_losses_full'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1


class nnUNetTrainerCPU_SingleModality_Base(nnUNetTrainerCPU):
    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetFineTuiningDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0, modality=self.modality)
        dataset_val = nnUNetFineTuiningDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0, modality=self.modality)
        return dataset_tr, dataset_val


class nnUNetTrainerCPU_FineTune_Base(nnUNetTrainerCPU_SingleModality_Base):
    def configure_optimizers(self):
        self.initial_lr = 1e-3
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRSchedulerWarmUp(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerCPU_LowLR(nnUNetTrainerCPU):
    def configure_optimizers(self):
        self.initial_lr = 1e-3
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRSchedulerWarmUp(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


# Look at this sick implementation as many classes as you want in 8 lines of code
MODALITIES = ['CT', 'Dermoscopy', 'Endoscopy', 'Fundus', 'Mammo', 'Microscopy', 'MR', 'OCT', 'PET', 'US', 'XRay']

for modality in MODALITIES:
    class_name = f"nnUNetTrainerCPU_FineTune_{modality}"
    class_dict = {"modality": modality}
    class_obj = type(class_name, (nnUNetTrainerCPU_FineTune_Base,), class_dict)
    globals()[class_name] = class_obj


for modality in MODALITIES:
    class_name = f"nnUNetTrainerCPU_SingleModality_{modality}"
    class_dict = {"modality": modality}
    class_obj = type(class_name, (nnUNetTrainerCPU_SingleModality_Base,), class_dict)
    globals()[class_name] = class_obj