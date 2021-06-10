"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

from pl_vae import VAE
from pytorch_lightning import Trainer, seed_everything, loggers
from pl_callbacks import CheckpointEveryNSteps

seed_everything(42)


def main(args):
    """ Main training routine specific for this project. """
    model = VAE(**vars(args))
    #logger = loggers.WandbLogger(name='ABC', project='test', offline=True)
    logger = loggers.TensorBoardLogger('./', name='logs')
    trainer = Trainer.from_argparse_args(
        args, callbacks=[CheckpointEveryNSteps(2000)], logger=logger)

    #ckpt_file = "N-Step-Checkpoint.ckpt"
    ## ckpt_file = "epoch=20.ckpt"
    #version_num = 78
    #base_fname = f"/home/kirills/Projects/ipam-coarse2fine/c2f/lightning_logs/version_{version_num}"
    #ckpt_fname = f"{base_fname}/checkpoints/{ckpt_file}"

    #model = VAE.load_from_checkpoint(
    #    ckpt_fname, use_recon_energy_loss=True, skip_E_calcs=False
    #)

    #trainer = Trainer(
    #    resume_from_checkpoint=ckpt_fname,
    #    gpus=1,
    #    val_check_interval=0.25,
    #    callbacks=[CheckpointEveryNSteps(2000)],
    #    max_epochs=2000,
    #    # gradient_clip_val=0.1,
    #)
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # each LightningModule defines arguments relevant to it
    parser = VAE.add_model_specific_args(parent_parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1)
    parser.set_defaults(val_check_interval=0.25)
    # parser.set_defaults(limit_val_batches=0.0)
    # parser.set_defaults(distributed_backend="ddp")
    # parser.set_defaults(num_nodes=4)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
