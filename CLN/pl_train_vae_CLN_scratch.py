"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser
import pandas as pd

from pl_vae_CLN import VAE
from pytorch_lightning import Trainer, seed_everything
from pl_callbacks import CheckpointEveryNSteps
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger

# seed_everything(42)


def main(args):
    """ Main training routine specific for this project. """
    seed_everything(42)
    # model = VAE(**vars(args))
    # trainer = Trainer.from_argparse_args(
    #    args,
    #    callbacks=[CheckpointEveryNSteps(2000)],
    #    # distributed_backend="ddp",
    # )

    ckpt_file = "N-Step-Checkpoint.ckpt"
    # ckpt_file = "epoch=6.ckpt"
    version_num = "version_0"
    base_fname = f"/scratch/midway3/kirills/c2f/CLN_twoGPU_beta2_V2/{version_num}"
    ckpt_fname = f"{base_fname}/checkpoints/{ckpt_file}"

    # hparams_fname = f"{base_fname}/hparams.yaml"
    # hparams = load_hparams_from_yaml(hparams_fname)
    # hparams["path"] = "/home/kirills/Projects/ipam-coarse2fine/c2f/ADP_new/"
    # model = VAE(**hparams)

    hparams_fname = f"{base_fname}/meta_tags.csv"
    hparams = read_csv_file(hparams_fname)
    hparams["path"] = "/scratch/midway3/kirills/c2f_eval/chig_new/"
    hparams["GMM_estimate"] = 0
    hparams["beta"] = 1.0
    # hparams["batch_size"] = 16
    hparams["hallucinate_every_n_epochs"] = 150
    # hparams["learning_rate"] = 2 * hparams["learning_rate"]
    model = VAE(**hparams)

    # #model = VAE.load_from_checkpoint(
    # #    ckpt_fname,
    # #    use_recon_energy_loss=True,
    # #    skip_E_calcs=False,
    # #    GMM_estimate=0,
    # #    beta=1.0,
    # #    path="/project2/andrewferguson/Kirill/midway3_c2f/chig_new/",
    # #    # strict=False,
    # #    # n_frames=3,
    # #    # use_diff_loss=True,
    # #)

    # logger = TestTubeLogger("./", name="CLN_distributedTest", version=12)
    logger = TensorBoardLogger(
        "./", name="CLN_twoGPU_beta2_V2", version=f"{version_num}_ConstantBeta"
    )
    trainer = Trainer(
        # resume_from_checkpoint=ckpt_fname,
        gpus=2,
        accelerator="ddp",
        val_check_interval=0.25,
        callbacks=[CheckpointEveryNSteps(2000)],
        max_epochs=3000,
        # gradient_clip_val=0.01,
        logger=logger,
    )
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
    # parser.set_defaults(gradient_clip_val=1.25)
    # parser.set_defaults(limit_val_batches=0.0)
    # parser.set_defaults(distributed_backend="ddp")
    # parser.set_defaults(num_nodes=4)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


def read_csv_file(hparams):
    import csv
    import ast

    with open(hparams, encoding="utf-8") as f:
        reader = csv.reader(f)

        hparams = dict()
        for i, row in enumerate(reader):
            if i != 0:
                k, v = row
                try:
                    hparams[k] = ast.literal_eval(v)
                except:
                    hparams[k] = v

    return hparams


if __name__ == "__main__":
    run_cli()
