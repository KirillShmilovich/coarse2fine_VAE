"""
Runs a model on a single node across multiple gpus.
"""
import os

from pl_vae_CLN import VAE
from pytorch_lightning import Trainer, seed_everything, loggers
from pl_callbacks import CheckpointEveryNSteps

from test_tube import HyperOptArgumentParser, SlurmCluster


def main(args, cluster_manager):
    """ Main training routine specific for this project. """
    seed_everything(42)
    model = VAE(**vars(args))
    # logger = loggers.TensorBoardLogger('./',
    #                                   name='ADP_Decoder7v3',
    #                                   version=cluster_manager.hpc_exp_number)
    logger = loggers.TestTubeLogger(
        "./", name="CLN_single", version=cluster_manager.hpc_exp_number
    )
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[CheckpointEveryNSteps(2000)],
        val_check_interval=0.25,
        gpus=1,
        # accelerator="ddp",
        terminate_on_nan=True,
        # gradient_clip_val=5.0,
        logger=logger,
    )
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


def remove_options(parser, options):
    for option in options:
        for action in parser._actions:
            if vars(action)["option_strings"][0] == option:
                parser._handle_conflict_resolve(None, [(option, action)])
                break


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # parent_parser = ArgumentParser(add_help=False)
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy="random_search")
    parser = VAE.add_model_specific_args(parser, root_dir)
    parser.add_argument("--log_path", default="./")
    remove_options(
        parser,
        [
            "--sigma",
            "--learning_rate",
            "--latent_dim",
            "--resolution",
            "--fac_encoder",
            "--fac_decoder",
            "--batch_size",
            "--length",
            "--noise",
            "--beta",
            "--use_cg_loss",
            "--use_edm_loss",
            "--bonds_edm_weight",
            "--cg_coord_weight",
            "--coord_weight",
            "--use_edm_loss",
            "--learning_gamma",
        ],
    )
    parser.opt_range(
        "--sigma",
        default=0.01,
        low=0.01,
        high=1.0,
        log_base=10,
        nb_samples=16,
        type=float,
        tunable=True,
    )
    parser.opt_range(
        "--learning_rate",
        default=1e-4,
        low=1e-4,
        high=1e-3,
        log_base=10,
        nb_samples=16,
        type=float,
        tunable=True,
    )
    parser.opt_list("--latent_dim", default=32, options=[64], type=int, tunable=True)
    parser.opt_list("--resolution", default=12, options=[12], type=int, tunable=True)
    parser.opt_list("--fac_encoder", default=16, options=[16], type=int, tunable=True)
    parser.opt_list("--fac_decoder", default=16, options=[8], type=int, tunable=True)
    parser.opt_list("--batch_size", default=32, options=[32], type=int, tunable=True)
    parser.opt_list(
        "--learning_gamma", default=1.0, options=[1.0], type=float, tunable=True
    )
    parser.opt_list("--length", default=5.5, options=[5.5], type=float, tunable=True)
    parser.opt_list("--noise", default=0.00, options=[0.0], type=float, tunable=True)
    # parser.opt_range(
    #    "--beta",
    #    default=1.0,
    #    low=1e-4,
    #    high=1.0,
    #    log_base=10,
    #    nb_samples=12,
    #    type=float,
    #    tunable=False,
    # )
    parser.opt_list(
        "--beta",
        default=1.0,
        options=[1.0],
        # default="250000",
        # options=["250000"],
        # default="0.0025,10000,4,25000",
        # options=['10000,4,15000'],
        # options=["0.0025,10000,4,25000"],
        # options=["0.0025,35000,4,80000"],
        type=str,
        tunable=True,
    )
    parser.opt_list("--use_cg_loss", default=1, options=[0, 1], type=int, tunable=False)
    parser.opt_list(
        "--use_edm_loss", default=1, options=[0, 1], type=int, tunable=False
    )
    parser.opt_range(
        "--bonds_edm_weight",
        default=1.0,
        low=0.01,
        high=1.0,
        nb_samples=8,
        type=float,
        tunable=False,
    )
    parser.opt_range(
        "--coord_weight",
        default=1.0,
        low=0.01,
        high=1.0,
        nb_samples=8,
        type=float,
        tunable=False,
    )
    parser.opt_range(
        "--cg_coord_weight",
        default=1.0,
        low=0.01,
        high=1.0,
        nb_samples=6,
        type=float,
        tunable=False,
    )
    hyperparams = parser.parse_args()

    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.log_path,
        python_cmd="/project/andrewferguson/Kirill/torch_kirill/bin/python",
    )

    # Email results if your hpc supports it.
    cluster.notify_job_status(email="kirills@uchicago.edu", on_done=True, on_fail=True)

    # SLURM Module to load.
    cluster.load_modules(["python"])

    # Add commands to the non-SLURM portion.
    # cluster.add_command(
    #    "conda activate /project2/andrewferguson/Kirill/conda_env")

    # Add custom SLURM commands which show up as:
    # #comment
    # #SBATCH --cmd=value
    # ############
    cluster.add_slurm_cmd(
        cmd="partition", value="andrewferguson-gpu", comment="partition"
    )
    cluster.add_slurm_cmd(cmd="account", value="pi-andrewferguson", comment="account")

    # cluster.add_slurm_cmd(cmd="partition", value="gpu", comment="partition")
    # cluster.add_slurm_cmd(cmd="partition",
    #                       value="gm4-pmext",
    #                       comment="partition")
    # cluster.add_slurm_cmd(cmd="qos", value="gm4", comment="qos")
    # cluster.add_slurm_cmd(cmd="account",
    #                       value="early-users",
    #                       comment="partition")

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.per_experiment_nb_cpus = 10
    cluster.memory_mb_per_node = 17000
    cluster.minutes_to_checkpoint_before_walltime = 0
    # cluster.job_time = "12:00:00"
    cluster.job_time = "168:00:00"

    # Each hyperparameter combination will use 8 gpus.
    cluster.optimize_parallel_cluster_gpu(
        # Function to execute:
        main,
        # Number of hyperparameter combinations to search:
        nb_trials=4,
        # This is w will display in the slurm queue:
        job_name="CLN_single",
    )
