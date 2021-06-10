import pydantic
import mdtraj
import pathlib


class Config(pydantic.BaseModel):
    class Model(pydantic.BaseModel):
        name: str
        noise_dim: int = 16
        n_atoms_aa: int = 53
        n_atoms_cg: int = 5
        n_specnorm: int = 7
        n_specnorm_generator: int = 0
        fac_generator: float = 4
        fac_critic: float = 4
        upsampling_model: bool = False

    model: Model

    class Training(pydantic.BaseModel):
        batch_size: int = 64
        n_pretrain_generator: int = 2500  # 2500
        n_pretrain_critic: int = 0
        n_epochs: int = 60000
        n_critic: int = 7
        n_save: int = 1000
        n_val: int = 20
        n_samples: int = 1000
        n_keep_ckpt: int = 3
        split_fraction: float = 0.8
        output_directory: str = ""
        epsilon: float = 1e-3

    training: Training

    class Grid(pydantic.BaseModel):
        resolution: int = 16
        length: float = 2.0
        sigma: float = 0.005

        @property
        def delta_s(self):
            return self.length / self.resolution

    grid: Grid

    class Data(pydantic.BaseModel):
        path: str
        aa_traj: str
        aa_pdb: str
        cg_traj: str
        cg_pdb: str
        lagtime: int = 3
        noise: float = 0

        def load_aa(self):
            return mdtraj.load(
                str(pathlib.Path(self.path).joinpath(self.aa_traj)),
                top=str(pathlib.Path(self.path).joinpath(self.aa_pdb)),
            )

        def load_cg(self):
            return mdtraj.load(
                str(pathlib.Path(self.path).joinpath(self.cg_traj)),
                top=str(pathlib.Path(self.path).joinpath(self.cg_pdb)),
            )

        @property
        def aa_pdb_fname(self):
            return str(pathlib.Path(self.path).joinpath(self.aa_pdb))

    data: Data

    class Opt(pydantic.BaseModel):
        optimizer_type: str = "rmsprop"
        lr_generator: float = 0.00005
        lr_generator_pretrain: float = 0.00005
        lr_critic: float = 0.0001

        @pydantic.validator("optimizer_type")
        def optimizer_valid(cls, value):
            valid_optimizers = ("adam", "rmsprop")
            if value not in valid_optimizers:
                raise ValueError(
                    f"Unsupported optimizer type {value}, supported are {valid_optimizers}"
                )
            return value

    opt: Opt

    class Vae(pydantic.BaseModel):
        fac_encoder: float
        fac_decoder: float
        n_specnorm: int = 0
        latent_dim: int
        lr: float
        n_epochs: int = 2
        beta: float = 1.0
        w_e: float
        recon_e: bool
        E_mu: float = 147.53787
        E_std: float = 29.141317
        # E_mu: float = 368.00990310481467
        # E_std: float = 72.68840703499076

    vae: Vae


if __name__ == "__main__":
    cfg = Config(
        model=Config.Model(name="testmodel"),
        training=Config.Training(),
        grid=Config.Grid(),
        data=Config.Data(
            path=".",
            aa_traj="aa.xtc",
            aa_pdb="aa.pdb",
            cg_traj="cg.xtc",
            cg_pdb="cg.pdb",
        ),
        opt=Config.Opt(),
        vae=Config.Vae(),
    )
    print(cfg.json())
