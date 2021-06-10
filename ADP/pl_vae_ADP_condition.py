"""
Example template for defining a system.
"""
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule

from data import C2FDataSet3
import mdtraj as md
import pathlib
from models import Encoder3, Decoder7
from openmmBridge import OpenMMEnergy
from util import avg_blob, make_grid_np, voxel_gauss, to_distmat
import numpy as np
from tqdm import tqdm


class VAE(LightningModule):
    """
    Sample model to show how to define a template.
    Example:
        >>> # define simple Net for MNIST dataset
        >>> params = dict(
        ...     in_features=28 * 28,
        ...     hidden_dim=1000,
        ...     out_features=10,
        ...     drop_prob=0.2,
        ...     learning_rate=0.001 * 8,
        ...     batch_size=2,
        ...     data_root='./datasets',
        ...     num_workers=4,
        ... )
        >>> model = LightningTemplateModel(**params)
    """
    def __init__(
        self,
        path,
        aa_traj,
        aa_pdb,
        cg_traj,
        cg_pdb,
        n_atoms_aa,
        n_atoms_cg,
        sigma,
        resolution,
        length,
        noise,
        n_frames,
        num_workers,
        batch_size,
        learning_rate,
        n_frames_past,
        latent_dim,
        fac_encoder,
        fac_decoder,
        train_percent,
        E_mu,
        E_std,
        save_every_n_steps,
        hallucinate_every_n_epochs,
        use_recon_energy_loss,
        use_diff_loss,
        use_edm_loss,
        use_coord_loss,
        use_cg_loss,
        bonds_edm_weight,
        cg_coord_weight,
        coord_weight,
        GMM_estimate,
        beta,
        learning_gamma,
        gamma,
        default_save_path,
        clip_E,
        **kwargs,
    ):
        # init superclass
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()
        delta_s = self.hparams.length / self.hparams.resolution
        self.grid = make_grid_np(delta_s, self.hparams.resolution)
        self.grid_shape = (self.hparams.resolution, ) * 3
        self.build_model()

    def build_model(self):

        # n_channels_encoder = (self.hparams.n_frames_past + 1) * (
        #    self.hparams.n_atoms_aa + self.hparams.n_atoms_cg
        # )
        # n_channels_decoder = n_channels_encoder - self.hparams.n_atoms_aa

        # n_channels_encoder = self.hparams.n_atoms_aa + self.hparams.n_atoms_cg
        # n_channels_decoder = self.hparams.n_atoms_cg

        # n_channels_encoder = 2 * self.hparams.n_atoms_aa + self.hparams.n_atoms_cg
        # n_channels_decoder = self.hparams.n_atoms_cg + self.hparams.n_atoms_aa
        n_channels_encoder = (
            self.hparams.n_frames_past +
            1) * self.hparams.n_atoms_aa + self.hparams.n_atoms_cg
        n_channels_decoder = n_channels_encoder - self.hparams.n_atoms_aa

        self.encoder = Encoder3(
            self.hparams.resolution,
            self.hparams.resolution,
            self.hparams.resolution,
            in_channels=n_channels_encoder,
            latent_dim=self.hparams.latent_dim,
            fac=self.hparams.fac_encoder,
            sn=0,
            device=self.device,
        )

        self.decoder = Decoder7(
            z_dim=self.hparams.latent_dim,
            condition_n_channels=n_channels_decoder,
            fac=self.hparams.fac_decoder,
            out_channels=self.hparams.n_atoms_aa,
            resolution=self.hparams.resolution,
            sn=0,
            device=self.device,
        )

        hidden_dim = 256
        self.Emlp = nn.Sequential(
            nn.Linear(self.hparams.latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

        aa_pdb_fname = str(
            pathlib.Path(self.hparams.path).joinpath(self.hparams.aa_pdb))
        self.ELoss = OpenMMEnergy(self.hparams.n_atoms_aa * 3, aa_pdb_fname)

    def encode(self, input):
        mu, logvar = self.encoder(input)
        return (mu, logvar)

    def decode(self, z, condition):
        result = self.decoder(z, condition)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def cg_func(self, aa_fake):
        # BUTANE
        # bead_0 = aa_fake[:, self.bead_0_idxs, ...].mean(axis=1, keepdim=True)
        # bead_1 = aa_fake[:, self.bead_1_idxs, ...].mean(axis=1, keepdim=True)
        # CG = torch.cat((bead_0, bead_1), axis=1)

        # ADP
        CG = aa_fake[:, self.ADP_cg_idxs, ...]
        # Chignolin
        # CG = aa_fake[:, self.Chignolin_cg_idxs, ...]
        return CG

    def forward(self, encoder_input, condition):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        mu, logvar = self.encode(encoder_input)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, condition)
        return (output, mu, logvar, z)

    def hallucinate_forward(
        self,
        aa_vox_past,
        cg_vox_past,
        cg_vox_current,
        aa_vox_current,
        n_steps,
    ):
        bs = aa_vox_current.size(0)
        dtype = aa_vox_current.dtype
        hallucinated = torch.empty((bs, n_steps, *aa_vox_current.shape[1:]),
                                   dtype=dtype,
                                   device=self.device)
        hallucinated_coords = torch.empty(
            (bs, n_steps, self.hparams.n_atoms_aa, 3),
            dtype=dtype,
            device=self.device)
        mus = torch.empty((bs, n_steps, self.hparams.latent_dim),
                          dtype=dtype,
                          device=self.device)
        logvars = torch.empty((bs, n_steps, self.hparams.latent_dim),
                              dtype=dtype,
                              device=self.device)
        zs = torch.empty((bs, n_steps, self.hparams.latent_dim),
                         dtype=dtype,
                         device=self.device)

        for i in range(n_steps):
            # condition = torch.cat((cg_vox_past, aa_vox_past, cg_vox_current), dim=1)

            # condition = torch.cat((cg_vox_past, aa_vox_past), dim=1)

            condition = torch.cat((cg_vox_current, aa_vox_past), dim=1)

            encoder_input = torch.cat((condition, aa_vox_current), dim=1)
            (recon_aa_vox, mu, logvar,
             z) = self.forward(encoder_input, condition)
            hallucinated[:, i, ...] = recon_aa_vox
            mus[:, i, ...] = mu
            logvars[:, i, ...] = logvar
            zs[:, i, ...] = z

            aa_fake = avg_blob(
                recon_aa_vox,
                res=self.hparams.resolution,
                width=self.hparams.length,
                sigma=self.hparams.sigma,
                device=self.device,
            )
            hallucinated_coords[:, i, ...] = aa_fake

            # cg_fake = aa_fake[:, self.alpha_carbon_idxs, ...]
            cg_fake = self.cg_func(aa_fake)
            aa_vox_recon = voxel_gauss(
                aa_fake,
                res=self.hparams.resolution,
                width=self.hparams.length,
                sigma=self.hparams.sigma,
                device=self.device,
            )
            cg_vox_recon = voxel_gauss(
                cg_fake,
                res=self.hparams.resolution,
                width=self.hparams.length,
                sigma=self.hparams.sigma,
                device=self.device,
            )

            aa_vox_past = torch.cat((aa_vox_past, aa_vox_current),
                                    dim=1)[:, self.hparams.n_atoms_aa:, ...]
            cg_vox_past = torch.cat((cg_vox_past, cg_vox_current),
                                    dim=1)[:, self.hparams.n_atoms_cg:, ...]
            cg_vox_current = cg_vox_recon
            aa_vox_current = aa_vox_recon

        return (hallucinated, hallucinated_coords, mus, logvars, zs)

    def process_batch(self, batch):
        aa, aa_vox, cg, cg_vox = batch
        bs = aa.size(0)

        aa = aa.view(bs, -1, self.hparams.n_atoms_aa, 3)
        cg = cg.view(bs, -1, self.hparams.n_atoms_cg, 3)
        aa_vox = aa_vox.view(bs, -1, self.hparams.n_atoms_aa, *self.grid_shape)
        cg_vox = cg_vox.view(bs, -1, self.hparams.n_atoms_cg, *self.grid_shape)

        aa_past = aa[:, :self.hparams.n_frames_past, ...]
        aa_future = aa[:, self.hparams.n_frames_past:, ...]
        aa_vox_past = aa_vox[:, :self.hparams.n_frames_past, ...]
        aa_vox_current = aa_vox[:, self.hparams.n_frames_past, ...]
        aa_vox_future = aa_vox[:, self.hparams.n_frames_past:, ...]

        cg_past = cg[:, :self.hparams.n_frames_past, ...]
        cg_future = cg[:, self.hparams.n_frames_past:, ...]
        cg_vox_past = cg_vox[:, :self.hparams.n_frames_past, ...]
        cg_vox_current = cg_vox[:, self.hparams.n_frames_past, ...]
        cg_vox_future = cg_vox[:, self.hparams.n_frames_past:, ...]
        return (
            aa_past,
            aa_future,
            aa_vox_past,
            aa_vox_current,
            aa_vox_future,
            cg_past,
            cg_vox_past,
            cg_vox_current,
            cg_future,
            cg_vox_future,
        )

    def loss_hallucinate(self, batch):
        bs = batch[0].size(0)
        future_steps = self.hparams.n_frames - self.hparams.n_frames_past
        (
            aa_past,
            aa_future,
            aa_vox_past,
            aa_vox_current,
            aa_vox_future,
            cg_past,
            cg_vox_past,
            cg_vox_current,
            cg_future,
            cg_vox_future,
        ) = self.process_batch(batch)

        (
            hallucinate_vox,
            hallucinate_coords,
            mus,
            logvars,
            zs,
        ) = self.hallucinate_forward(
            aa_vox_past.view(bs, -1, *self.grid_shape),
            cg_vox_past.view(bs, -1, *self.grid_shape),
            cg_vox_current.view(bs, -1, *self.grid_shape),
            aa_vox_current.view(bs, -1, *self.grid_shape),
            n_steps=future_steps,
        )

        def mean_center(x):
            return x - x.mean(axis=(-1, -2), keepdims=True)

        voxel_loss = F.mse_loss(hallucinate_vox, aa_vox_future)
        coords_loss = F.mse_loss(mean_center(hallucinate_coords), aa_future)
        recon_loss = voxel_loss + coords_loss

        hallucinated_CG_coords = self.cg_func(
            hallucinate_coords.view(bs * future_steps, self.hparams.n_atoms_aa,
                                    3))
        hallucinated_CG_vox = voxel_gauss(
            hallucinated_CG_coords,
            res=self.hparams.resolution,
            width=self.hparams.length,
            sigma=self.hparams.sigma,
            device=self.device,
        )
        cg_voxel_loss = F.mse_loss(
            hallucinated_CG_vox,
            cg_vox_future.reshape(bs * future_steps, self.hparams.n_atoms_cg,
                                  *self.grid_shape),
        )
        cg_coord_loss = F.mse_loss(
            mean_center(hallucinated_CG_coords),
            cg_future.reshape(bs * future_steps, self.hparams.n_atoms_cg, 3),
        )

        hallucinate_edm = to_distmat(
            hallucinate_coords.reshape(bs * future_steps,
                                       self.hparams.n_atoms_aa, 3))
        real_edm = to_distmat(
            aa_future.reshape(bs * future_steps, self.hparams.n_atoms_aa, 3))
        edm_loss = F.mse_loss(hallucinate_edm, real_edm)

        real_bonds_edm = real_edm[:, self.bond_idxs[:, 0], self.bond_idxs[:,
                                                                          1]]
        hallucinate_bonds_edm = hallucinate_edm[:, self.bond_idxs[:, 0],
                                                self.bond_idxs[:, 1]]
        bonds_edm_loss = F.mse_loss(hallucinate_bonds_edm, real_bonds_edm)

        def diff(x):
            return x[:, 1:] - x[:, :-1]

        voxel_diff = F.mse_loss(diff(hallucinate_vox), diff(aa_vox_future))
        coords_diff = F.mse_loss(diff(hallucinate_coords), diff(aa_future))
        diff_loss = (voxel_diff + coords_diff) / 2.0

        mu = mus.view(bs * future_steps, self.hparams.latent_dim)
        logvar = logvars.view(-1, self.hparams.latent_dim)
        KLD_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1),
            dim=0)
        KLD_weight = mu.size(0) / len(self.ds_train)

        if self.global_step >= 175000:
            self.trainer.gradient_clip_val = self.hparams.clip_E
            self.hparams.use_recon_energy_loss = True
            self.hparams.skip_E_calcs = False

            if hasattr(self, "recon_energy_loss_weight"):
                if self.training:
                    # goal = 5e-2 # 750K
                    # factor = 1.0000113563220710922852453836718548440896019691959288165191

                    # goal = 1e-1 # 1M
                    # factor = 1.0000115129917389509449561379274639104559958866285946533811

                    # goal = 5e-2  # 500K
                    # factor = (
                    #    1.0000170345314688160103988560411565154983591534774133574086100524
                    # )

                    goal = 1.0  # 1.25M
                    factor = (
                        1.0000092103827872912862930047032391734439796534302560512742
                    )

                    self.recon_energy_loss_weight *= factor
                    if self.recon_energy_loss_weight > goal:
                        self.recon_energy_loss_weight = goal
            else:
                self.recon_energy_loss_weight = 1e-5

        if self.hparams.skip_E_calcs:
            energy_loss = torch.tensor([0.0], device=self.device)
        else:
            energies = (
                self.ELoss.energy(aa_future.reshape(bs * future_steps, -1)) -
                self.hparams.E_mu) / self.hparams.E_std
            z = zs.view(bs * future_steps, self.hparams.latent_dim)
            pred_energies = self.Emlp(z)
            energy_loss = F.mse_loss(pred_energies, energies)

            recon_energies = (self.ELoss.energy(
                hallucinate_coords.reshape(bs * future_steps, -1)) -
                              self.hparams.E_mu) / self.hparams.E_std
            recon_energy_loss = torch.nn.functional.mse_loss(recon_energies,
                                                             energies,
                                                             reduction="none")

        if isinstance(self.hparams.beta, str):

            beta_params = self.hparams.beta.split(",")
            if len(beta_params) == 4:
                k, x0, M, R = [float(f) for f in beta_params]
                if self.global_step < (M * R):
                    tau = np.mod(self.global_step, R)
                    beta_weight = float(1.0 / (1.0 + np.exp(-k * (tau - x0))))
                else:
                    beta_weight = 1.0
            if len(beta_params) == 3:
                T, M, R = [float(f) for f in beta_params]
                if self.global_step < (M * R):
                    tau = np.mod(self.global_step, R)
                    beta_weight = min(tau / T, 1.0)
                else:
                    beta_weight = 1.0
            if len(beta_params) == 2:
                k, x0 = beta_params
                beta_weight = float(
                    1.0 / (1.0 + np.exp(-float(k) *
                                        (self.global_step - float(x0)))))
            if len(beta_params) == 1:
                R = float(beta_params[0])
                beta_weight = min(self.global_step / R, 1.0)

            # loss = (voxel_loss + beta_weight * KLD_loss +
            #        self.hparams.gamma * energy_loss)
            loss = (voxel_loss + beta_weight * KLD_weight * KLD_loss +
                    self.hparams.gamma * energy_loss)
        else:
            loss = (voxel_loss + self.hparams.beta * KLD_weight * KLD_loss +
                    self.hparams.gamma * energy_loss)

        if self.hparams.use_recon_energy_loss:
            #recon_energy_loss = recon_energy_loss.clamp_max(1e5).mean()
            recon_energy_loss = recon_energy_loss.mean()
            loss = loss + self.recon_energy_loss_weight * recon_energy_loss
            # loss = loss + 1e-4 * recon_energy_loss
            # loss = loss + torch.log(recon_energy_loss)
        else:
            if self.hparams.skip_E_calcs:
                recon_energy_loss = torch.tensor([0.0], device=self.device)
        if self.hparams.use_diff_loss:
            loss = loss + diff_loss
        if self.hparams.use_edm_loss:
            loss = (loss + self.hparams.bonds_edm_weight * edm_loss / 2.0
                    )  # + self.hparams.bonds_edm_weight * bonds_edm_loss
            # loss = loss + self.hparams.bonds_edm_weight * bonds_edm_loss / 2.
        if self.hparams.use_coord_loss:
            loss = loss + self.hparams.coord_weight * (coords_loss)  # +
            # cg_coord_loss)
        if self.hparams.use_cg_loss:
            loss = (loss + self.hparams.cg_coord_weight * cg_coord_loss
                    )  # + cg_voxel_loss

        loss_dict = {
            "loss": loss,
            "KLD": KLD_loss,
            "recon": recon_loss,
            "VOX": voxel_loss,
            "COORD": coords_loss,
            "Energy_latent": energy_loss,
            "Energy_recon": recon_energy_loss,
            "diff": diff_loss,
            "EDM": edm_loss,
            "bonds_EDM": bonds_edm_loss,
            "CG_coord": cg_coord_loss,
            "CG_voxel": cg_voxel_loss,
        }
        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        # self.is_training = True
        loss_dict = self.loss_hallucinate(batch)
        # self.is_training = False
        loss = loss_dict["loss"]
        tensorboard_logs = {("train/" + key): val
                            for key, val in loss_dict.items()}

        return {"loss": loss, "log": tensorboard_logs}

    def on_train_epoch_start(self, ):
        print(f"Optimizer state after epoch {self.current_epoch}...")
        print(self.optimizers())

    def on_train_epoch_end(self, _):
        if self.current_epoch == 0:
            return
        if self.current_epoch % self.hparams.hallucinate_every_n_epochs == 0:
            test_loader = self.test_dataloader(shuffle=False)
            fake_coords = list()
            real_coords = list()
            if self.hparams.GMM_estimate:
                self.estimate_latent_space_density()

            with torch.no_grad():
                for batch_idx, batch in tqdm(
                        enumerate(test_loader),
                        total=len(test_loader),
                        leave=False,
                        desc="Hallucinating",
                ):
                    batch = self.transfer_batch_to_device(batch, self.device)
                    aa, aa_vox, cg, cg_vox = batch
                    if batch_idx < self.hparams.n_frames_past:
                        if batch_idx == 0:
                            aa_vox_past = aa_vox
                            cg_vox_past = cg_vox
                        else:
                            aa_vox_past = torch.cat((aa_vox_past, aa_vox),
                                                    dim=1)
                            cg_vox_past = torch.cat((cg_vox_past, cg_vox),
                                                    dim=1)

                        cg_vox_current = cg_vox
                        aa_vox_current = aa_vox
                    else:
                        condition = torch.cat((cg_vox, aa_vox_current), dim=1)
                        # condition = torch.cat((cg_vox_current, aa_vox_past),
                        #                      dim=1)

                        if self.hparams.GMM_estimate:
                            z = self.sample_GMM_latent_space()
                        else:
                            z = torch.empty(
                                (1, self.hparams.latent_dim),
                                dtype=condition.dtype,
                                device=self.device,
                            ).normal_()

                        recon_aa_vox = self.decode(z, condition)

                        aa_fake = avg_blob(
                            recon_aa_vox,
                            res=self.hparams.resolution,
                            width=self.hparams.length,
                            sigma=self.hparams.sigma,
                            device=self.device,
                        )
                        fake_coords.append(aa_fake)
                        real_coords.append(aa)

                        # cg_fake = aa_fake[:, self.alpha_carbon_idxs, ...]
                        cg_fake = self.cg_func(aa_fake)
                        aa_vox_recon = voxel_gauss(
                            aa_fake,
                            res=self.hparams.resolution,
                            width=self.hparams.length,
                            sigma=self.hparams.sigma,
                            device=self.device,
                        )
                        cg_vox_recon = voxel_gauss(
                            cg_fake,
                            res=self.hparams.resolution,
                            width=self.hparams.length,
                            sigma=self.hparams.sigma,
                            device=self.device,
                        )
                        cg_vox_current = cg_vox_recon
                        aa_vox_current = aa_vox_recon

                        aa_vox_past = torch.cat(
                            (aa_vox_past, aa_vox_recon),
                            dim=1)[:, self.hparams.n_atoms_aa:, ...]
                        cg_vox_past = torch.cat(
                            (cg_vox_past, cg_vox_recon),
                            dim=1)[:, self.hparams.n_atoms_cg:, ...]

                hallucinate_coords = torch.cat(fake_coords, dim=0)
                real_coords = torch.cat(real_coords, dim=0)

                hallucinate_trj = md.Trajectory(
                    hallucinate_coords.detach().cpu().numpy(),
                    topology=self.aa_traj.top)
                real_trj = md.Trajectory(real_coords.detach().cpu().numpy(),
                                         topology=self.aa_traj.top)
                samples_path = (pathlib.Path(self.logger.save_dir).joinpath(
                    self.logger.name).joinpath(
                        f"version_{self.logger.version}").joinpath("samples"))
                samples_path.mkdir(parents=True, exist_ok=True)

                print(f"Saving hallucination on step {self.global_step}...")
                # hallucinate_trj.save_pdb(
                #    str(samples_path / f"hallucination_{self.global_step}.pdb")
                # )
                # real_trj.save_pdb(
                #    str(samples_path / f"hallucination_real_{self.global_step}.pdb")
                # )
                hallucinate_trj.save_pdb(
                    str(samples_path / f"hallucination.pdb"))
                real_trj.save_pdb(str(samples_path /
                                      f"hallucination_real.pdb"))

    def on_train_batch_end(self, _, batch, batch_idx, dataloader_idx):
        if self.global_step % self.hparams.save_every_n_steps == 0:
            batch = next(iter(self.val_dataloader(shuffle=True)))
            batch = self.transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                bs = batch[0].size(0)
                future_steps = self.hparams.n_frames - self.hparams.n_frames_past
                (
                    aa_past,
                    aa_future,
                    aa_vox_past,
                    aa_vox_current,
                    aa_vox_future,
                    cg_past,
                    cg_vox_past,
                    cg_vox_current,
                    cg_future,
                    cg_vox_future,
                ) = self.process_batch(batch)

                (
                    hallucinate_vox,
                    hallucinate_coords,
                    mus,
                    logvars,
                    zs,
                ) = self.hallucinate_forward(
                    aa_vox_past.view(bs, -1, *self.grid_shape),
                    cg_vox_past.view(bs, -1, *self.grid_shape),
                    cg_vox_current.view(bs, -1, *self.grid_shape),
                    aa_vox_current.view(bs, -1, *self.grid_shape),
                    n_steps=future_steps,
                )
                fake_coords = hallucinate_coords.reshape(
                    bs * future_steps, self.hparams.n_atoms_aa, 3)
                real_coords = aa_future.reshape(bs * future_steps,
                                                self.hparams.n_atoms_aa, 3)

                real_trj = md.Trajectory(np.array(real_coords.detach().cpu()),
                                         topology=self.aa_traj.top)
                fake_trj = md.Trajectory(np.array(fake_coords.detach().cpu()),
                                         topology=self.aa_traj.top)

                # condition = torch.cat(
                #     (
                #         cg_vox_current.view(bs, -1, *self.grid_shape),
                #         aa_vox_past.view(bs, -1, *self.grid_shape),
                #     ),
                #     dim=1,
                # )
                condition = torch.cat(
                    (
                        cg_vox_current.view(bs, -1, *self.grid_shape),
                        aa_vox_past.view(bs, -1, *self.grid_shape),
                    ),
                    dim=1,
                )

                repeats = 100
                z = torch.empty(
                    [repeats, self.hparams.latent_dim],
                    dtype=condition.dtype,
                    device=self.device,
                ).normal_()
                c = torch.repeat_interleave(condition[-1].unsqueeze(0),
                                            repeats=repeats,
                                            dim=0)
                fake = self.decoder(z, c)
                fake_coords = avg_blob(
                    fake,
                    res=self.hparams.resolution,
                    width=self.hparams.length,
                    sigma=self.hparams.sigma,
                    device=self.device,
                )

                condition_coords = torch.repeat_interleave(
                    aa_past[-1][-1].unsqueeze(0),
                    repeats=repeats,
                    dim=0,
                )

                z_trj = md.Trajectory(np.array(fake_coords.detach().cpu()),
                                      topology=self.aa_traj.top)
                condition_trj = md.Trajectory(np.array(
                    condition_coords.detach().cpu()),
                                              topology=self.aa_traj.top)

                samples_path = (pathlib.Path(self.logger.save_dir).joinpath(
                    self.logger.name).joinpath(
                        f"version_{self.logger.version}").joinpath("samples"))
                samples_path.mkdir(parents=True, exist_ok=True)

                print(f"Saving samples on step {self.global_step}")
                # real_trj.save_pdb(
                #    str(samples_path / f"real_step_{self.global_step}.pdb")
                # )
                # fake_trj.save_pdb(
                #    str(samples_path / f"fake_step_{self.global_step}.pdb")
                # )
                # z_trj.save_pdb(str(samples_path / f"z_samples_{self.global_step}.pdb"))
                real_trj.save_pdb(str(samples_path / f"real_step.pdb"))
                fake_trj.save_pdb(str(samples_path / f"fake_step.pdb"))
                z_trj.save_pdb(str(samples_path / f"z_samples.pdb"))
                condition_trj.save_pdb(str(samples_path / f"condition.pdb"))

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        loss_dict = self.loss_hallucinate(batch)
        tensorboard_logs = {("val/" + key): val
                            for key, val in loss_dict.items()}
        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        keys = outputs[0].keys()
        tensorboard_logs = {
            key: torch.stack([x[key] for x in outputs]).mean()
            for key in keys
        }
        avg_loss = tensorboard_logs["val/loss"]
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def estimate_latent_space_density(self):

        dataloader = self.train_dataloader()
        latent_codes = list()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    total=len(dataloader),
                    leave=False,
                    desc="Estimating latent space density",
            ):
                batch = self.transfer_batch_to_device(batch, self.device)
                (
                    aa_past,
                    aa_future,
                    aa_vox_past,
                    aa_vox_current,
                    aa_vox_future,
                    cg_past,
                    cg_vox_past,
                    cg_vox_current,
                    cg_future,
                    cg_vox_future,
                ) = self.process_batch(batch)
                bs = batch[0].size(0)

                # condition = torch.cat(
                #     (
                #         cg_vox_current.view(bs, self.hparams.n_atoms_cg, *
                #                             self.grid_shape),
                #         aa_vox_past.view(bs, self.hparams.n_atoms_aa, *
                #                          self.grid_shape),
                #     ),
                #     dim=1,
                # )
                condition = torch.cat(
                    (
                        cg_vox_current.view(bs, self.hparams.n_atoms_cg, *
                                            self.grid_shape),
                        aa_vox_past.view(bs, self.hparams.n_atoms_aa, *
                                         self.grid_shape),
                    ),
                    dim=1,
                )

                encoder_input = torch.cat(
                    (
                        condition,
                        aa_vox_current.view(bs, self.hparams.n_atoms_aa, *
                                            self.grid_shape),
                    ),
                    dim=1,
                )
                (recon_aa_vox, mu, logvar,
                 z) = self.forward(encoder_input, condition)
                latent_codes.append(z.detach().cpu().numpy())

        from sklearn.mixture import GaussianMixture

        z = np.concatenate(latent_codes)

        self.GMM = GaussianMixture(n_components=10)
        print("Fitting GMM to posterior latent space density...")
        self.GMM.fit(z)

    def sample_GMM_latent_space(self, n_samples=1):
        z_GMM, _ = self.GMM.sample(n_samples)
        z_GMM = torch.Tensor(z_GMM).to(self.device)
        return z_GMM

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.learning_gamma)
        return [optimizer], [scheduler]

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.aa_traj = md.load(
            str(
                pathlib.Path(self.hparams.path).joinpath(
                    self.hparams.aa_traj)),
            top=str(
                pathlib.Path(self.hparams.path).joinpath(self.hparams.aa_pdb)),
        )

        # Penta alanine
        # self.alpha_carbon_idxs = self.aa_traj.top.select_atom_indices("alpha")

        # BUTANE
        # self.bead_0_idxs = [
        #    [atom.index for atom in self.aa_traj.top.atoms if atom.name == f"C{i}"][0]
        #    for i in [1, 2]
        # ]

        # self.bead_1_idxs = [
        #    [atom.index for atom in self.aa_traj.top.atoms if atom.name == f"C{i}"][0]
        #    for i in [3, 4]
        # ]

        # ADP
        self.ADP_cg_idxs = [4, 6, 8, 10, 14, 16]
        # Chignolin
        # self.Chignolin_cg_idxs = self.aa_traj.top.select("backbone")

        self.bond_idxs = torch.LongTensor(
            [[b.atom1.index, b.atom2.index] for b in self.aa_traj.top.bonds],
            device=self.device,
        )
        self.cg_traj = md.load(
            str(
                pathlib.Path(self.hparams.path).joinpath(
                    self.hparams.cg_traj)),
            top=str(
                pathlib.Path(self.hparams.path).joinpath(self.hparams.cg_pdb)),
        )
        print(
            f"AA trajectory with {self.aa_traj.n_frames} frames, {self.aa_traj.n_atoms} atoms"
        )
        print(
            f"CG trajectory with {self.cg_traj.n_frames} frames, {self.cg_traj.n_atoms} atoms"
        )

        if self.aa_traj.n_frames != self.cg_traj.n_frames:
            raise ValueError(
                "Number of frames in AA and CG trajectory must be the same")

        print(f"Atoms in AA: {[a.name for a in self.aa_traj.topology.atoms]}")

        self.aa_traj.xyz -= md.compute_center_of_geometry(
            self.aa_traj)[..., None, :]
        self.cg_traj.xyz -= md.compute_center_of_geometry(
            self.cg_traj)[..., None, :]

        n_train = int(self.aa_traj.n_frames * self.hparams.train_percent)
        print(
            f"Using {n_train} frames for training, {self.aa_traj.n_frames - n_train} for validation."
        )

        aa_coords_train = self.aa_traj.xyz[:n_train]
        cg_coords_train = self.cg_traj.xyz[:n_train]
        self.ds_train = C2FDataSet3(
            coords_aa=aa_coords_train,
            coords_cg=cg_coords_train,
            noise=self.hparams.noise,
            sigma=self.hparams.sigma,
            resolution=self.hparams.resolution,
            length=self.hparams.length,
            rand_rot=True,
            n_frames=self.hparams.n_frames,
            # cg_idxs=self.ADP_cg_idxs,
        )

        aa_coords_val = self.aa_traj.xyz[n_train:]
        cg_coords_val = self.cg_traj.xyz[n_train:]
        self.ds_val = C2FDataSet3(
            coords_aa=aa_coords_val,
            coords_cg=cg_coords_val,
            noise=0,
            sigma=self.hparams.sigma,
            resolution=self.hparams.resolution,
            length=self.hparams.length,
            rand_rot=False,
            n_frames=self.hparams.n_frames,
            # cg_idxs=self.ADP_cg_idxs,
        )

        self.ds_test = C2FDataSet3(
            coords_aa=aa_coords_val,
            coords_cg=cg_coords_val,
            noise=0,
            sigma=self.hparams.sigma,
            resolution=self.hparams.resolution,
            length=self.hparams.length,
            rand_rot=False,
            n_frames=1,
            # cg_idxs=self.ADP_cg_idxs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.ds_val,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            self.ds_test,
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        # parser = ArgumentParser(parents=[parent_parser])

        # parser.add_argument(
        #    "--path",
        #    default="/home/kirills/Projects/ipam-coarse2fine/butane/allAtom_amber",
        #    type=str,
        # )
        parser.add_argument(
            "--path",
            default="/project2/andrewferguson/Kirill/c2f_Chig/ADP_data",
            type=str,
        )
        # parser.add_argument(
        #    "--path",
        #    default="/project2/andrewferguson/Kirill/c2f_Chig/Chignolin",
        #    type=str,
        # )
        parser.add_argument("--aa_traj", default="AA.dcd", type=str)
        parser.add_argument("--aa_pdb", default="AA.pdb", type=str)
        parser.add_argument("--cg_traj", default="CG.dcd", type=str)
        parser.add_argument("--cg_pdb", default="CG.pdb", type=str)
        parser.add_argument("--n_atoms_aa", default=22, type=int)
        parser.add_argument("--n_atoms_cg", default=6, type=int)

        parser.add_argument("--sigma", default=0.01, type=float)
        parser.add_argument("--resolution", default=12, type=float)
        parser.add_argument("--length", default=1.8, type=float)
        parser.add_argument("--noise", default=0.0, type=float)

        parser.add_argument("--n_frames", default=2, type=int)
        parser.add_argument("--n_frames_past", default=1, type=int)

        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)

        parser.add_argument("--latent_dim", default=32, type=int)
        parser.add_argument("--fac_encoder", default=8, type=int)
        parser.add_argument("--fac_decoder", default=8, type=int)

        parser.add_argument("--train_percent", default=0.95, type=float)

        # Penta alanine
        # parser.add_argument("--E_mu", default=147.53787, type=float)
        # parser.add_argument("--E_std", default=29.141317, type=float)

        # Butane
        # parser.add_argument("--E_mu", default=23.7596, type=float)
        # parser.add_argument("--E_std", default=3.7297, type=float)

        # ADP
        # parser.add_argument("--E_mu", default=-44.2997, type=float)
        # parser.add_argument("--E_std", default=5.0728, type=float)
        # AMBER 96
        # parser.add_argument("--E_mu", default=-19.6224, type=float)
        # parser.add_argument("--E_std", default=6.0085, type=float)
        # AMBER 96sbildn
        parser.add_argument("--E_mu", default=-7.3970, type=float)
        parser.add_argument("--E_std", default=5.7602, type=float)
        # AMBER 96sbildn (small)
        # parser.add_argument("--E_mu", default=-7.3398, type=float)
        # parser.add_argument("--E_std", defalt=5.7864, type=float)

        # Chignolin
        # parser.add_argument("--E_mu", default=50.7496, type=float)
        # parser.add_argument("--E_std", default=77.7321, type=float)

        parser.add_argument("--save_every_n_steps", default=6000, type=int)
        parser.add_argument("--hallucinate_every_n_epochs",
                            default=5,
                            type=int)

        parser.add_argument("--use_recon_energy_loss",
                            default=False,
                            type=bool)
        parser.add_argument("--use_diff_loss", default=False, type=bool)
        parser.add_argument("--use_edm_loss", default=True, type=bool)
        parser.add_argument("--use_coord_loss", default=True, type=bool)
        parser.add_argument("--use_cg_loss", default=True, type=bool)
        parser.add_argument("--skip_E_calcs", default=True, type=bool)
        parser.add_argument("--GMM_estimate", default=False, type=bool)
        parser.add_argument("--bonds_edm_weight", default=0.1, type=float)
        parser.add_argument("--cg_coord_weight", default=0.1, type=float)
        parser.add_argument("--coord_weight", default=1.0, type=float)
        parser.add_argument("--default_save_path", default=None)

        parser.add_argument("--beta", default=1.0, type=float)
        parser.add_argument("--clip_E", default=0.1, type=float)
        parser.add_argument("--learning_gamma", default=1.0, type=float)
        parser.add_argument("--gamma", default=0.0, type=float)
        return parser
