import mdtraj
import numpy as np
from torch.utils.data import Dataset, DataLoader

import config
from config import Config
from util import make_grid_np, rand_rotation_matrix, voxelize_gauss

import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit as u


class C2FDataSet(Dataset):
    def __init__(
        self,
        coords_aa,
        coords_cg,
        noise,
        lagtime,
        rand_rot: bool,
        grid_cfg: config.Config.Grid,
    ):
        self.sigma = grid_cfg.sigma
        self.coords_aa = coords_aa
        self.coords_cg = coords_cg
        self.grid = make_grid_np(grid_cfg.delta_s, grid_cfg.resolution)
        self.noise = noise
        if self.noise > 0:
            self.noisify = lambda x: (x + np.random.normal(
                scale=self.noise, size=x.shape)).astype(np.float32)
            self.noisify = lambda x: x.astype(np.float32)
        else:
            self.noisify = lambda x: x.astype(np.float32)
        self.lagtime = lagtime
        self.rand_rot = rand_rot

    def __len__(self):
        return len(self.coords_aa) - self.lagtime

    def __getitem__(self, item):
        if self.rand_rot:
            R = rand_rotation_matrix()
        else:
            R = np.eye(3)
        aa_prev = self.noisify(self.coords_aa[item] @ R.T)

        aa_prev = np.concatenate(
            (aa_prev, self.noisify(self.coords_aa[item + 1] @ R.T)))
        aa_prev = np.concatenate(
            (aa_prev, self.noisify(self.coords_aa[item + 2] @ R.T)))

        aa_prev_vox = voxelize_gauss(aa_prev, self.sigma, self.grid)
        aa = self.noisify(self.coords_aa[item + self.lagtime] @ R.T)
        aa_vox = voxelize_gauss(aa, self.sigma, self.grid)
        cg_prev = self.noisify(self.coords_cg[item] @ R.T)

        cg_prev = np.concatenate(
            (cg_prev, self.noisify(self.coords_cg[item + 1] @ R.T)))
        cg_prev = np.concatenate(
            (cg_prev, self.noisify(self.coords_cg[item + 2] @ R.T)))

        cg_prev_vox = voxelize_gauss(cg_prev, self.sigma, self.grid)
        cg = self.noisify(self.coords_cg[item + self.lagtime] @ R.T)
        cg_vox = voxelize_gauss(cg, self.sigma, self.grid)
        return aa_prev_vox, aa_prev, aa_vox, aa, cg_prev_vox, cg_prev, cg_vox, cg


class C2FDataSet3(Dataset):
    def __init__(
        self,
        coords_aa,
        coords_cg,
        noise,
        sigma,
        resolution,
        length,
        rand_rot: bool,
        n_frames: int,
    ):
        delta_s = length / resolution
        self.sigma = sigma
        self.coords_aa = coords_aa
        self.coords_cg = coords_cg
        self.grid = make_grid_np(delta_s, resolution)
        self.noise = noise
        self.n_frames = n_frames
        if self.noise > 0:
            #self.noisify = lambda x: (
            #    x + np.random.normal(scale=self.noise, size=x.shape)
            #).astype(np.float32)
            self.noisify = lambda x: x.astype(np.float32)
        else:
            self.noisify = lambda x: x.astype(np.float32)
        self.rand_rot = rand_rot

    def __len__(self):
        return len(self.coords_aa) - self.n_frames

    def __getitem__(self, item):
        if self.rand_rot:
            R = rand_rotation_matrix()
        else:
            R = np.eye(3)

        noise = np.float32(np.random.normal(scale=self.noise))

        aa = self.noisify(self.coords_aa[item] @ R.T) + noise
        for i in range(1, self.n_frames):
            aa = np.concatenate(
                (aa, self.noisify(self.coords_aa[item + i] @ R.T) + noise))

        aa_vox = voxelize_gauss(aa, self.sigma, self.grid)

        cg = self.noisify(self.coords_cg[item] @ R.T) + noise
        for i in range(1, self.n_frames):
            cg = np.concatenate(
                (cg, self.noisify(self.coords_cg[item + i] @ R.T) + noise))

        cg_vox = voxelize_gauss(cg, self.sigma, self.grid)

        # aa_vox = aa_vox / aa_vox.sum(axis=(-1, -2, -3), keepdims=True)
        # cg_vox = cg_vox / cg_vox.sum(axis=(-1, -2, -3), keepdims=True)

        return aa, aa_vox, cg, cg_vox


class C2FDataSet2(Dataset):
    def __init__(
        self,
        coords_aa,
        coords_cg,
        noise,
        lagtime,
        rand_rot: bool,
        grid_cfg: config.Config.Grid,
        pdb_fname,
    ):
        self.sigma = grid_cfg.sigma
        self.coords_aa = coords_aa
        self.coords_cg = coords_cg
        self.grid = make_grid_np(grid_cfg.delta_s, grid_cfg.resolution)
        self.noise = noise
        self.pdb_fname = pdb_fname

        if self.noise > 0:
            self.noisify = lambda x: (x + np.random.normal(
                scale=self.noise, size=x.shape)).astype(np.float32)
        else:
            self.noisify = lambda x: x.astype(np.float32)
        self.lagtime = lagtime
        self.rand_rot = rand_rot

        self.pdb_fname = pdb_fname
        self.pdb = app.PDBFile(self.pdb_fname)
        self.forcefield = app.ForceField("amber03.xml", "amber99_obc.xml")
        self.system = self.forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=mm.app.HBonds)
        self.integrator = mm.LangevinIntegrator(300 * u.kelvin,
                                                1.0 / u.picoseconds,
                                                2.0 * u.femtoseconds)

    def __len__(self):
        return len(self.coords_aa) - self.lagtime

    def __getitem__(self, item):
        if self.rand_rot:
            R = rand_rotation_matrix()
        else:
            R = np.eye(3)
        aa_prev = self.noisify(self.coords_aa[item] @ R.T)
        aa_prev_vox = voxelize_gauss(aa_prev, self.sigma, self.grid)

        aa = self.noisify(self.coords_aa[item + self.lagtime] @ R.T)
        aa_vox = voxelize_gauss(aa, self.sigma, self.grid)

        cg_prev = self.noisify(self.coords_cg[item] @ R.T)
        cg_prev_vox = voxelize_gauss(cg_prev, self.sigma, self.grid)

        cg = self.noisify(self.coords_cg[item + self.lagtime] @ R.T)
        cg_vox = voxelize_gauss(cg, self.sigma, self.grid)

        return aa_prev_vox, aa_prev, aa_vox, aa, cg_prev_vox, cg_prev, cg_vox, cg


def setup_loaders(aa_traj, cg_traj, cfg: Config, log):
    log.debug(
        f"AA trajectory with {aa_traj.n_frames} frames, {aa_traj.n_atoms} atoms"
    )
    log.debug(
        f"CG trajectory with {cg_traj.n_frames} frames, {cg_traj.n_atoms} atoms"
    )

    assert aa_traj.n_frames == cg_traj.n_frames

    log.debug(f"Atoms in AA: {[a.name for a in aa_traj.topology.atoms]}")

    aa_traj.xyz -= mdtraj.compute_center_of_geometry(aa_traj)[..., None, :]
    cg_traj.xyz -= mdtraj.compute_center_of_geometry(cg_traj)[..., None, :]

    n_train = int(aa_traj.n_frames * cfg.training.split_fraction)
    log.debug(
        f"Using {n_train} frames for training, the remainder for validation.")

    aa_coords_train = aa_traj.xyz[:n_train]
    cg_coords_train = cg_traj.xyz[:n_train]

    ds_train = C2FDataSet(
        aa_coords_train,
        cg_coords_train,
        cfg.data.noise,
        cfg.data.lagtime,
        True,
        cfg.grid,
    )

    aa_coords_val = aa_traj.xyz[n_train:]
    cg_coords_val = cg_traj.xyz[n_train:]
    ds_val = C2FDataSet(aa_coords_val, cg_coords_val, 0, cfg.data.lagtime,
                        False, cfg.grid)

    log.debug("Setting up data loaders")
    batch_size = cfg.training.batch_size

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=0,
    )
    loader_val = DataLoader(ds_val,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False)

    return loader_train, loader_val


class C2FDataSet4(Dataset):
    def __init__(
        self,
        coords_aa,
        coords_cg,
        noise,
        sigma,
        resolution,
        length,
        cg_idxs,
        rand_rot: bool,
        n_frames: int,
    ):
        delta_s = length / resolution
        self.sigma = sigma
        self.coords_aa = coords_aa
        self.coords_cg = coords_cg
        self.grid = make_grid_np(delta_s, resolution)
        self.noise = noise
        self.n_frames = n_frames
        self.cg_idxs = np.array(cg_idxs)

        self.rand_rot = rand_rot

    def __len__(self):
        return len(self.coords_aa) - self.n_frames

    def __getitem__(self, item):
        if self.rand_rot:
            R = rand_rotation_matrix()
        else:
            R = np.eye(3)

        noise = np.float32(np.random.normal(scale=self.noise))

        aa = np.float32(self.coords_aa[item] @ R.T) + noise
        cg = aa[self.cg_idxs]
        for i in range(1, self.n_frames):
            aa_next = np.float32(self.coords_aa[item + i] @ R.T) + noise
            cg_next = aa_next[self.cg_idxs]
            aa = np.concatenate((aa, aa_next))
            cg = np.concatenate((cg, cg_next))

        aa_vox = voxelize_gauss(aa, self.sigma, self.grid)
        cg_vox = voxelize_gauss(cg, self.sigma, self.grid)

        # aa_vox = aa_vox / aa_vox.sum(axis=(-1, -2, -3), keepdims=True)
        # cg_vox = cg_vox / cg_vox.sum(axis=(-1, -2, -3), keepdims=True)

        return aa, aa_vox, cg, cg_vox
