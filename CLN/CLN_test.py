# In[25]:

from pl_vae_CLN import VAE
import numpy as np
import torch
import mdtraj as md
from data import C2FDataSetCLN
from tqdm import tqdm
from util import avg_blob, voxel_gauss
import pathlib
from joblib import dump, load

# In[2]:


def main():
    def rigid_transform(A, B):
        """Returns the rotation matrix (R) and translation (t) to solve 'A @ R + t = B'
        A,B are N x 3 matricies"""
        # http://nghiaho.com/?page_id=671

        centoid_A = A.mean(0, keepdims=True)
        centoid_B = B.mean(0, keepdims=True)

        H = (A - centoid_A).T @ (B - centoid_B)

        U, S, Vt = np.linalg.svd(H, full_matrices=False)

        R = Vt.T @ U.T  # 3x3

        # Reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        R = R.T  # Transpose because A is not 3 x N

        t = centoid_B - centoid_A @ R  # 1x3

        return R, t

    def estimate_latent_space_density(self):
        dataloader = self.train_dataloader()
        latent_codes = list()
        Energies = list()
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

                energies = self.ELoss.energy(aa_future.reshape(bs, -1))

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
                latent_codes.append(mu.detach().cpu().numpy())
                Energies.append(energies.detach().cpu().numpy())

        from sklearn.mixture import GaussianMixture

        z = np.concatenate(latent_codes)
        Energies = np.concatenate(Energies)

        self.GMM = GaussianMixture(n_components=10)
        print("Fitting GMM to posterior latent space density...")
        self.GMM.fit(z)
        return z, Energies

    # In[9]:

    ckpt_file = "N-Step-Checkpoint.ckpt"
    # ckpt_file = "epoch=6.ckpt"/
    #version_name = 'version_8RRRRRRRRRRRR'
    version_name = 'version_1RR'
    base_fname = f"/project2/andrewferguson/Kirill/midway3_c2f/CLN_twoGPU_V2/{version_name}"
    ckpt_fname = f"{base_fname}/checkpoints/{ckpt_file}"
    save_fname = 'CLN_testset_1RR'
    model = VAE.load_from_checkpoint(ckpt_fname).to(
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    print(model.device)

    # In[19]:

    aa_traj = md.load(
        str(pathlib.Path(model.hparams.path).joinpath(model.hparams.aa_pdb)))

    # In[22]:

    cg_traj = md.load(
        str(pathlib.Path(model.hparams.path).joinpath(model.hparams.cg_pdb)))

    # In[20]:

    trj_fnames = [
        str(f) for f in sorted(
            pathlib.Path(model.hparams.path).joinpath("AA_xtc").glob("*xtc"))
    ]
    aa_trajs = [
        md.load(trj_fname, top=aa_traj.top).center_coordinates()
        for trj_fname in trj_fnames
    ]

    # In[23]:

    trj_fnames = [
        str(f) for f in sorted(
            pathlib.Path(model.hparams.path).joinpath("CG_xtc").glob("*xtc"))
    ]
    cg_trajs = [
        md.load(trj_fname, top=cg_traj.top).center_coordinates()
        for trj_fname in trj_fnames
    ]

    # In[24]:

    n_train = int(len(aa_trajs) * model.hparams.train_percent)

    # In[45]:

    CLN_cg_idxs = [at.index for at in aa_traj.top.atoms if at.name == "CA"]

    # In[26]:

    aa_coords_train = [trj.xyz for trj in aa_trajs[:n_train]]
    cg_coords_train = [trj.xyz for trj in cg_trajs[:n_train]]
    model.ds_train = C2FDataSetCLN(coords_aa=aa_coords_train,
                                   coords_cg=cg_coords_train,
                                   noise=model.hparams.noise,
                                   sigma=model.hparams.sigma,
                                   resolution=model.hparams.resolution,
                                   length=model.hparams.length,
                                   rand_rot=False,
                                   n_frames=model.hparams.n_frames)

    # In[6]:

    z, Energies = estimate_latent_space_density(model)
    np.save(f'{save_fname}/CLN_energies.npy', Energies)
    np.save(f'{save_fname}/CLN_z.npy', z)
    dump(model.GMM, f'{save_fname}/GMM.joblib')

    # In[54]:

    def hallucinate(cg_test_trj, aa_test_trj, name_num, GMM=True):

        best_min = np.inf
        for i, cg_traj in tqdm(enumerate(cg_trajs[:n_train]),
                               total=len(cg_trajs[:n_train]),
                               leave=False):
            rmsd_min = md.rmsd(cg_traj, cg_test_trj, frame=0).min()
            if rmsd_min < best_min:
                aa_trj_min = i
                best_min = rmsd_min
                aa_argmin = md.rmsd(cg_traj, cg_test_trj, frame=0).argmin()
            #if i == 0:
            #    aa_trj_min = 0
            #    best_min = np.inf
            #    continue
            #else:
            #    rmsd_min = md.rmsd(cg_traj, cg_test_trj, frame=0).min()
            #    if rmsd_min < best_min:
            #        aa_trj_min = i
            #        best_min = rmsd_min
            #        aa_argmin = md.rmsd(cg_traj, cg_test_trj, frame=0).argmin()

        aa_0 = aa_trajs[:n_train][aa_trj_min][aa_argmin]
        R, t = rigid_transform(
            aa_0.atom_slice(CLN_cg_idxs).xyz[0], cg_test_trj.xyz[0])
        aa_0 = aa_trajs[:n_train][aa_trj_min][aa_argmin - 1]
        aa_0.xyz = aa_0.xyz @ R + t

        with torch.no_grad():
            aa_vox_current = voxel_gauss(torch.Tensor(aa_0[0].xyz).to(
                model.device),
                                         res=model.hparams.resolution,
                                         width=model.hparams.length,
                                         sigma=model.hparams.sigma,
                                         device=model.device)

            fake_coords = list()
            for i in tqdm(range(cg_test_trj.n_frames), leave=False):
                cg_vox = voxel_gauss(torch.Tensor(cg_test_trj[i].xyz).to(
                    model.device),
                                     res=model.hparams.resolution,
                                     width=model.hparams.length,
                                     sigma=model.hparams.sigma,
                                     device=model.device)

                condition = torch.cat((cg_vox, aa_vox_current), dim=1)

                if GMM:
                    z = model.sample_GMM_latent_space()
                else:
                    z = torch.empty(
                        (1, model.hparams.latent_dim),
                        dtype=condition.dtype,
                        device=model.device,
                    ).normal_()

                recon_aa_vox = model.decode(z, condition)

                aa_fake = avg_blob(
                    recon_aa_vox,
                    res=model.hparams.resolution,
                    width=model.hparams.length,
                    sigma=model.hparams.sigma,
                    device=model.device,
                )
                fake_coords.append(aa_fake)

                aa_vox_current = voxel_gauss(aa_fake,
                                             res=model.hparams.resolution,
                                             width=model.hparams.length,
                                             sigma=model.hparams.sigma,
                                             device=model.device)

        hallucinate_coords = torch.cat(fake_coords, dim=0)
        hallucinate_trj = md.Trajectory(
            hallucinate_coords.detach().cpu().numpy(),
            topology=aa_traj.top).center_coordinates()
        hallucinate_trj.save_dcd(
            f'/project2/andrewferguson/Kirill/midway3_c2f/{save_fname}/hallucinated_{name_num}.dcd'
        )
        cg_test_trj.save_dcd(
            f'/project2/andrewferguson/Kirill/midway3_c2f/{save_fname}/CG_{name_num}.dcd'
        )
        aa_test_trj.save_dcd(
            f'/project2/andrewferguson/Kirill/midway3_c2f/{save_fname}/AA_{name_num}.dcd'
        )

    # In[55]:
    # In[ ]:

    for i, (cg,
            aa) in tqdm(enumerate(zip(cg_trajs[n_train:], aa_trajs[n_train:])),
                        total=len(cg_trajs[n_train:])):
        hallucinate(cg, aa, i, GMM=True)


if __name__ == '__main__':
    main()
