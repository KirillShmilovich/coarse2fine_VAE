from pl_vae_CLN import VAE
import numpy as np
import torch
import mdtraj as md
from tqdm import tqdm
from util import avg_blob, voxel_gauss
from pathlib import Path
from sklearn.mixture import GaussianMixture
import sys
from joblib import load


def main(frame):
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

    ckpt_file = "N-Step-Checkpoint.ckpt"
    # ckpt_file = "epoch=6.ckpt"/
    version_name = 'version_8RRRRRRRRRRRR'
    base_fname = f"/project2/andrewferguson/Kirill/midway3_c2f/CLN_twoGPU/{version_name}"
    ckpt_fname = f"{base_fname}/checkpoints/{ckpt_file}"
    model = VAE.load_from_checkpoint(ckpt_fname).to(
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    print(model.device)

    aa_pdb_fname = '/project2/andrewferguson/Kirill/midway3_c2f/chig_new/AA.pdb'
    cg_pdb_fname = '/project2/andrewferguson/Kirill/midway3_c2f/chig_new/CG.pdb'

    aa_pdb = md.load(aa_pdb_fname)
    cg_pdb = md.load(cg_pdb_fname)

    aa_trj_fnames = [
        str(f) for f in sorted(
            Path('/project2/andrewferguson/Kirill/midway3_c2f/chig_new/').
            joinpath("AA_xtc").glob("*xtc"))
    ]
    cg_trj_fnames = [
        str(f) for f in sorted(
            Path('/project2/andrewferguson/Kirill/midway3_c2f/chig_new/').
            joinpath("CG_xtc").glob("*xtc"))
    ]

    n_train = int(len(aa_trj_fnames) * model.hparams.train_percent)
    CLN_cg_idxs = [at.index for at in aa_pdb.top.atoms if at.name == "CA"]

    ################## CHANGE ##################
    #z = np.load('CLN_testset_k16/CLN_z.npy')
    #model.GMM = GaussianMixture(n_components=10)
    #print("Fitting GMM to posterior latent space density...")
    #model.GMM.fit(z[::10])

    model.GMM = load('GMM_k5.joblib')

    cg_data = np.load(
        '/project2/andrewferguson/Kirill/midway3_c2f/cln_ca_regular_avg_cgtraj.npy'
    ) / 10.

    #all_coords = list()
    #for frame in tqdm(range(cg_data.shape[0])):
    cg_data_trj = md.Trajectory(cg_data[frame],
                                topology=cg_pdb.top).center_coordinates()
    best_min = np.inf
    for i, cg_traj_fname in tqdm(enumerate(cg_trj_fnames[:n_train]),
                                 total=len(cg_trj_fnames[:n_train]),
                                 leave=False):
        cg_traj = md.load(cg_traj_fname, top=cg_pdb_fname)
        rmsd = md.rmsd(cg_traj, cg_data_trj, frame=0)
        rmsd_min = rmsd.min()
        if rmsd_min < best_min:
            aa_trj_min = i
            best_min = rmsd_min
            aa_argmin = rmsd.argmin()

    aa_0_fname = aa_trj_fnames[:n_train][aa_trj_min]
    aa_0 = md.load(aa_0_fname, top=aa_pdb_fname)[aa_argmin]
    R, t = rigid_transform(
        aa_0.atom_slice(CLN_cg_idxs).xyz[0], cg_data_trj.xyz[0])
    aa_0_fname = aa_trj_fnames[:n_train][aa_trj_min]
    aa_0 = md.load(aa_0_fname, top=aa_pdb_fname)[aa_argmin - 1]
    aa_0.xyz = aa_0.xyz @ R + t

    with torch.no_grad():
        aa_vox_current = voxel_gauss(torch.Tensor(aa_0[0].xyz).to(
            model.device),
                                     res=model.hparams.resolution,
                                     width=model.hparams.length,
                                     sigma=model.hparams.sigma,
                                     device=model.device)

        fake_coords = list()
        for i in tqdm(range(cg_data_trj.n_frames), leave=False):
            cg_vox = voxel_gauss(torch.Tensor(cg_data_trj[i].xyz).to(
                model.device),
                                 res=model.hparams.resolution,
                                 width=model.hparams.length,
                                 sigma=model.hparams.sigma,
                                 device=model.device)

            condition = torch.cat((cg_vox, aa_vox_current), dim=1)

            z = model.sample_GMM_latent_space()

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
    #all_coords.append(hallucinate_coords)
    hallucinate_trj = md.Trajectory(hallucinate_coords.detach().cpu().numpy(),
                                    topology=aa_pdb.top)
    hallucinate_trj.save_dcd(
        f'/project2/andrewferguson/Kirill/midway3_c2f/CLNcg2aa_k5/aa_hullucinate_{frame}.dcd'
    )
    cg_data_trj.save_dcd(
        f'/project2/andrewferguson/Kirill/midway3_c2f/CLNcg2aa_k5/cg_hullucinate_{frame}.dcd'
    )

    #aa_data = torch.cat([coords.unsqueeze(0)
    #                     for coords in all_coords]).cpu().numpy()

    #np.save('CLN2_backmapped.npy', aa_data)


if __name__ == '__main__':
    frame = int(sys.argv[1])
    main(frame)
