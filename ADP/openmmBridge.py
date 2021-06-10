"""Energy and Force computation in OpenMM
"""

import warnings
import multiprocessing as mp

import simtk.openmm as mm
from simtk.openmm import app
from simtk import unit as u

import numpy as np
import pickle
import torch

# from ...utils.types import assert_numpy
# from .base import Energy


_OPENMM_FLOATING_TYPE = np.float64
_SPATIAL_DIM = 3


class _OpenMMEnergyWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, openmm_energy_bridge):
        energy, force, *_ = openmm_energy_bridge.evaluate(input)
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        (neg_force,) = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None


_evaluate_openmm_energy = _OpenMMEnergyWrapper.apply


class OpenMMBridge:
    """Bridge object to evaluate energies in OpenMM.
    Input positions are in nm, returned energies are dimensionless (units of kT), returned forces are in kT/nm.

    Parameters
    ----------
    openmm_system : simtk.openmm.System
        The OpenMM system object that contains all force objects.
    openmm_integrator : simtk.openmm.Integrator
        A thermostated OpenMM integrator (has to have a method `getTemperature()`.
    platform_name : str, optional
        An OpenMM platform name ('CPU', 'CUDA', 'Reference', or 'OpenCL')
    err_handling : str, optional
        How to handle infinite energies (one of {"warning", "ignore", "exception"}).
    n_workers : int, optional
        The number of processes used to compute energies in batches. This should not exceed the
        most-used batch size or the number of accessible CPU cores. The default is the number
        of logical cpu cores. If a GPU platform is used (CUDA or OpenCL), n_workers is always set to 1
        to sidestep multiprocessing (due to performance issues when using multiprocessing with GPUs).
    n_simulation_steps : int, optional
        If > 0, perform a number of simulation steps and compute energy and forces for the resulting state.
    """

    def __init__(
        self,
        openmm_system,
        openmm_integrator,
        platform_name="CPU",
        err_handling="warning",
        n_workers=mp.cpu_count(),
        n_simulation_steps=0,
    ):
        from simtk import unit

        platform_properties = (
            {"Threads": str(max(1, mp.cpu_count() // n_workers))}
            if platform_name == "CPU"
            else {}
        )

        # Compute all energies in child processes due to a bug in the OpenMM's PME code.
        # This might be problematic if an energy has already been computed in the same program on the parent thread,
        # see https://github.com/openmm/openmm/issues/2602
        self._openmm_system = openmm_system
        self._openmm_integrator = openmm_integrator
        if platform_name in ["CUDA", "OpenCL"] or n_workers == 1:
            self.context_wrapper = SingleContext(
                1, openmm_system, openmm_integrator, platform_name, platform_properties
            )
        else:
            self.context_wrapper = MultiContext(
                n_workers,
                openmm_system,
                openmm_integrator,
                platform_name,
                platform_properties,
            )
        self._err_handling = err_handling
        self._n_simulation_steps = n_simulation_steps
        self._unit_reciprocal = 1 / (
            openmm_integrator.getTemperature() * unit.MOLAR_GAS_CONSTANT_R
        ).value_in_unit(unit.kilojoule_per_mole)
        self.last_energies = None
        self.last_forces = None

    @property
    def integrator(self):
        return self._openmm_integrator

    @property
    def n_simulation_steps(self):
        return self._n_simulation_steps

    def _reduce_units(self, x):
        if x is None:
            return None
        return x * self._unit_reciprocal

    def evaluate(
        self,
        batch,
        evaluate_force=True,
        evaluate_energy=True,
        evaluate_positions=False,
        evaluate_path_probability_ratio=False,
    ):
        """
        Compute energies/forces for a batch of positions.

        Parameters:
        -----------
        batch : np.ndarray or torch.Tensor
            A batch of particle positions that has shape (batch_size, num_particles * 3).
        evaluate_force : bool, optional
            Whether to compute forces.
        evaluate_energy : bool, optional
            Whether to compute energies.
        evaluate_positions : bool, optional
            Whether to return positions.
        evaluate_path_probability_ratio : bool, optional
            Whether to compute the log path probability ratio. Makes only sense for PathProbabilityIntegrator instances.

        Returns
        -------
        energies : torch.Tensor or None
            The energies in units of kilojoule/mole; its shape  is (len(batch), )
        forces : torch.Tensor or None
            The forces in units of kilojoule/mole/nm; its shape is (len(batch), num_particles*3)
        new_positions : torch.Tensor or None
            The positions in units of nm; its shape is (len(batch), num_particles*3)
        log_path_probability_ratio : torch.Tensor or None
            The logarithmic path probability ratios; its shape  is (len(batch), )
        """

        # make a list of positions
        # batch_array = assert_numpy(batch, arr_type=_OPENMM_FLOATING_TYPE)
        batch_array = batch.cpu().numpy()

        # assert correct number of positions
        assert (
            batch_array.shape[1] == self._openmm_system.getNumParticles() * _SPATIAL_DIM
        )

        # reshape to (B, N, D)
        batch_array = batch_array.reshape(batch.shape[0], -1, _SPATIAL_DIM)
        (
            energies,
            forces,
            new_positions,
            log_path_probability_ratio,
        ) = self.context_wrapper.evaluate(
            batch_array,
            evaluate_energy=evaluate_energy,
            evaluate_force=evaluate_force,
            evaluate_positions=evaluate_positions,
            evaluate_path_probability_ratio=evaluate_path_probability_ratio,
            err_handling=self._err_handling,
            n_simulation_steps=self._n_simulation_steps,
        )

        # divide by kT
        energies = self._reduce_units(energies)
        forces = self._reduce_units(forces)

        # to PyTorch tensors
        energies = (
            torch.tensor(energies).to(batch).reshape(-1, 1) if evaluate_energy else None
        )
        forces = (
            torch.tensor(forces)
            .to(batch)
            .reshape(
                batch.shape[0], self._openmm_system.getNumParticles() * _SPATIAL_DIM
            )
            if evaluate_force
            else None
        )
        new_positions = (
            torch.tensor(new_positions)
            .to(batch)
            .reshape(
                batch.shape[0], self._openmm_system.getNumParticles() * _SPATIAL_DIM
            )
            if evaluate_positions
            else None
        )
        log_path_probability_ratio = (
            torch.tensor(log_path_probability_ratio).to(batch).reshape(-1, 1)
            if evaluate_path_probability_ratio
            else None
        )

        # store
        self.last_energies = energies
        self.last_forces = forces

        return energies, forces, new_positions, log_path_probability_ratio


class MultiContext:
    """A container for multiple OpenMM Contexts that are operated by different worker processes.

    Parameters:
    -----------
    n_workers : int
        The number of workers which operate one context each.
    system : simtk.openmm.System
        The system that contains all forces.
    integrator : simtk.openmm.Integrator
        An OpenMM integrator.
    platform_name : str
        The name of an OpenMM platform ('Reference', 'CPU', 'CUDA', or 'OpenCL')
    platform_properties : dict, optional
        A dictionary of platform properties.
    """

    def __init__(
        self, n_workers, system, integrator, platform_name, platform_properties={}
    ):
        """Set up workers and queues."""
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._workers = []
        for i in range(n_workers):
            worker = MultiContext.Worker(
                self._task_queue,
                self._result_queue,
                system,
                integrator,
                platform_name,
                platform_properties,
            )
            self._workers.append(worker)
            worker.start()

    def evaluate(
        self,
        positions,
        box_vectors=None,
        evaluate_energy=True,
        evaluate_force=True,
        evaluate_positions=False,
        evaluate_path_probability_ratio=False,
        err_handling="warning",
        n_simulation_steps=0,
    ):
        """Delegate energy and force computations to the workers.

        Parameters
        ----------
        positions : numpy.ndarray
            The particle positions in nanometer; its shape is (batch_size, num_particles, 3).
        box_vectors : numpy.ndarray, optional
            The periodic box vectors in nanometer; its shape is (batch_size, 3, 3).
            If not specified, don't change the box vectors.
        evaluate_energy : bool, optional
            Whether to compute energies.
        evaluate_force : bool, optional
            Whether to compute forces.
        evaluate_positions : bool, optional
            Whether to return positions.
        evaluate_path_probability_ratio : bool, optional
            Whether to compute the log path probability ratio. Makes only sense for PathProbabilityIntegrator instances.
        _err_handling : str, optional
            How to handle infinite energies (one of {"warning", "ignore", "exception"}).
        n_simulation_steps : int, optional
            If > 0, perform a number of simulation steps and compute energy and forces for the resulting state.

        Returns
        -------
        energies : np.ndarray or None
            The energies in units of kilojoule/mole; its shape  is (len(positions), )
        forces : np.ndarray or None
            The forces in units of kilojoule/mole/nm; its shape is (len(positions), num_particles, 3)
        new_positions : np.ndarray or None
            The positions in units of nm; its shape is (len(positions), num_particles, 3)
        log_path_probability_ratio : np.ndarray or None
            The logarithmic path probability ratios; its shape  is (len(positions), )
        """
        assert box_vectors is None or len(box_vectors) == len(
            positions
        ), "box_vectors and positions have to be the same length"
        box_vectors = [None for _ in positions] if box_vectors is None else box_vectors
        for i, (p, bv) in enumerate(zip(positions, box_vectors)):
            self._task_queue.put(
                [
                    i,
                    p,
                    bv,
                    evaluate_energy,
                    evaluate_force,
                    evaluate_positions,
                    evaluate_path_probability_ratio,
                    err_handling,
                    n_simulation_steps,
                ]
            )
        results = [self._result_queue.get() for _ in positions]
        results = sorted(results, key=lambda x: x[0])
        return (
            np.array([res[1] for res in results]) if evaluate_energy else None,
            np.array([res[2] for res in results]) if evaluate_force else None,
            np.array([res[3] for res in results]) if evaluate_positions else None,
            np.array([res[4] for res in results])
            if evaluate_path_probability_ratio
            else None,
        )

    def __del__(self):
        """Terminate the workers."""
        # soft termination
        for _ in self._workers:
            self._task_queue.put(None)
        # hard termination
        for worker in self._workers:
            worker.terminate()

    class Worker(mp.Process):
        """A worker process that computes energies in its own context.

        Parameters:
        -----------
        task_queue : multiprocessing.Queue
            The queue that the MultiContext pushes tasks to.
        result_queue : multiprocessing.Queue
            The queue that the MultiContext receives results from.
        system : simtk.openmm.System
            The system that contains all forces.
        integrator : simtk.openmm.Integrator
            An OpenMM integrator.
        platform_name : str
            The name of an OpenMM platform ('Reference', 'CPU', 'CUDA', or 'OpenCL')
        platform_properties : dict
            A dictionary of platform properties.
        """

        def __init__(
            self,
            task_queue,
            result_queue,
            system,
            integrator,
            platform_name,
            platform_properties,
        ):
            super(MultiContext.Worker, self).__init__()
            self._task_queue = task_queue
            self._result_queue = result_queue
            self._openmm_system = system
            self._openmm_integrator = pickle.loads(pickle.dumps(integrator))
            self._openmm_platform_name = platform_name
            self._openmm_platform_properties = platform_properties
            self._openmm_context = None

        def run(self):
            """Run the process: set positions and compute energies and forces.
            Positions and box vectors are received from the task_queue in units of nanometers.
            Energies and forces are pushed to the result_queue in units of kJ/mole and kJ/mole/nm, respectively.
            """
            from simtk import unit
            from simtk.openmm import Platform, Context

            # create the context
            # it is crucial to do that in the run function and not in the constructor
            # for some reason, the CPU platform hangs if the context is created in the constructor
            # see also https://github.com/openmm/openmm/issues/2602
            openmm_platform = Platform.getPlatformByName(self._openmm_platform_name)
            self._openmm_context = Context(
                self._openmm_system,
                self._openmm_integrator,
                openmm_platform,
                self._openmm_platform_properties,
            )
            self._openmm_context.reinitialize(preserveState=True)

            # get tasks from the task queue
            for task in iter(self._task_queue.get, None):
                (
                    index,
                    positions,
                    box_vectors,
                    evaluate_energy,
                    evaluate_force,
                    evaluate_positions,
                    evaluate_path_probability_ratio,
                    err_handling,
                    n_simulation_steps,
                ) = task
                try:
                    # initialize state
                    self._openmm_context.setPositions(positions)
                    if box_vectors is not None:
                        self._openmm_context.setPeriodicBoxVectors(box_vectors)
                    log_path_probability_ratio = self._openmm_integrator.step(
                        n_simulation_steps
                    )

                    # compute energy and forces
                    state = self._openmm_context.getState(
                        getEnergy=evaluate_energy,
                        getForces=evaluate_force,
                        getPositions=evaluate_positions,
                    )
                    energy = (
                        state.getPotentialEnergy().value_in_unit(
                            unit.kilojoule_per_mole
                        )
                        if evaluate_energy
                        else None
                    )
                    forces = (
                        state.getForces(asNumpy=True).value_in_unit(
                            unit.kilojoule_per_mole / unit.nanometer
                        )
                        if evaluate_force
                        else None
                    )
                    new_positions = (
                        state.getPositions().value_in_unit(unit.nanometers)
                        if evaluate_positions
                        else None
                    )
                except Exception as e:
                    if err_handling == "warning":
                        warnings.warn("Suppressed exception: {}".format(e))
                    elif err_handling == "exception":
                        raise e

                # push energies and forces to the results queue
                self._result_queue.put(
                    [index, energy, forces, new_positions, log_path_probability_ratio]
                )


class SingleContext:
    """Mimics the MultiContext API but does not spawn worker processes.

    Parameters:
    -----------
    n_workers : int
        Needs to be 1.
    system : simtk.openmm.System
        The system that contains all forces.
    integrator : simtk.openmm.Integrator
        An OpenMM integrator.
    platform_name : str
        The name of an OpenMM platform ('Reference', 'CPU', 'CUDA', or 'OpenCL')
    platform_properties : dict, optional
        A dictionary of platform properties.
    """

    def __init__(
        self, n_workers, system, integrator, platform_name, platform_properties={}
    ):
        """Set up workers and queues."""
        from simtk.openmm import Platform, Context

        assert n_workers == 1
        openmm_platform = Platform.getPlatformByName(platform_name)
        self._openmm_context = Context(
            system, integrator, openmm_platform, platform_properties
        )

    def evaluate(
        self,
        positions,
        box_vectors=None,
        evaluate_energy=True,
        evaluate_force=True,
        evaluate_positions=False,
        evaluate_path_probability_ratio=False,
        err_handling="warning",
        n_simulation_steps=0,
    ):
        """Compute energies and/or forces.

        Parameters:
        -----------
        positions : numpy.ndarray
            The particle positions in nanometer; its shape is (batch_size, num_particles, 3).
        box_vectors : numpy.ndarray, optional
            The periodic box vectors in nanometer; its shape is (batch_size, 3, 3).
            If not specified, don't change the box vectors.
        evaluate_energy : bool, optional
            Whether to compute energies.
        evaluate_force : bool, optional
            Whether to compute forces.
        evaluate_positions : bool, optional
            Whether to return positions.
        evaluate_path_probability_ratio : bool, optional
            Whether to compute the log path probability ratio. Makes only sense for PathProbabilityIntegrator instances.
        _err_handling : str, optional
            How to handle infinite energies (one of {"warning", "ignore", "exception"}).
        n_simulation_steps : int, optional
            If > 0, perform a number of simulation steps and compute energy and forces for the resulting state.

        Returns:
        --------
        energies : np.ndarray or None
            The energies in units of kilojoule/mole; its shape  is (len(positions), )
        forces : np.ndarray or None
            The forces in units of kilojoule/mole/nm; its shape is (len(positions), num_particles, 3)
        new_positions : np.ndarray or None
            The positions in units of nm; its shape is (len(positions), num_particles, 3)
        log_path_probability_ratio : np.ndarray or None
            The logarithmic path probability ratios; its shape  is (len(positions), )
        """
        from simtk import unit

        assert box_vectors is None or len(box_vectors) == len(
            positions
        ), "box_vectors and positions have to be the same length"
        box_vectors = [None for _ in positions] if box_vectors is None else box_vectors

        forces = np.zeros_like(positions)
        energies = np.zeros_like(positions[:, 0, 0])
        new_positions = np.zeros_like(positions)
        log_path_probability_ratios = np.zeros_like(positions[:, 0, 0])

        for i, (p, bv) in enumerate(zip(positions, box_vectors)):

            try:
                # initialize state
                self._openmm_context.setPositions(p)
                if bv is not None:
                    self._openmm_context.setPeriodicBoxVectors(bv)
                log_path_probability_ratio = self._openmm_context.getIntegrator().step(
                    n_simulation_steps
                )

                # compute energy and forces
                state = self._openmm_context.getState(
                    getEnergy=evaluate_energy,
                    getForces=evaluate_force,
                    getPositions=evaluate_positions,
                )
                energy = (
                    state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    if evaluate_energy
                    else None
                )
                force = (
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer
                    )
                    if evaluate_force
                    else None
                )
                new_pos = (
                    state.getPositions().value_in_unit(unit.nanometers)
                    if evaluate_positions
                    else None
                )

                energies[i] = energy if evaluate_energy else 0.0
                forces[i, :, :] = force if evaluate_force else 0.0
                new_positions[i, :, :] = new_pos if evaluate_positions else 0.0
                log_path_probability_ratios[i] = (
                    log_path_probability_ratio
                    if evaluate_path_probability_ratio
                    else 0.0
                )

            except Exception as e:
                if err_handling == "warning":
                    warnings.warn("Suppressed exception: {}".format(e))
                elif err_handling == "exception":
                    raise e

        return (
            energies if evaluate_energy else None,
            forces if evaluate_force else None,
            new_positions if evaluate_positions else None,
            log_path_probability_ratios if evaluate_path_probability_ratio else None,
        )


class Energy(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.0
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x, create_graph=True, retain_graph=True)[0]


class OpenMMEnergy(Energy):
    def __init__(self, dimension, pdb_fname):
        super().__init__(dimension)
        self.pdb_fname = pdb_fname
        self.pdb = app.PDBFile(self.pdb_fname)
        # self.forcefield = app.ForceField("amber03.xml")
        # self.system = self.forcefield.createSystem(
        #    self.pdb.topology, nonbondedMethod=app.NoCutoff
        # )

        # BUTANE
        # prmtop = app.AmberPrmtopFile(
        #     "/home/kirills/Projects/ipam-coarse2fine/butane/allAtom_amber/MOL_AC.prmtop"
        # )
        # self.system = prmtop.createSystem(nonbondedMethod=app.NoCutoff)

        # ADP
        # from openmmtools.testsystems import AlanineDipeptideImplicit
        # alanine = AlanineDipeptideImplicit()
        # self.system = alanine.system
        # self.forcefield = app.ForceField("amber96.xml")
        # self.system = self.forcefield.createSystem(
        #    self.pdb.topology, nonbondedMethod=app.NoCutoff
        # )

        # Chignolin
        self.forcefield = app.ForceField("amberfb15.xml")
        self.system = self.forcefield.createSystem(
            self.pdb.topology, nonbondedMethod=app.NoCutoff
        )

        self.integrator = mm.LangevinIntegrator(
            300 * u.kelvin, 1.0 / u.picoseconds, 2.0 * u.femtoseconds
        )
        self._openmm_energy_bridge = OpenMMBridge(self.system, self.integrator)
        self._last_batch = None

    def _energy(self, batch, no_grads=False):
        # check if we have already computed this energy (hash of string representation should be sufficient)
        if hash(str(batch)) == self._last_batch:
            return self._openmm_energy_bridge.last_energies
        else:
            self._last_batch = hash(str(batch))
            return _evaluate_openmm_energy(batch, self._openmm_energy_bridge)

    def force(self, batch, temperature=None):
        # check if we have already computed this energy
        if hash(str(batch)) == self._last_batch:
            return self._openmm_energy_bridge.last_forces
        else:
            self._last_batch = hash(str(batch))
            return self._openmm_energy_bridge.evaluate(batch)[1]
