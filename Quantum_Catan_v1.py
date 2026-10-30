from __future__ import annotations

# Quantum_Catan

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import qiskit
import scipy

from qiskit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate, RYGate
from qiskit.quantum_info import Operator, Pauli, Statevector

from scipy.linalg import expm_frechet
from scipy.optimize import minimize
from scipy.special import logsumexp


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_2977.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4710_k86_loto_plus_1733.csv"),
)


M = 39
K = 7
QUBITS = 6
DIM = 64
STATES = M + 1
START = M

MIXER_PARAMETERS = 30
AMPLITUDE_PARAMETERS = K * M
CRITIC_PARAMETERS = K * 6

TIME_STEP = 1.0
HISTORY_SCALE = 0.65
HISTORY_WEIGHTS = (1.0, 0.5, 0.25)

MIXER_BOUND = 0.75
FILTER_BOUND = 1.40

RIDGES = (0.3, 3.0)
MAX_PPO_ROUNDS = 2

BOOTSTRAP_STEPS = 60
CRITIC_STEPS = 20
PPO_STEPS = 12
MAX_FUNCTION_CALLS = 350

CLIP_RATIO = 0.15
KL_TARGET = 0.02
KL_PENALTY = 0.20
IMITATION_WEIGHT = 0.05
CRITIC_RIDGE = 1.0e-4

GRADIENT_LIMIT = 5.0e-5
CHECKPOINT_EVERY = 5

VERSION = "QUANTUM_CATAN_V1_SEPARATE_ACTOR_20261029"

CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "Quantum_Catan_V1_separate_actor_checkpoints"
)


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def finite(values, message):
    require(np.all(np.isfinite(values)), message)


def safe_log(values):
    result = np.zeros_like(values, dtype=float)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result


class SelectionGraph:
    """
    Rastuća putanja predstavlja jedan skup brojeva.

    P(S) proporcionalna je proizvodu potencijala cele putanje.
    Normalizacija obuhvata sve legalne skupove.

    Masa nastavaka odnosi se na preostale izbore iste sedmorke.
    Ne koristi buduća CSV kola.
    """

    def __init__(self, numbers, picks):
        self.m = int(numbers)
        self.k = int(picks)
        self.s = self.m + 1
        self.start = self.m

        require(
            1 <= self.k <= self.m,
            "Neispravna veličina grafa.",
        )

        self.mask = np.zeros(
            (self.k, self.s, self.m),
            dtype=bool,
        )

        for step in range(self.k):
            origins = (
                (self.start,)
                if step == 0
                else range(
                    step - 1,
                    self.m - self.k + step,
                )
            )

            maximum = self.m - self.k + step

            for origin in origins:
                previous = -1 if origin == self.start else origin

                self.mask[
                    step,
                    origin,
                    previous + 1:maximum + 1,
                ] = True

        self.active = np.any(self.mask, axis=2)

    def backward(self, a, b):
        require(
            a.ndim == 3
            and a.shape[1:] == (self.k, self.m)
            and b.shape == (self.s, self.m),
            "Neispravan oblik potencijala.",
        )

        require(
            np.all(np.isfinite(a))
            and np.all(np.isfinite(b))
            and np.all(a > 0.0)
            and np.all(b > 0.0),
            "Potencijali nisu konačni i pozitivni.",
        )

        suffix = np.full(
            (len(a), self.k + 1, self.s),
            -np.inf,
        )

        suffix[:, self.k, :] = 0.0

        log_a = np.log(a)
        log_b = np.log(b)

        for step in range(self.k - 1, -1, -1):
            origins = np.flatnonzero(self.active[step])

            scores = (
                log_a[:, step, None, :]
                + log_b[origins][None, :, :]
                + suffix[:, step + 1, :self.m][:, None, :]
            )

            scores = np.where(
                self.mask[step, origins][None, :, :],
                scores,
                -np.inf,
            )

            suffix[:, step, origins] = logsumexp(
                scores,
                axis=2,
            )

        finite(
            suffix[:, 0, self.start],
            "Globalna normalizacija nije konačna.",
        )

        return suffix

    def policy(self, a, b, step, suffix):
        result = np.zeros(
            (len(a), self.s, self.m),
        )

        origins = np.flatnonzero(self.active[step])

        log_probability = (
            np.log(a[:, step, :])[:, None, :]
            + np.log(b[origins])[None, :, :]
            + suffix[:, step + 1, :self.m][:, None, :]
            - suffix[:, step, origins][:, :, None]
        )

        log_probability = np.where(
            self.mask[step, origins][None, :, :],
            log_probability,
            -np.inf,
        )

        result[:, origins, :] = np.exp(log_probability)

        require(
            np.max(
                np.abs(
                    np.sum(result[:, origins, :], axis=2) - 1.0
                )
            ) < 2.0e-9,
            "Uslovna politika nije normalizovana.",
        )

        return result

    def empty_adjoints(self, a, b):
        return (
            np.zeros_like(a),
            np.zeros_like(b),
            np.zeros(
                (len(a), self.k + 1, self.s),
            ),
        )

    def seed_observation(
        self,
        draws,
        coefficient,
        g_log_a,
        g_log_b,
        g_suffix,
    ):
        rows = np.arange(len(draws))

        previous = np.full(
            len(draws),
            self.start,
            dtype=np.int64,
        )

        g_suffix[:, 0, self.start] += coefficient

        for step in range(self.k):
            chosen = draws[:, step]

            g_log_a[rows, step, chosen] -= coefficient

            np.add.at(
                g_log_b,
                (previous, chosen),
                -coefficient,
            )

            previous = chosen

    def backward_gradient(
        self,
        a,
        b,
        suffix,
        g_log_a,
        g_log_b,
        g_suffix,
    ):
        for step in range(self.k):
            p = self.policy(a, b, step, suffix)

            edge_gradient = (
                g_suffix[:, step, :, None] * p
            )

            action_sum = np.sum(edge_gradient, axis=1)

            g_log_a[:, step, :] += action_sum
            g_log_b += np.sum(edge_gradient, axis=0)
            g_suffix[:, step + 1, :self.m] += action_sum

        return g_log_a / a, g_log_b / b

    def observed(
        self,
        a,
        b,
        draws,
        gradient=False,
        suffix=None,
    ):
        count = len(a)

        require(
            draws.shape == (count, self.k)
            and np.all(draws >= 0)
            and np.all(draws < self.m)
            and np.all(np.diff(draws, axis=1) > 0),
            "Ciljni skup nije legalan.",
        )

        if suffix is None:
            suffix = self.backward(a, b)

        rows = np.arange(count)
        log_probability = -suffix[:, 0, self.start].copy()

        previous = np.full(
            count,
            self.start,
            dtype=np.int64,
        )

        for step in range(self.k):
            chosen = draws[:, step]

            log_probability += (
                np.log(a[rows, step, chosen])
                + np.log(b[previous, chosen])
            )

            previous = chosen

        finite(
            log_probability,
            "Logaritamska verovatnoća nije konačna.",
        )

        require(
            np.all(log_probability <= 2.0e-8),
            "Verovatnoća skupa prelazi jedan.",
        )

        value = -float(np.mean(log_probability))

        if not gradient:
            return value, None, None

        ga, gb, gs = self.empty_adjoints(a, b)

        self.seed_observation(
            draws,
            1.0 / count,
            ga,
            gb,
            gs,
        )

        ga, gb = self.backward_gradient(
            a, b, suffix, ga, gb, gs
        )

        return value, ga, gb

    def rollout(self, a, b, rewards=None, suffix=None):
        if suffix is None:
            suffix = self.backward(a, b)

        visits = np.zeros(
            (len(a), self.k + 1, self.s),
        )

        visits[:, 0, self.start] = 1.0

        for step in range(self.k):
            p = self.policy(a, b, step, suffix)

            visits[:, step + 1, :self.m] = np.einsum(
                "ni,nij->nj",
                visits[:, step],
                p,
                optimize=True,
            )

        require(
            np.max(
                np.abs(np.sum(visits, axis=2) - 1.0)
            ) < 2.0e-8,
            "Putanje ne čuvaju ukupnu verovatnoću.",
        )

        inclusion = np.sum(
            visits[:, 1:, :self.m],
            axis=1,
        )

        require(
            np.all(inclusion >= 0.0)
            and np.all(inclusion <= 1.0 + 2.0e-8),
            "Marginalna verovatnoća je izvan granica.",
        )

        require(
            np.max(
                np.abs(np.sum(inclusion, axis=1) - self.k)
            ) < 2.0e-7,
            "Zbir marginala nije jednak broju izbora.",
        )

        values = None

        if rewards is not None:
            require(
                rewards.shape == (len(a), self.m),
                "Neispravan oblik nagrada.",
            )

            values = np.zeros_like(visits)

            for step in range(self.k - 1, -1, -1):
                p = self.policy(a, b, step, suffix)

                future = (
                    rewards + values[:, step + 1, :self.m]
                )

                values[:, step] = np.sum(
                    p * future[:, None, :],
                    axis=2,
                )

            expected = np.sum(
                inclusion * rewards,
                axis=1,
            )

            require(
                np.max(
                    np.abs(
                        expected - values[:, 0, self.start]
                    )
                ) < 2.0e-8,
                "Bellmanova i marginalna nagrada se ne slažu.",
            )

        return visits, inclusion, values

    def maximum(self, a, b, suffix=None):
        if suffix is None:
            suffix = self.backward(a, b)

        count = len(a)
        rows = np.arange(count)

        future = np.zeros((count, self.s))

        choices = np.full(
            (count, self.k, self.s),
            -1,
            dtype=np.int64,
        )

        log_a = np.log(a)
        log_b = np.log(b)

        for step in range(self.k - 1, -1, -1):
            current = np.full_like(future, -np.inf)

            for origin in np.flatnonzero(self.active[step]):
                legal = np.flatnonzero(self.mask[step, origin])

                scores = (
                    log_a[:, step, :][:, legal]
                    + log_b[origin, legal][None, :]
                    + future[:, legal]
                )

                best = np.argmax(scores, axis=1)

                current[:, origin] = scores[rows, best]
                choices[:, step, origin] = legal[best]

            future = current

        log_probability = (
            future[:, self.start] - suffix[:, 0, self.start]
        )

        paths = np.empty(
            (count, self.k),
            dtype=np.int64,
        )

        previous = np.full(
            count,
            self.start,
            dtype=np.int64,
        )

        for step in range(self.k):
            paths[:, step] = choices[rows, step, previous]
            previous = paths[:, step]

        require(
            np.all(np.diff(paths, axis=1) > 0),
            "MAP putanja nije legalna.",
        )

        require(
            np.all(
                log_probability
                >= -math.log(math.comb(self.m, self.k)) - 2.0e-8
            )
            and np.all(log_probability <= 2.0e-8),
            "MAP verovatnoća ne prolazi granice.",
        )

        nll, _, _ = self.observed(
            a, b, paths, suffix=suffix
        )

        require(
            abs(nll + float(np.mean(log_probability))) < 2.0e-9,
            "MAP i direktna verovatnoća putanje se ne slažu.",
        )

        return paths, log_probability


GRAPH = SelectionGraph(M, K)


def pauli_label(operators):
    labels = ["I"] * QUBITS

    for qubit, axis in operators:
        labels[QUBITS - 1 - qubit] = axis

    return "".join(labels)


class QuantumActor:
    """
    Tri odvojene grupe parametara:

    1. Kvantni mešač: Hamiltonijan bez fiksnog dodatka.
    2. Amplitudni filteri za svaki slot i broj.
    3. Prelazni filteri za svaki legalni par.

    Nijedan parametar prelaza ne menja amplitudne filtere.
    Nijedan parametar amplitudnog filtera ne menja prelaze.
    """

    def __init__(self):
        labels = []

        for axis in ("X", "Y", "Z"):
            for qubit in range(QUBITS):
                labels.append(pauli_label(((qubit, axis),)))

        for axis in ("Z", "X"):
            for qubit in range(QUBITS):
                labels.append(
                    pauli_label(
                        (
                            (qubit, axis),
                            ((qubit + 1) % QUBITS, axis),
                        )
                    )
                )

        self.basis = np.asarray(
            [Pauli(label).to_matrix() for label in labels],
            dtype=np.complex128,
        )

        require(
            len(self.basis) == MIXER_PARAMETERS,
            "Neispravna baza kvantnog mešača.",
        )

        self.edge_i, self.edge_j = np.where(
            np.any(GRAPH.mask, axis=0)
        )

        self.edge_count = len(self.edge_i)

        self.a_start = MIXER_PARAMETERS
        self.b_start = self.a_start + AMPLITUDE_PARAMETERS
        self.size = self.b_start + self.edge_count

        self.bounds = (
            [(-MIXER_BOUND, MIXER_BOUND)] * MIXER_PARAMETERS
            + [(-FILTER_BOUND, FILTER_BOUND)]
            * (AMPLITUDE_PARAMETERS + self.edge_count)
        )

        preparation = QuantumCircuit(QUBITS)

        for qubit in range(QUBITS):
            preparation.h(qubit)

        self.preparation = preparation

        self.initial = np.asarray(
            Statevector.from_instruction(preparation).data,
            dtype=np.complex128,
        )

        require(
            np.max(
                np.abs(np.abs(self.initial) ** 2 - 1.0 / DIM)
            ) < 1.0e-12,
            "Referentno pripremno kolo nije usaglašeno.",
        )

    def initial_parameters(self):
        return np.zeros(self.size)

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=float)

        require(
            theta.shape == (self.size,),
            "Pogrešan broj parametara kvantnog aktera.",
        )

        finite(theta, "Parametri aktera nisu konačni.")

        return (
            theta[:self.a_start],
            theta[self.a_start:self.b_start].reshape(K, M),
            theta[self.b_start:],
        )

    @staticmethod
    def ry_probabilities(offsets):
        offsets = np.asarray(offsets, dtype=float)
        flat = offsets.ravel()

        probabilities = np.empty(len(flat))

        # Izlazne težine dolaze iz matrica Qiskit RY kapija.
        for index, offset in enumerate(flat):
            angle = math.pi / 2.0 + float(offset)
            matrix = RYGate(angle).to_matrix()
            probabilities[index] = abs(matrix[1, 0]) ** 2

        derivatives = 0.5 * np.cos(flat)

        require(
            np.all(probabilities > 0.0)
            and np.all(probabilities < 1.0),
            "Amplitudni filter je izašao iz otvorenog domena.",
        )

        return (
            probabilities.reshape(offsets.shape),
            derivatives.reshape(offsets.shape),
        )

    def filter_values(self, theta):
        _, a_offsets, b_offsets = self.unpack(theta)

        fa, dfa = self.ry_probabilities(a_offsets)
        fb, dfb = self.ry_probabilities(b_offsets)

        b = np.full((STATES, M), 0.5)
        b[self.edge_i, self.edge_j] = fb

        return fa, dfa, b, dfb

    def operator(self, theta):
        mixer, _, _ = self.unpack(theta)

        h = np.einsum(
            "p,pij->ij",
            mixer,
            self.basis,
            optimize=True,
        )

        require(
            np.max(np.abs(h - h.conj().T)) < 1.0e-12,
            "Hamiltonijan nije Hermitski.",
        )

        circuit = QuantumCircuit(QUBITS)

        circuit.append(
            HamiltonianGate(h, time=TIME_STEP),
            list(range(QUBITS)),
        )

        u = np.asarray(
            Operator(circuit).data,
            dtype=np.complex128,
        )

        require(
            np.max(
                np.abs(u.conj().T @ u - np.eye(DIM))
            ) < 2.0e-10,
            "Qiskit operator nije unitaran.",
        )

        return h, u

    def contexts(self, occupancy):
        features = np.zeros_like(occupancy, dtype=float)

        for lag, weight in enumerate(HISTORY_WEIGHTS):
            if lag == 0:
                features += weight * occupancy
            else:
                features[lag:] += weight * occupancy[:-lag]

        phases = np.zeros(
            (len(features), K, DIM),
        )

        phases[:, :, :M] = (
            HISTORY_SCALE * features[:, None, :]
        )

        labels = np.arange(1, DIM + 1, dtype=float)

        for step in range(K):
            phases[:, step] += (
                0.17
                * (step + 1)
                * np.sin(0.30 * labels)
            )

        contexts = (
            self.initial[None, None, :]
            * np.exp(1j * phases)
        )

        require(
            np.max(
                np.abs(
                    np.sum(np.abs(contexts) ** 2, axis=2) - 1.0
                )
            ) < 1.0e-11,
            "Istorijska kvantna stanja nisu normalizovana.",
        )

        return np.ascontiguousarray(contexts)

    def forward(self, theta, contexts):
        h, u = self.operator(theta)
        fa, dfa, b, dfb = self.filter_values(theta)

        output = (
            contexts.reshape(-1, DIM) @ u.T
        ).reshape(contexts.shape)

        finite(output, "Kvantne amplitude nisu konačne.")

        require(
            np.max(
                np.abs(
                    np.sum(np.abs(output) ** 2, axis=2) - 1.0
                )
            ) < 2.0e-10,
            "Kvantni izlaz nije normalizovan.",
        )

        q = np.abs(output[:, :, :M]) ** 2
        a = q * fa[None, :, :]

        require(
            np.all(a > 0.0) and np.all(b > 0.0),
            "Kandidat je na nultoj granici logaritamskog domena.",
        )

        cache = (output, h, u, q, fa, dfa, dfb)

        return a, b, cache

    def gradient(self, contexts, cache, ga, gb):
        output, h, u, q, fa, dfa, dfb = cache

        gradient = np.zeros(self.size)

        g_a_offsets = np.sum(
            ga * q * dfa[None, :, :],
            axis=0,
        )

        gradient[self.a_start:self.b_start] = g_a_offsets.ravel()

        gradient[self.b_start:] = (
            gb[self.edge_i, self.edge_j] * dfb
        )

        adjoint = np.zeros_like(output)

        adjoint[:, :, :M] = (
            2.0
            * output[:, :, :M]
            * ga
            * fa[None, :, :]
        )

        g_u = (
            adjoint.reshape(-1, DIM).T
            @ contexts.reshape(-1, DIM).conj()
        )

        generator = -1j * TIME_STEP * h

        exponential, g_generator = expm_frechet(
            generator.conj().T,
            g_u,
            compute_expm=True,
        )

        require(
            np.max(
                np.abs(exponential - u.conj().T)
            ) < 2.0e-9,
            "Gradijent mešača odstupa od Qiskit operatora.",
        )

        directions = -1j * TIME_STEP * self.basis

        gradient[:self.a_start] = np.real(
            np.einsum(
                "ij,pij->p",
                g_generator.conj(),
                directions,
                optimize=True,
            )
        )

        finite(gradient, "Gradijent aktera nije konačan.")

        return gradient

    def regularizer(self, theta, ridge):
        gradient = np.zeros_like(theta)
        value = 0.0

        groups = (
            slice(0, self.a_start),
            slice(self.a_start, self.b_start),
            slice(self.b_start, self.size),
        )

        for group in groups:
            values = theta[group]
            value += 0.5 * ridge * float(np.mean(values ** 2))
            gradient[group] = ridge * values / len(values)

        return value, gradient

    def explicit_policy(
        self,
        theta,
        context,
        origin,
        step,
        suffix_next,
    ):
        h, _ = self.operator(theta)
        fa, _, b, _ = self.filter_values(theta)

        legal = np.flatnonzero(GRAPH.mask[step, origin])

        log_filter = np.full(DIM, -np.inf)

        log_filter[legal] = (
            np.log(fa[step, legal])
            + np.log(b[origin, legal])
            + suffix_next[legal]
        )

        finite(
            log_filter[legal],
            "Kvantni filter nema konačne nastavke.",
        )

        shift = float(np.max(log_filter[legal]))

        filter_probability = np.zeros(DIM)

        filter_probability[legal] = np.exp(
            log_filter[legal] - shift
        )

        require(
            np.all(filter_probability >= 0.0)
            and np.all(filter_probability <= 1.0),
            "Neispravna uspešna grana filtera.",
        )

        success = np.diag(np.sqrt(filter_probability))
        failure = np.diag(np.sqrt(1.0 - filter_probability))

        unitary_filter = np.block(
            [
                [success, -failure],
                [failure, success],
            ]
        )

        require(
            np.max(
                np.abs(
                    unitary_filter.conj().T
                    @ unitary_filter
                    - np.eye(2 * DIM)
                )
            ) < 2.0e-10,
            "Kvantni filter nije unitaran.",
        )

        phase = context / self.initial

        require(
            np.max(
                np.abs(np.abs(phase) - 1.0)
            ) < 2.0e-10,
            "Istorijski faktor nije fazna kapija.",
        )

        circuit = QuantumCircuit(QUBITS + 1)

        circuit.compose(
            self.preparation,
            qubits=list(range(QUBITS)),
            inplace=True,
        )

        circuit.unitary(
            np.diag(phase),
            list(range(QUBITS)),
            label="History_and_slot",
        )

        circuit.append(
            HamiltonianGate(h, time=TIME_STEP),
            list(range(QUBITS)),
        )

        circuit.unitary(
            unitary_filter,
            list(range(QUBITS + 1)),
            label="Learned_filters_and_continuation",
        )

        state = np.asarray(
            Statevector.from_instruction(circuit).data
        )

        probabilities = np.abs(state[:DIM]) ** 2
        mass = float(np.sum(probabilities))

        require(mass > 0.0, "Kvantni filter nema uspešnu masu.")

        return probabilities[:M] / mass


class QuantumCritic:
    def __init__(self):
        self.inputs = np.empty(
            (STATES, 2, 4),
            dtype=np.complex128,
        )

        for origin in range(STATES):
            angle = 0.20 + 2.40 * (origin + 1) / STATES

            for code in range(2):
                circuit = QuantumCircuit(2)
                circuit.ry(angle, 1)
                circuit.rz(0.37, 1)
                circuit.ry(0.29, 0)

                if code:
                    circuit.x(0)

                self.inputs[origin, code] = (
                    Statevector.from_instruction(circuit).data
                )

        first = QuantumCircuit(2)
        first.cx(0, 1)

        second = QuantumCircuit(2)
        second.cx(1, 0)

        self.cx01 = Operator(first).data
        self.cx10 = Operator(second).data

        self.generators = (
            Pauli("IY").to_matrix(),
            Pauli("IZ").to_matrix(),
            Pauli("YI").to_matrix(),
            Pauli("ZI").to_matrix(),
            Pauli("IY").to_matrix(),
            Pauli("XI").to_matrix(),
        )

    @staticmethod
    def initial_parameters():
        return (
            0.25
            + 0.05
            * np.sin(np.arange(CRITIC_PARAMETERS) + 0.3)
        )

    def stage(self, theta):
        circuit = QuantumCircuit(2)

        circuit.ry(float(theta[0]), 0)
        circuit.rz(float(theta[1]), 0)
        circuit.ry(float(theta[2]), 1)
        circuit.rz(float(theta[3]), 1)
        circuit.cx(0, 1)
        circuit.ry(float(theta[4]), 0)
        circuit.rx(float(theta[5]), 1)
        circuit.cx(1, 0)

        u = np.asarray(Operator(circuit).data)

        entries = (
            (0, self.generators[0]),
            (1, self.generators[1]),
            (2, self.generators[2]),
            (3, self.generators[3]),
            (None, self.cx01),
            (4, self.generators[4]),
            (5, self.generators[5]),
            (None, self.cx10),
        )

        gates = []

        for parameter, value in entries:
            if parameter is None:
                gates.append(value)
            else:
                gates.append(
                    math.cos(theta[parameter] / 2.0)
                    * np.eye(4)
                    - 1j
                    * math.sin(theta[parameter] / 2.0)
                    * value
                )

        prefixes = [np.eye(4, dtype=complex)]

        for gate in gates:
            prefixes.append(gate @ prefixes[-1])

        require(
            np.max(np.abs(prefixes[-1] - u)) < 2.0e-11,
            "Razlaganje kritičara odstupa od Qiskit kola.",
        )

        derivatives = np.empty((6, 4, 4), dtype=complex)
        suffix = np.eye(4, dtype=complex)

        for index in range(len(entries) - 1, -1, -1):
            parameter, generator = entries[index]

            if parameter is not None:
                derivatives[parameter] = (
                    suffix
                    @ (
                        -0.5j
                        * generator
                        @ prefixes[index + 1]
                    )
                )

            suffix = suffix @ gates[index]

        flat = self.inputs.reshape(-1, 4)
        output = flat @ u.T

        probability = np.sum(
            np.abs(output[:, (1, 3)]) ** 2,
            axis=1,
        )

        jacobian = np.empty((len(flat), 6))

        for parameter in range(6):
            delta = flat @ derivatives[parameter].T

            jacobian[:, parameter] = (
                2.0
                * np.real(
                    np.sum(
                        output[:, (1, 3)].conj()
                        * delta[:, (1, 3)],
                        axis=1,
                    )
                )
            )

        require(
            np.all(probability >= 0.0)
            and np.all(probability <= 1.0 + 1.0e-10),
            "Kritičar daje neispravnu verovatnoću.",
        )

        return (
            probability.reshape(STATES, 2),
            jacobian.reshape(STATES, 2, 6),
        )

    def lookup(self, theta):
        theta = np.asarray(theta).reshape(K, 6)

        table = np.empty((K, STATES, 2))
        jacobian = np.empty((K, STATES, 2, 6))

        for step in range(K):
            p, derivative = self.stage(theta[step])
            scale = (K - step) / K

            table[step] = scale * p
            jacobian[step] = scale * derivative

        return table, jacobian

    @staticmethod
    def select(table, codes):
        origins = np.arange(STATES)[None, :]

        return [
            table[step][origins, codes]
            for step in range(K)
        ] + [np.zeros((len(codes), STATES))]


@dataclass
class Data:
    contexts: np.ndarray
    codes: np.ndarray
    draws: np.ndarray
    occupancy: np.ndarray

    def __len__(self):
        return len(self.draws)

    def subset(self, start, end=None):
        return Data(
            self.contexts[start:end],
            self.codes[start:end],
            self.draws[start:end],
            self.occupancy[start:end],
        )

    def prefix(self, count):
        return self.subset(0, count)


def bootstrap_objective(theta, actor, data, ridge):
    a, b, cache = actor.forward(theta, data.contexts)

    suffix = GRAPH.backward(a, b)

    value, ga, gb = GRAPH.observed(
        a,
        b,
        data.draws,
        gradient=True,
        suffix=suffix,
    )

    gradient = actor.gradient(
        data.contexts, cache, ga, gb
    )

    penalty, penalty_gradient = actor.regularizer(theta, ridge)

    return (
        value + penalty,
        gradient + penalty_gradient,
        {"kl": 0.0},
    )


def critic_objective(theta, critic, data, visits, targets):
    table, jacobian = critic.lookup(theta)

    gradient = np.zeros((K, 6))
    value = 0.0

    origins = np.arange(STATES)[None, :]
    denominator = len(data) * K

    for step in range(K):
        prediction = table[step][origins, data.codes]
        derivative = jacobian[step][origins, data.codes]

        delta = prediction - targets[:, step]

        weighted = (
            visits[:, step]
            * delta
            / denominator
        )

        value += (
            0.5
            * float(
                np.sum(visits[:, step] * delta ** 2)
            )
            / denominator
        )

        gradient[step] = np.sum(
            weighted[:, :, None] * derivative,
            axis=(0, 1),
        )

    value += (
        0.5
        * CRITIC_RIDGE
        * float(np.mean(theta ** 2))
    )

    result = gradient.ravel()
    result += CRITIC_RIDGE * theta / len(theta)

    finite(result, "Gradijent kritičara nije konačan.")

    return value, result, {"kl": 0.0}


def ppo_objective(
    theta,
    actor,
    data,
    ridge,
    old_a,
    old_b,
    old_suffix,
    visits,
    critic_values,
):
    a, b, cache = actor.forward(theta, data.contexts)
    suffix = GRAPH.backward(a, b)

    count = len(data)
    rewards = data.occupancy / K

    ga, gb, gs = GRAPH.empty_adjoints(a, b)

    surrogate_loss = 0.0
    kl = 0.0

    for step in range(K):
        old_p = GRAPH.policy(
            old_a, old_b, step, old_suffix
        )

        new_p = GRAPH.policy(a, b, step, suffix)

        advantage = (
            rewards[:, None, :]
            + critic_values[step + 1][:, None, :M]
            - critic_values[step][:, :, None]
        )

        ratio = np.ones_like(new_p)

        np.divide(
            new_p,
            old_p,
            out=ratio,
            where=old_p > 0.0,
        )

        clipped = np.clip(
            ratio,
            1.0 - CLIP_RATIO,
            1.0 + CLIP_RATIO,
        )

        old_weight = (
            visits[:, step, :, None] * old_p
        )

        surrogate_loss -= (
            float(
                np.sum(
                    old_weight
                    * np.minimum(
                        ratio * advantage,
                        clipped * advantage,
                    )
                )
            )
            / count
        )

        active_derivative = (
            (
                (advantage >= 0.0)
                & (ratio <= 1.0 + CLIP_RATIO)
            )
            | (
                (advantage < 0.0)
                & (ratio >= 1.0 - CLIP_RATIO)
            )
        )

        g_probability = (
            -visits[:, step, :, None]
            * advantage
            * active_derivative
            * (old_p > 0.0)
            / count
        )

        kl += (
            float(
                np.sum(
                    old_weight
                    * (
                        safe_log(old_p)
                        - safe_log(new_p)
                    )
                )
            )
            / (count * K)
        )

        g_kl = np.zeros_like(new_p)

        np.divide(
            -old_weight,
            new_p,
            out=g_kl,
            where=new_p > 0.0,
        )

        g_probability += (
            KL_PENALTY * g_kl / (count * K)
        )

        g_log_policy = new_p * g_probability
        action_sum = np.sum(g_log_policy, axis=1)

        ga[:, step] += action_sum
        gb += np.sum(g_log_policy, axis=0)
        gs[:, step + 1, :M] += action_sum
        gs[:, step] -= np.sum(g_log_policy, axis=2)

    nll, _, _ = GRAPH.observed(
        a,
        b,
        data.draws,
        suffix=suffix,
    )

    GRAPH.seed_observation(
        data.draws,
        IMITATION_WEIGHT / count,
        ga,
        gb,
        gs,
    )

    ga, gb = GRAPH.backward_gradient(
        a, b, suffix, ga, gb, gs
    )

    gradient = actor.gradient(
        data.contexts, cache, ga, gb
    )

    penalty, penalty_gradient = actor.regularizer(theta, ridge)

    value = (
        surrogate_loss
        + IMITATION_WEIGHT * nll
        + KL_PENALTY * kl
        + penalty
    )

    gradient += penalty_gradient

    require(
        np.isfinite(value) and kl >= -2.0e-8,
        "Neispravna PPO funkcija ili KL divergencija.",
    )

    finite(gradient, "PPO gradijent nije konačan.")

    return value, gradient, {"kl": max(0.0, kl)}


def digest_values(*values):
    digest = hashlib.sha256()

    configuration = (
        VERSION,
        qiskit.__version__,
        np.__version__,
        scipy.__version__,
        TIME_STEP,
        HISTORY_SCALE,
        HISTORY_WEIGHTS,
        MIXER_BOUND,
        FILTER_BOUND,
        RIDGES,
        BOOTSTRAP_STEPS,
        CRITIC_STEPS,
        PPO_STEPS,
        MAX_FUNCTION_CALLS,
        CLIP_RATIO,
        KL_TARGET,
        KL_PENALTY,
        IMITATION_WEIGHT,
        CRITIC_RIDGE,
    )

    digest.update(repr(configuration).encode("utf-8"))

    for value in values:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)

            digest.update(
                repr(
                    (array.shape, str(array.dtype))
                ).encode("utf-8")
            )

            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))

    return digest.hexdigest()


def save_checkpoint(path, theta, iterations, done, message):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=np.asarray(VERSION),
            theta=theta,
            iterations=np.asarray(iterations),
            done=np.asarray(done),
            message=np.asarray(message),
        )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def projected_norm(theta, gradient, bounds):
    lower = np.asarray([pair[0] for pair in bounds])
    upper = np.asarray([pair[1] for pair in bounds])

    result = gradient.copy()

    result[
        (theta <= lower + 1.0e-10)
        & (result > 0.0)
    ] = 0.0

    result[
        (theta >= upper - 1.0e-10)
        & (result < 0.0)
    ] = 0.0

    return float(np.max(np.abs(result)))


class StopForKL(Exception):
    pass


@dataclass
class Optimization:
    theta: np.ndarray
    gradient_norm: float
    message: str
    kl: float

    @property
    def converged(self):
        return self.gradient_norm <= GRADIENT_LIMIT


def optimize(
    function,
    initial,
    steps,
    bounds,
    key,
    label,
    limit_kl=False,
):
    path = (
        CHECKPOINT_DIRECTORY
        / f"{digest_values(key, initial, steps, tuple(bounds))}.npz"
    )

    theta = np.asarray(initial, dtype=float).copy()
    lower = np.asarray([pair[0] for pair in bounds])
    upper = np.asarray([pair[1] for pair in bounds])

    used = 0
    done = False
    message = "Početni kandidat"

    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                require(
                    saved["version"].item() == VERSION,
                    "Nepodudarna verzija kontrolne tačke.",
                )

                candidate = saved["theta"].copy()
                candidate_used = int(saved["iterations"].item())

                require(
                    candidate.shape == theta.shape
                    and np.all(np.isfinite(candidate))
                    and np.all(candidate >= lower - 1.0e-10)
                    and np.all(candidate <= upper + 1.0e-10)
                    and 0 <= candidate_used <= steps,
                    "Neispravna kontrolna tačka.",
                )

                candidate_done = bool(saved["done"].item())
                candidate_message = str(saved["message"].item())

            _, _, info = function(candidate)

            require(
                not limit_kl
                or info["kl"] <= KL_TARGET + 1.0e-10,
                "Sačuvani kandidat prelazi KL prag.",
            )

            theta = candidate
            used = candidate_used
            done = candidate_done
            message = candidate_message

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            BadZipFile,
            EOFError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            say(
                f"Neupotrebljiva kontrolna tačka: {label}",
                str(error),
            )

            theta = np.asarray(initial, dtype=float).copy()
            used = 0
            done = False
            message = "Početak bez neispravne kontrolne tačke"

    value, gradient, info = function(theta)

    require(
        not limit_kl
        or info["kl"] <= KL_TARGET + 1.0e-10,
        "Početni PPO kandidat prelazi KL prag.",
    )

    best = {
        "theta": theta.copy(),
        "value": float(value),
        "gradient": gradient.copy(),
        "info": dict(info),
    }

    prior = used
    local_iterations = 0
    started = time.monotonic()

    cached_theta = None
    cached_result = None

    say("", label)

    if done:
        say("Koristim završenu kontrolnu tačku ove konstrukcije.")

    def evaluate(candidate):
        nonlocal cached_theta, cached_result

        if (
            cached_theta is not None
            and np.array_equal(candidate, cached_theta)
        ):
            return cached_result

        result = function(candidate)
        candidate_value, candidate_gradient, candidate_info = result

        require(
            np.isfinite(candidate_value),
            "Funkcija optimizacije nije konačna.",
        )

        finite(
            candidate_gradient,
            "Gradijent optimizacije nije konačan.",
        )

        eligible = (
            not limit_kl
            or candidate_info["kl"] <= KL_TARGET + 1.0e-10
        )

        if eligible and candidate_value < best["value"]:
            best["theta"] = candidate.copy()
            best["value"] = float(candidate_value)
            best["gradient"] = candidate_gradient.copy()
            best["info"] = dict(candidate_info)

        cached_theta = candidate.copy()
        cached_result = result

        return result

    def fun(candidate):
        candidate_value, candidate_gradient, _ = evaluate(candidate)
        return candidate_value, candidate_gradient

    def callback(candidate):
        nonlocal local_iterations, used

        local_iterations += 1
        used = min(steps, prior + local_iterations)

        _, _, candidate_info = evaluate(candidate)

        if local_iterations % CHECKPOINT_EVERY == 0:
            say(
                f"Iteracija {used}/{steps}"
                f" | cilj {best['value']:.9f}"
                f" | KL {best['info']['kl']:.6f}"
                f" | proteklo {time.monotonic() - started:.1f} s"
            )

            save_checkpoint(
                path,
                best["theta"],
                used,
                False,
                "Trening je u toku",
            )

        if limit_kl and candidate_info["kl"] > KL_TARGET:
            raise StopForKL()

    if not done and used < steps:
        save_checkpoint(
            path,
            best["theta"],
            used,
            False,
            message,
        )

        try:
            result = minimize(
                fun,
                theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                callback=callback,
                options={
                    "maxiter": steps - used,
                    "maxfun": MAX_FUNCTION_CALLS,
                    "maxls": 25,
                    "maxcor": 12,
                    "ftol": 1.0e-11,
                    "gtol": 1.0e-6,
                },
            )

            used = min(steps, prior + int(result.nit))
            message = str(result.message)

        except StopForKL:
            message = (
                "Dostignut KL prag; zadržan najbolji kandidat unutar praga."
            )

        except KeyboardInterrupt:
            save_checkpoint(
                path,
                best["theta"],
                used,
                False,
                "Korisnički prekid",
            )
            raise

        except (
            RuntimeError,
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            message = (
                "Zadržan najbolji validan kandidat. "
                f"Razlog prekida: {error}"
            )

    value, gradient, info = function(best["theta"])

    norm = projected_norm(
        best["theta"],
        gradient,
        bounds,
    )

    save_checkpoint(
        path,
        best["theta"],
        used,
        True,
        message,
    )

    say(
        f"Zadržani cilj: {value:.9f}",
        f"Max|proj.grad|: {norm:.6e}",
        f"KL: {info['kl']:.6f}",
        f"Status: {message}",
    )

    if norm > GRADIENT_LIMIT:
        say(
            "DIJAGNOSTIKA: gradijentni uslov nije potvrđen.",
            "Kandidat je validan; minimum nije potvrđen.",
        )

    return Optimization(
        best["theta"].copy(),
        norm,
        message,
        float(info["kl"]),
    )


@dataclass
class Model:
    actor: np.ndarray
    critic: np.ndarray
    rounds: int
    warnings: tuple
    gradient_norm: float


def train_models(
    actor,
    critic,
    data,
    ridge,
    rounds,
    label,
    warm_actor=None,
):
    initial = actor.initial_parameters()

    if warm_actor is not None:
        cold_value, _, _ = bootstrap_objective(
            initial, actor, data, ridge
        )

        warm_value, _, _ = bootstrap_objective(
            warm_actor, actor, data, ridge
        )

        if warm_value < cold_value:
            initial = warm_actor.copy()

    key = digest_values(
        data.contexts,
        data.codes,
        data.draws,
        data.occupancy,
        actor.basis,
        actor.edge_i,
        actor.edge_j,
        ridge,
    )

    bootstrap = optimize(
        lambda theta: bootstrap_objective(
            theta, actor, data, ridge
        ),
        initial,
        BOOTSTRAP_STEPS,
        actor.bounds,
        (key, "bootstrap"),
        f"{label} / početno kvantno učenje",
    )

    actor_theta = bootstrap.theta.copy()
    critic_theta = critic.initial_parameters()

    warnings = []

    if not bootstrap.converged:
        warnings.append(
            "Početno učenje nema potvrđen gradijentni uslov."
        )

    models = [
        Model(
            actor_theta.copy(),
            critic_theta.copy(),
            0,
            tuple(warnings),
            bootstrap.gradient_norm,
        )
    ]

    for round_index in range(1, rounds + 1):
        old_a, old_b, _ = actor.forward(
            actor_theta,
            data.contexts,
        )

        old_suffix = GRAPH.backward(old_a, old_b)

        visits, _, value_targets = GRAPH.rollout(
            old_a,
            old_b,
            rewards=data.occupancy / K,
            suffix=old_suffix,
        )

        old_actor_theta = actor_theta.copy()

        critic_fit = optimize(
            lambda theta: critic_objective(
                theta,
                critic,
                data,
                visits,
                value_targets,
            ),
            critic_theta,
            CRITIC_STEPS,
            [(-math.pi, math.pi)] * CRITIC_PARAMETERS,
            (
                key,
                "critic",
                round_index,
                digest_values(old_actor_theta),
            ),
            f"{label} / PPO {round_index} / kvantni kritičar",
        )

        critic_theta = critic_fit.theta.copy()

        table, _ = critic.lookup(critic_theta)
        critic_values = critic.select(table, data.codes)

        actor_fit = optimize(
            lambda theta: ppo_objective(
                theta,
                actor,
                data,
                ridge,
                old_a,
                old_b,
                old_suffix,
                visits,
                critic_values,
            ),
            old_actor_theta,
            PPO_STEPS,
            actor.bounds,
            (
                key,
                "ppo",
                round_index,
                digest_values(old_actor_theta, critic_theta),
            ),
            f"{label} / PPO {round_index} / kvantni akter",
            limit_kl=True,
        )

        actor_theta = actor_fit.theta.copy()

        if not critic_fit.converged:
            warnings.append(
                f"Kritičar u PPO rundi {round_index} nema potvrđen gradijentni uslov."
            )

        if not actor_fit.converged:
            warnings.append(
                f"Akter u PPO rundi {round_index} nema potvrđen gradijentni uslov."
            )

        models.append(
            Model(
                actor_theta.copy(),
                critic_theta.copy(),
                round_index,
                tuple(warnings),
                actor_fit.gradient_norm,
            )
        )

    return models


def evaluate(actor, model, data, hits=False):
    a, b, _ = actor.forward(
        model.actor,
        data.contexts,
    )

    suffix = GRAPH.backward(a, b)

    nll, _, _ = GRAPH.observed(
        a,
        b,
        data.draws,
        suffix=suffix,
    )

    histogram = None

    if hits:
        paths, _ = GRAPH.maximum(
            a, b, suffix=suffix
        )

        matches = np.sum(
            np.take_along_axis(
                data.occupancy,
                paths,
                axis=1,
            ),
            axis=1,
        ).astype(np.int64)

        histogram = np.bincount(
            matches,
            minlength=K + 1,
        )

        require(
            int(np.sum(histogram)) == len(data),
            "Neispravna statistika test pogodaka.",
        )

    return nll, histogram


def load_draws(path):
    fingerprint = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    rows = []
    empty = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, row in enumerate(
            csv.reader(handle),
            start=1,
        ):
            if not row or all(not item.strip() for item in row):
                empty += 1
                continue

            require(
                len(row) == K,
                f"{path.name}, red {line_number}: očekujem sedam kolona.",
            )

            try:
                values = [int(item.strip()) for item in row]
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(values)) == K
                and all(1 <= value <= M for value in values),
                f"{path.name}, red {line_number}: nelegalna sedmorka.",
            )

            rows.append(sorted(value - 1 for value in values))

    require(
        len(rows) >= 100,
        f"{path.name}: premalo redova.",
    )

    draws = np.asarray(rows, dtype=np.int64)
    occupancy = np.zeros((len(draws), M))

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {empty}",
        f"SHA256: {fingerprint}",
        "Redosled CSV-a ostaje hronološki.",
    )

    return draws, occupancy


def make_data(actor, draws, occupancy):
    contexts = actor.contexts(occupancy)

    codes = np.zeros(
        (len(draws), STATES),
        dtype=np.int64,
    )

    codes[:, :M] = occupancy.astype(np.int64)

    return (
        Data(
            contexts[:-1],
            codes[:-1],
            draws[1:],
            occupancy[1:],
        ),
        contexts,
    )


def gradient_check(function, theta, directions, label):
    _, analytic, _ = function(theta)

    epsilon = 1.0e-6
    maximum_error = 0.0

    for direction in directions:
        direction = np.asarray(direction, dtype=float)

        plus, _, _ = function(theta + epsilon * direction)
        minus, _, _ = function(theta - epsilon * direction)

        numerical = (plus - minus) / (2.0 * epsilon)
        expected = float(analytic @ direction)

        error = abs(numerical - expected)
        maximum_error = max(maximum_error, error)

        require(
            error < 8.0e-5 * (1.0 + abs(expected)),
            f"{label}: gradijent ne prolazi numeričku proveru.",
        )

    return maximum_error


def actor_directions(actor):
    directions = []

    indices = list(range(MIXER_PARAMETERS))

    indices += [
        actor.a_start,
        actor.a_start + M // 2,
        actor.b_start - 1,
        actor.b_start,
        actor.b_start + actor.edge_count // 2,
        actor.size - 1,
    ]

    for index in sorted(set(indices)):
        direction = np.zeros(actor.size)
        direction[index] = 1.0
        directions.append(direction)

    for start, end in (
        (0, actor.a_start),
        (actor.a_start, actor.b_start),
        (actor.b_start, actor.size),
    ):
        direction = np.zeros(actor.size)

        direction[start:end] = np.cos(
            0.37 * np.arange(end - start) + 0.4
        )

        direction /= np.linalg.norm(direction)
        directions.append(direction)

    return directions


def self_checks(actor, critic):
    say("", "Pokrećem ugrađene provere pre treninga.")

    graph = SelectionGraph(6, 3)

    toy_a = np.asarray(
        [
            [
                [
                    0.2 + 0.03 * (step + 1) * (number + 1)
                    for number in range(6)
                ]
                for step in range(3)
            ]
        ]
    )

    toy_b = np.asarray(
        [
            [
                0.1 + 0.01 * (origin + 1) * (number + 1)
                for number in range(6)
            ]
            for origin in range(7)
        ]
    )

    legal = list(combinations(range(6), 3))
    raw = []

    for subset in legal:
        previous = graph.start
        weight = 1.0

        for step, chosen in enumerate(subset):
            weight *= (
                toy_a[0, step, chosen]
                * toy_b[previous, chosen]
            )

            previous = chosen

        raw.append(weight)

    raw = np.asarray(raw)
    probabilities = raw / np.sum(raw)

    suffix = graph.backward(toy_a, toy_b)

    require(
        abs(
            suffix[0, 0, graph.start]
            - math.log(float(np.sum(raw)))
        ) < 1.0e-11,
        "Normalizacija odstupa od male enumeracije.",
    )

    expected_inclusion = np.zeros(6)

    for subset, probability in zip(legal, probabilities):
        nll, _, _ = graph.observed(
            toy_a,
            toy_b,
            np.asarray([subset]),
            suffix=suffix,
        )

        require(
            abs(math.exp(-nll) - probability) < 1.0e-11,
            "Verovatnoća putanje odstupa od enumeracije.",
        )

        expected_inclusion[list(subset)] += probability

    _, inclusion, values = graph.rollout(
        toy_a,
        toy_b,
        rewards=np.asarray(
            [[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]]
        ) / 3.0,
        suffix=suffix,
    )

    require(
        np.max(
            np.abs(inclusion[0] - expected_inclusion)
        ) < 1.0e-11,
        "Marginale odstupaju od enumeracije.",
    )

    require(
        np.isfinite(values[0, 0, graph.start]),
        "Bellmanova vrednost nije konačna.",
    )

    paths, logs = graph.maximum(
        toy_a, toy_b, suffix=suffix
    )

    best = int(np.argmax(probabilities))

    require(
        tuple(paths[0]) == legal[best]
        and abs(
            logs[0] - math.log(probabilities[best])
        ) < 1.0e-11,
        "MAP odstupa od enumeracije.",
    )

    for numbers, picks in ((1, 1), (6, 1), (6, 6)):
        boundary = SelectionGraph(numbers, picks)

        a = np.ones((1, picks, numbers))
        b = np.ones((numbers + 1, numbers))

        _, marginal, _ = boundary.rollout(a, b)

        require(
            abs(float(np.sum(marginal)) - picks) < 1.0e-10,
            "Granični slučaj grafa nije ispravan.",
        )

    occupancy = np.zeros((4, M))

    for row in range(4):
        labels = (5 * np.arange(K) + 3 * row) % M
        occupancy[row, labels] = 1.0

    draws = np.asarray(
        [np.flatnonzero(row) for row in occupancy],
        dtype=np.int64,
    )

    data, contexts = make_data(actor, draws, occupancy)
    zero = actor.initial_parameters()

    neutral_a, neutral_b, _ = actor.forward(
        zero, contexts[:1]
    )

    require(
        np.max(
            np.abs(neutral_a - 0.5 / DIM)
        ) < 1.0e-12
        and np.max(
            np.abs(neutral_b - 0.5)
        ) < 1.0e-12,
        "Akter ne može da predstavi referentne neutralne potencijale.",
    )

    neutral_suffix = GRAPH.backward(neutral_a, neutral_b)
    reference_nll = math.log(math.comb(M, K))

    for subset in (
        tuple(range(K)),
        tuple(range(M - K, M)),
        (0, 5, 11, 17, 23, 29, 38),
    ):
        nll, _, _ = GRAPH.observed(
            neutral_a,
            neutral_b,
            np.asarray([subset]),
            suffix=neutral_suffix,
        )

        require(
            abs(nll - reference_nll) < 2.0e-10,
            "Referentno kolo favorizuje oznake ili prinudni završetak.",
        )

    change_a = zero.copy()
    change_a[actor.a_start] = 0.10

    changed_a, unchanged_b, _ = actor.forward(
        change_a, contexts[:1]
    )

    require(
        np.array_equal(neutral_b, unchanged_b)
        and abs(changed_a[0, 0, 0] - neutral_a[0, 0, 0]) > 1.0e-8,
        "Parametri amplituda nisu odvojeni od prelaza.",
    )

    change_b = zero.copy()
    change_b[actor.b_start] = 0.10

    unchanged_a, changed_b, _ = actor.forward(
        change_b, contexts[:1]
    )

    require(
        np.array_equal(neutral_a, unchanged_a)
        and np.max(np.abs(changed_b - neutral_b)) > 1.0e-8,
        "Parametri prelaza nisu odvojeni od amplituda.",
    )

    theta = 0.02 * np.sin(
        np.arange(actor.size) + 0.3
    )

    directions = actor_directions(actor)

    actor_error = gradient_check(
        lambda parameters: bootstrap_objective(
            parameters, actor, data, 0.3
        ),
        theta,
        directions,
        "Kvantni akter",
    )

    old_a, old_b, _ = actor.forward(
        theta, data.contexts
    )

    old_suffix = GRAPH.backward(old_a, old_b)

    visits, _, targets = GRAPH.rollout(
        old_a,
        old_b,
        rewards=data.occupancy / K,
        suffix=old_suffix,
    )

    critic_theta = critic.initial_parameters()
    critic_directions = []

    for index in range(CRITIC_PARAMETERS):
        direction = np.zeros(CRITIC_PARAMETERS)
        direction[index] = 1.0
        critic_directions.append(direction)

    critic_error = gradient_check(
        lambda parameters: critic_objective(
            parameters,
            critic,
            data,
            visits,
            targets,
        ),
        critic_theta,
        critic_directions,
        "Kvantni kritičar",
    )

    table, _ = critic.lookup(critic_theta)
    critic_values = critic.select(table, data.codes)

    ppo_theta = theta + 0.002 * np.cos(
        np.arange(actor.size) + 0.7
    )

    ppo_error = gradient_check(
        lambda parameters: ppo_objective(
            parameters,
            actor,
            data,
            0.3,
            old_a,
            old_b,
            old_suffix,
            visits,
            critic_values,
        ),
        ppo_theta,
        directions,
        "PPO akter",
    )

    a, b, _ = actor.forward(
        theta, contexts[:1]
    )

    suffix = GRAPH.backward(a, b)
    paths, _ = GRAPH.maximum(a, b, suffix=suffix)

    previous = START
    filter_error = 0.0

    for step, number in enumerate(paths[0]):
        expected = GRAPH.policy(
            a, b, step, suffix
        )[0, previous]

        actual = actor.explicit_policy(
            theta,
            contexts[0, step],
            previous,
            step,
            suffix[0, step + 1],
        )

        filter_error = max(
            filter_error,
            float(np.max(np.abs(expected - actual))),
        )

        previous = int(number)

    require(
        filter_error < 2.0e-9,
        "Kvantni filter odstupa od politike.",
    )

    prefix = actor.contexts(occupancy[:2])

    require(
        np.max(
            np.abs(prefix - contexts[:2])
        ) < 1.0e-12,
        "Budući podaci menjaju ranije ulaze.",
    )

    say(
        "Mala enumeracija, granice, normalizacija i MAP: potvrđeni.",
        "Referentno Qiskit kolo može da predstavi neutralne potencijale.",
        "Parametri amplituda i prelaza: nezavisnost potvrđena.",
        f"Greška proverenih gradijenata aktera: {actor_error:.3e}",
        f"Greška gradijenata kritičara: {critic_error:.3e}",
        f"Greška proverenih PPO gradijenata: {ppo_error:.3e}",
        f"Greška eksplicitnog Qiskit filtera: {filter_error:.3e}",
        "Hronološka konstrukcija ulaza: potvrđena na provernom primeru.",
    )


def longest_run(numbers):
    longest = 0
    current = 0
    previous = None

    for number in numbers:
        if previous is not None and number == previous + 1:
            current += 1
        else:
            current = 1

        longest = max(longest, current)
        previous = number

    return longest


def process_file(actor, critic, path, draws, occupancy):
    data, contexts = make_data(
        actor, draws, occupancy
    )

    count = len(data)

    s60 = int(0.60 * count)
    s70 = int(0.70 * count)
    s80 = int(0.80 * count)

    require(
        20 < s60 < s70 < s80 < count,
        "Neispravne hronološke podele.",
    )

    scores = {}
    warm_by_ridge = {}

    for ridge in RIDGES:
        totals = np.zeros(MAX_PPO_ROUNDS + 1)
        evaluated = 0
        warm = None

        for fold, (train_end, validation_end) in enumerate(
            ((s60, s70), (s70, s80)),
            start=1,
        ):
            models = train_models(
                actor,
                critic,
                data.prefix(train_end),
                ridge,
                MAX_PPO_ROUNDS,
                f"{path.name} / validacija {fold} / ridge {ridge}",
                warm_actor=warm,
            )

            warm = models[0].actor.copy()

            validation_data = data.subset(
                train_end, validation_end
            )

            for rounds, model in enumerate(models):
                nll, _ = evaluate(
                    actor, model, validation_data
                )

                totals[rounds] += (
                    nll * len(validation_data)
                )

                say(
                    f"Validacija {fold}"
                    f" | ridge {ridge}"
                    f" | PPO rundi {rounds}"
                    f" | NLL {nll:.9f}"
                )

            evaluated += len(validation_data)

        warm_by_ridge[ridge] = warm

        for rounds in range(MAX_PPO_ROUNDS + 1):
            scores[(ridge, rounds)] = (
                totals[rounds] / evaluated
            )

    chosen_ridge, chosen_rounds = min(
        scores,
        key=lambda item: (
            scores[item],
            item[1],
            item[0],
        ),
    )

    validation_nll = scores[
        (chosen_ridge, chosen_rounds)
    ]

    say(
        "",
        f"{path.name}: izabrani ridge {chosen_ridge}",
        f"Izabrani broj PPO rundi: {chosen_rounds}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_models = train_models(
        actor,
        critic,
        data.prefix(s80),
        chosen_ridge,
        chosen_rounds,
        f"{path.name} / model za odvojeni test",
        warm_actor=warm_by_ridge[chosen_ridge],
    )

    test_data = data.subset(s80)

    test_nll, histogram = evaluate(
        actor,
        test_models[-1],
        test_data,
        hits=True,
    )

    bootstrap_test_nll, _ = evaluate(
        actor,
        test_models[0],
        test_data,
    )

    reference = Model(
        actor.initial_parameters(),
        critic.initial_parameters(),
        0,
        (),
        0.0,
    )

    reference_test_nll, _ = evaluate(
        actor,
        reference,
        test_data,
    )

    final_models = train_models(
        actor,
        critic,
        data,
        chosen_ridge,
        chosen_rounds,
        f"{path.name} / završno učenje na svim prelazima",
        warm_actor=test_models[0].actor,
    )

    model = final_models[-1]

    a, b, _ = actor.forward(
        model.actor,
        contexts[-1:],
    )

    suffix = GRAPH.backward(a, b)

    visits, inclusion, _ = GRAPH.rollout(
        a, b, suffix=suffix
    )

    paths, log_probability = GRAPH.maximum(
        a, b, suffix=suffix
    )

    selected = paths[0]

    previous = START
    filter_error = 0.0

    for step, number in enumerate(selected):
        expected = GRAPH.policy(
            a, b, step, suffix
        )[0, previous]

        explicit = actor.explicit_policy(
            model.actor,
            contexts[-1, step],
            previous,
            step,
            suffix[0, step + 1],
        )

        filter_error = max(
            filter_error,
            float(np.max(np.abs(expected - explicit))),
        )

        previous = int(number)

    require(
        filter_error < 2.0e-9,
        "NEXT politika odstupa od Qiskit filtera.",
    )

    recent_count = min(8, len(contexts))

    recent_a, recent_b, _ = actor.forward(
        model.actor,
        contexts[-recent_count:],
    )

    recent_suffix = GRAPH.backward(
        recent_a, recent_b
    )

    _, recent_inclusion, _ = GRAPH.rollout(
        recent_a,
        recent_b,
        suffix=recent_suffix,
    )

    sensitivity = float(
        np.max(np.ptp(recent_inclusion, axis=0))
    )

    warnings = list(model.warnings)

    if chosen_rounds == 0:
        warnings.append(
            "PPO nije izabran kao poboljšanje hronološke validacije."
        )

    elif test_nll >= bootstrap_test_nll:
        warnings.append(
            "PPO nije dao niži test NLL od početnog kvantnog učenja."
        )

    test_improved = test_nll < reference_test_nll

    if not test_improved:
        warnings.append(
            "Odvojeni test ne potvrđuje poboljšanje u odnosu na referentno Qiskit kolo."
        )

    run = longest_run(selected)

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if selected[-1] - selected[0] <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se u rasponu od deset brojeva."
        )

    if np.any(
        np.all(draws == selected[None, :], axis=1)
    ):
        warnings.append(
            "NEXT se već pojavio u istoriji ovog CSV-a."
        )

    if np.array_equal(selected, draws[-1]):
        warnings.append(
            "NEXT je jednak poslednjem kolu."
        )

    if sensitivity < 1.0e-8:
        warnings.append(
            "Poslednje proverene raspodele slabo reaguju na istoriju."
        )

    mean_hits = float(
        np.dot(np.arange(K + 1), histogram)
        / np.sum(histogram)
    )

    mixer, a_offsets, b_offsets = actor.unpack(
        model.actor
    )

    say(
        "",
        path.name,
        "NEXT: " + " ".join(
            str(number + 1)
            for number in selected
        ),
        f"Validacioni NLL: {validation_nll:.9f}",
        f"Odvojeni test NLL: {test_nll:.9f}",
        f"Test NLL početnog kvantnog modela: {bootstrap_test_nll:.9f}",
        f"Test NLL referentnog Qiskit kola: {reference_test_nll:.9f}",
        f"Izabrani broj PPO rundi: {chosen_rounds}",
        f"Modelna verovatnoća NEXT: {math.exp(log_probability[0]):.12g}",
        f"Ukupna masa završnih putanja: {np.sum(visits[0, -1]):.12f}",
        f"Zbir marginalnih verovatnoća: {np.sum(inclusion[0]):.12f}",
        f"Greška eksplicitnog Qiskit filtera: {filter_error:.3e}",
        f"Dijagnostička osetljivost: {sensitivity:.6e}",
        f"Završni Max|proj.grad|: {model.gradient_norm:.6e}",
        f"Norma parametara kvantnog mešača: {np.linalg.norm(mixer):.6e}",
        f"Norma amplitudnih pomeraja: {np.linalg.norm(a_offsets):.6e}",
        f"Norma prelaznih pomeraja: {np.linalg.norm(b_offsets):.6e}",
        f"Podrška raspodele: {math.comb(M, K):,} sedmorki",
        "MAP je izračunat tačno dinamičkim programiranjem.",
        "Test: broj kola prema broju pogodaka",
        " | ".join(
            f"{hits}: {int(value)}"
            for hits, value in enumerate(histogram)
        ),
        f"Prosečan broj pogodaka MAP kombinacije: {mean_hits:.6f}",
        "Marginalne verovatnoće uključivanja brojeva:",
    )

    for start in range(0, M, 6):
        say(
            " | ".join(
                f"{number + 1:02d}: {inclusion[0, number]:.8f}"
                for number in range(
                    start, min(start + 6, M)
                )
            )
        )

    say(
        "TEST STATUS: "
        + (
            "NIŽI NLL OD REFERENTNOG KOLA"
            if test_improved
            else "POBOLJŠANJE NIJE POTVRĐENO"
        )
    )

    for warning in warnings:
        say(f"DIJAGNOSTIKA: {warning}")

    return {
        "name": path.name,
        "selected": tuple(int(number) for number in selected),
        "sensitivity": sensitivity,
        "test_improved": test_improved,
    }


def main():
    say(
        "QUANTUM CATAN V1 — ODVOJENI KVANTNI AKTERI",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "",
        "Uklonjen je zajednički fiksni Hamiltonijan amplituda i prelaza.",
        "Mešač, amplitudni filteri i prelazni filteri imaju odvojene parametre.",
        "Svaki legalni par ima zaseban ugao Qiskit RY filtera.",
        "Procenjivač vrednosti koristi zasebna Qiskit kola.",
        "",
        "Referentno pripremno stanje služi početku učenja i proverama.",
        "Njegova raspodela nije nametnuta završnom naučenom modelu.",
        "Uče se uglovi kvantnih kapija iz istorijskih prelaza.",
        "",
        "Politika gradi sedmorku kroz sedam uslovnih odluka.",
        "Cela raspodela ima globalnu normalizaciju preko svih legalnih putanja.",
        "Mase nastavaka ulaze u politiku, gradijente i Qiskit filter.",
        "Izbor sledećeg broja zavisi i od prethodnog izbora.",
        "",
        "Šest kubita predstavlja registar jednog slota.",
        "Sedam slotova računa se uslovno, bez gustog zajedničkog vektora.",
        "Amplitudni filter koristi pomoćni kubit.",
        "Kritičar koristi dva kubita.",
        "",
        "PPO nagrada je broj pogodaka podeljen sa sedam.",
        "PPO zadržava član za verovatnoću cele istorijske sedmorke.",
        "Očekivanja i posećenosti računaju se tačno, bez uzorkovanja poteza.",
        "Hronološka validacija bira regularizaciju i broj PPO rundi.",
        "",
        "Završno učenje koristi sve istorijske prelaze svakog CSV-a.",
        "Odvojeni test ne bira parametre.",
        "Nema filtera parnosti, zbira, zona, nizova ili istorijskih ponavljanja.",
        "Isti rezultati se ne menjaju radi veštačkog razlikovanja.",
        "",
        "Klasični deo računa gradijente, normalizaciju i PPO očekivanja.",
        "Kvantne težine i procene dolaze iz Qiskit kapija i amplituda.",
        "Nema treniranog klasičnog regresora iza kvantnog izlaza.",
        "Optimizacija je lokalna; globalni minimum nije potvrđen.",
        "",
        "Numerička ispravnost i kvalitet odvojenog testa ispisuju se zasebno.",
        "Stare kontrolne tačke drugih konstrukcija se ne učitavaju.",
        f"Kontrolne tačke: {CHECKPOINT_DIRECTORY}",
    )

    actor = QuantumActor()
    critic = QuantumCritic()

    say(
        "",
        f"Parametara mešača: {MIXER_PARAMETERS}",
        f"Parametara amplitudnih filtera: {AMPLITUDE_PARAMETERS}",
        f"Parametara legalnih prelaza: {actor.edge_count}",
        f"Ukupno parametara kvantnog aktera: {actor.size}",
        f"Parametara kvantnog kritičara: {CRITIC_PARAMETERS}",
    )

    loaded = []

    for path in FILES:
        draws, occupancy = load_draws(path)
        loaded.append((path, draws, occupancy))

    self_checks(actor, critic)

    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for path, draws, occupancy in loaded:
        results.append(
            process_file(
                actor,
                critic,
                path,
                draws,
                occupancy,
            )
        )

    require(
        len(results) == len(FILES),
        "Nisu završena sva tri CSV-a.",
    )

    say("", "ZAVRŠNI NEXT REZULTATI")

    for result in results:
        status = (
            "NIŽI TEST NLL OD REFERENTNOG KOLA"
            if result["test_improved"]
            else "TEST POBOLJŠANJE NIJE POTVRĐENO"
        )

        say(
            result["name"]
            + " | NEXT: "
            + " ".join(
                str(number + 1)
                for number in result["selected"]
            )
            + " | "
            + status
        )

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            if (
                results[first]["selected"]
                == results[second]["selected"]
            ):
                say(
                    "DIJAGNOSTIKA: isti NEXT za "
                    f"{results[first]['name']} i "
                    f"{results[second]['name']}.",
                    "Osetljivosti na istoriju: "
                    f"{results[first]['sensitivity']:.6e} i "
                    f"{results[second]['sensitivity']:.6e}.",
                    "Zajednički maksimum ne potvrđuje jednakost celih raspodela.",
                )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        say(
            "",
            "PREKINUTO na zahtev korisnika.",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Sačuvane kontrolne tačke ostaju dostupne.",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM CATAN V1 — ODVOJENI KVANTNI AKTERI
Python: 3.11.13
Qiskit: 1.4.4

Uklonjen je zajednički fiksni Hamiltonijan amplituda i prelaza.
Mešač, amplitudni filteri i prelazni filteri imaju odvojene parametre.
Svaki legalni par ima zaseban ugao Qiskit RY filtera.
Procenjivač vrednosti koristi zasebna Qiskit kola.

Referentno pripremno stanje služi početku učenja i proverama.
Njegova raspodela nije nametnuta završnom naučenom modelu.
Uče se uglovi kvantnih kapija iz istorijskih prelaza.

Politika gradi sedmorku kroz sedam uslovnih odluka.
Cela raspodela ima globalnu normalizaciju preko svih legalnih putanja.
Mase nastavaka ulaze u politiku, gradijente i Qiskit filter.
Izbor sledećeg broja zavisi i od prethodnog izbora.

Šest kubita predstavlja registar jednog slota.
Sedam slotova računa se uslovno, bez gustog zajedničkog vektora.
Amplitudni filter koristi pomoćni kubit.
Kritičar koristi dva kubita.

PPO nagrada je broj pogodaka podeljen sa sedam.
PPO zadržava član za verovatnoću cele istorijske sedmorke.
Očekivanja i posećenosti računaju se tačno, bez uzorkovanja poteza.
Hronološka validacija bira regularizaciju i broj PPO rundi.

Završno učenje koristi sve istorijske prelaze svakog CSV-a.
Odvojeni test ne bira parametre.
Nema filtera parnosti, zbira, zona, nizova ili istorijskih ponavljanja.
Isti rezultati se ne menjaju radi veštačkog razlikovanja.

Klasični deo računa gradijente, normalizaciju i PPO očekivanja.
Kvantne težine i procene dolaze iz Qiskit kapija i amplituda.
Nema treniranog klasičnog regresora iza kvantnog izlaza.
Optimizacija je lokalna; globalni minimum nije potvrđen.

Numerička ispravnost i kvalitet odvojenog testa ispisuju se zasebno.
Stare kontrolne tačke drugih konstrukcija se ne učitavaju.
Kontrolne tačke: /Quantum_Catan/Quantum_Catan_V1_separate_actor_checkpoints

Parametara mešača: 30
Parametara amplitudnih filtera: 273
Parametara legalnih prelaza: 759
Ukupno parametara kvantnog aktera: 1062
Parametara kvantnog kritičara: 42

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled CSV-a ostaje hronološki.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled CSV-a ostaje hronološki.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled CSV-a ostaje hronološki.

Pokrećem ugrađene provere pre treninga.
Mala enumeracija, granice, normalizacija i MAP: potvrđeni.
Referentno Qiskit kolo može da predstavi neutralne potencijale.
Parametri amplituda i prelaza: nezavisnost potvrđena.
Greška proverenih gradijenata aktera: 3.330e-09
Greška gradijenata kritičara: 3.255e-11
Greška proverenih PPO gradijenata: 9.339e-10
Greška eksplicitnog Qiskit filtera: 4.441e-16
Hronološka konstrukcija ulaza: potvrđena na provernom primeru.

loto7_4710_k86.csv / validacija 1 / ridge 0.3 / početno kvantno učenje
Iteracija 5/60 | cilj 16.543082549 | KL 0.000000 | proteklo 3.5 s
Iteracija 10/60 | cilj 16.530287170 | KL 0.000000 | proteklo 6.0 s
Iteracija 15/60 | cilj 16.510243640 | KL 0.000000 | proteklo 9.0 s
Iteracija 20/60 | cilj 16.500312149 | KL 0.000000 | proteklo 11.5 s
Iteracija 25/60 | cilj 16.492109262 | KL 0.000000 | proteklo 14.1 s
Iteracija 30/60 | cilj 16.482580877 | KL 0.000000 | proteklo 16.5 s
Iteracija 35/60 | cilj 16.477115877 | KL 0.000000 | proteklo 19.0 s
Iteracija 40/60 | cilj 16.471855903 | KL 0.000000 | proteklo 21.5 s
Iteracija 45/60 | cilj 16.468559529 | KL 0.000000 | proteklo 24.6 s
Iteracija 50/60 | cilj 16.465982496 | KL 0.000000 | proteklo 27.1 s
Iteracija 55/60 | cilj 16.463699142 | KL 0.000000 | proteklo 29.6 s
Iteracija 60/60 | cilj 16.461082869 | KL 0.000000 | proteklo 32.6 s
Zadržani cilj: 16.461082869
Max|proj.grad|: 5.473458e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / validacija 1 / ridge 0.3 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001904703 | KL 0.000000 | proteklo 0.3 s
Iteracija 10/20 | cilj 0.000448569 | KL 0.000000 | proteklo 0.4 s
Iteracija 15/20 | cilj 0.000411626 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000405026 | KL 0.000000 | proteklo 0.7 s
Zadržani cilj: 0.000405026
Max|proj.grad|: 1.770134e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / validacija 1 / ridge 0.3 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.825248994 | KL 0.001561 | proteklo 9.1 s
Iteracija 10/12 | cilj 0.824272506 | KL 0.001803 | proteklo 15.6 s
Zadržani cilj: 0.823890848
Max|proj.grad|: 5.736421e-03
KL: 0.002319
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / validacija 1 / ridge 0.3 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000414200 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000413758 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000412915 | KL 0.000000 | proteklo 0.5 s
Iteracija 20/20 | cilj 0.000412753 | KL 0.000000 | proteklo 0.6 s
Zadržani cilj: 0.000412753
Max|proj.grad|: 9.590253e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / validacija 1 / ridge 0.3 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.826340551 | KL 0.000692 | proteklo 9.0 s
Iteracija 10/12 | cilj 0.826005526 | KL 0.000918 | proteklo 15.5 s
Zadržani cilj: 0.825756057
Max|proj.grad|: 7.784215e-03
KL: 0.001122
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 1 | ridge 0.3 | PPO rundi 0 | NLL 16.627463225
Validacija 1 | ridge 0.3 | PPO rundi 1 | NLL 16.630903296
Validacija 1 | ridge 0.3 | PPO rundi 2 | NLL 16.649858946

loto7_4710_k86.csv / validacija 2 / ridge 0.3 / početno kvantno učenje
Iteracija 5/60 | cilj 16.483508628 | KL 0.000000 | proteklo 4.0 s
Iteracija 10/60 | cilj 16.478543218 | KL 0.000000 | proteklo 6.9 s
Iteracija 15/60 | cilj 16.474542114 | KL 0.000000 | proteklo 10.3 s
Iteracija 20/60 | cilj 16.471871841 | KL 0.000000 | proteklo 13.3 s
Iteracija 25/60 | cilj 16.467904490 | KL 0.000000 | proteklo 16.1 s
Iteracija 30/60 | cilj 16.466367638 | KL 0.000000 | proteklo 19.0 s
Iteracija 35/60 | cilj 16.465188195 | KL 0.000000 | proteklo 21.9 s
Iteracija 40/60 | cilj 16.463417672 | KL 0.000000 | proteklo 24.8 s
Iteracija 45/60 | cilj 16.462629654 | KL 0.000000 | proteklo 28.3 s
Iteracija 50/60 | cilj 16.461892872 | KL 0.000000 | proteklo 31.1 s
Iteracija 55/60 | cilj 16.460859200 | KL 0.000000 | proteklo 34.0 s
Iteracija 60/60 | cilj 16.459733217 | KL 0.000000 | proteklo 36.9 s
Zadržani cilj: 16.459733217
Max|proj.grad|: 2.578132e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / validacija 2 / ridge 0.3 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001887763 | KL 0.000000 | proteklo 0.3 s
Iteracija 10/20 | cilj 0.000436749 | KL 0.000000 | proteklo 0.5 s
Iteracija 15/20 | cilj 0.000401457 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000394642 | KL 0.000000 | proteklo 0.8 s
Zadržani cilj: 0.000394642
Max|proj.grad|: 4.405428e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / validacija 2 / ridge 0.3 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.827267012 | KL 0.001251 | proteklo 9.1 s
Iteracija 10/12 | cilj 0.826336211 | KL 0.001811 | proteklo 16.6 s
Zadržani cilj: 0.825828431
Max|proj.grad|: 7.399689e-03
KL: 0.001761
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / validacija 2 / ridge 0.3 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000407768 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000407364 | KL 0.000000 | proteklo 0.4 s
Iteracija 15/20 | cilj 0.000406573 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000406228 | KL 0.000000 | proteklo 0.8 s
Zadržani cilj: 0.000406228
Max|proj.grad|: 9.204267e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / validacija 2 / ridge 0.3 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.827506352 | KL 0.000689 | proteklo 9.0 s
Iteracija 10/12 | cilj 0.826932291 | KL 0.001224 | proteklo 16.5 s
Zadržani cilj: 0.826673899
Max|proj.grad|: 9.041452e-03
KL: 0.001522
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 2 | ridge 0.3 | PPO rundi 0 | NLL 16.620325476
Validacija 2 | ridge 0.3 | PPO rundi 1 | NLL 16.627543570
Validacija 2 | ridge 0.3 | PPO rundi 2 | NLL 16.642153798

loto7_4710_k86.csv / validacija 1 / ridge 3.0 / početno kvantno učenje
Iteracija 5/60 | cilj 16.543185292 | KL 0.000000 | proteklo 3.5 s
Iteracija 10/60 | cilj 16.530461402 | KL 0.000000 | proteklo 5.9 s
Iteracija 15/60 | cilj 16.513569742 | KL 0.000000 | proteklo 8.9 s
Iteracija 20/60 | cilj 16.507001416 | KL 0.000000 | proteklo 11.8 s
Iteracija 25/60 | cilj 16.499128498 | KL 0.000000 | proteklo 14.3 s
Iteracija 30/60 | cilj 16.493321653 | KL 0.000000 | proteklo 17.3 s
Iteracija 35/60 | cilj 16.489561661 | KL 0.000000 | proteklo 19.8 s
Iteracija 40/60 | cilj 16.486794014 | KL 0.000000 | proteklo 22.8 s
Iteracija 45/60 | cilj 16.485353314 | KL 0.000000 | proteklo 25.2 s
Iteracija 50/60 | cilj 16.483874835 | KL 0.000000 | proteklo 27.7 s
Iteracija 55/60 | cilj 16.483123277 | KL 0.000000 | proteklo 30.6 s
Iteracija 60/60 | cilj 16.481260693 | KL 0.000000 | proteklo 33.1 s
Zadržani cilj: 16.481260693
Max|proj.grad|: 2.574717e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / validacija 1 / ridge 3.0 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001901114 | KL 0.000000 | proteklo 0.3 s
Iteracija 10/20 | cilj 0.000445349 | KL 0.000000 | proteklo 0.4 s
Iteracija 15/20 | cilj 0.000409731 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000403138 | KL 0.000000 | proteklo 0.7 s
Zadržani cilj: 0.000403138
Max|proj.grad|: 4.606054e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / validacija 1 / ridge 3.0 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.839679008 | KL 0.001756 | proteklo 7.7 s
Iteracija 10/12 | cilj 0.832746473 | KL 0.002662 | proteklo 14.2 s
Zadržani cilj: 0.831722635
Max|proj.grad|: 3.191718e-02
KL: 0.003719
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / validacija 1 / ridge 3.0 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000415722 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000415269 | KL 0.000000 | proteklo 0.4 s
Iteracija 15/20 | cilj 0.000414572 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000414053 | KL 0.000000 | proteklo 0.7 s
Zadržani cilj: 0.000414053
Max|proj.grad|: 3.751443e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / validacija 1 / ridge 3.0 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.830722399 | KL 0.001029 | proteklo 7.8 s
Iteracija 10/12 | cilj 0.829817707 | KL 0.001035 | proteklo 14.3 s
Zadržani cilj: 0.829163025
Max|proj.grad|: 7.225814e-03
KL: 0.001092
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 1 | ridge 3.0 | PPO rundi 0 | NLL 16.593014147
Validacija 1 | ridge 3.0 | PPO rundi 1 | NLL 16.584752383
Validacija 1 | ridge 3.0 | PPO rundi 2 | NLL 16.573457053

loto7_4710_k86.csv / validacija 2 / ridge 3.0 / početno kvantno učenje
Iteracija 5/60 | cilj 16.498714716 | KL 0.000000 | proteklo 4.1 s
Iteracija 10/60 | cilj 16.495179644 | KL 0.000000 | proteklo 7.0 s
Iteracija 15/60 | cilj 16.493047512 | KL 0.000000 | proteklo 10.5 s
Iteracija 20/60 | cilj 16.491936032 | KL 0.000000 | proteklo 13.4 s
Iteracija 25/60 | cilj 16.490431624 | KL 0.000000 | proteklo 16.9 s
Iteracija 30/60 | cilj 16.489582224 | KL 0.000000 | proteklo 19.7 s
Iteracija 35/60 | cilj 16.488722629 | KL 0.000000 | proteklo 22.6 s
Iteracija 40/60 | cilj 16.488178556 | KL 0.000000 | proteklo 25.5 s
Iteracija 45/60 | cilj 16.487953278 | KL 0.000000 | proteklo 28.4 s
Iteracija 50/60 | cilj 16.487656217 | KL 0.000000 | proteklo 31.3 s
Iteracija 55/60 | cilj 16.487403901 | KL 0.000000 | proteklo 34.2 s
Iteracija 60/60 | cilj 16.487195959 | KL 0.000000 | proteklo 37.1 s
Zadržani cilj: 16.487195959
Max|proj.grad|: 7.146391e-03
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / validacija 2 / ridge 3.0 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001884907 | KL 0.000000 | proteklo 0.3 s
Iteracija 10/20 | cilj 0.000435177 | KL 0.000000 | proteklo 0.5 s
Iteracija 15/20 | cilj 0.000400290 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000393550 | KL 0.000000 | proteklo 0.8 s
Zadržani cilj: 0.000393550
Max|proj.grad|: 4.267008e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / validacija 2 / ridge 3.0 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.838108614 | KL 0.001549 | proteklo 9.1 s
Iteracija 10/12 | cilj 0.831382954 | KL 0.003004 | proteklo 16.6 s
Zadržani cilj: 0.830130408
Max|proj.grad|: 2.855959e-02
KL: 0.003753
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / validacija 2 / ridge 3.0 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000404317 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000403867 | KL 0.000000 | proteklo 0.4 s
Iteracija 15/20 | cilj 0.000403167 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000402636 | KL 0.000000 | proteklo 0.8 s
Zadržani cilj: 0.000402636
Max|proj.grad|: 2.548949e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / validacija 2 / ridge 3.0 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.831127801 | KL 0.000851 | proteklo 9.1 s
Iteracija 10/12 | cilj 0.829854908 | KL 0.001494 | proteklo 16.7 s
Zadržani cilj: 0.829164657
Max|proj.grad|: 4.440804e-03
KL: 0.001165
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 2 | ridge 3.0 | PPO rundi 0 | NLL 16.589906637
Validacija 2 | ridge 3.0 | PPO rundi 1 | NLL 16.578662775
Validacija 2 | ridge 3.0 | PPO rundi 2 | NLL 16.576664210

loto7_4710_k86.csv: izabrani ridge 3.0
Izabrani broj PPO rundi: 2
Validacioni NLL: 16.575060631

loto7_4710_k86.csv / model za odvojeni test / početno kvantno učenje
Iteracija 5/60 | cilj 16.501408131 | KL 0.000000 | proteklo 4.7 s
Iteracija 10/60 | cilj 16.499666230 | KL 0.000000 | proteklo 8.1 s
Iteracija 15/60 | cilj 16.497900197 | KL 0.000000 | proteklo 11.4 s
Iteracija 20/60 | cilj 16.497355711 | KL 0.000000 | proteklo 15.5 s
Iteracija 25/60 | cilj 16.496647267 | KL 0.000000 | proteklo 18.9 s
Iteracija 30/60 | cilj 16.495984599 | KL 0.000000 | proteklo 22.2 s
Iteracija 35/60 | cilj 16.495572488 | KL 0.000000 | proteklo 25.6 s
Iteracija 40/60 | cilj 16.495347820 | KL 0.000000 | proteklo 29.6 s
Iteracija 45/60 | cilj 16.495213712 | KL 0.000000 | proteklo 33.6 s
Iteracija 50/60 | cilj 16.495086298 | KL 0.000000 | proteklo 37.0 s
Iteracija 55/60 | cilj 16.494983784 | KL 0.000000 | proteklo 40.4 s
Iteracija 60/60 | cilj 16.494889387 | KL 0.000000 | proteklo 43.7 s
Zadržani cilj: 16.494889387
Max|proj.grad|: 8.698810e-03
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / model za odvojeni test / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001889665 | KL 0.000000 | proteklo 0.3 s
Iteracija 10/20 | cilj 0.000436473 | KL 0.000000 | proteklo 0.5 s
Iteracija 15/20 | cilj 0.000401333 | KL 0.000000 | proteklo 0.7 s
Iteracija 20/20 | cilj 0.000394427 | KL 0.000000 | proteklo 1.0 s
Zadržani cilj: 0.000394427
Max|proj.grad|: 3.193473e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / model za odvojeni test / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.837689999 | KL 0.002092 | proteklo 10.6 s
Iteracija 10/12 | cilj 0.830759980 | KL 0.002910 | proteklo 19.5 s
Zadržani cilj: 0.830101135
Max|proj.grad|: 6.459371e-03
KL: 0.003173
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / model za odvojeni test / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000405081 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000404772 | KL 0.000000 | proteklo 0.4 s
Iteracija 15/20 | cilj 0.000403938 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000403546 | KL 0.000000 | proteklo 0.8 s
Zadržani cilj: 0.000403546
Max|proj.grad|: 6.872459e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / model za odvojeni test / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.831392978 | KL 0.000616 | proteklo 10.5 s
Iteracija 10/12 | cilj 0.830113549 | KL 0.001134 | proteklo 19.2 s
Zadržani cilj: 0.829455357
Max|proj.grad|: 6.928854e-03
KL: 0.000879
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / završno učenje na svim prelazima / početno kvantno učenje
Iteracija 5/60 | cilj 16.515775988 | KL 0.000000 | proteklo 5.9 s
Iteracija 10/60 | cilj 16.513298524 | KL 0.000000 | proteklo 10.1 s
Iteracija 15/60 | cilj 16.510543137 | KL 0.000000 | proteklo 14.3 s
Iteracija 20/60 | cilj 16.509530553 | KL 0.000000 | proteklo 18.5 s
Iteracija 25/60 | cilj 16.508183955 | KL 0.000000 | proteklo 22.9 s
Iteracija 30/60 | cilj 16.507191721 | KL 0.000000 | proteklo 27.2 s
Iteracija 35/60 | cilj 16.506709053 | KL 0.000000 | proteklo 32.2 s
Iteracija 40/60 | cilj 16.506460601 | KL 0.000000 | proteklo 37.3 s
Iteracija 45/60 | cilj 16.506225745 | KL 0.000000 | proteklo 41.6 s
Iteracija 50/60 | cilj 16.506034193 | KL 0.000000 | proteklo 45.8 s
Iteracija 55/60 | cilj 16.505864456 | KL 0.000000 | proteklo 50.0 s
Iteracija 60/60 | cilj 16.505675094 | KL 0.000000 | proteklo 55.1 s
Zadržani cilj: 16.505675094
Max|proj.grad|: 6.591254e-03
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv / završno učenje na svim prelazima / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001884951 | KL 0.000000 | proteklo 0.4 s
Iteracija 10/20 | cilj 0.000435157 | KL 0.000000 | proteklo 0.7 s
Iteracija 15/20 | cilj 0.000400456 | KL 0.000000 | proteklo 0.9 s
Iteracija 20/20 | cilj 0.000393831 | KL 0.000000 | proteklo 1.2 s
Zadržani cilj: 0.000393831
Max|proj.grad|: 4.398536e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86.csv / završno učenje na svim prelazima / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.835438704 | KL 0.001755 | proteklo 13.4 s

loto7_4710_k86.csv / završno učenje na svim prelazima / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.831019927 | KL 0.000537 | proteklo 13.1 s
Iteracija 10/12 | cilj 0.830209778 | KL 0.000708 | proteklo 24.1 s
Zadržani cilj: 0.829520431
Max|proj.grad|: 1.460323e-02
KL: 0.001115
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86.csv
NEXT: 1 x 5 y 34 z 36
Validacioni NLL: 16.575060631
Odvojeni test NLL: 16.565193552
Test NLL početnog kvantnog modela: 16.589473522
Test NLL referentnog Qiskit kola: 16.548639443
Izabrani broj PPO rundi: 2
Modelna verovatnoća NEXT: 1.644862648e-07
Ukupna masa završnih putanja: 1.000000000000
Zbir marginalnih verovatnoća: 7.000000000000
Greška eksplicitnog Qiskit filtera: 4.441e-16
Dijagnostička osetljivost: 1.088386e-02
Završni Max|proj.grad|: 1.460323e-02
Norma parametara kvantnog mešača: 9.769981e-02
Norma amplitudnih pomeraja: 3.498835e-01
Norma prelaznih pomeraja: 8.415801e-01
Podrška raspodele: 15,380,937 sedmorki
MAP je izračunat tačno dinamičkim programiranjem.
Test: broj kola prema broju pogodaka
0: 201 | 1: 402 | 2: 254 | 3: 79 | 4: 6 | 5: 0 | 6: 0 | 7: 0
Prosečan broj pogodaka MAP kombinacije: 1.243100
Marginalne verovatnoće uključivanja brojeva:
01: 0.20151578 | 02: 0.19313320 | 03: 0.19425283 | 04: 0.19052613 | 05: 0.18406736 | 06: 0.18280199
07: 0.18420057 | 08: 0.19596621 | 09: 0.17386382 | 10: 0.17431798 | 11: 0.18236606 | 12: 0.17526876
13: 0.16132299 | 14: 0.16532342 | 15: 0.16759771 | 16: 0.16950688 | 17: 0.16603700 | 18: 0.16956431
19: 0.16775884 | 20: 0.16398044 | 21: 0.17128162 | 22: 0.17654272 | 23: 0.17759003 | 24: 0.17681372
25: 0.17118537 | 26: 0.18400556 | 27: 0.17952849 | 28: 0.17892139 | 29: 0.17399857 | 30: 0.16747272
31: 0.17046064 | 32: 0.18193529 | 33: 0.18983208 | 34: 0.19805591 | 35: 0.19128080 | 36: 0.18740980
37: 0.18670854 | 38: 0.18476047 | 39: 0.18884402
TEST STATUS: POBOLJŠANJE NIJE POTVRĐENO
DIJAGNOSTIKA: Početno učenje nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Akter u PPO rundi 1 nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Akter u PPO rundi 2 nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Odvojeni test ne potvrđuje poboljšanje u odnosu na referentno Qiskit kolo.

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.3 / početno kvantno učenje
Iteracija 5/60 | cilj 16.538353499 | KL 0.000000 | proteklo 2.3 s
Iteracija 10/60 | cilj 16.519757286 | KL 0.000000 | proteklo 3.9 s
Iteracija 15/60 | cilj 16.478436693 | KL 0.000000 | proteklo 5.8 s
Iteracija 20/60 | cilj 16.470241094 | KL 0.000000 | proteklo 7.7 s
Iteracija 25/60 | cilj 16.451593330 | KL 0.000000 | proteklo 9.6 s
Iteracija 30/60 | cilj 16.439498154 | KL 0.000000 | proteklo 11.1 s
Iteracija 35/60 | cilj 16.430474233 | KL 0.000000 | proteklo 13.0 s
Iteracija 40/60 | cilj 16.419507706 | KL 0.000000 | proteklo 14.6 s
Iteracija 45/60 | cilj 16.416788577 | KL 0.000000 | proteklo 16.1 s
Iteracija 50/60 | cilj 16.411812325 | KL 0.000000 | proteklo 18.0 s
Iteracija 55/60 | cilj 16.406273238 | KL 0.000000 | proteklo 20.0 s
Iteracija 60/60 | cilj 16.402500022 | KL 0.000000 | proteklo 21.5 s
Zadržani cilj: 16.402500022
Max|proj.grad|: 8.400019e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.3 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001919511 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000458334 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000422223 | KL 0.000000 | proteklo 0.4 s
Iteracija 20/20 | cilj 0.000415719 | KL 0.000000 | proteklo 0.5 s
Zadržani cilj: 0.000415719
Max|proj.grad|: 4.480537e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.3 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.825144019 | KL 0.001015 | proteklo 5.6 s
Iteracija 10/12 | cilj 0.823812512 | KL 0.002239 | proteklo 9.6 s
Zadržani cilj: 0.823534885
Max|proj.grad|: 4.338841e-03
KL: 0.001938
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.3 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000432128 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000431673 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000430951 | KL 0.000000 | proteklo 0.3 s
Iteracija 20/20 | cilj 0.000430355 | KL 0.000000 | proteklo 0.5 s
Zadržani cilj: 0.000430355
Max|proj.grad|: 2.937859e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 0.3 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.825543256 | KL 0.000753 | proteklo 5.6 s
Iteracija 10/12 | cilj 0.824817371 | KL 0.001423 | proteklo 9.6 s
Zadržani cilj: 0.824507513
Max|proj.grad|: 2.755395e-03
KL: 0.001348
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 1 | ridge 0.3 | PPO rundi 0 | NLL 16.734442675
Validacija 1 | ridge 0.3 | PPO rundi 1 | NLL 16.728285512
Validacija 1 | ridge 0.3 | PPO rundi 2 | NLL 16.733474854

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.3 / početno kvantno učenje
Iteracija 5/60 | cilj 16.447414154 | KL 0.000000 | proteklo 2.6 s
Iteracija 10/60 | cilj 16.440298173 | KL 0.000000 | proteklo 5.0 s
Iteracija 15/60 | cilj 16.431504626 | KL 0.000000 | proteklo 7.5 s
Iteracija 20/60 | cilj 16.428303374 | KL 0.000000 | proteklo 9.4 s
Iteracija 25/60 | cilj 16.423372003 | KL 0.000000 | proteklo 11.2 s
Iteracija 30/60 | cilj 16.420896271 | KL 0.000000 | proteklo 13.0 s
Iteracija 35/60 | cilj 16.418505929 | KL 0.000000 | proteklo 14.9 s
Iteracija 40/60 | cilj 16.417246748 | KL 0.000000 | proteklo 17.0 s
Iteracija 45/60 | cilj 16.416418095 | KL 0.000000 | proteklo 18.9 s
Iteracija 50/60 | cilj 16.414510640 | KL 0.000000 | proteklo 20.7 s
Iteracija 55/60 | cilj 16.413377526 | KL 0.000000 | proteklo 22.9 s
Iteracija 60/60 | cilj 16.411968742 | KL 0.000000 | proteklo 25.1 s
Zadržani cilj: 16.411968742
Max|proj.grad|: 2.250687e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.3 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001907919 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000449417 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000413013 | KL 0.000000 | proteklo 0.4 s
Iteracija 20/20 | cilj 0.000406328 | KL 0.000000 | proteklo 0.6 s
Zadržani cilj: 0.000406328
Max|proj.grad|: 4.030674e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.3 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.827605869 | KL 0.001278 | proteklo 6.6 s
Iteracija 10/12 | cilj 0.826483891 | KL 0.001649 | proteklo 11.3 s
Zadržani cilj: 0.826074624
Max|proj.grad|: 1.141919e-02
KL: 0.002235
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.3 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000421653 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000421268 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000420484 | KL 0.000000 | proteklo 0.4 s
Iteracija 20/20 | cilj 0.000420020 | KL 0.000000 | proteklo 0.5 s
Zadržani cilj: 0.000420020
Max|proj.grad|: 1.050819e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 0.3 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.826885520 | KL 0.000650 | proteklo 5.6 s
Iteracija 10/12 | cilj 0.826475086 | KL 0.001514 | proteklo 10.3 s
Zadržani cilj: 0.826071788
Max|proj.grad|: 4.117847e-03
KL: 0.001109
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 2 | ridge 0.3 | PPO rundi 0 | NLL 16.608707098
Validacija 2 | ridge 0.3 | PPO rundi 1 | NLL 16.629549938
Validacija 2 | ridge 0.3 | PPO rundi 2 | NLL 16.645365605

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 3.0 / početno kvantno učenje
Iteracija 5/60 | cilj 16.538501816 | KL 0.000000 | proteklo 2.2 s
Iteracija 10/60 | cilj 16.520090866 | KL 0.000000 | proteklo 3.7 s
Iteracija 15/60 | cilj 16.485986239 | KL 0.000000 | proteklo 5.6 s
Iteracija 20/60 | cilj 16.479806462 | KL 0.000000 | proteklo 7.4 s
Iteracija 25/60 | cilj 16.465752174 | KL 0.000000 | proteklo 9.0 s
Iteracija 30/60 | cilj 16.458222847 | KL 0.000000 | proteklo 10.5 s
Iteracija 35/60 | cilj 16.451804897 | KL 0.000000 | proteklo 12.3 s
Iteracija 40/60 | cilj 16.445182547 | KL 0.000000 | proteklo 13.9 s
Iteracija 45/60 | cilj 16.443484700 | KL 0.000000 | proteklo 15.4 s
Iteracija 50/60 | cilj 16.441826503 | KL 0.000000 | proteklo 17.0 s
Iteracija 55/60 | cilj 16.440082658 | KL 0.000000 | proteklo 18.5 s
Iteracija 60/60 | cilj 16.438367290 | KL 0.000000 | proteklo 20.4 s
Zadržani cilj: 16.438367290
Max|proj.grad|: 4.711929e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 3.0 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001906327 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000451558 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000416128 | KL 0.000000 | proteklo 0.4 s
Iteracija 20/20 | cilj 0.000409216 | KL 0.000000 | proteklo 0.5 s
Zadržani cilj: 0.000409216
Max|proj.grad|: 3.852740e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 3.0 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.843635722 | KL 0.001772 | proteklo 4.8 s
Iteracija 10/12 | cilj 0.834862817 | KL 0.003357 | proteklo 8.7 s
Zadržani cilj: 0.832406653
Max|proj.grad|: 1.990155e-02
KL: 0.004021
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 3.0 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000410462 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000410019 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000409068 | KL 0.000000 | proteklo 0.3 s
Iteracija 20/20 | cilj 0.000408811 | KL 0.000000 | proteklo 0.4 s
Zadržani cilj: 0.000408811
Max|proj.grad|: 2.939531e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / validacija 1 / ridge 3.0 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.832233288 | KL 0.000791 | proteklo 4.8 s
Iteracija 10/12 | cilj 0.830664615 | KL 0.000629 | proteklo 9.5 s
Zadržani cilj: 0.829413625
Max|proj.grad|: 1.022368e-02
KL: 0.001231
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 1 | ridge 3.0 | PPO rundi 0 | NLL 16.657668592
Validacija 1 | ridge 3.0 | PPO rundi 1 | NLL 16.613252027
Validacija 1 | ridge 3.0 | PPO rundi 2 | NLL 16.600599874

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 3.0 / početno kvantno učenje
Iteracija 5/60 | cilj 16.470733568 | KL 0.000000 | proteklo 2.9 s
Iteracija 10/60 | cilj 16.465360169 | KL 0.000000 | proteklo 4.7 s
Iteracija 15/60 | cilj 16.460196130 | KL 0.000000 | proteklo 6.5 s
Iteracija 20/60 | cilj 16.459006086 | KL 0.000000 | proteklo 8.6 s
Iteracija 25/60 | cilj 16.456135114 | KL 0.000000 | proteklo 10.8 s
Iteracija 30/60 | cilj 16.455034017 | KL 0.000000 | proteklo 12.6 s
Iteracija 35/60 | cilj 16.454453445 | KL 0.000000 | proteklo 14.4 s
Iteracija 40/60 | cilj 16.453772968 | KL 0.000000 | proteklo 16.2 s
Iteracija 45/60 | cilj 16.453306673 | KL 0.000000 | proteklo 18.0 s
Iteracija 50/60 | cilj 16.452806621 | KL 0.000000 | proteklo 19.8 s
Iteracija 55/60 | cilj 16.452613682 | KL 0.000000 | proteklo 21.6 s
Iteracija 60/60 | cilj 16.452174104 | KL 0.000000 | proteklo 23.7 s
Zadržani cilj: 16.452174104
Max|proj.grad|: 1.605736e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 3.0 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001897646 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000445231 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000409717 | KL 0.000000 | proteklo 0.4 s
Iteracija 20/20 | cilj 0.000402939 | KL 0.000000 | proteklo 0.6 s
Zadržani cilj: 0.000402939
Max|proj.grad|: 4.331624e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 3.0 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.844806831 | KL 0.001718 | proteklo 5.6 s
Iteracija 10/12 | cilj 0.833669649 | KL 0.004321 | proteklo 10.2 s
Zadržani cilj: 0.832500518
Max|proj.grad|: 1.403315e-02
KL: 0.004552
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 3.0 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000410188 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000409735 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000408907 | KL 0.000000 | proteklo 0.4 s
Iteracija 20/20 | cilj 0.000408500 | KL 0.000000 | proteklo 0.5 s
Zadržani cilj: 0.000408500
Max|proj.grad|: 8.394386e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / validacija 2 / ridge 3.0 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.832958315 | KL 0.000768 | proteklo 5.6 s
Iteracija 10/12 | cilj 0.831512256 | KL 0.000658 | proteklo 10.3 s
Zadržani cilj: 0.829874444
Max|proj.grad|: 8.897958e-03
KL: 0.001282
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 2 | ridge 3.0 | PPO rundi 0 | NLL 16.587597379
Validacija 2 | ridge 3.0 | PPO rundi 1 | NLL 16.584160232
Validacija 2 | ridge 3.0 | PPO rundi 2 | NLL 16.577633279

loto7_4710_k86_loto_2977.csv: izabrani ridge 3.0
Izabrani broj PPO rundi: 2
Validacioni NLL: 16.589135877

loto7_4710_k86_loto_2977.csv / model za odvojeni test / početno kvantno učenje
Iteracija 5/60 | cilj 16.470813966 | KL 0.000000 | proteklo 3.3 s
Iteracija 10/60 | cilj 16.465803501 | KL 0.000000 | proteklo 5.3 s
Iteracija 15/60 | cilj 16.463436322 | KL 0.000000 | proteklo 7.8 s
Iteracija 20/60 | cilj 16.462550354 | KL 0.000000 | proteklo 10.3 s
Iteracija 25/60 | cilj 16.461379115 | KL 0.000000 | proteklo 12.4 s
Iteracija 30/60 | cilj 16.460565052 | KL 0.000000 | proteklo 14.8 s
Iteracija 35/60 | cilj 16.459894476 | KL 0.000000 | proteklo 17.3 s
Iteracija 40/60 | cilj 16.459570819 | KL 0.000000 | proteklo 19.3 s
Iteracija 45/60 | cilj 16.459310142 | KL 0.000000 | proteklo 21.8 s
Iteracija 50/60 | cilj 16.459114852 | KL 0.000000 | proteklo 23.9 s
Iteracija 55/60 | cilj 16.458923864 | KL 0.000000 | proteklo 26.3 s
Iteracija 60/60 | cilj 16.458849170 | KL 0.000000 | proteklo 28.4 s
Zadržani cilj: 16.458849170
Max|proj.grad|: 3.978256e-03
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / model za odvojeni test / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001892546 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000440790 | KL 0.000000 | proteklo 0.4 s
Iteracija 15/20 | cilj 0.000404668 | KL 0.000000 | proteklo 0.5 s
Iteracija 20/20 | cilj 0.000397798 | KL 0.000000 | proteklo 0.6 s
Zadržani cilj: 0.000397798
Max|proj.grad|: 4.220477e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / model za odvojeni test / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.843954531 | KL 0.002961 | proteklo 6.4 s
Iteracija 10/12 | cilj 0.834072753 | KL 0.003932 | proteklo 11.7 s
Zadržani cilj: 0.832927707
Max|proj.grad|: 1.171204e-02
KL: 0.004878
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / model za odvojeni test / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000413229 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000412869 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000412247 | KL 0.000000 | proteklo 0.4 s
Iteracija 20/20 | cilj 0.000411659 | KL 0.000000 | proteklo 0.5 s
Zadržani cilj: 0.000411659
Max|proj.grad|: 1.240616e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / model za odvojeni test / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.832499367 | KL 0.000733 | proteklo 6.4 s
Iteracija 10/12 | cilj 0.830083631 | KL 0.001253 | proteklo 11.7 s
Zadržani cilj: 0.829316277
Max|proj.grad|: 4.655793e-03
KL: 0.001183
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / završno učenje na svim prelazima / početno kvantno učenje
Iteracija 5/60 | cilj 16.494464647 | KL 0.000000 | proteklo 3.6 s
Iteracija 10/60 | cilj 16.490463247 | KL 0.000000 | proteklo 6.2 s
Iteracija 15/60 | cilj 16.485958916 | KL 0.000000 | proteklo 8.8 s
Iteracija 20/60 | cilj 16.484061266 | KL 0.000000 | proteklo 11.5 s
Iteracija 25/60 | cilj 16.482534973 | KL 0.000000 | proteklo 14.1 s
Iteracija 30/60 | cilj 16.481114414 | KL 0.000000 | proteklo 16.8 s
Iteracija 35/60 | cilj 16.479759460 | KL 0.000000 | proteklo 20.1 s
Iteracija 40/60 | cilj 16.479356182 | KL 0.000000 | proteklo 22.7 s
Iteracija 45/60 | cilj 16.478845283 | KL 0.000000 | proteklo 25.8 s
Iteracija 50/60 | cilj 16.478656130 | KL 0.000000 | proteklo 28.4 s
Iteracija 55/60 | cilj 16.478360997 | KL 0.000000 | proteklo 31.6 s
Iteracija 60/60 | cilj 16.478209253 | KL 0.000000 | proteklo 34.2 s
Zadržani cilj: 16.478209253
Max|proj.grad|: 4.964780e-03
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / završno učenje na svim prelazima / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001896243 | KL 0.000000 | proteklo 0.3 s
Iteracija 10/20 | cilj 0.000442366 | KL 0.000000 | proteklo 0.4 s
Iteracija 15/20 | cilj 0.000406392 | KL 0.000000 | proteklo 0.6 s
Iteracija 20/20 | cilj 0.000399781 | KL 0.000000 | proteklo 0.8 s
Zadržani cilj: 0.000399781
Max|proj.grad|: 4.162083e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / završno učenje na svim prelazima / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.841731268 | KL 0.002094 | proteklo 8.1 s
Iteracija 10/12 | cilj 0.833129621 | KL 0.003304 | proteklo 14.9 s
Zadržani cilj: 0.831724306
Max|proj.grad|: 2.924912e-02
KL: 0.004754
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv / završno učenje na svim prelazima / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000407444 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000407025 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000406345 | KL 0.000000 | proteklo 0.5 s
Iteracija 20/20 | cilj 0.000405863 | KL 0.000000 | proteklo 0.7 s
Zadržani cilj: 0.000405863
Max|proj.grad|: 6.654762e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_2977.csv / završno učenje na svim prelazima / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.832113648 | KL 0.000945 | proteklo 8.4 s
Iteracija 10/12 | cilj 0.830517347 | KL 0.001091 | proteklo 15.3 s
Zadržani cilj: 0.829900081
Max|proj.grad|: 7.906661e-03
KL: 0.001471
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_2977.csv
NEXT: 16 x 33 y 35 z 39
Validacioni NLL: 16.589135877
Odvojeni test NLL: 16.574039601
Test NLL početnog kvantnog modela: 16.618772168
Test NLL referentnog Qiskit kola: 16.548639443
Izabrani broj PPO rundi: 2
Modelna verovatnoća NEXT: 1.68764824036e-07
Ukupna masa završnih putanja: 1.000000000000
Zbir marginalnih verovatnoća: 7.000000000000
Greška eksplicitnog Qiskit filtera: 9.992e-16
Dijagnostička osetljivost: 1.679737e-02
Završni Max|proj.grad|: 7.906661e-03
Norma parametara kvantnog mešača: 9.868910e-02
Norma amplitudnih pomeraja: 3.458448e-01
Norma prelaznih pomeraja: 1.151112e+00
Podrška raspodele: 15,380,937 sedmorki
MAP je izračunat tačno dinamičkim programiranjem.
Test: broj kola prema broju pogodaka
0: 123 | 1: 243 | 2: 171 | 3: 52 | 4: 7 | 5: 0 | 6: 0 | 7: 0
Prosečan broj pogodaka MAP kombinacije: 1.290268
Marginalne verovatnoće uključivanja brojeva:
01: 0.20159669 | 02: 0.17928439 | 03: 0.19514923 | 04: 0.18924219 | 05: 0.18811240 | 06: 0.18501632
07: 0.18027640 | 08: 0.20326885 | 09: 0.18108507 | 10: 0.17228020 | 11: 0.16815904 | 12: 0.17232018
13: 0.16604235 | 14: 0.17219565 | 15: 0.17303593 | 16: 0.18243170 | 17: 0.16301704 | 18: 0.16071654
19: 0.16733517 | 20: 0.16061430 | 21: 0.16959862 | 22: 0.18086837 | 23: 0.17652797 | 24: 0.18240986
25: 0.16293534 | 26: 0.17703431 | 27: 0.17591399 | 28: 0.17755757 | 29: 0.18211147 | 30: 0.17190852
31: 0.17460406 | 32: 0.18264438 | 33: 0.18973271 | 34: 0.19090404 | 35: 0.18927679 | 36: 0.18057983
37: 0.18822014 | 38: 0.19316932 | 39: 0.19282307
TEST STATUS: POBOLJŠANJE NIJE POTVRĐENO
DIJAGNOSTIKA: Početno učenje nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Akter u PPO rundi 1 nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Akter u PPO rundi 2 nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Odvojeni test ne potvrđuje poboljšanje u odnosu na referentno Qiskit kolo.


loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.3 / početno kvantno učenje
Iteracija 5/60 | cilj 16.533113837 | KL 0.000000 | proteklo 1.2 s
Iteracija 10/60 | cilj 16.499084764 | KL 0.000000 | proteklo 2.1 s
Iteracija 15/60 | cilj 16.440721083 | KL 0.000000 | proteklo 3.2 s
Iteracija 20/60 | cilj 16.423803425 | KL 0.000000 | proteklo 4.1 s
Iteracija 25/60 | cilj 16.401660345 | KL 0.000000 | proteklo 5.0 s
Iteracija 30/60 | cilj 16.376714606 | KL 0.000000 | proteklo 6.0 s
Iteracija 35/60 | cilj 16.358180442 | KL 0.000000 | proteklo 6.9 s
Iteracija 40/60 | cilj 16.339484077 | KL 0.000000 | proteklo 7.8 s
Iteracija 45/60 | cilj 16.331160838 | KL 0.000000 | proteklo 8.7 s
Iteracija 50/60 | cilj 16.323928894 | KL 0.000000 | proteklo 9.7 s
Iteracija 55/60 | cilj 16.315002605 | KL 0.000000 | proteklo 10.6 s
Iteracija 60/60 | cilj 16.310424797 | KL 0.000000 | proteklo 11.7 s
Zadržani cilj: 16.310424797
Max|proj.grad|: 5.940156e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.3 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001881930 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000437122 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000404261 | KL 0.000000 | proteklo 0.2 s
Iteracija 20/20 | cilj 0.000397362 | KL 0.000000 | proteklo 0.3 s
Zadržani cilj: 0.000397362
Max|proj.grad|: 1.419700e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.3 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.824347051 | KL 0.001256 | proteklo 2.8 s
Iteracija 10/12 | cilj 0.822728828 | KL 0.002303 | proteklo 5.2 s
Zadržani cilj: 0.822289577
Max|proj.grad|: 8.309808e-03
KL: 0.002292
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.3 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000403028 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000402531 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000401773 | KL 0.000000 | proteklo 0.2 s
Iteracija 20/20 | cilj 0.000401376 | KL 0.000000 | proteklo 0.3 s
Zadržani cilj: 0.000401376
Max|proj.grad|: 1.179576e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 0.3 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.824096364 | KL 0.000673 | proteklo 2.8 s
Iteracija 10/12 | cilj 0.823221492 | KL 0.001338 | proteklo 5.0 s
Zadržani cilj: 0.822747078
Max|proj.grad|: 1.042693e-02
KL: 0.001820
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 1 | ridge 0.3 | PPO rundi 0 | NLL 16.720151878
Validacija 1 | ridge 0.3 | PPO rundi 1 | NLL 16.722503938
Validacija 1 | ridge 0.3 | PPO rundi 2 | NLL 16.722386512

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.3 / početno kvantno učenje
Iteracija 5/60 | cilj 16.364029848 | KL 0.000000 | proteklo 1.5 s
Iteracija 10/60 | cilj 16.354401994 | KL 0.000000 | proteklo 2.5 s
Iteracija 15/60 | cilj 16.343552405 | KL 0.000000 | proteklo 3.6 s
Iteracija 20/60 | cilj 16.337178733 | KL 0.000000 | proteklo 4.9 s
Iteracija 25/60 | cilj 16.330140461 | KL 0.000000 | proteklo 6.2 s
Iteracija 30/60 | cilj 16.326848338 | KL 0.000000 | proteklo 7.2 s
Iteracija 35/60 | cilj 16.322890635 | KL 0.000000 | proteklo 8.2 s
Iteracija 40/60 | cilj 16.320214171 | KL 0.000000 | proteklo 9.3 s
Iteracija 45/60 | cilj 16.317412754 | KL 0.000000 | proteklo 10.5 s
Iteracija 50/60 | cilj 16.315186325 | KL 0.000000 | proteklo 11.5 s
Iteracija 55/60 | cilj 16.313589874 | KL 0.000000 | proteklo 12.6 s
Iteracija 60/60 | cilj 16.309936568 | KL 0.000000 | proteklo 13.6 s
Zadržani cilj: 16.309936568
Max|proj.grad|: 3.068442e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.3 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001898882 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000444255 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000411167 | KL 0.000000 | proteklo 0.3 s
Iteracija 20/20 | cilj 0.000403537 | KL 0.000000 | proteklo 0.3 s
Zadržani cilj: 0.000403537
Max|proj.grad|: 2.474629e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.3 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.826778142 | KL 0.001467 | proteklo 3.7 s
Iteracija 10/12 | cilj 0.825028059 | KL 0.002676 | proteklo 6.3 s
Zadržani cilj: 0.824697972
Max|proj.grad|: 3.724072e-03
KL: 0.002424
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.3 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000414893 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000414335 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000413872 | KL 0.000000 | proteklo 0.3 s
Iteracija 20/20 | cilj 0.000413557 | KL 0.000000 | proteklo 0.4 s
Zadržani cilj: 0.000413557
Max|proj.grad|: 4.999194e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 0.3 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.826166931 | KL 0.000638 | proteklo 3.2 s
Iteracija 10/12 | cilj 0.825098013 | KL 0.001730 | proteklo 5.8 s
Zadržani cilj: 0.824629362
Max|proj.grad|: 4.045139e-03
KL: 0.001584
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 2 | ridge 0.3 | PPO rundi 0 | NLL 16.866973620
Validacija 2 | ridge 0.3 | PPO rundi 1 | NLL 16.894541229
Validacija 2 | ridge 0.3 | PPO rundi 2 | NLL 16.912069991

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 3.0 / početno kvantno učenje
Iteracija 5/60 | cilj 16.533286594 | KL 0.000000 | proteklo 1.2 s
Iteracija 10/60 | cilj 16.500447779 | KL 0.000000 | proteklo 2.1 s
Iteracija 15/60 | cilj 16.450970674 | KL 0.000000 | proteklo 3.0 s
Iteracija 20/60 | cilj 16.436872855 | KL 0.000000 | proteklo 3.9 s
Iteracija 25/60 | cilj 16.417064791 | KL 0.000000 | proteklo 4.9 s
Iteracija 30/60 | cilj 16.400373169 | KL 0.000000 | proteklo 5.8 s
Iteracija 35/60 | cilj 16.389507883 | KL 0.000000 | proteklo 6.7 s
Iteracija 40/60 | cilj 16.381438965 | KL 0.000000 | proteklo 7.6 s
Iteracija 45/60 | cilj 16.375079228 | KL 0.000000 | proteklo 8.4 s
Iteracija 50/60 | cilj 16.371451034 | KL 0.000000 | proteklo 9.5 s
Iteracija 55/60 | cilj 16.368056943 | KL 0.000000 | proteklo 10.4 s
Iteracija 60/60 | cilj 16.365588208 | KL 0.000000 | proteklo 11.4 s
Zadržani cilj: 16.365588208
Max|proj.grad|: 2.718710e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 3.0 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001878873 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000432171 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000399798 | KL 0.000000 | proteklo 0.2 s
Iteracija 20/20 | cilj 0.000392308 | KL 0.000000 | proteklo 0.3 s
Zadržani cilj: 0.000392308
Max|proj.grad|: 2.137993e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 3.0 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.856332116 | KL 0.002885 | proteklo 2.7 s
Iteracija 10/12 | cilj 0.840160565 | KL 0.005589 | proteklo 4.9 s
Zadržani cilj: 0.837534066
Max|proj.grad|: 1.007880e-02
KL: 0.006212
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 3.0 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000398488 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000398094 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000397592 | KL 0.000000 | proteklo 0.2 s
Iteracija 20/20 | cilj 0.000397148 | KL 0.000000 | proteklo 0.3 s
Zadržani cilj: 0.000397148
Max|proj.grad|: 7.080418e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / validacija 1 / ridge 3.0 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.834250906 | KL 0.001235 | proteklo 3.1 s
Iteracija 10/12 | cilj 0.832119089 | KL 0.001285 | proteklo 5.4 s
Zadržani cilj: 0.829928916
Max|proj.grad|: 1.740471e-02
KL: 0.002345
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 1 | ridge 3.0 | PPO rundi 0 | NLL 16.629171259
Validacija 1 | ridge 3.0 | PPO rundi 1 | NLL 16.585230346
Validacija 1 | ridge 3.0 | PPO rundi 2 | NLL 16.598946192

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 3.0 / početno kvantno učenje
Iteracija 5/60 | cilj 16.406388534 | KL 0.000000 | proteklo 1.5 s
Iteracija 10/60 | cilj 16.401528661 | KL 0.000000 | proteklo 2.5 s
Iteracija 15/60 | cilj 16.391833617 | KL 0.000000 | proteklo 3.5 s
Iteracija 20/60 | cilj 16.389746076 | KL 0.000000 | proteklo 4.5 s
Iteracija 25/60 | cilj 16.387362190 | KL 0.000000 | proteklo 5.6 s
Iteracija 30/60 | cilj 16.384072098 | KL 0.000000 | proteklo 6.8 s
Iteracija 35/60 | cilj 16.382982443 | KL 0.000000 | proteklo 8.0 s
Iteracija 40/60 | cilj 16.381422616 | KL 0.000000 | proteklo 9.1 s
Iteracija 45/60 | cilj 16.380754195 | KL 0.000000 | proteklo 10.3 s
Iteracija 50/60 | cilj 16.380054426 | KL 0.000000 | proteklo 11.3 s
Iteracija 55/60 | cilj 16.379400714 | KL 0.000000 | proteklo 12.6 s
Iteracija 60/60 | cilj 16.378885413 | KL 0.000000 | proteklo 13.6 s
Zadržani cilj: 16.378885413
Max|proj.grad|: 2.141448e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 3.0 / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001891483 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000440056 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000407233 | KL 0.000000 | proteklo 0.3 s
Iteracija 20/20 | cilj 0.000399571 | KL 0.000000 | proteklo 0.4 s
Zadržani cilj: 0.000399571
Max|proj.grad|: 1.555206e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 3.0 / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.857080696 | KL 0.002177 | proteklo 3.2 s
Iteracija 10/12 | cilj 0.839781651 | KL 0.005983 | proteklo 5.8 s
Zadržani cilj: 0.837456979
Max|proj.grad|: 2.543946e-02
KL: 0.006708
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 3.0 / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000402378 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000401936 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000401354 | KL 0.000000 | proteklo 0.2 s
Iteracija 20/20 | cilj 0.000401169 | KL 0.000000 | proteklo 0.3 s
Zadržani cilj: 0.000401169
Max|proj.grad|: 1.078290e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / validacija 2 / ridge 3.0 / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.834705817 | KL 0.001188 | proteklo 3.2 s
Iteracija 10/12 | cilj 0.831853721 | KL 0.001267 | proteklo 5.8 s
Zadržani cilj: 0.829224744
Max|proj.grad|: 9.735580e-03
KL: 0.002364
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.
Validacija 2 | ridge 3.0 | PPO rundi 0 | NLL 16.745211232
Validacija 2 | ridge 3.0 | PPO rundi 1 | NLL 16.672740480
Validacija 2 | ridge 3.0 | PPO rundi 2 | NLL 16.642527736

loto7_4710_k86_loto_plus_1733.csv: izabrani ridge 3.0
Izabrani broj PPO rundi: 2
Validacioni NLL: 16.620736964

loto7_4710_k86_loto_plus_1733.csv / model za odvojeni test / početno kvantno učenje
Iteracija 5/60 | cilj 16.427945875 | KL 0.000000 | proteklo 1.6 s
Iteracija 10/60 | cilj 16.423163353 | KL 0.000000 | proteklo 2.8 s
Iteracija 15/60 | cilj 16.416997852 | KL 0.000000 | proteklo 4.0 s
Iteracija 20/60 | cilj 16.415735131 | KL 0.000000 | proteklo 5.6 s
Iteracija 25/60 | cilj 16.412975273 | KL 0.000000 | proteklo 6.8 s
Iteracija 30/60 | cilj 16.411331166 | KL 0.000000 | proteklo 8.2 s
Iteracija 35/60 | cilj 16.410417527 | KL 0.000000 | proteklo 9.4 s
Iteracija 40/60 | cilj 16.409473854 | KL 0.000000 | proteklo 10.8 s
Iteracija 45/60 | cilj 16.409023402 | KL 0.000000 | proteklo 12.0 s
Iteracija 50/60 | cilj 16.408591677 | KL 0.000000 | proteklo 13.2 s
Iteracija 55/60 | cilj 16.408269278 | KL 0.000000 | proteklo 14.6 s
Iteracija 60/60 | cilj 16.408073532 | KL 0.000000 | proteklo 15.8 s
Zadržani cilj: 16.408073532
Max|proj.grad|: 1.159846e-02
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / model za odvojeni test / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001885049 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000437872 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000405533 | KL 0.000000 | proteklo 0.3 s
Iteracija 20/20 | cilj 0.000397791 | KL 0.000000 | proteklo 0.4 s
Zadržani cilj: 0.000397791
Max|proj.grad|: 2.553716e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / model za odvojeni test / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.853961938 | KL 0.003047 | proteklo 3.6 s
Iteracija 10/12 | cilj 0.839892045 | KL 0.005195 | proteklo 6.7 s
Zadržani cilj: 0.837531812
Max|proj.grad|: 3.591298e-02
KL: 0.006088
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / model za odvojeni test / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000402666 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000402303 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000401768 | KL 0.000000 | proteklo 0.3 s
Iteracija 20/20 | cilj 0.000401480 | KL 0.000000 | proteklo 0.4 s
Zadržani cilj: 0.000401480
Max|proj.grad|: 7.708164e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / model za odvojeni test / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.835022336 | KL 0.001114 | proteklo 3.6 s
Iteracija 10/12 | cilj 0.832795260 | KL 0.000884 | proteklo 6.7 s
Zadržani cilj: 0.830782564
Max|proj.grad|: 2.151913e-02
KL: 0.002528
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / završno učenje na svim prelazima / početno kvantno učenje
Iteracija 5/60 | cilj 16.453042161 | KL 0.000000 | proteklo 2.1 s
Iteracija 10/60 | cilj 16.443779258 | KL 0.000000 | proteklo 3.5 s
Iteracija 15/60 | cilj 16.437520333 | KL 0.000000 | proteklo 5.0 s
Iteracija 20/60 | cilj 16.436162081 | KL 0.000000 | proteklo 6.8 s
Iteracija 25/60 | cilj 16.432675519 | KL 0.000000 | proteklo 8.3 s
Iteracija 30/60 | cilj 16.430954725 | KL 0.000000 | proteklo 10.1 s
Iteracija 35/60 | cilj 16.429706218 | KL 0.000000 | proteklo 11.5 s
Iteracija 40/60 | cilj 16.428729161 | KL 0.000000 | proteklo 13.0 s
Iteracija 45/60 | cilj 16.428289788 | KL 0.000000 | proteklo 14.5 s
Iteracija 50/60 | cilj 16.427582223 | KL 0.000000 | proteklo 16.3 s
Iteracija 55/60 | cilj 16.427245860 | KL 0.000000 | proteklo 17.8 s
Iteracija 60/60 | cilj 16.426822990 | KL 0.000000 | proteklo 19.6 s
Zadržani cilj: 16.426822990
Max|proj.grad|: 8.695425e-03
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / završno učenje na svim prelazima / PPO 1 / kvantni kritičar
Iteracija 5/20 | cilj 0.001875404 | KL 0.000000 | proteklo 0.2 s
Iteracija 10/20 | cilj 0.000432007 | KL 0.000000 | proteklo 0.3 s
Iteracija 15/20 | cilj 0.000399578 | KL 0.000000 | proteklo 0.4 s
Iteracija 20/20 | cilj 0.000391565 | KL 0.000000 | proteklo 0.5 s
Zadržani cilj: 0.000391565
Max|proj.grad|: 1.937186e-05
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / završno učenje na svim prelazima / PPO 1 / kvantni akter
Iteracija 5/12 | cilj 0.851979078 | KL 0.001832 | proteklo 4.6 s
Iteracija 10/12 | cilj 0.837082767 | KL 0.005340 | proteklo 8.4 s
Zadržani cilj: 0.835548504
Max|proj.grad|: 1.288236e-02
KL: 0.005230
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv / završno učenje na svim prelazima / PPO 2 / kvantni kritičar
Iteracija 5/20 | cilj 0.000396646 | KL 0.000000 | proteklo 0.1 s
Iteracija 10/20 | cilj 0.000396198 | KL 0.000000 | proteklo 0.2 s
Iteracija 15/20 | cilj 0.000395536 | KL 0.000000 | proteklo 0.3 s
Iteracija 20/20 | cilj 0.000395215 | KL 0.000000 | proteklo 0.4 s
Zadržani cilj: 0.000395215
Max|proj.grad|: 5.150780e-06
KL: 0.000000
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

loto7_4710_k86_loto_plus_1733.csv / završno učenje na svim prelazima / PPO 2 / kvantni akter
Iteracija 5/12 | cilj 0.834378753 | KL 0.001123 | proteklo 4.6 s
Iteracija 10/12 | cilj 0.832107622 | KL 0.000869 | proteklo 8.5 s
Zadržani cilj: 0.829976148
Max|proj.grad|: 2.558406e-02
KL: 0.002802
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: gradijentni uslov nije potvrđen.
Kandidat je validan; minimum nije potvrđen.

loto7_4710_k86_loto_plus_1733.csv
NEXT: 1 x 5 y 7 z 34
Validacioni NLL: 16.620736964
Odvojeni test NLL: 16.582914957
Test NLL početnog kvantnog modela: 16.605846889
Test NLL referentnog Qiskit kola: 16.548639443
Izabrani broj PPO rundi: 2
Modelna verovatnoća NEXT: 3.21726613202e-07
Ukupna masa završnih putanja: 1.000000000000
Zbir marginalnih verovatnoća: 7.000000000000
Greška eksplicitnog Qiskit filtera: 2.637e-16
Dijagnostička osetljivost: 3.209223e-02
Završni Max|proj.grad|: 2.558406e-02
Norma parametara kvantnog mešača: 1.212115e-01
Norma amplitudnih pomeraja: 4.315018e-01
Norma prelaznih pomeraja: 1.159859e+00
Podrška raspodele: 15,380,937 sedmorki
MAP je izračunat tačno dinamičkim programiranjem.
Test: broj kola prema broju pogodaka
0: 79 | 1: 141 | 2: 94 | 3: 28 | 4: 5 | 5: 0 | 6: 0 | 7: 0
Prosečan broj pogodaka MAP kombinacije: 1.247839
Marginalne verovatnoće uključivanja brojeva:
01: 0.20375426 | 02: 0.20379529 | 03: 0.18787479 | 04: 0.18998547 | 05: 0.20282414 | 06: 0.19808775
07: 0.19226806 | 08: 0.19278712 | 09: 0.17455407 | 10: 0.17484156 | 11: 0.17752804 | 12: 0.16787500
13: 0.17807479 | 14: 0.17527703 | 15: 0.16571864 | 16: 0.15943042 | 17: 0.15009274 | 18: 0.17036141
19: 0.14608160 | 20: 0.15562308 | 21: 0.15274672 | 22: 0.18196188 | 23: 0.17043444 | 24: 0.16386775
25: 0.17402373 | 26: 0.19581698 | 27: 0.17359472 | 28: 0.18378332 | 29: 0.17692028 | 30: 0.17491507
31: 0.17012512 | 32: 0.20292466 | 33: 0.18294962 | 34: 0.21215778 | 35: 0.16855779 | 36: 0.18037780
37: 0.19197396 | 38: 0.18548916 | 39: 0.19054395
TEST STATUS: POBOLJŠANJE NIJE POTVRĐENO
DIJAGNOSTIKA: Početno učenje nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Akter u PPO rundi 1 nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Akter u PPO rundi 2 nema potvrđen gradijentni uslov.
DIJAGNOSTIKA: Odvojeni test ne potvrđuje poboljšanje u odnosu na referentno Qiskit kolo.

ZAVRŠNI NEXT REZULTATI
loto7_4710_k86.csv | NEXT: 1 2 5 33 34 35 36 | TEST POBOLJŠANJE NIJE POTVRĐENO
loto7_4710_k86_loto_2977.csv | NEXT: 16 22 33 34 35 38 39 | TEST POBOLJŠANJE NIJE POTVRĐENO
loto7_4710_k86_loto_plus_1733.csv | NEXT: 1 2 5 6 7 26 34 | TEST POBOLJŠANJE NIJE POTVRĐENO
"""



"""
Sekvencijalno biranje poteza uz učenje politike i procenu vrednosti. 

Za moj Loto najzanimljivije su tri stvari:

Ideja iz PyCatan	                    Moguća primena
Politika zavisi od trenutnog stanja	    Qiskit kolo bira naredni broj na osnovu istorije i već izabranih brojeva. Tako sedmorka može imati naučene unutrašnje zavisnosti.
Maska legalnih poteza	                Odmah isključuje brojeve izvan 1-39 i već izabrane brojeve. Bez naknadnog popravljanja duplikata i kozmetičkih filtera.
Politika i procena vrednosti	        Kvantna politika daje verovatnoće izbora, a kvantni procenjivač ocenjuje delimično sastavljenu sedmorku.

CSV sadrži izvučene brojeve, ali nema agentove odluke, njihove tadašnje verovatnoće ni nagrade. 
Te podatke ne smem izmisliti niti istorijske rezultate pogrešno proglasiti agentovim potezima. 
Takođe rešiti učenje bez slučajnog uzorkovanja kombinacija.

Qiskit politika koja gradi sedmorku kroz sedam zavisnih odluka, uz procenjivač vrednosti. 
V1 sa kvantnom politikom koja bira sedam brojeva redom i kvantnim procenjivačem vrednosti delimične sedmorke. 
Nagradu definišem kao broj pogodaka podeljen sa sedam, uz zadržavanje učenja verovatnoće cele sedmorke. 
PPO očekivanja računaću tačno preko stabla odluka, bez slučajnih poteza.

Računanje mogu da svedem na stanje „korak izbora + poslednji izabrani broj“, pa ista stanja ne moram ponovo obilaziti za svaku sedmorku. 
Time dobijamo tačna PPO očekivanja i tačan maksimum zajedničke raspodele. 
Validacija će uporediti početno kvantno učenje sa jednom i dve PPO runde; ako PPO ne pomogne, to će ispis jasno pokazati.

V1 zadržava početno učenje, kvantnu politiku, kvantnog procenjivača vrednosti i PPO sa ograničenom promenom politike. 
Sedam izbora čini zajedničku raspodelu; naredni izbor zavisi i od prethodno izabranog broja.
Koristi težine svih mogućih nastavaka sedmorke. 

Razdvojam parametre kvantnih amplituda i prelaza, uklonio sam fiksni Hamiltonijan koji ograničava oba dela i dodao sam proveru da kvantni akter može da predstavlja neutralne težine bez favorizovanja oznaka brojeva. PPO, globalnu normalizaciju i tačan izbor maksimuma.

Amplitudni deo ima svoje parametre, svaki legalni prelaz svoj ugao kvantnog filtera, a procenjivač vrednosti ostaje zaseban. 
Dodajem i test referentnog Qiskit kola pre treninga: time se proverava da arhitektura sama ne nameće stare obrasce. 
Ispis će odvojeno pokazati numeričku ispravnost i rezultat na odvojenom testu.
Amplitude, prelazi i procenjivač vrednosti ne dele iste parametre. 
Prelazi imaju zasebne uglove Qiskit kapija. Ne fiksni Hamiltonijan. Globalna normalizacija i PPO.

Kod uključuje proveru referentnog kvantnog stanja, nezavisnosti parametara, gradijenata i maksimuma. 
Zaseban direktorijum kontrolnih tačaka.
"""
