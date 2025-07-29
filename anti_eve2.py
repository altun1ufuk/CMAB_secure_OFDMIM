# Ufuk Altun

# Ufuk Altun

from itertools import product
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import os
from scipy.stats import entropy
from multiprocessing import Pool
from tqdm import tqdm

import itertools
import math
import random

from math import floor, log2, sqrt
from numpy.random import randn, rand
from scipy.signal import lfilter
from numpy.fft import ifft, fft





class ContextualBandit:
    def __init__(self, action_space, n_training, epsilon_start=1.0, epsilon_min=0.01):
        """Contextual Bandit with Decaying Epsilon-Greedy Algorithm."""
        self.action_space = action_space
        self.n_training = n_training
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_factor = n_training / 5  # Adjust for decay speed
        self.q_values = {}  # Q-values for each (context, action) pair
        self.action_counts = {}  # Track action selections per context

    def get_q_value(self, context, action):
        """Retrieve Q-value for (context, action), initialize if not present."""
        if (context, action) not in self.q_values:
            self.q_values[(context, action)] = 0.0  # Default Q-value
            self.action_counts[(context, action)] = 0  # Initialize action count
        return self.q_values[(context, action)]

    def select_action(self, context, step):
        """Select an action using an adaptive epsilon-greedy strategy."""
        # Adaptively decay epsilon based on n_training
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(-step / self.decay_factor)

        if np.random.rand() < self.epsilon:  # Explore
            return random.choice(self.action_space)
        else:  # Exploit: Pick the action with the highest Q-value for the given context
            return max(self.action_space, key=lambda action: self.get_q_value(context, action))

    def update_q_values(self, context, action, reward):
        """Update Q-values using an incremental mean update rule."""
        # Ensure (context, action) exists in Q-table
        self.get_q_value(context, action)

        self.action_counts[(context, action)] += 1
        learning_rate = 1 / self.action_counts[(context, action)]  # Adaptive learning rate
        self.q_values[(context, action)] += learning_rate * (reward - self.q_values[(context, action)])

    def entropy_to_context(self, x):
        """Categorizes a single x value."""
        if 0 <= x < 1:
            return 0
        elif 1 <= x < 1.8:
            return 1
        elif 1.8 <= x < 2.6:
            return 2
        else:  # x >= 2.6
            return 3



class RuleBased:
    """Baseline model with no reinforcement learning. Actions are chosen by fixed rules."""
    def __init__(self):
        pass
        
    def select_action(self, context, n_training):
        """Selects an action based on rules."""
        if context == 0:
            return (1,1,0,2)  # No transmission
        elif context == 1:
            return (1,1,2,2)  
        elif context == 2:
            return (1,1,4,4) 
        elif context == 3:
            return (1,1,8,8)  
        else:
            return (1,1,2,2)  # Default action if something goes wrong

    def entropy_to_context(self, x):
        """Categorizes a single x value."""
        if 0 <= x < 1:
            return 0
        elif 1 <= x < 1.8:
            return 1
        elif 1.8 <= x < 2.6:
            return 2
        else:  # x >= 2.6
            return 3






class OFDM_IM_Env:
    """
    Simulate an OFDM-IM communication system between Alice, Bob and Eve.
    
    Parameters:
    - EbNo_dB_b, EbNo_dB_e: Signal-to-noise ratio between Alice and Bob/Eve.
    - nSym: Number of OFDM-IM symbols sent at each iteration (n_training).
    - N, Ncp, L: Number of subcarriers/Cyclic prefix/channel taps.
    - ICSI: Imperfect channel state information. If True, adds noise to channel estimations
    - BER_limit_dB: Bit error rate limit/threshold required to correctly decode a frame. 
                    If BER of a frame is above BER_limit_dB, it is discarded.
    - channel_type: Defines how rician channel parameters (K_factor) change.
                    "Rayleigh" -> K_factor=0, "Rician_K=1" -> K_factor=1, "Rician_K=10" -> K_factor=10, 
                    "Dynamic" -> K_factor follows a uniform distribution between np.logspace(-2, 2, 20)
    - diagnostics: Defines how much information is displayed at each iteration. 
                0 -> No text is displayed
                1 -> At each iteration: iteration count, throughput, BER_b, BER_e, Entropy, Context, Action, nBits
                2 -> At each symbol: nErr, Quantization levels, key_a, key_b, key_e, key missmatch Bob, key missmatch Eve
    """
    def __init__(self, EbNo_dB_b, EbNo_dB_e, nSym, N, Ncp, L, ICSI, BER_limit_dB, channel_type, keyEC=True, diagnostics=-1, secure=True):
        self.EbNo_dB_b = EbNo_dB_b
        self.EbNo_dB_e = EbNo_dB_e
        self.nSym = nSym
        self.N = N
        self.Ncp = Ncp
        self.L = L
        self.ICSI = ICSI
        self.Total_length = Ncp + N
        self.subcarrier_map = {
            (4, 2): (2, np.array([[1, 2], [2, 3], [3, 4], [1, 4]])),
            (4, 3): (2, np.array([[1, 2, 3], [2, 3, 4], [1, 3, 4], [1, 2, 4]])),
            (6, 2): (3, np.array([[1, 2], [1, 3], [1, 6], [2, 3], [3, 4], [3, 5], [5, 6], [4, 6]])),
            (6, 3): (4, np.array([
                [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6],
                [1, 3, 4], [1, 4, 5], [1, 4, 6], [1, 5, 6],
                [2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 5, 6],
                [3, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6]
            ])),
            (1, 1): (0, np.array([[1]]))
        }
        self.BER_limit_dB = BER_limit_dB
        self.BER_limit_lin = (10 ** (-BER_limit_dB / 10))
        self.channel_type = channel_type
        self.keyEC = keyEC
        self.diagnostics = diagnostics
        self.secure = secure
        
    def _unpackbits(self, x, num_bits):
        """A Modified version of numpy's unpackbits."""
        if np.issubdtype(x.dtype, np.floating):
            raise ValueError("numpy data type needs to be int-like")
        xshape = list(x.shape)
        x = x.reshape([-1, 1])
        mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
        return np.flip((x & mask).astype(bool).astype(int).reshape(xshape + [num_bits]))

    def _hamming_correct_key(self, key_a, key_b):
        """Corrects single-bit errors in key_b using Hamming (7,4) error correction, based on key_a."""
        corrected_key_b = np.copy(key_b)
        block_size = 4  # Number of data bits per block
        for i in range(0, len(key_a), block_size):
            if i + 3 >= len(key_a):  # Ensure we don't exceed the key length
                break
            # Extract 4-bit blocks from Alice and Bob
            d1, d2, d3, d4 = key_a[i:i+4]
            r1, r2, r3, r4 = key_b[i:i+4]

            # Compute correct parity bits from Alice?s key
            p1 = d1 ^ d2 ^ d4
            p2 = d1 ^ d3 ^ d4
            p3 = d2 ^ d3 ^ d4

            # Compute received parity bits from Bob?s key (as if received incorrectly)
            p1_b = r1 ^ r2 ^ r4
            p2_b = r1 ^ r3 ^ r4
            p3_b = r2 ^ r3 ^ r4

            # Calculate syndrome bits (where the error is)
            s1 = p1 ^ p1_b
            s2 = p2 ^ p2_b
            s3 = p3 ^ p3_b

            # Find the error position in the 7-bit encoded block
            error_pos = (s3 << 2) | (s2 << 1) | s1  # Convert binary syndrome to decimal

            # Map the 7-bit position to the 4-bit key_b index
            error_map = {3: 0, 5: 1, 6: 2, 7: 3}  # Only correct data bits

            if error_pos in error_map:
                bit_index = i + error_map[error_pos]
                corrected_key_b[bit_index] ^= 1  # Flip the incorrect bit

        return corrected_key_b
     
    def _generate_constellation(self, M):
        """Generate constellation mapping based on M (modulation order)."""
        if M == 2:
            return sqrt(1) * np.array([-1, 1])
        elif M == 4:  # QPSK
            return sqrt(1 / 2) * np.array([1 + 1j, -1 + 1j, 1 - 1j, - 1 - 1j])
        elif M == 8:  # 8-PSK
            return sqrt(1 / 5) * np.array([
                -3 / sqrt(2) + 1j * 3 / sqrt(2), 3 / sqrt(2) + 1j * 3 / sqrt(2),
                -3 / sqrt(2) - 1j * 3 / sqrt(2), 3 / sqrt(2) - 1j * 3 / sqrt(2), 
                1j, 1, -1, -1j
            ])
        elif M == 16:  # 16-QAM
            return sqrt(1 / 10) * np.array([
                -3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j, -1 - 3j,
                -1 - 1j, -1 + 3j, -1 + 1j, 3 - 3j, 3 - 1j, 
                3 + 3j, 3 + 1j, 1 - 3j, 1 - 1j, 1 + 3j, 1 + 1j
            ])
        else:
            raise ValueError("Invalid M value")

    def _generate_subcarrier_combinations(self, n, k):
        """Define active subcarrier index combinations."""
        if (n, k) in self.subcarrier_map:
            return self.subcarrier_map[(n, k)]
        else:
            raise ValueError("Invalid n, k values")
            
    def _generate_symbol_space(self, c, M, k, n, scComb, constmap):
        """Generate the symbol space based on subcarrier combinations and modulation."""
        symbol_space = np.zeros((c * M ** k, n), dtype=np.complex128)
        x = np.array(list(itertools.combinations(np.tile(range(1, M + 1), (1, k))[0], k)))
        perm = np.unique(x, axis=0)
        for ll in range(1, c + 1):
            symbol_space[(ll - 1) * M ** k: ll * M ** k, scComb[ll - 1, :] - 1] = constmap[perm - 1]
        return symbol_space
        
    def _OFDM_signal_generation(self, data, g, p, n, K, symbol_space):
        """OFDM signal generation."""
        X_Freq = np.zeros((1, self.N), dtype=np.complex128) # assume that data is in frequency domain
        for jj in range(g):
            temp = data[p * jj: p * (jj + 1)]
            X_Freq[0, n * jj: n * (jj + 1)] = symbol_space[int("".join(map(str, temp)), 2), :]
        x_time = self.N / sqrt(K) * ifft(X_Freq, n=self.N, axis=-1)
        x_time_cp = np.hstack((x_time[:, self.N - self.Ncp:], x_time))
        return x_time_cp
        
    def _simulate_channel(self, x_time_cp, EbNo_dB, Eb, K_factor):
        """Simulate Rician Channel."""
        noise = sqrt(0.5) * (np.random.randn(1, self.Total_length) + 1j * np.random.randn(1, self.Total_length))
        No_t = (10 ** (-EbNo_dB / 10)) * Eb
        if K_factor == 0:
            h_time = sqrt(0.5 / (self.L + 1)) * (np.random.randn(self.L + 1, 1) + 1j * np.random.randn(self.L + 1, 1))
        else:        
            LOS = np.sqrt(K_factor / (K_factor + 1))  # Line-of-sight component
            NLOS = np.sqrt(0.5 / ((K_factor + 1)*(self.L + 1))) * (np.random.randn(self.L+1, 1) + 1j * np.random.randn(self.L+1, 1))
            h_time = (LOS + NLOS) / np.sqrt(np.sum(np.abs(LOS + NLOS) ** 2))
        y_time = lfilter(h_time.flatten(), 1, x_time_cp) + sqrt(No_t) * noise
        return h_time, y_time
    
    def _simulate_receiver(self, h_time, y_time, EbNo_dB, nErr, K, Eb, g, n, k, c, M, symbol_space, p, data_raw, key):
        """Simulate cyclic prefix removal, FFT, ML algorithm and error count steps."""
        data = data_raw ^ key
        y_parallel = y_time[:, self.Ncp:(self.N + self.Ncp)] # Remove cyclic prefix
        Y_fre = sqrt(K) / self.N * fft(y_parallel, n=self.N, axis=-1)  # FFT
        if self.ICSI == True:
            No_t = (10 ** (-EbNo_dB / 10)) * Eb
            No_f = K / self.N * No_t
            h_noise = sqrt(0.5 * No_f) * (np.random.randn(1, self.N) + 1j * np.random.randn(1, self.N))
            H_fre = (fft(h_time.transpose(), n=self.N)) + h_noise
        else:
            H_fre = (fft(h_time.transpose(), n=self.N))
        # ----------------- ML Algorithm ---------------------
        for hh in range(1, g + 1):
            metrik = np.sum(abs(np.tile(Y_fre[0, (hh - 1) * n: n + (hh - 1) * n], (c * M ** k, 1)) - np.tile(
                H_fre[0, (hh - 1) * n: n + (hh - 1) * n], (c * M ** k, 1)) * symbol_space) ** 2, axis=1)
            indexx = np.argmin(metrik)
            # ------------------- Error Counting -------------------
            data_hat = self._unpackbits(np.array([[indexx]]), p)
            numErrors = np.count_nonzero(data[((hh - 1) * p):p + ((hh - 1) * p)] - data_hat)
            nErr = nErr + numErrors  
        if self.diagnostics == 2:
            print("nErr: ", numErrors)
        return nErr
        
    def _calculate_entropy(self, h_samples):
        """Calculate the entropy observed at each subcarrier and return their average."""
        entropies = []
        bin_edges = np.linspace(0.0, 2.5, num=11)  # Define fixed bin edges
        H_fre = fft(h_samples, n=self.N, axis=0)
        for i in range(self.N):  
            magnitudes = np.abs(H_fre[i, :])  # Extract column (subcarrier)
            if np.isnan(magnitudes).any():
                print("nan yüzünden")
            hist, _ = np.histogram(magnitudes, bins=bin_edges, density=True)  # Normalize histogram
            hist = hist + 1e-10  # Avoid log(0) issues
            entropies.append(entropy(hist, base=2))  # Compute entropy for this subcarrier
        avg_entropy = np.nanmean(entropies)  # Compute the average entropy across subcarriers
        return avg_entropy
    
    def _quantize_channel(self, h_samples, Q):
        """Quantize channel observations with Q quantization levels and convert into log2(Q) bits"""
        magnitudes = np.abs(h_samples)  # Use magnitude of the complex channel
        min_val, max_val = np.min(magnitudes), np.max(magnitudes)  
        levels = np.linspace(min_val, max_val, Q) # Uniform Quantization
        if self.diagnostics == 2:
            print("Quantization levels: ", levels)
        quantized_indices = np.digitize(magnitudes, levels) - 1  # Convert to index
        key_bits = self._unpackbits(quantized_indices, int(np.log2(Q))).reshape(-1) # Convert indices to binary
        return key_bits
    
    def _physical_layer_key_generation(self, K_factor, action, Eb, K, key_len):
        """Simulate PLKG process."""
        if self.secure == False:
            return np.zeros(key_len, dtype=int), np.zeros(key_len, dtype=int), np.zeros(key_len, dtype=int)
        # ----------------- Alice-Bob channel ---------------------
        if K_factor == 0: # Alice-Bob channel generation
            h_time = sqrt(0.5 / (self.L + 1)) * (np.random.randn(self.L + 1, 1) + 1j * np.random.randn(self.L + 1, 1))
        else:        
            LOS = np.sqrt(K_factor / (K_factor + 1)) 
            NLOS = np.sqrt(0.5 / ((K_factor + 1)*(self.L + 1))) * (np.random.randn(self.L, 1) + 1j * np.random.randn(self.L, 1))
            h_time = (LOS + NLOS) / np.sqrt(np.sum(np.abs(LOS + NLOS) ** 2))
        H_fre_a = (fft(h_time.transpose(), n=self.N)) # Alice's channel samples at each subcarrier
        n, k, M, Q = action
        No_t = (10 ** (-self.EbNo_dB_b / 10)) * Eb # Bob noise generation
        No_f = K / self.N * No_t
        h_noise = sqrt(0.5 * No_f) * (np.random.randn(1, self.N) + 1j * np.random.randn(1, self.N))
        H_fre_b = (fft(h_time.transpose(), n=self.N)) + h_noise # Bob's channel samples at each subcarrier
        # ----------------- Alice-Eve channel ---------------------
        No_t = (10 ** (-self.EbNo_dB_e / 10)) * Eb # Eve noise generation
        No_f = K / self.N * No_t
        h_noise = sqrt(0.5 * No_f) * (np.random.randn(1, self.N) + 1j * np.random.randn(1, self.N))
        if K_factor == 0: # Eve channel generation
            h_time = sqrt(0.5 / (self.L + 1)) * (np.random.randn(self.L + 1, 1) + 1j * np.random.randn(self.L + 1, 1))
        else:        
            LOS = np.sqrt(K_factor / (K_factor + 1))  
            NLOS = np.sqrt(0.5 / ((K_factor + 1)*(self.L + 1))) * (np.random.randn(self.L+1, 1) + 1j * np.random.randn(self.L+1, 1))
            h_time = (LOS + NLOS) / np.sqrt(np.sum(np.abs(LOS + NLOS) ** 2))
        H_fre_e = (fft(h_time.transpose(), n=self.N)) + h_noise # Eve's channel samples at each subcarrier
        # ----------------- Key generation with channel quantization ---------------------
        key_a = self._quantize_channel(H_fre_a, Q)[:key_len] # Alice key generation
        key_b = self._quantize_channel(H_fre_b, Q)[:key_len] # Bob key generation
        key_e = self._quantize_channel(H_fre_e, Q)[:key_len] # Eve key generation
        if self.keyEC:
            key_b = self._hamming_correct_key(key_a, key_b)
            key_e = self._hamming_correct_key(key_a, key_e)
        if self.diagnostics == 2:
            print("key_a: ", key_a[:35])
            print("key_b: ", key_b[:35])
            print("key_e: ", key_e[:35]) 
        return key_a, key_b, key_e 
        
    def symbol_transmission(self, action, K_factor):
        """
        Simulate the transmission of nSym symbols from Alice to Bob.
        
        Parameters:
        - action: The tuple containing n: # of subcarriers in a subblock, 
                                       k: # of active subcarriers in a subblock, 
                                       M: modulation order,
                                       Q: quantization levels).
        - K_factor: Rician fading parameter. Defines power ratio beterrn LOS and NLOS components.
        
        Returns: 
        - nErr_b: Total number of errors at Bob,
        - nErr_e: Total number of errors at Eve,
        - nSym * m: Total number of transmitted bits,
        - self._calculate_entropy(h_time_total): Average entropy of subcarriers calculated from channel observations.
        """
        n, k, M, Q = action
        if M == 0:
            constmap = self._generate_constellation(2)
            symbol_space = self._generate_symbol_space(1, 2, 1, 1, np.array([[1]]), constmap) 
            h_time_total = np.zeros((self.L+1,self.nSym), dtype=np.complex128)
            for ind_nSym in range(self.nSym):
                x_time_cp = self._OFDM_signal_generation(np.random.randint(2, size=self.N), self.N, 1, 1, self.N, symbol_space)
                h_time, y_time = self._simulate_channel(x_time_cp, self.EbNo_dB_b, self.Total_length / self.N, K_factor)
                h_time_total[:,ind_nSym] = h_time.flatten()
            return 0, 0, 0, self._calculate_entropy(h_time_total)  
        g = self.N // n  # Total number of subblocks per symbol
        K = k * g  # Total number of active subcarriers per symbol
        constmap = self._generate_constellation(M)
        p1, scComb = self._generate_subcarrier_combinations(n,k) # Bits carried by indices per subblock
        c = 2 ** p1  # Number of active subcarrier combinations
        p2 = int(k * np.log2(M))  # Bits carried by modulation per subblock
        p = p1 + p2  # Bits per subblock
        m = g * p  # Bits per symbol
        Eb = self.Total_length / m  # Bit energy
        symbol_space = self._generate_symbol_space(c, M, k, n, scComb, constmap)

        nErr_b, nErr_e = 0, 0
        h_time_total = np.zeros((self.L+1,self.nSym), dtype=np.complex128)
        
        for ind_nSym in range(self.nSym):
            data = np.random.randint(2, size=m) # Random data generation
            key_a, key_b, key_e = self._physical_layer_key_generation(K_factor, action, Eb, K, len(data))
            data_encrypted = data ^ key_a # Data encryption with XOR
            x_time_cp = self._OFDM_signal_generation(data_encrypted, g, p, n, K, symbol_space)
            # ----------------- Bob's reception ---------------------
            h_time, y_time = self._simulate_channel(x_time_cp, self.EbNo_dB_b, Eb, K_factor)
            nErr_b = self._simulate_receiver(h_time, y_time, self.EbNo_dB_b, nErr_b, 
                                             K, Eb, g, n, k, c, M, symbol_space, p, data, key_b)
            h_time_total[:,ind_nSym] = h_time.flatten()
            # ----------------- Eve's reception ---------------------
            h_time_e, y_time_e = self._simulate_channel(x_time_cp, self.EbNo_dB_e, Eb, K_factor)
            nErr_e = self._simulate_receiver(h_time_e, y_time_e, self.EbNo_dB_e, nErr_e, 
                                             K, Eb, g, n, k, c, M, symbol_space, p, data, key_e)
            if self.diagnostics == 2:
                print("key missmatch Bob: ", np.count_nonzero(key_a - key_b), " / ", len(key_a), "key missmatch Eve: ", np.count_nonzero(key_a - key_e), " / ", len(key_a)  )
        return nErr_b, nErr_e, self.nSym * m, self._calculate_entropy(h_time_total)
        
    def reward_function(self, action, nErr_b, nErr_e, nBits):
        """
        Simulate the reward function.
        
        Parameters:
        - action: The tuple containing n: # of subcarriers in a subblock, 
                                       k: # of active subcarriers in a subblock, 
                                       M: modulation order,
                                       Q: quantization levels).
        - nErr_b, nErr_e, nBits: Number of errors at Bob/Eve.
        - nBits: Number of sent bits.
        
        Returns: 
        - reward: A reward proportional to correctly sent bits per symbol. If BER_limit is violated, penaltizes with -N
        - throughput: Total number of errors at Eve,
        - BER_b, BER_e: Total number of transmitted bits,
        """
        if action == (1,1,0,2):
            reward = 0
            BER_b = np.nan
            BER_e = np.nan
        elif nErr_b / nBits < self.BER_limit_lin:
            reward = (nBits - nErr_b) / self.nSym
            BER_b = nErr_b / nBits
            BER_e = nErr_e / nBits
        else:
            reward = -self.N
            BER_b = nErr_b / nBits
            BER_e = nErr_e / nBits
        throughput = max(0, reward)
        return reward, throughput, BER_b, BER_e


def main_run(sim_mode, action_space, 
                nSym=100, N=128, Ncp=16, L=5, ICSI=True,
                EbNo_dB_b=15, EbNo_dB_e=10, BER_limit_dB=30, 
                n_MC=2, n_training=5, n_test=3, channel_type="Rayleigh", keyEC= True, diagnostics=0):
    """
    Unified simulation function that runs Contextual Bandit or Rule-Based models.

    Parameters:
    - sim_mode: "contextual_bandit" or "rulebased"
    - action_space: List of possible actions (n,k,M,Q tuples)
    - nSym: Number of OFDM-IM symbols sent at each iteration (n_training).
    - N, Ncp, L: Number of subcarriers/Cyclic prefix/channel taps.
    - ICSI: Imperfect channel state information. If True, adds noise to channel estimations
    - EbNo_dB_b, EbNo_dB_e: Signal-to-noise ratio between Alice and Bob/Eve.
    - BER_limit_dB: Bit error rate limit/threshold required to correctly decode a frame. 
                    If BER of a frame is above BER_limit_dB, it is discarded.
    - n_MC: Number of Monte Carlo simulations.
    - n_training: Number of iterations at each Monte Carlo simulation.
    - channel_type: Defines how rician channel parameters (K_factor) change.
                    "Rayleigh" -> K_factor=0, "Rician_K=1" -> K_factor=1, "Rician_K=10" -> K_factor=10, 
                    "Dynamic" -> K_factor follows a uniform distribution between np.logspace(-2, 2, 20)
    - diagnostics: Defines how much information is displayed at each iteration and returned as output. 
                -1 -> Production mode, no class returned. avg_throughput, avg_BER_b, avg_BER_e are scalars.
                0 -> No text is displayed. avg_throughput, avg_BER_b, avg_BER_e are matrices with (n_MC x n_training) size
                1 -> At each iteration: iteration count, throughput, BER_b, BER_e, Entropy, Context, Action, nBits.
                2 -> At each symbol: nErr, Quantization levels, key_a, key_b, key_e, key missmatch Bob, key missmatch Eve.
                
    Returns: 
    - avg_throughput: Average throughput of Alice towards Bob across n_MC Monte Carlo iterations,
    - avg_BER_b, avg_BER_e: Average BER of Bob/Eve across n_MC Monte Carlo iterations,
    - env, model: Class objects of OFDM_IM_Env and ContextualBandit. Not returned if diagnostics is -1.
    """
    
    avg_throughput = np.zeros((n_MC, n_test))
    avg_BER_b = np.zeros((n_MC, n_test))
    avg_BER_e = np.zeros((n_MC, n_test))

    progress_bar = tqdm(range(n_MC), desc="Simulation Progress", leave=True)

    for ii in range(n_MC):
        if sim_mode == "insecure_icsi":
            env = OFDM_IM_Env(EbNo_dB_b, EbNo_dB_e, nSym, N, Ncp, L, ICSI, BER_limit_dB, channel_type, keyEC, diagnostics, False)
        elif sim_mode == "insecure_pcsi": 
            env = OFDM_IM_Env(EbNo_dB_b, EbNo_dB_e, nSym, N, Ncp, L, False, BER_limit_dB, channel_type, keyEC, diagnostics, False)
        else:
            env = OFDM_IM_Env(EbNo_dB_b, EbNo_dB_e, nSym, N, Ncp, L, ICSI, BER_limit_dB, channel_type, keyEC, diagnostics, True)
        
        # Initialize model based on sim_mode
        if sim_mode in ["contextual_bandit", "insecure_icsi", "insecure_pcsi"]:
            model = ContextualBandit(action_space=action_space, n_training=n_training)
            context = random.choice([1, 2, 3])
        elif sim_mode == "rulebased":
            model = RuleBased()
            context = random.choice([1, 2, 3])
        else:
            raise ValueError("Invalid sim_mode.")

        # Set initial K-factor
        K_factor = {"Rayleigh": 0, "Rician_K=1": 1, "Rician_K=10": 10}.get(channel_type, None)
        if sim_mode != "rulebased":
            #for i in tqdm(range(n_training), desc=f"Training {sim_mode}", leave=True):
            for i in range(n_training):
                if channel_type == "Dynamic" and i % 10 == 0:
                    K_factor = np.random.choice(np.logspace(-2, 2, 10))
    
                action = model.select_action(context, i) if sim_mode in ["contextual_bandit", "insecure_icsi", "insecure_pcsi"] else model.select_action(i)
                nErr_b, nErr_e, nBits, entropy = env.symbol_transmission(action, K_factor)
                reward, throughput, BER_b, BER_e = env.reward_function(action, nErr_b, nErr_e, nBits)
    
                # Update learning model
                if sim_mode in ["contextual_bandit", "insecure_icsi", "insecure_pcsi"]:
                    model.update_q_values(context, action, reward)
                    context = model.entropy_to_context(entropy)
                else:
                    model.update_q_values(action, reward)
                if diagnostics > 0:
                    print(f"{i:03d} | Mode: {sim_mode} | Action: {action} | Throughput: {throughput:8.4f} | "
                          f"BER_b: {BER_b:.4f} | BER_e: {BER_e:.4f} | Entropy: {entropy:.2f} | Context: {context}")
                
        for i in range(n_test):
            if channel_type == "Dynamic" and i % 10 == 0:
                K_factor = np.random.choice(np.logspace(-2, 2, 10))
            action = model.select_action(context, n_training) if sim_mode in ["contextual_bandit", "rulebased", "insecure_icsi", "insecure_pcsi"] else model.select_action(n_training)
            nErr_b, nErr_e, nBits, entropy = env.symbol_transmission(action, K_factor)
            reward, throughput, BER_b, BER_e = env.reward_function(action, nErr_b, nErr_e, nBits)
            
            avg_throughput[ii, i] = throughput
            avg_BER_b[ii, i] = BER_b
            avg_BER_e[ii, i] = BER_e
        # Update progress bar
        progress_bar.update(1)
    progress_bar.close()
    if diagnostics == -1:
        return np.mean(avg_throughput), np.nanmean(avg_BER_b), np.nanmean(avg_BER_e)
    else:
        return avg_throughput, avg_BER_b, avg_BER_e, env, model
        
def generate_param_sets(param_name, param_values, base_params):
    """
    Generates parameter sets for parallel execution, varying only the specified parameter.
    
    :param param_name: The name of the parameter to vary ("EbNo_dB_b", "EbNo_dB_e", or "BER_limit_dB").
    :param param_values: A list or array of values to iterate over for the chosen parameter.
    :return: A combined list of parameter sets for all models.
    """
    action_space_ofdmim = [(1,1,0,2), (1,1,2,2), (1,1,4,4), (1,1,8,8),
                            (4,2,2,2), (4,2,4,4), (4,2,8,4),
                            (4,3,2,4), (4,3,4,4), (4,3,8,8),
                            (6,2,2,2), (6,2,4,4), (6,2,8,4),
                            (6,3,2,4), (6,3,4,4), (6,3,8,8)]

    action_space_ofdm = [(1,1,0,2), (1,1,2,2), (1,1,4,4), (1,1,8,8)]

    param_sets = []

    for value in param_values:
        # Update the specific parameter
        params = base_params.copy()
        params[param_name] = value  

        param_sets.extend([
            ('contextual_bandit', action_space_ofdmim, *params.values()),
            ('contextual_bandit', action_space_ofdm, *params.values()),
            ('rulebased', action_space_ofdm, *params.values()),
            ('insecure_pcsi', action_space_ofdm, *params.values()),
            ('insecure_icsi', action_space_ofdm, *params.values())
        ])

    return param_sets




if __name__ == "__main__":

    # ----------------- Set simulation parameter ---------------------
    # "EbNo_dB_b", "EbNo_dB_e", "BER_limit_dB", np.arange(-2, 30, 2)
    base_params = {
        "nSym": 100,
        "N": 128,
        "Ncp": 16,
        "L": 5,
        "ICSI": True,
        "EbNo_dB_b": 25,
        "EbNo_dB_e": 15,
        "BER_limit_dB": 15,
        "n_MC": 10,
        "n_training": 700,
        "n_test": 500,
        "channel_type": "Dynamic",
        "keyEC": True,
        "diagnostics": -1
    }
    param_values = np.arange(-2, 30, 2)  #np.arange(20, 300, 20)  # np.arange(-2, 30, 2)
    param_name = "EbNo_dB_e"
    param_sets = generate_param_sets(param_name, param_values, base_params)

    # ----------------- Parallel Run ---------------------
    with Pool(112) as p:
        results_ = p.starmap(main_run, param_sets)

    # ----------------- Post-process ---------------------
    # Define model names in the same order as they were added to param_sets
    model_names = ["Contextual_OFDM_IM", "Contextual_OFDM", 
                 "Rule_Based_OFDM", 
                "Contextual_Incesure_PCSI_OFDM", "Contextual_Incesure_ICSI_OFDM"]
    
    # Organize results by model
    results_by_model = {model: {"throughput": [], "BER_b": [], "BER_e": []} for model in model_names}
    
    for i, (avg_throughput, avg_BER_b, avg_BER_e) in enumerate(results_):
        model_index = i % 7  # Assign result to correct model
        model_name = model_names[model_index]

        results_by_model[model_name]["throughput"].append(avg_throughput)
        results_by_model[model_name]["BER_b"].append(avg_BER_b)
        results_by_model[model_name]["BER_e"].append(avg_BER_e)
    results_by_model[param_name] = param_values

    # ----------------- Plot Figures ---------------------
    plt.figure(figsize=(12, 5))

    # Plot Throughput
    plt.subplot(1, 2, 1)
    for model_name in model_names:
        plt.plot(param_values, results_by_model[model_name]["throughput"], label=model_name, marker="o")
    plt.xlabel("Parameter Value (e.g., EbNo_dB_e)")
    plt.ylabel("Throughput")
    plt.title("Throughput vs. Parameter")
    plt.legend()
    plt.grid()

    # Plot BER (Bob & Eve) on a log scale
    plt.subplot(1, 2, 2)
    for model_name in model_names:
        plt.plot(param_values, results_by_model[model_name]["BER_b"], label=f"{model_name} - Bob", marker="o")
        plt.plot(param_values, results_by_model[model_name]["BER_e"], label=f"{model_name} - Eve", marker="s")

    plt.xlabel("Parameter Value (e.g., EbNo_dB_e)")
    plt.ylabel("BER (log scale)")
    plt.title("BER of Bob & Eve vs. Parameter")
    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid()
    plt.savefig("ber_plot_anti_eve.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ----------------- Save Results ---------------------
    # Save results as a MATLAB .mat file
    scipy.io.savemat("simulation_results.mat", results_by_model)
    print("Results saved to simulation_results.mat!")

