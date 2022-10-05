import numpy as np
from numpy import math as math
from math import factorial as factorial


def Qj_fn(j, N):  # Inverse Laplace Transform Weights
	Qj = 0

	a = int(np.floor((j + 1) / 2))
	b = int(np.min([j, N / 2]))

	for k in range(a, b + 1):
		_q = k ** (N / 2) * factorial(2 * k)
		_q /= factorial((N / 2) - k)
		_q /= factorial(k)
		_q /= factorial(k - 1)
		_q /= factorial(j - k)
		_q /= factorial(2 * k - j)

		Qj += _q

	Qj *= (-1) ** (N / 2 + j)

	return Qj


class BarrierOptionPricer:
	def __init__(self, S, sigma, r, d, _lambda, q1, q2, eta1, eta2, T, t, K, S_u, S_d, si, phi_bar_u, phi_bar_d, N):
		# Market Parameters
		self.S = S  # current price
		self.sigma = sigma  #
		self.r = r  # risk free rate/drift
		self.d = d  # dividend
		self._lambda = _lambda  # jumps rate
		self.q2 = q1  # probability of positive jump opposite to Sepp's paper
		self.q1 = q2  # probability of negative jump
		self.eta2 = eta1  # positive jump size opposite to Sepp's paper
		self.eta1 = eta2  # negative jump size

		# Barrier Option Parameters
		self.T = T  # maturity
		self.t = t  # current time
		self.K = K  # strike price
		self.S_u = S_u  # barrier
		self.S_d = S_d  # barrier
		self.si = si  # +1 for call, -1 for put
		self.phi_bar_u = phi_bar_u  # rebate
		self.phi_bar_d = phi_bar_d

		# helper Variables
		self.m = self.q2 / (1 - self.eta2) + self.q1 / (1 + self.eta1) - 1
		self.mu = self.r - self.d - self._lambda * self.m - 0.5 * self.sigma ** 2
		self.tau = T - t
		self.x = np.log(self.S / self.K)
		self.x_u = np.log(self.S_u / self.K)
		self.x_d = np.log(self.S_d / self.K)

		# Laplace Transform
		self.N = N

	def get_psi(self, p):  # get roots of characteristic function
		pc4 = 0.5 * self.sigma ** 2 * self.eta1 * self.eta2  # polynomial coeff of power 4
		pc3 = self.mu * self.eta1 * self.eta2 - 0.5 * self.sigma ** 2 * (self.eta1 - self.eta2)
		pc2 = -(0.5 * self.sigma ** 2 + self.mu * (self.eta1 - self.eta2) + (
				self.r + p + self._lambda) * self.eta1 * self.eta2)
		pc1 = -self.mu + (self.r + p + self._lambda) * (self.eta1 - self.eta2)
		pc1 -= self._lambda * (self.q2 * self.eta1 - self.q1 * self.eta2)
		pc0 = self.r + p

		coeff = [pc4, pc3, pc2, pc1, pc0]
		psi_arr = np.roots(coeff)
		psi_arr.sort()
		psi3, psi2, psi1, psi0 = list(psi_arr)

		# check eq 4.3
		if self._lambda != 0:
			if not ((-np.infty < psi3) & (psi3 < -1 / self.eta1)):
				return 0
			if not ((-1 / self.eta1 < psi2) & (psi2 < 0)):
				return 0
			if not ((0 < psi1) & (psi1 < 1 / self.eta2)):
				return 0
			if not ((1 / self.eta2 < psi0) & (psi0 < np.infty)):
				return 0

		return psi3, psi2, psi1, psi0

	def get_C0123(self, p, psi3, psi2, psi1, psi0):
		A = np.array([
			[1, 1, -1, -1],
			[psi0, psi1, -psi2, -psi3],
			[1 / (psi0 * self.eta1 + 1), 1 / (psi1 * self.eta1 + 1), -1 / (psi2 * self.eta1 + 1),
			 -1 / (psi3 * self.eta1 + 1)],
			[1 / (psi0 * self.eta2 - 1), 1 / (psi1 * self.eta2 - 1), -1 / (psi2 * self.eta2 - 1),
			 -1 / (psi3 * self.eta2 - 1)]])

		B = np.array([(1 / (self.d + p)) - (1 / (self.r + p)),
					  1 / (self.d + p),
					  1 / ((self.d + p) * (self.eta1 + 1)) - 1 / (self.r + p),
					  1 / ((self.d + p) * (self.eta2 - 1)) + 1 / (self.r + p)])

		C = np.linalg.solve(A, B)

		return C[0], C[1], C[2], C[3]

	def get_C4567(self, p, psi3, psi2, psi1, psi0, C0, C1, C2, C3):

		A = np.array([
			[np.exp(psi0 * self.x_d) / ((psi0 * self.eta1) + 1),
			 np.exp(psi1 * self.x_d) / ((psi1 * self.eta1) + 1),
			 np.exp(psi2 * self.x_d) / ((psi2 * self.eta1) + 1),
			 np.exp(psi3 * self.x_d) / ((psi3 * self.eta1) + 1)],

			[np.exp(psi0 * self.x_d),
			 np.exp(psi1 * self.x_d),
			 np.exp(psi2 * self.x_d),
			 np.exp(psi3 * self.x_d)],

			[np.exp(psi0 * self.x_u),
			 np.exp(psi1 * self.x_u),
			 np.exp(psi2 * self.x_u),
			 np.exp(psi3 * self.x_u)],

			[np.exp(psi0 * self.x_u) / ((psi0 * self.eta2) - 1),
			 np.exp(psi1 * self.x_u) / ((psi1 * self.eta2) - 1),
			 np.exp(psi2 * self.x_u) / ((psi2 * self.eta2) - 1),
			 np.exp(psi3 * self.x_u) / ((psi3 * self.eta2) - 1)]])

		B0 = (-(self.si - 1) / 2) * (np.exp(self.x_d) / ((self.d + p) * (self.eta1 + 1)) - (1 / (self.r + p)))
		B0 += self.phi_bar_d
		B0 -= np.exp(psi0 * self.x_d) * C0 / (psi0 * self.eta1 + 1)
		B0 -= np.exp(psi1 * self.x_d) * C1 / (psi1 * self.eta1 + 1)

		B1 = (-(self.si - 1) / 2) * (np.exp(self.x_d) / (self.d + p) - (1 / (self.r + p)))
		B1 += self.phi_bar_d
		B1 -= np.exp(psi0 * self.x_d) * C0
		B1 -= np.exp(psi1 * self.x_d) * C1

		B2 = (-(self.si + 1) / 2) * (np.exp(self.x_u) / (self.d + p) - (1 / (self.r + p)))
		B2 += self.phi_bar_u
		B2 -= np.exp(psi2 * self.x_u) * C2
		B2 -= np.exp(psi3 * self.x_u) * C3

		B3 = (-(self.si + 1) / 2) * (np.exp(self.x_u) / ((self.d + p) * (self.eta2 - 1)) + (1 / (self.r + p)))
		B3 -= self.phi_bar_u
		B3 -= np.exp(psi2 * self.x_u) * C2 / (psi2 * self.eta2 - 1)
		B3 -= np.exp(psi3 * self.x_u) * C3 / (psi3 * self.eta2 - 1)

		B = np.array([B0, B1, B2, B3])
		C = np.linalg.solve(A, B)
		return C[0], C[1], C[2], C[3]

	def U_DB(self, x, p):  # eq 5.2
		#     print(p)
		#     p0=0.01
		psi3, psi2, psi1, psi0 = self.get_psi(p)
		C0, C1, C2, C3 = self.get_C0123(p, psi3, psi2, psi1, psi0)
		C4, C5, C6, C7 = self.get_C4567(p, psi3, psi2, psi1, psi0, C0, C1, C2, C3)

		U = 0
		if x < 0:
			U += (C0 + C4) * np.exp(psi0 * x)
			U += (C1 + C5) * np.exp(psi1 * x)
			U += C6 * np.exp(psi2 * x) + C7 * np.exp(psi3 * x)
			U += (self.si - 1) / 2 * (np.exp(x) / (self.d + p) - 1 / (self.r + p))

		if x >= 0:
			U += (C2 + C6) * np.exp(psi2 * x)
			U += (C3 + C7) * np.exp(psi3 * x)
			U += C4 * np.exp(psi0 * x) + C5 * np.exp(psi1 * x)
			U += (self.si + 1) / 2 * (np.exp(x) / (self.d + p) - 1 / (self.r + p))

		return U

	def U_DB_fn(self, p):
		return self.U_DB(self.x, p)

	def price(self):
		V = 0

		for j in range(1, self.N + 1):
			_u = self.U_DB_fn(j * np.log(2) / self.tau)
			_q = Qj_fn(j, self.N)
			V += _u * _q
		#     print(j, _q, _u)

		V *= np.log(2) / self.tau
		V *= self.K

		return V