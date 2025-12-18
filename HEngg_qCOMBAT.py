# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 22:58:14 2025

@author: 373591
"""

# Print times, fidelities, U_k, J_k^a, J_k^b as Python array literals.
# If the optimized objects aren't available, rebuild a quick optimized schedule first.

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def ensure_optimized_schedule():
    global pulse, t_grid, F_t
    have = True
    for name in ("pulse","t_grid","F_t"):
        if name not in globals():
            have = False
            break
    if have:
        return

    # ---- Minimal rebuild (same model as used above) ----
    Gamma = np.array([[0,1,1,0],
                      [1,0,0,1],
                      [1,0,0,1],
                      [0,1,1,0]], dtype=float)
    I4 = np.eye(4, dtype=complex)
    def kron(a,b): return np.kron(a,b)
    def projector(site_idx):
        e = np.zeros((4,1), dtype=complex); e[site_idx] = 1.0
        return e @ e.T.conj()

    P_a_sites = [kron(projector(s), I4) for s in range(4)]
    P_b_sites = [kron(I4, projector(t)) for t in range(4)]
    Hint_base = sum(kron(projector(s), projector(s)) for s in range(4))
    Hhop_a = kron(Gamma, I4)
    Hhop_b = kron(I4, Gamma)

    class PulseFull:
        def __init__(self, n, T, seed=11):
            self.n = int(n); self.T = float(T); self.dt = self.T/(2*self.n)
            rng = np.random.default_rng(seed)
            self.Va = rng.uniform(-1.0, 1.0, size=(self.n, 4))
            self.Vb = rng.uniform(-1.0, 1.0, size=(self.n, 4))
            self.U  = rng.uniform(0.2, 2.0, size=self.n)
            self.Ja = rng.uniform(0.0, 1.0, size=self.n)
            self.Jb = rng.uniform(0.0, 1.0, size=self.n)
            self.bounds = {'Va': (-3.0, 3.0), 'Vb': (-3.0, 3.0), 'U': (0.0, 6.0), 'Ja': (0.0, 5.0), 'Jb': (0.0, 5.0)}
            self.clip()
        def clip(self):
            self.Va = np.clip(self.Va, *self.bounds['Va'])
            self.Vb = np.clip(self.Vb, *self.bounds['Vb'])
            self.U  = np.clip(self.U,  *self.bounds['U'])
            self.Ja = np.clip(self.Ja, *self.bounds['Ja'])
            self.Jb = np.clip(self.Jb, *self.bounds['Jb'])
        def H_step(self, m):
            if m % 2 == 0:
                k = m // 2
                H = np.zeros((16,16), dtype=complex)
                for s in range(4): H += self.Va[k,s] * P_a_sites[s]
                for t in range(4): H += self.Vb[k,t] * P_b_sites[t]
                H += self.U[k] * Hint_base
                return H
            else:
                k = m // 2
                return self.Ja[k]*Hhop_a + self.Jb[k]*Hhop_b
        def unitaries(self):
            return [expm(-1j * self.H_step(m) * self.dt) for m in range(2*self.n)]

    class GRAPEAdam:
        def __init__(self, pulse, lr=0.08, beta1=0.9, beta2=0.999, eps=1e-8, l2=1e-5):
            self.p = pulse; self.lr=lr; self.beta1=beta1; self.beta2=beta2; self.eps=eps; self.l2=l2
            self.m = { 'Va': np.zeros_like(self.p.Va), 'Vb': np.zeros_like(self.p.Vb),
                       'U': np.zeros_like(self.p.U), 'Ja': np.zeros_like(self.p.Ja), 'Jb': np.zeros_like(self.p.Jb) }
            self.v = { k: np.zeros_like(v) for k,v in self.m.items() }
            self.t = 0
        def forward(self, psi0):
            Us = self.p.unitaries()
            psis=[psi0]
            for U in Us: psis.append(U @ psis[-1])
            return psis, Us
        def gradients(self, psi0, psi_target):
            n=self.p.n; dt=self.p.dt
            psis,Us=self.forward(psi0)
            psiT=psis[-1]; overlap=np.vdot(psi_target, psiT)
            F=float(np.real(overlap*overlap.conjugate()))
            chis=[None]*(2*n+1); chis[-1]=psi_target*overlap.conjugate()
            for m in range(2*n-1,-1,-1): chis[m]=Us[m].conj().T @ chis[m+1]
            gVa=np.zeros_like(self.p.Va); gVb=np.zeros_like(self.p.Vb)
            gU=np.zeros_like(self.p.U); gJa=np.zeros_like(self.p.Ja); gJb=np.zeros_like(self.p.Jb)
            for m in range(2*n):
                k=m//2; psi_m1=psis[m+1]; chi_mp1=chis[m+1]; fac=overlap.conjugate()
                if m%2==0:
                    for s in range(4):
                        cont=-1j*dt*np.vdot(chi_mp1, P_a_sites[s] @ psi_m1); gVa[k,s]+=2.0*np.real(cont*fac)
                    for t in range(4):
                        cont=-1j*dt*np.vdot(chi_mp1, P_b_sites[t] @ psi_m1); gVb[k,t]+=2.0*np.real(cont*fac)
                    cont=-1j*dt*np.vdot(chi_mp1, Hint_base @ psi_m1); gU[k]+=2.0*np.real(cont*fac)
                else:
                    cont_a=-1j*dt*np.vdot(chi_mp1, Hhop_a @ psi_m1); gJa[k]+=2.0*np.real(cont_a*fac)
                    cont_b=-1j*dt*np.vdot(chi_mp1, Hhop_b @ psi_m1); gJb[k]+=2.0*np.real(cont_b*fac)
            if self.l2>0:
                gVa -= 2*self.l2*self.p.Va; gVb -= 2*self.l2*self.p.Vb
                gU  -= 2*self.l2*self.p.U;  gJa -= 2*self.l2*self.p.Ja; gJb -= 2*self.l2*self.p.Jb
            return F,overlap,{'Va':gVa,'Vb':gVb,'U':gU,'Ja':gJa,'Jb':gJb}
        def adam_step(self,key,g):
            self.t+=1
            m=self.m[key]; v=self.v[key]
            m[:]=self.beta1*m + (1-self.beta1)*g
            v[:]=self.beta2*v + (1-self.beta2)*(g*g)
            m_hat=m/(1-self.beta1**self.t)
            v_hat=v/(1-self.beta2**self.t)
            return self.lr*m_hat/(np.sqrt(v_hat)+self.eps)
        def optimize(self,psi0,psi_target,iters=360):
            hist=[]
            for it in range(iters):
                F,ov,grads=self.gradients(psi0,psi_target); hist.append(F)
                for key in grads:
                    step=self.adam_step(key,grads[key]); getattr(self.p,key)[:] += step
                self.p.clip()
            return np.array(hist)
        def fidelity_vs_time(self,psi0,psi_target):
            psis,_=self.forward(psi0)
            times=np.linspace(0.0,self.p.T,2*self.p.n+1)
            F_t=[float(np.abs(np.vdot(psi_target,psi))**2) for psi in psis]
            return times,np.array(F_t)

    def idx(s,t): return s*4+t
    psi0=np.zeros(16,dtype=complex); psi0[idx(0,1)]=1.0
    psi1=np.zeros(16,dtype=complex); psi1[idx(0,1)]=1/np.sqrt(2); psi1[idx(2,3)]=1/np.sqrt(2)

    T=2*np.pi; n=36
    pulse = PulseFull(n=n, T=T, seed=11)
    opt = GRAPEAdam(pulse, lr=0.08, l2=1e-5)
    opt.optimize(psi0, psi1, iters=360)
    global t_grid, F_t
    t_grid, F_t = opt.fidelity_vs_time(psi0, psi1)

ensure_optimized_schedule()

# Convert to plain Python lists with decent precision
np.set_printoptions(precision=16, suppress=False)

times = list(map(float, np.asarray(t_grid).ravel()))
fidelities = list(map(float, np.asarray(F_t).ravel()))
U_k = list(map(float, np.asarray(pulse.U).ravel()))
J_k_a = list(map(float, np.asarray(pulse.Ja).ravel()))
J_k_b = list(map(float, np.asarray(pulse.Jb).ravel()))

print("times = ", times)
print("fidelities = ", fidelities)
print("U_k = ", U_k)
print("J_k_a = ", J_k_a)
print("J_k_b = ", J_k_b)



# --- Extract arrays from your optimization run ---
# Assumes variables exist: pulse (with .U and .Ja), t_grid, and F_t
U = np.asarray(pulse.U)*100    # even-step interaction strengths U_k
Ja = np.asarray(pulse.Ja)*100  # odd-step hoppings J_k^a
Jb = np.asarray(pulse.Jb)*100   # odd-step hoppings J_k^a
t = np.asarray(t_grid)/(2*np.pi)*10      # time grid at step boundaries
F = np.asarray(F_t)         # fidelities at those times
V_a=np.asarray(pulse.Va)*100
V_b=np.asarray(pulse.Vb)*100

# --- 1) Fidelity vs time after optimization ---
plt.figure()
plt.plot(t, F, marker='o')
plt.xlabel('time')
plt.ylabel('fidelity with $\\psi_1$')
plt.title('Fidelity vs time after optimization')
plt.tight_layout()
plt.show()

# --- 2) U_k vs k (even steps) ---
k_even = np.arange(U.size)
plt.figure()
plt.plot(k_even, U, marker='o')
plt.plot(k_even, V_a[:, 1]-V_a[:, 0])
plt.xlabel('k (even-step index)')
plt.ylabel('$U_k$')
plt.title('$U_k$ vs $k$')
plt.tight_layout()
plt.show()

# --- 3) J_k^a vs k (odd steps) ---
k_odd = np.arange(Ja.size)
plt.figure()
plt.plot(k_odd, Ja, marker='o')
plt.plot(k_odd, Jb, marker='o')

plt.xlabel('k (odd-step index)')
plt.ylabel('$J_k^a$')
plt.title('$J_k^a$ vs $k$')
plt.tight_layout()
plt.show()


V1a=V_a[:, 1]-V_a[:, 0]
V2a=V_a[:, 2]-V_a[:, 0]
V3a=V_a[:, 3]-V_a[:, 0]

plt.rcParams['font.size'] = 14          # base size for all text
plt.rcParams['axes.titlesize'] = 14     # title of each subplot
plt.rcParams['axes.labelsize'] = 14     # x/y labels
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 16   # suptitle


plt.figure(300)
plt.plot(t, F, '.-k')
plt.xlabel('time')
plt.ylabel('fidelity with SPDC-like state')
plt.title('Fidelity vs time after optimization')
plt.tight_layout()

plt.savefig('Fig1.png')

plt.figure(100)
for i in range(len(t)-1):
    if i%2 ==0:
        plt.plot([t[i], t[i], t[i+1], t[i+1]], [0, U[i//2], U[i//2], 0], 'k')
        plt.plot([t[i], t[i], t[i+1], t[i+1]], [0, V1a[i//2], V1a[i//2], 0], 'b')
        plt.plot([t[i], t[i], t[i+1], t[i+1]], [0, V2a[i//2], V2a[i//2], 0], 'g')
        plt.plot([t[i], t[i], t[i+1], t[i+1]], [0, V3a[i//2], V3a[i//2], 0], 'c')
    if i%2==1:
        plt.plot([t[i], t[i+1]], [0,  0], 'k')
        plt.plot([t[i], t[i+1]], [0,  0], 'b')
plt.plot([0], [0], 'b', label=r'$V_{12}$')
plt.plot([0], [0], 'g', label=r'$V_{21}$')
plt.plot([0], [0], 'c', label=r'$V_{22}$')
plt.plot([0], [0], 'k', label=r'$U$')      
plt.xlabel('Time (ms)')
plt.ylabel('Parameter (Hz)')
plt.legend(ncol=2)
plt.savefig('Fig2.png')
        
        
plt.figure(200)
for i in range(len(t)-1):
    if i%2 ==1:
        plt.plot([t[i], t[i], t[i+1], t[i+1]], [0, Ja[i//2], Ja[i//2], 0], 'k')
        plt.plot([t[i], t[i], t[i+1], t[i+1]], [0, Jb[i//2], Jb[i//2], 0], 'b')
    if i%2==0:
        plt.plot([t[i], t[i+1]], [0,  0], 'k')
        plt.plot([t[i], t[i+1]], [0,  0], 'b')
plt.plot([0], [0], 'b', label=r'$J^{a}$')
plt.plot([0], [0], 'k', label=r'$J^{d}$')  
plt.xlabel('Time (ms)')
plt.ylabel('Parameter (Hz)')
plt.legend()
plt.savefig('Fig3.png')