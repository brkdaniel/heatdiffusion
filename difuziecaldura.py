import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.path import Path
from scipy.interpolate import griddata

def k_func(x, y):
    return 1 + 0.5 * x**2 + 0.2 * y

def u_exact(x, y):
    return -np.exp(x) - (5/6) * y**3

def u_boundary(x, y):
    return u_exact(x, y)

def f(x, y):
    return np.exp(x) + 5 * y

def conditie_pe_frontiera(indice_muchie):
    return "Dirichlet" if indice_muchie % 2 == 0 else "Neumann"

def gN(x, y, muchie, puncte_rezistor):
    if conditie_pe_frontiera(muchie) != "Neumann":
        return 0.0
    v1 = puncte_rezistor[muchie]
    v2 = puncte_rezistor[(muchie + 1) % len(puncte_rezistor)]
    tangent = v2 - v1
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    du_dx = -np.exp(x)
    du_dy = -0.5 * y**2
    grad_u = np.array([du_dx, du_dy])
    return k_func(x, y) * np.dot(grad_u, normal)

N = 100
x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1, 1, N)
xx, yy = np.meshgrid(x, y)
h = x[1] - x[0]

puncte_rezistor = np.array([
    [-1.2, -0.2], [-1.0, -0.2], [-1.0, -0.5], [-0.5, -0.5], [-0.5, 0.5],
    [0.5, 0.5], [0.5, -0.5], [1.0, -0.5], [1.0, -0.2], [1.2, -0.2],
    [1.2, 0.2], [1.0, 0.2], [1.0, 0.5], [0.5, 0.5], [0.5, -0.5],
    [-0.5, -0.5], [-0.5, 0.5], [-1.0, 0.5], [-1.0, 0.2], [-1.2, 0.2],
    [-1.2, -0.2]
])
num_muchii = len(puncte_rezistor) - 1
poligon = Path(puncte_rezistor)
masca = poligon.contains_points(np.vstack([xx.ravel(), yy.ravel()]).T).reshape(N, N)

indexare = -np.ones_like(masca, dtype=int)
idx_curent = 0
for i in range(N):
    for j in range(N):
        if masca[i, j]:
            indexare[i, j] = idx_curent
            idx_curent += 1
numar_unknowns = idx_curent

frontiera = []
for i in range(N):
    for j in range(N):
        if not masca[i, j]:
            vecini = [(i+di, j+dj) for (di, dj) in [(-1,0), (1,0), (0,-1), (0,1)]
                      if 0 <= i+di < N and 0 <= j+dj < N]
            interior_langa = any(masca[ni, nj] for (ni, nj) in vecini)
            if interior_langa:
                x_p, y_p = xx[i, j], yy[i, j]
                dist_min = 1e10
                muchie_cea_mai_apropiata = 0
                for k in range(num_muchii):
                    v1 = puncte_rezistor[k]
                    v2 = puncte_rezistor[(k + 1) % len(puncte_rezistor)]
                    v = v2 - v1
                    w = np.array([x_p, y_p]) - v1
                    t = np.dot(w, v) / np.dot(v, v)
                    t = np.clip(t, 0, 1)
                    proj = v1 + t * v
                    d = np.linalg.norm(np.array([x_p, y_p]) - proj)
                    if d < dist_min:
                        dist_min = d
                        muchie_cea_mai_apropiata = k
                frontiera.append((i, j, muchie_cea_mai_apropiata))

A = lil_matrix((numar_unknowns, numar_unknowns))
b = np.zeros(numar_unknowns)

for i in range(N):
    for j in range(N):
        idx = indexare[i, j]
        if idx == -1:
            continue
        diag_coef = 0.0
        b_val = f(xx[i, j], yy[i, j])

        for (di, dj) in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N:
                idx_n = indexare[ni, nj]
                k_ij = k_func(xx[i,j], yy[i,j])
                k_nij = k_func(xx[ni,nj], yy[ni,nj])
                kmed = 0.5 * (k_ij + k_nij)
                if idx_n != -1:
                    A[idx, idx_n] = kmed / h**2
                    diag_coef += kmed / h**2
                else:
                    muchie = next((km for (fi, fj, km) in frontiera if fi == ni and fj == nj), None)
                    if muchie is not None:
                        tip = conditie_pe_frontiera(muchie)
                        if tip == "Dirichlet":
                            b_val -= kmed * u_boundary(xx[ni,nj], yy[ni,nj]) / h**2
                            diag_coef += kmed / h**2
                        else:
                            b_val += gN(xx[ni,nj], yy[ni,nj], muchie, puncte_rezistor) / h

        A[idx, idx] = -diag_coef
        b[idx] = b_val

u_sol = spsolve(A.tocsr(), b)

u_grid = np.full_like(xx, np.nan)
for i in range(N):
    for j in range(N):
        idx = indexare[i, j]
        if idx != -1:
            u_grid[i, j] = u_sol[idx]

u_ex_grid = np.where(masca, u_exact(xx, yy), np.nan)

puncte_cunoscute = [(xx[i,j], yy[i,j]) for i in range(N) for j in range(N) if indexare[i,j] != -1]
valori_cunoscute = [u_sol[indexare[i,j]] for i in range(N) for j in range(N) if indexare[i,j] != -1]

puncte_cunoscute = np.array(puncte_cunoscute)
valori_cunoscute = np.array(valori_cunoscute)

N_fina = 300
x_fina = np.linspace(x.min(), x.max(), N_fina)
y_fina = np.linspace(y.min(), y.max(), N_fina)
xx_fina, yy_fina = np.meshgrid(x_fina, y_fina)
u_interp = griddata(puncte_cunoscute, valori_cunoscute, (xx_fina, yy_fina), method='cubic')
masca_fina = poligon.contains_points(np.vstack([xx_fina.ravel(), yy_fina.ravel()]).T).reshape(N_fina, N_fina)
u_interp_masked = np.where(masca_fina, u_interp, np.nan)


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.contourf(xx_fina, yy_fina, u_interp_masked, levels=50, cmap='cividis')
plt.colorbar()
plt.title("Solutie numerica interpolata")
plt.gca().set_aspect('equal')


plt.subplot(1, 2, 2)
plt.contourf(xx, yy, u_ex_grid, levels=50, cmap='cividis')
plt.colorbar()
plt.title("Solutie exacta")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()


eroare = np.abs(u_grid - u_ex_grid)
plt.figure(figsize=(7,6))
plt.contourf(xx, yy, eroare, levels=50, cmap='viridis')
plt.colorbar(label='Eroare')
plt.title('Eroare absoluta')
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()


valori_N = [20, 40, 60, 80, 100, 120]
pasuri = []
eroari = []
for N in valori_N:
    x = np.linspace(-1.5, 1.5, N)
    y = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, y)
    h = x[1] - x[0]
    pasuri.append(h)

    poligon = Path(puncte_rezistor)
    masca = poligon.contains_points(np.vstack([xx.ravel(), yy.ravel()]).T).reshape(N, N)
    indexare = -np.ones_like(masca, dtype=int)
    idx_curent = 0
    for i in range(N):
        for j in range(N):
            if masca[i,j]:
                indexare[i,j] = idx_curent
                idx_curent += 1
    numar_unknowns = idx_curent

    A = lil_matrix((numar_unknowns, numar_unknowns))
    b = np.zeros(numar_unknowns)

    for i in range(N):
        for j in range(N):
            idx = indexare[i, j]
            if idx == -1:
                continue
            diag_coef = 0.0
            b_val = f(xx[i, j], yy[i, j])
            for (di, dj) in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    idx_n = indexare[ni, nj]
                    k_ij = k_func(xx[i,j], yy[i,j])
                    k_nij = k_func(xx[ni,nj], yy[ni,nj])
                    kmed = 0.5 * (k_ij + k_nij)
                    if idx_n != -1:
                        A[idx, idx_n] = kmed / h**2
                        diag_coef += kmed / h**2
                    else:
                        diag_coef += kmed / h**2
                        b_val -= kmed * u_boundary(xx[ni,nj], yy[ni,nj]) / h**2
            A[idx, idx] = -diag_coef
            b[idx] = b_val

    u_sol = spsolve(A.tocsr(), b)
    suma_eroare = 0.0
    suma_ex = 0.0
    for i in range(N):
        for j in range(N):
            if masca[i,j]:
                idx = indexare[i, j]
                u_ex = u_exact(xx[i,j], yy[i,j])
                suma_eroare += (u_sol[idx] - u_ex)**2
                suma_ex += u_ex**2
    eroare_rel = np.sqrt(suma_eroare / suma_ex)
    eroari.append(eroare_rel)

plt.figure(figsize=(7,5))
plt.loglog(pasuri, eroari, 'o-', label='Eroare L2')
plt.xlabel("Pas h")
plt.ylabel("Eroare relativă L2")
plt.title("Convergenta solutiei numerice")
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()