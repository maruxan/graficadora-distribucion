# Importando modulos necesarios
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

np.random.seed(2016)  # replicar random

# parametros esteticos de seaborn
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})


# Definimos funciones

def binomial(N, p): # función para graficar distribución binomial
    binomial = stats.binom(N, p)  # Distribución
    x = np.arange(binomial.ppf(0.01),
                  binomial.ppf(0.99))
    fmp = binomial.pmf(x)  # Función de Masa de Probabilidad
    plt.plot(x, fmp, '--')
    plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
    plt.title('Distribución Binomial')
    plt.ylabel('probabilidad')
    plt.xlabel('valores')
    plt.show()


def poisson(mu): # función para graficar distribución de poisson
    poisson = stats.poisson(mu)  # Distribución
    x = np.arange(poisson.ppf(0.01),
                  poisson.ppf(0.99))
    fmp = poisson.pmf(x)  # Función de Masa de Probabilidad
    plt.plot(x, fmp, '--')
    plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
    plt.title('Distribución Poisson')
    plt.ylabel('probabilidad')
    plt.xlabel('valores')
    plt.show()


def hipergeometrica(M, n, N): # función para graficar distribución hipergeométrica
    hipergeometrica = stats.hypergeom(M, n, N)  # Distribución
    x = np.arange(0, n + 1)
    fmp = hipergeometrica.pmf(x)  # Función de Masa de Probabilidad
    plt.plot(x, fmp, '--')
    plt.vlines(x, 0, fmp, colors='b', lw=5, alpha=0.5)
    plt.title('Distribución Hipergeométrica')
    plt.ylabel('probabilidad')
    plt.xlabel('valores')
    plt.show()


# Menu
print("Bienvenido a la graficadora de probabilidad")
print("Seleccione una operación:")
print("[1] Distribución Binomial")
print("[2] Distribución de Poisson")
print("[3] Distribución Hipergeométrica")
print("[4] Salir")
print("")

# Leemos opción
choice = input("Opción: ")
print("")

# Operamos según opción
if choice == '1':
    print("Distribución Binomial")

    # Leemos los parámetros
    N = int(input("Ingrese N: "))
    p = float(input("Ingrese p: "))

    # LLamamos a la función
    binomial(N, p)

elif choice == '2':
    print("Distribución de Poisson")

    # Leemos los parámetros
    mu = float(input("Ingrese Lambda: "))

    # LLamamos a la función
    poisson(mu)

elif choice == '3':
    print("Distribución Hipergeométrica")

    # Leemos los parámetros
    N = int(input("Ingrese N: "))
    X = int(input("Ingrese X: "))
    n = int(input("Ingrese n: "))

    # LLamamos a la función
    hipergeometrica(N, X, n)

elif choice == '4':
    print("")
else:
    print("Opción inválida")


print("\nAdios.")
