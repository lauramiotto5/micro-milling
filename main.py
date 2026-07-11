"""
Otimização de Parâmetros de Microusinagem
==========================================
Análise estatística e otimização multi-objetivo para microfresamento.

Fator: Avanço por dente (fz) - 4 níveis: 5, 10, 15, 20 µm/dente
Respostas: Área de rebarba, Ra, Rq, Rz, Rt, Rsk, Rku

Métodos:
- Regressão polinomial (linear e quadrática)
- ANOVA
- Otimização por função desirabilidade
- Análise gráfica com intervalos de confiança
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. DADOS EXPERIMENTAIS
# =============================================================================
# Parâmetros fixos
PROFUNDIDADE_CORTE = 20  # µm
ROTACAO = 20000  # rpm
VELOCIDADE_CORTE = 25.13  # m/min

# Fator variável: Avanço por dente (µm/dente)
fz = np.array([5, 10, 15, 20])

# -----------------------------------------------------------------------
# INSIRA SEUS DADOS EXPERIMENTAIS ABAIXO
# Cada condição deve ter 3 repetições (ou ajuste conforme seus dados)
# -----------------------------------------------------------------------

# Área de rebarba (µm²) - 3 medições por condição (início, meio, fim do canal)
# Formato: cada linha = uma condição, cada coluna = uma repetição
rebarba = np.array([
    [None, None, None],  # Condição 1: fz = 5 µm/dente
    [None, None, None],  # Condição 2: fz = 10 µm/dente
    [None, None, None],  # Condição 3: fz = 15 µm/dente
    [None, None, None],  # Condição 4: fz = 20 µm/dente
], dtype=float)

# Rugosidade Ra (µm) - 3 medições por condição
ra = np.array([
    [None, None, None],  # Condição 1: fz = 5
    [None, None, None],  # Condição 2: fz = 10
    [None, None, None],  # Condição 3: fz = 15
    [None, None, None],  # Condição 4: fz = 20
], dtype=float)

# Rugosidade Rq (µm) - 3 medições por condição
rq = np.array([
    [None, None, None],
    [None, None, None],
    [None, None, None],
    [None, None, None],
], dtype=float)

# Rugosidade Rz (µm) - 3 medições por condição
rz = np.array([
    [None, None, None],
    [None, None, None],
    [None, None, None],
    [None, None, None],
], dtype=float)

# Rugosidade Rt (µm) - 3 medições por condição
rt = np.array([
    [None, None, None],
    [None, None, None],
    [None, None, None],
    [None, None, None],
], dtype=float)

# Rugosidade Rsk - 3 medições por condição
rsk = np.array([
    [None, None, None],
    [None, None, None],
    [None, None, None],
    [None, None, None],
], dtype=float)

# Rugosidade Rku - 3 medições por condição
rku = np.array([
    [None, None, None],
    [None, None, None],
    [None, None, None],
    [None, None, None],
], dtype=float)

# =============================================================================
# EXEMPLO COM DADOS FICTÍCIOS (remova quando inserir dados reais)
# =============================================================================
USAR_DADOS_EXEMPLO = True

if USAR_DADOS_EXEMPLO:
    print("=" * 60)
    print("ATENÇÃO: Usando dados fictícios de exemplo!")
    print("Substitua pelos seus dados reais e defina USAR_DADOS_EXEMPLO = False")
    print("=" * 60, "\n")

    rebarba = np.array([
        [1200, 1350, 1280],
        [2100, 2250, 2050],
        [3500, 3200, 3400],
        [5200, 4900, 5100],
    ], dtype=float)

    ra = np.array([
        [0.35, 0.38, 0.36],
        [0.42, 0.45, 0.40],
        [0.55, 0.58, 0.52],
        [0.72, 0.68, 0.75],
    ], dtype=float)

    rq = np.array([
        [0.44, 0.47, 0.45],
        [0.53, 0.56, 0.50],
        [0.69, 0.72, 0.65],
        [0.90, 0.85, 0.94],
    ], dtype=float)

    rz = np.array([
        [2.1, 2.3, 2.2],
        [2.8, 3.0, 2.7],
        [3.6, 3.9, 3.4],
        [4.8, 4.5, 5.0],
    ], dtype=float)

    rt = np.array([
        [2.8, 3.0, 2.9],
        [3.5, 3.8, 3.4],
        [4.5, 4.8, 4.3],
        [6.0, 5.7, 6.3],
    ], dtype=float)

    rsk = np.array([
        [-0.2, -0.15, -0.18],
        [0.05, 0.10, 0.02],
        [0.25, 0.30, 0.20],
        [0.45, 0.50, 0.42],
    ], dtype=float)

    rku = np.array([
        [2.8, 2.9, 2.85],
        [3.0, 3.1, 2.95],
        [3.2, 3.4, 3.1],
        [3.5, 3.7, 3.3],
    ], dtype=float)


# =============================================================================
# 2. FUNÇÕES AUXILIARES
# =============================================================================

def verificar_dados(dados, nome):
    """Verifica se os dados foram preenchidos."""
    if np.any(np.isnan(dados)):
        print(f"AVISO: Dados de '{nome}' contêm valores NaN. Verifique o preenchimento.")
        return False
    return True


def regressao_polinomial(x, y_medio, y_std, grau=2):
    """
    Ajusta regressão polinomial e retorna coeficientes, R², e modelo.
    """
    coefs = np.polyfit(x, y_medio, grau)
    modelo = np.poly1d(coefs)
    y_pred = modelo(x)

    # R²
    ss_res = np.sum((y_medio - y_pred) ** 2)
    ss_tot = np.sum((y_medio - np.mean(y_medio)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # R² ajustado
    n = len(x)
    p = grau  # número de parâmetros (excluindo intercepto)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r2

    return coefs, modelo, r2, r2_adj


def anova_one_way(dados):
    """
    Realiza ANOVA one-way para verificar efeito do avanço.
    dados: array (n_condicoes x n_repeticoes)
    """
    grupos = [dados[i, :] for i in range(dados.shape[0])]
    f_stat, p_valor = stats.f_oneway(*grupos)
    return f_stat, p_valor


def teste_normalidade(dados):
    """Teste de Shapiro-Wilk para normalidade dos resíduos."""
    todos = dados.flatten()
    if len(todos) < 3:
        return None, None
    stat, p_valor = stats.shapiro(todos)
    return stat, p_valor


def desirabilidade_individual(y, y_min, y_max, objetivo="minimizar", peso=1):
    """
    Calcula desirabilidade individual (Derringer & Suich, 1980).
    objetivo: 'minimizar', 'maximizar', 'alvo'
    """
    if objetivo == "minimizar":
        if y <= y_min:
            d = 1.0
        elif y >= y_max:
            d = 0.0
        else:
            d = ((y_max - y) / (y_max - y_min)) ** peso
    elif objetivo == "maximizar":
        if y <= y_min:
            d = 0.0
        elif y >= y_max:
            d = 1.0
        else:
            d = ((y - y_min) / (y_max - y_min)) ** peso
    else:  # alvo
        y_alvo = (y_min + y_max) / 2
        if y < y_min or y > y_max:
            d = 0.0
        elif y <= y_alvo:
            d = ((y - y_min) / (y_alvo - y_min)) ** peso
        else:
            d = ((y_max - y) / (y_max - y_alvo)) ** peso
    return d


# =============================================================================
# 3. ANÁLISE ESTATÍSTICA
# =============================================================================

def executar_analise():
    """Executa toda a análise estatística e otimização."""

    print("\n" + "=" * 60)
    print("   ANÁLISE ESTATÍSTICA - MICROUSINAGEM")
    print("=" * 60)

    # Dicionário com todas as respostas
    respostas = {
        "Rebarba (µm²)": rebarba,
        "Ra (µm)": ra,
        "Rq (µm)": rq,
        "Rz (µm)": rz,
        "Rt (µm)": rt,
        "Rsk": rsk,
        "Rku": rku,
    }

    # Verificar dados
    dados_validos = True
    for nome, dados in respostas.items():
        if not verificar_dados(dados, nome):
            dados_validos = False

    if not dados_validos:
        print("\nERRO: Corrija os dados antes de prosseguir.")
        return

    # --- Estatísticas descritivas ---
    print("\n" + "-" * 60)
    print("3.1 ESTATÍSTICAS DESCRITIVAS")
    print("-" * 60)

    medias = {}
    desvios = {}

    for nome, dados in respostas.items():
        media = np.mean(dados, axis=1)
        desvio = np.std(dados, axis=1, ddof=1)
        medias[nome] = media
        desvios[nome] = desvio

        print(f"\n{nome}:")
        print(f"  {'fz (µm/dente)':<15} {'Média':<12} {'Desvio Padrão':<15} {'CV (%)':<10}")
        for i, f in enumerate(fz):
            cv = (desvio[i] / media[i] * 100) if media[i] != 0 else 0
            print(f"  {f:<15} {media[i]:<12.4f} {desvio[i]:<15.4f} {cv:<10.2f}")

    # --- ANOVA ---
    print("\n" + "-" * 60)
    print("3.2 ANOVA ONE-WAY (α = 0.05)")
    print("-" * 60)
    print(f"\n  {'Resposta':<20} {'F-estatística':<15} {'p-valor':<12} {'Significativo?'}")
    print(f"  {'-'*60}")

    for nome, dados in respostas.items():
        f_stat, p_valor = anova_one_way(dados)
        sig = "SIM ***" if p_valor < 0.001 else "SIM **" if p_valor < 0.01 else "SIM *" if p_valor < 0.05 else "NÃO"
        print(f"  {nome:<20} {f_stat:<15.4f} {p_valor:<12.6f} {sig}")

    # --- Teste de Normalidade ---
    print("\n" + "-" * 60)
    print("3.3 TESTE DE NORMALIDADE (Shapiro-Wilk, α = 0.05)")
    print("-" * 60)
    print(f"\n  {'Resposta':<20} {'W-estatística':<15} {'p-valor':<12} {'Normal?'}")
    print(f"  {'-'*50}")

    for nome, dados in respostas.items():
        stat, p_valor = teste_normalidade(dados)
        if stat is not None:
            normal = "SIM" if p_valor > 0.05 else "NÃO"
            print(f"  {nome:<20} {stat:<15.4f} {p_valor:<12.6f} {normal}")

    # --- Regressão Polinomial ---
    print("\n" + "-" * 60)
    print("3.4 REGRESSÃO POLINOMIAL")
    print("-" * 60)

    modelos = {}
    melhor_grau = {}

    for nome in respostas.keys():
        media = medias[nome]
        desvio = desvios[nome]

        print(f"\n{nome}:")

        melhor_r2_adj = -np.inf
        melhor_g = 1

        for grau in [1, 2]:  # Linear e Quadrático (máx. grau 2 com 4 pontos)
            coefs, modelo, r2, r2_adj = regressao_polinomial(fz, media, desvio, grau)
            print(f"  Grau {grau}: R² = {r2:.4f}, R²_adj = {r2_adj:.4f}")

            if grau == 1:
                print(f"    Equação: y = {coefs[0]:.4f}·fz + {coefs[1]:.4f}")
            elif grau == 2:
                print(f"    Equação: y = {coefs[0]:.4f}·fz² + {coefs[1]:.4f}·fz + {coefs[2]:.4f}")

            if r2_adj > melhor_r2_adj:
                melhor_r2_adj = r2_adj
                melhor_g = grau
                modelos[nome] = modelo
                melhor_grau[nome] = grau

        print(f"  → Melhor modelo: grau {melhor_g} (R²_adj = {melhor_r2_adj:.4f})")

    # --- Otimização por Desirabilidade ---
    print("\n" + "-" * 60)
    print("3.5 OTIMIZAÇÃO MULTI-OBJETIVO (Função Desirabilidade)")
    print("-" * 60)
    print("\nObjetivo: Minimizar rebarba E rugosidade simultaneamente")
    print("Método: Derringer & Suich (1980)\n")

    # Respostas a minimizar (todas exceto Rsk que pode ter comportamento diferente)
    respostas_otim = ["Rebarba (µm²)", "Ra (µm)", "Rq (µm)", "Rz (µm)", "Rt (µm)"]

    # Pesos (importância relativa - ajuste conforme necessário)
    pesos = {
        "Rebarba (µm²)": 1.0,
        "Ra (µm)": 1.0,
        "Rq (µm)": 0.8,
        "Rz (µm)": 0.8,
        "Rt (µm)": 0.6,
    }

    print("  Pesos das respostas:")
    for resp, peso in pesos.items():
        print(f"    {resp}: {peso}")
    print("\n  (Ajuste os pesos conforme a importância relativa de cada resposta)\n")

    # Limites para desirabilidade
    limites = {}
    for nome in respostas_otim:
        media = medias[nome]
        limites[nome] = (np.min(media), np.max(media))

    def desirabilidade_global(fz_val):
        """Calcula desirabilidade global para um dado fz."""
        d_individuais = []
        for nome in respostas_otim:
            y_pred = modelos[nome](fz_val)
            y_min, y_max = limites[nome]
            d = desirabilidade_individual(y_pred, y_min, y_max, "minimizar", pesos[nome])
            d_individuais.append(d)

        # Desirabilidade global (média geométrica)
        d_global = np.prod(d_individuais) ** (1.0 / len(d_individuais))
        return d_global

    # Busca do ótimo no intervalo de fz
    fz_range = np.linspace(fz.min(), fz.max(), 1000)
    d_global_values = np.array([desirabilidade_global(f) for f in fz_range])

    # Encontrar máximo da desirabilidade
    idx_otimo = np.argmax(d_global_values)
    fz_otimo = fz_range[idx_otimo]
    d_otimo = d_global_values[idx_otimo]

    print(f"  RESULTADO DA OTIMIZAÇÃO:")
    print(f"  {'─' * 40}")
    print(f"  Avanço ótimo (fz):     {fz_otimo:.2f} µm/dente")
    print(f"  Desirabilidade global: {d_otimo:.4f}")
    print(f"\n  Valores preditos no ponto ótimo:")
    for nome in respostas_otim:
        y_pred = modelos[nome](fz_otimo)
        print(f"    {nome}: {y_pred:.4f}")

    # =============================================================================
    # 4. GRÁFICOS
    # =============================================================================

    print("\n" + "-" * 60)
    print("3.6 GERANDO GRÁFICOS...")
    print("-" * 60)

    fz_plot = np.linspace(fz.min() - 1, fz.max() + 1, 200)

    # --- Gráfico 1: Respostas individuais com regressão ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    respostas_plot = ["Rebarba (µm²)", "Ra (µm)", "Rq (µm)", "Rz (µm)", "Rt (µm)", "Rsk"]

    for idx, nome in enumerate(respostas_plot):
        if idx >= len(axes):
            break
        ax = axes[idx]
        media = medias[nome]
        desvio = desvios[nome]

        # Pontos experimentais com barras de erro
        ax.errorbar(fz, media, yerr=desvio, fmt='ko', capsize=4,
                    markersize=6, label='Dados experimentais')

        # Curva de regressão
        if nome in modelos:
            y_fit = modelos[nome](fz_plot)
            grau = melhor_grau[nome]
            ax.plot(fz_plot, y_fit, 'b-', linewidth=1.5,
                    label=f'Regressão (grau {grau})')

            # Intervalo de confiança (aproximado)
            n = len(fz)
            y_pred_pts = modelos[nome](fz)
            residuos = media - y_pred_pts
            se = np.sqrt(np.sum(residuos**2) / (n - grau - 1)) if (n - grau - 1) > 0 else 0
            t_crit = stats.t.ppf(0.975, n - grau - 1) if (n - grau - 1) > 0 else 2
            ax.fill_between(fz_plot, y_fit - t_crit * se, y_fit + t_crit * se,
                            alpha=0.15, color='blue', label='IC 95%')

        # Linha vertical no ótimo
        if nome in respostas_otim:
            ax.axvline(fz_otimo, color='red', linestyle='--', alpha=0.7, label=f'Ótimo ({fz_otimo:.1f})')

        ax.set_xlabel('Avanço por dente (µm/dente)')
        ax.set_ylabel(nome)
        ax.set_title(nome)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets\\graficos_respostas.png', dpi=150, bbox_inches='tight')
    print("  → Salvo: graficos_respostas.png")

    # --- Gráfico 2: Desirabilidade ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(fz_range, d_global_values, 'b-', linewidth=2, label='Desirabilidade Global')
    ax.axvline(fz_otimo, color='red', linestyle='--', linewidth=1.5,
               label=f'Ótimo: fz = {fz_otimo:.2f} µm/dente')
    ax.scatter([fz_otimo], [d_otimo], color='red', s=100, zorder=5)
    ax.set_xlabel('Avanço por dente (µm/dente)', fontsize=12)
    ax.set_ylabel('Desirabilidade Global (D)', fontsize=12)
    ax.set_title('Otimização por Função Desirabilidade', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('assets\\desirabilidade.png', dpi=150, bbox_inches='tight')
    print("  → Salvo: desirabilidade.png")

    # --- Gráfico 3: Box plots ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, nome in enumerate(respostas_plot):
        if idx >= len(axes):
            break
        ax = axes[idx]
        dados = respostas[nome]
        bp = ax.boxplot([dados[i, :] for i in range(dados.shape[0])],
                        tick_labels=[str(f) for f in fz], patch_artist=True)
        cores = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
        for patch, cor in zip(bp['boxes'], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        ax.set_xlabel('Avanço por dente (µm/dente)')
        ax.set_ylabel(nome)
        ax.set_title(f'Box Plot - {nome}')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('assets\\boxplots.png', dpi=150, bbox_inches='tight')
    print("  → Salvo: boxplots.png")

    plt.show()

    # =============================================================================
    # 5. RESUMO E RECOMENDAÇÕES
    # =============================================================================
    print("\n" + "=" * 60)
    print("   RESUMO E RECOMENDAÇÕES")
    print("=" * 60)
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │ PARÂMETRO ÓTIMO ENCONTRADO                              │
  │                                                         │
  │   Avanço por dente (fz): {fz_otimo:.2f} µm/dente{' ' * (20 - len(f'{fz_otimo:.2f}'))}│
  │   Profundidade de corte:  {PROFUNDIDADE_CORTE} µm (fixo)              │
  │   Rotação:                {ROTACAO} rpm (fixo)            │
  │   Velocidade de corte:    {VELOCIDADE_CORTE} m/min (fixo)         │
  │                                                         │
  │   Desirabilidade global:  {d_otimo:.4f}                       │
  └─────────────────────────────────────────────────────────┘

  RECOMENDAÇÕES:
  1. Os resultados são válidos apenas dentro do intervalo testado
     (fz = {fz.min()} a {fz.max()} µm/dente).
  2. Para maior confiabilidade, considere aumentar o número de
     repetições (mínimo 5 por condição).
  3. Para um estudo mais completo, sugere-se expandir o DOE para
     incluir profundidade de corte e/ou velocidade como fatores.
  4. Realize ensaio de confirmação no ponto ótimo encontrado.
    """)


# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == "__main__":
    executar_analise()


