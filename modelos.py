"""
Módulo de Modelos e Otimização para Microusinagem
==================================================
Contém: Regressão polinomial, ANOVA, análise de resíduos,
         NSGA-II e função desirabilidade.
"""

import numpy as np
from scipy import stats
from deap import base, creator, tools, algorithms
import random
import warnings

warnings.filterwarnings("ignore")

try:
    assign_crowding_dist = tools.assignCrowdingDist
except AttributeError:
    from deap.tools import emo

    assign_crowding_dist = emo.assignCrowdingDist


# =============================================================================
# REGRESSÃO POLINOMIAL
# =============================================================================

def ajustar_regressao(x, y, grau):
    """
    Ajusta regressão polinomial de dado grau.

    Parâmetros:
        x: array com os valores do fator (fz)
        y: array com os valores da resposta
        grau: grau do polinômio (1, 2 ou 3)

    Retorna:
        dict com coeficientes, R², R² ajustado, RMSE, equação e IC 95%
    """
    n = len(x)
    p = grau  # número de parâmetros (excluindo intercepto)

    # Ajuste
    coefs = np.polyfit(x, y, grau)
    modelo = np.poly1d(coefs)
    y_pred = modelo(x)

    # Resíduos
    residuos = y - y_pred
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    # R²
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # R² ajustado
    if (n - p - 1) > 0:
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        r2_adj = r2

    # RMSE
    rmse = np.sqrt(ss_res / (n - p - 1)) if (n - p - 1) > 0 else np.sqrt(ss_res / n)

    # Intervalo de confiança (aproximação)
    if (n - p - 1) > 0:
        t_crit = stats.t.ppf(0.975, n - p - 1)
        se = rmse  # erro padrão simplificado
        ic_95 = t_crit * se
    else:
        ic_95 = 0

    # Equação como string
    equacao_str = _formatar_equacao(coefs, grau)

    return {
        "coefs": coefs,
        "grau": grau,
        "r2": r2,
        "r2_adj": r2_adj,
        "rmse": rmse,
        "ic_95": ic_95,
        "equacao_str": equacao_str,
        "residuos": residuos,
        "y_pred": y_pred,
    }


def _formatar_equacao(coefs, grau):
    """Formata equação polinomial como string legível."""
    termos = []
    for i, c in enumerate(coefs):
        potencia = grau - i
        if potencia == 0:
            termos.append(f"{c:+.4f}")
        elif potencia == 1:
            termos.append(f"{c:+.4f}·fz")
        else:
            termos.append(f"{c:+.4f}·fz^{potencia}")

    eq = " ".join(termos)
    if eq.startswith("+"):
        eq = eq[1:]
    return f"y = {eq.strip()}"


# =============================================================================
# VALIDAÇÃO ESTATÍSTICA
# =============================================================================

def validar_modelo(x, y, coefs):
    """
    Valida o modelo com métricas estatísticas.

    Retorna:
        dict com R², RMSE, MAE, MAPE
    """
    y_pred = np.polyval(coefs, x)
    n = len(x)
    p = len(coefs) - 1

    residuos = y - y_pred
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / max(n - p - 1, 1))
    mae = np.mean(np.abs(residuos))
    mape = np.mean(np.abs(residuos / y[y != 0])) * 100 if np.any(y != 0) else 0

    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def analise_residuos(x, y, coefs):
    """
    Realiza análise completa de resíduos.

    Retorna:
        dict com resíduos, resíduos padronizados, Durbin-Watson, etc.
    """
    y_pred = np.polyval(coefs, x)
    residuos = y - y_pred

    # Resíduos padronizados
    std_res = np.std(residuos, ddof=1) if len(residuos) > 1 else 1
    residuos_padronizados = residuos / std_res if std_res > 0 else residuos

    # Durbin-Watson (autocorrelação)
    if len(residuos) > 1:
        diff_res = np.diff(residuos)
        durbin_watson = np.sum(diff_res**2) / np.sum(residuos**2) if np.sum(residuos**2) > 0 else 2.0
    else:
        durbin_watson = 2.0

    # Teste de Breusch-Pagan simplificado (heterocedasticidade)
    # Regredimos resíduos² contra x
    res_sq = residuos**2
    if len(x) > 2:
        slope, intercept, r_bp, p_bp, se_bp = stats.linregress(x, res_sq)
        heterocedasticidade = p_bp < 0.05
    else:
        p_bp = 1.0
        heterocedasticidade = False

    return {
        "residuos": residuos,
        "residuos_padronizados": residuos_padronizados,
        "y_pred": y_pred,
        "durbin_watson": durbin_watson,
        "p_heterocedasticidade": p_bp,
        "heterocedasticidade": heterocedasticidade,
    }


def anova_one_way(grupos):
    """
    Realiza ANOVA one-way.

    Parâmetros:
        grupos: lista de arrays, cada array contém as observações de um grupo

    Retorna:
        (F-estatística, p-valor)
    """
    f_stat, p_valor = stats.f_oneway(*grupos)
    return f_stat, p_valor


def teste_normalidade(dados):
    """
    Teste de Shapiro-Wilk para normalidade.

    Retorna:
        (W-estatística, p-valor)
    """
    if len(dados) < 3:
        return None, None
    w_stat, p_valor = stats.shapiro(dados)
    return w_stat, p_valor


# =============================================================================
# OTIMIZAÇÃO MULTIOBJETIVO - NSGA-II
# =============================================================================

def otimizacao_nsga2(modelos, objetivos, fz_min, fz_max, n_geracoes=200, n_populacao=100):
    """
    Otimização multiobjetivo usando NSGA-II.

    Parâmetros:
        modelos: dict {nome_resposta: coeficientes_polinomiais}
        objetivos: dict {nome_resposta: 'minimizar' ou 'maximizar'}
        fz_min: limite inferior do fz
        fz_max: limite superior do fz
        n_geracoes: número de gerações
        n_populacao: tamanho da população

    Retorna:
        dict com fronteira de Pareto, fz ótimos, e solução de compromisso
    """
    n_objetivos = len(modelos)
    respostas = list(modelos.keys())

    # Definir pesos para DEAP (-1 = minimizar, 1 = maximizar)
    pesos_deap = []
    for resp in respostas:
        if objetivos[resp] == "minimizar":
            pesos_deap.append(-1.0)
        else:
            pesos_deap.append(1.0)

    # Limpar classes anteriores se existirem
    if "FitnessMulti" in creator.__dict__:
        del creator.FitnessMulti
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMulti", base.Fitness, weights=tuple(pesos_deap))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_fz", random.uniform, fz_min, fz_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_fz, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def avaliar(individual):
        fz_val = individual[0]
        # Restringir ao domínio
        fz_val = max(fz_min, min(fz_max, fz_val))
        fitness_values = []
        for resp in respostas:
            y = np.polyval(modelos[resp], fz_val)
            fitness_values.append(y)
        return tuple(fitness_values)

    toolbox.register("evaluate", avaliar)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=(fz_max - fz_min) * 0.1, indpb=1.0)
    toolbox.register("select", tools.selNSGA2)

    # Executar NSGA-II
    random.seed(42)
    pop = toolbox.population(n=n_populacao)

    # Avaliar população inicial
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for gen in range(n_geracoes):
        # Atribuir crowding distance antes do torneio
        assign_crowding_dist(pop)
        # Seleção e reprodução
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutação
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                # Restringir ao domínio
                mutant[0] = max(fz_min, min(fz_max, mutant[0]))
                del mutant.fitness.values

        # Avaliar novos indivíduos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Seleção para próxima geração
        pop = toolbox.select(pop + offspring, n_populacao)

    # Extrair fronteira de Pareto
    pareto_front_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    pareto_front = np.array([ind.fitness.values for ind in pareto_front_inds])
    fz_otimos = np.array([ind[0] for ind in pareto_front_inds])

    # Solução de compromisso (TOPSIS simplificado)
    idx_compromisso = _topsis(pareto_front, pesos_deap)

    return {
        "pareto_front": pareto_front,
        "fz_otimos": fz_otimos,
        "idx_compromisso": idx_compromisso,
        "respostas": respostas,
    }


def _topsis(pareto_front, pesos):
    """
    Método TOPSIS simplificado para selecionar solução de compromisso.
    """
    if len(pareto_front) == 0:
        return 0

    # Normalizar
    norm = np.sqrt(np.sum(pareto_front**2, axis=0))
    norm[norm == 0] = 1
    normalizado = pareto_front / norm

    # Solução ideal e anti-ideal
    ideal = np.zeros(normalizado.shape[1])
    anti_ideal = np.zeros(normalizado.shape[1])

    for j in range(normalizado.shape[1]):
        if pesos[j] < 0:  # minimizar
            ideal[j] = np.min(normalizado[:, j])
            anti_ideal[j] = np.max(normalizado[:, j])
        else:  # maximizar
            ideal[j] = np.max(normalizado[:, j])
            anti_ideal[j] = np.min(normalizado[:, j])

    # Distâncias
    dist_ideal = np.sqrt(np.sum((normalizado - ideal)**2, axis=1))
    dist_anti = np.sqrt(np.sum((normalizado - anti_ideal)**2, axis=1))

    # Score TOPSIS
    denom = dist_ideal + dist_anti
    denom[denom == 0] = 1
    score = dist_anti / denom

    return np.argmax(score)


def fronteira_pareto(pontos):
    """
    Identifica pontos na fronteira de Pareto (minimização).
    """
    is_pareto = np.ones(pontos.shape[0], dtype=bool)
    for i, p in enumerate(pontos):
        if is_pareto[i]:
            # Remove pontos dominados
            is_pareto[is_pareto] = np.any(pontos[is_pareto] < p, axis=1) | np.all(pontos[is_pareto] == p, axis=1)
            is_pareto[i] = True
    return is_pareto


# =============================================================================
# FUNÇÃO DESIRABILIDADE
# =============================================================================

def desirabilidade_individual(y, y_min, y_max, objetivo="minimizar", peso=1.0):
    """
    Calcula desirabilidade individual (Derringer & Suich, 1980).

    Parâmetros:
        y: valor predito
        y_min: limite inferior
        y_max: limite superior
        objetivo: 'minimizar', 'maximizar' ou 'alvo'
        peso: expoente (importância)

    Retorna:
        valor de desirabilidade [0, 1]
    """
    if y_max == y_min:
        return 1.0

    if objetivo == "minimizar":
        if y <= y_min:
            return 1.0
        elif y >= y_max:
            return 0.0
        else:
            return ((y_max - y) / (y_max - y_min)) ** peso
    elif objetivo == "maximizar":
        if y <= y_min:
            return 0.0
        elif y >= y_max:
            return 1.0
        else:
            return ((y - y_min) / (y_max - y_min)) ** peso
    else:  # alvo
        y_alvo = (y_min + y_max) / 2
        if y < y_min or y > y_max:
            return 0.0
        elif y <= y_alvo:
            return ((y - y_min) / (y_alvo - y_min)) ** peso
        else:
            return ((y_max - y) / (y_max - y_alvo)) ** peso


def desirabilidade_global(fz_val, modelos, respostas, pesos, objetivos):
    """
    Calcula desirabilidade global (média geométrica) para um valor de fz.

    Parâmetros:
        fz_val: valor do avanço por dente
        modelos: dict {resposta: dict com 'coefs'}
        respostas: lista de nomes de respostas
        pesos: dict {resposta: peso}
        objetivos: dict {resposta: 'minimizar'/'maximizar'}

    Retorna:
        desirabilidade global [0, 1]
    """
    d_individuais = []

    for resp in respostas:
        coefs = modelos[resp]["coefs"]
        y_pred = np.polyval(coefs, fz_val)

        # Calcular limites a partir do modelo na faixa
        fz_range = np.linspace(fz_val * 0.5, fz_val * 2.5, 100)
        y_range = np.polyval(coefs, fz_range)
        y_min = np.min(y_range)
        y_max = np.max(y_range)

        # Usar limites dos dados observados (passados via modelo)
        if "y_min" in modelos[resp]:
            y_min = modelos[resp]["y_min"]
            y_max = modelos[resp]["y_max"]

        objetivo = objetivos.get(resp, "minimizar")
        peso = pesos.get(resp, 1.0)

        d = desirabilidade_individual(y_pred, y_min, y_max, objetivo, peso)
        d_individuais.append(d)

    if not d_individuais:
        return 0.0

    # Média geométrica
    d_array = np.array(d_individuais)
    d_array = np.maximum(d_array, 1e-10)  # evitar log(0)
    d_global = np.prod(d_array) ** (1.0 / len(d_array))

    return d_global
