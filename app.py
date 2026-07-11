"""
Aplicativo Streamlit - Otimização de Microusinagem
====================================================
DOE, Regressão, Validação Estatística e Otimização Multiobjetivo (NSGA-II)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from io import BytesIO

from modelos import (
    ajustar_regressao,
    validar_modelo,
    analise_residuos,
    anova_one_way,
    teste_normalidade,
    otimizacao_nsga2,
    fronteira_pareto,
    desirabilidade_global,
)


# =============================================================================
# FUNÇÕES AUXILIARES (definidas antes do uso)
# =============================================================================

def gerar_template_csv():
    """Gera template CSV para download."""
    dados = {
        "fz": [5, 5, 5, 10, 10, 10, 15, 15, 15, 20, 20, 20],
        "repeticao": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "rebarba": ["", "", "", "", "", "", "", "", "", "", "", ""],
        "Ra": ["", "", "", "", "", "", "", "", "", "", "", ""],
        "Rq": ["", "", "", "", "", "", "", "", "", "", "", ""],
        "Rz": ["", "", "", "", "", "", "", "", "", "", "", ""],
        "Rt": ["", "", "", "", "", "", "", "", "", "", "", ""],
        "Rsk": ["", "", "", "", "", "", "", "", "", "", "", ""],
        "Rku": ["", "", "", "", "", "", "", "", "", "", "", ""],
    }
    return pd.DataFrame(dados).to_csv(index=False)


def gerar_dados_exemplo():
    """Gera dados fictícios para demonstração."""
    np.random.seed(42)
    fz_vals = [5, 10, 15, 20]
    repeticoes = 3
    dados = []

    for fz_val in fz_vals:
        for rep in range(1, repeticoes + 1):
            rebarba = 800 + 10 * fz_val**2 + 50 * fz_val + np.random.normal(0, 80)
            ra = 0.30 + 0.001 * fz_val**2 + 0.005 * fz_val + np.random.normal(0, 0.015)
            rq = 0.38 + 0.0013 * fz_val**2 + 0.006 * fz_val + np.random.normal(0, 0.02)
            rz_val = 1.8 + 0.005 * fz_val**2 + 0.05 * fz_val + np.random.normal(0, 0.1)
            rt_val = 2.5 + 0.008 * fz_val**2 + 0.03 * fz_val + np.random.normal(0, 0.15)
            rsk_val = -0.3 + 0.04 * fz_val + np.random.normal(0, 0.03)
            rku_val = 2.7 + 0.001 * fz_val**2 + 0.02 * fz_val + np.random.normal(0, 0.05)

            dados.append({
                "fz": fz_val,
                "repeticao": rep,
                "rebarba": round(rebarba, 2),
                "Ra": round(ra, 4),
                "Rq": round(rq, 4),
                "Rz": round(rz_val, 4),
                "Rt": round(rt_val, 4),
                "Rsk": round(rsk_val, 4),
                "Rku": round(rku_val, 4),
            })

    return pd.DataFrame(dados)


# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Otimização de Microusinagem",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 Otimização de Parâmetros de Microusinagem")
st.markdown("**DOE · Regressão · Validação Estatística · Otimização Multiobjetivo (NSGA-II)**")
st.divider()

# =============================================================================
# SIDEBAR - CONFIGURAÇÕES
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("Parâmetros Fixos")
    prof_corte = st.number_input("Profundidade de corte (µm)", value=20.0, step=1.0)
    rotacao = st.number_input("Rotação (rpm)", value=20000, step=1000)
    vel_corte = st.number_input("Velocidade de corte (m/min)", value=25.13, step=0.01)

    st.divider()
    st.subheader("Otimização NSGA-II")
    n_geracoes = st.slider("Número de gerações", 50, 500, 200, step=50)
    n_populacao = st.slider("Tamanho da população", 50, 300, 100, step=50)

    st.divider()
    st.subheader("Modelo de Regressão")
    grau_max = st.selectbox("Grau máximo do polinômio", [1, 2, 3], index=1)

# =============================================================================
# 1. IMPORTAÇÃO DE DADOS
# =============================================================================
st.header("📂 1. Importação de Dados Experimentais")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Formato esperado do arquivo (CSV ou Excel):**
    - Coluna `fz`: Avanço por dente (µm/dente)
    - Coluna `repeticao`: Número da repetição (1, 2, 3...)
    - Colunas de resposta: `rebarba`, `Ra`, `Rq`, `Rz`, `Rt`, `Rsk`, `Rku`

    As colunas de resposta são opcionais — inclua apenas as que você mediu.
    """)

with col2:
    st.download_button(
        "📥 Baixar template CSV",
        data=gerar_template_csv(),
        file_name="template_microusinagem.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader(
    "Carregar arquivo de dados",
    type=["csv", "xlsx", "xls"],
    help="Arquivo CSV ou Excel com os dados experimentais",
)

usar_exemplo = st.checkbox("Usar dados de exemplo (demonstração)", value=True)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    usar_exemplo = False
elif usar_exemplo:
    df = gerar_dados_exemplo()
else:
    st.info("⬆️ Carregue um arquivo de dados ou marque 'Usar dados de exemplo'.")
    st.stop()

# Validação dos dados
if "fz" not in df.columns:
    st.error("❌ O arquivo deve conter uma coluna 'fz' (avanço por dente).")
    st.stop()

# Identificar colunas de resposta disponíveis
respostas_possiveis = ["rebarba", "Ra", "Rq", "Rz", "Rt", "Rsk", "Rku"]
respostas_disponiveis = [col for col in respostas_possiveis if col in df.columns]

if not respostas_disponiveis:
    st.error("❌ Nenhuma coluna de resposta encontrada. Use: rebarba, Ra, Rq, Rz, Rt, Rsk, Rku")
    st.stop()

# Converter para numérico
for col in respostas_disponiveis:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df_limpo = df.dropna(subset=respostas_disponiveis, how="all")

st.success(
    f"✅ Dados carregados: {len(df_limpo)} observações, "
    f"{df_limpo['fz'].nunique()} níveis de fz, "
    f"respostas: {', '.join(respostas_disponiveis)}"
)

with st.expander("👁️ Visualizar dados carregados"):
    st.dataframe(df_limpo, width='stretch')

# =============================================================================
# 2. DOE - PLANEJAMENTO EXPERIMENTAL
# =============================================================================
st.header("📊 2. Planejamento Experimental (DOE)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estrutura do Experimento")
    niveis_fz = sorted(df_limpo["fz"].unique())
    n_reps = df_limpo.groupby("fz").size().values

    doe_info = pd.DataFrame({
        "Nível": range(1, len(niveis_fz) + 1),
        "fz (µm/dente)": niveis_fz,
        "Repetições": n_reps,
        "Prof. corte (µm)": prof_corte,
        "Rotação (rpm)": rotacao,
        "Vel. corte (m/min)": vel_corte,
    })
    st.dataframe(doe_info, width='stretch', hide_index=True)

    n_rep_str = str(n_reps[0]) if len(set(n_reps)) == 1 else str(n_reps.tolist())
    st.markdown(f"""
    - **Tipo de planejamento:** Unifatorial (1 fator, {len(niveis_fz)} níveis)
    - **Fator:** Avanço por dente (fz)
    - **Total de ensaios:** {len(df_limpo)}
    - **Repetições por nível:** {n_rep_str}
    """)

with col2:
    st.subheader("Estatísticas Descritivas")
    stats_df = df_limpo.groupby("fz")[respostas_disponiveis].agg(["mean", "std"])
    st.dataframe(stats_df.round(4), width='stretch')

# =============================================================================
# 3. AJUSTE DE MODELOS
# =============================================================================
st.header("📈 3. Ajuste de Modelos de Regressão")

modelos_ajustados = {}
resultados_modelos = []

for resp in respostas_disponiveis:
    dados_resp = df_limpo[["fz", resp]].dropna()
    if dados_resp.empty:
        continue

    x = dados_resp["fz"].values
    y = dados_resp[resp].values

    melhor_modelo = None
    melhor_r2_adj = -np.inf

    for grau in range(1, grau_max + 1):
        resultado = ajustar_regressao(x, y, grau)
        if resultado["r2_adj"] > melhor_r2_adj:
            melhor_r2_adj = resultado["r2_adj"]
            melhor_modelo = resultado

    # Adicionar limites observados ao modelo
    melhor_modelo["y_min"] = np.min(y)
    melhor_modelo["y_max"] = np.max(y)

    modelos_ajustados[resp] = melhor_modelo
    resultados_modelos.append({
        "Resposta": resp,
        "Grau": melhor_modelo["grau"],
        "R²": round(melhor_modelo["r2"], 4),
        "R² ajustado": round(melhor_modelo["r2_adj"], 4),
        "RMSE": round(melhor_modelo["rmse"], 4),
        "Equação": melhor_modelo["equacao_str"],
    })

# Tabela resumo dos modelos
df_modelos = pd.DataFrame(resultados_modelos)
st.dataframe(df_modelos, width='stretch', hide_index=True)

# Gráficos de regressão
st.subheader("Curvas de Regressão Ajustadas")

n_plots = len(respostas_disponiveis)
cols_per_row = min(3, n_plots)
rows = (n_plots + cols_per_row - 1) // cols_per_row

fig = make_subplots(
    rows=rows, cols=cols_per_row,
    subplot_titles=respostas_disponiveis,
    vertical_spacing=0.15,
)

fz_plot = np.linspace(min(niveis_fz) - 1, max(niveis_fz) + 1, 200)

for idx, resp in enumerate(respostas_disponiveis):
    row = idx // cols_per_row + 1
    col = idx % cols_per_row + 1

    dados_resp = df_limpo[["fz", resp]].dropna()
    modelo = modelos_ajustados[resp]

    # Pontos experimentais
    fig.add_trace(
        go.Scatter(
            x=dados_resp["fz"], y=dados_resp[resp],
            mode="markers", name=f"{resp} (dados)",
            marker=dict(size=8, color="black"),
            showlegend=False,
        ),
        row=row, col=col,
    )

    # Curva ajustada
    y_fit = np.polyval(modelo["coefs"], fz_plot)
    fig.add_trace(
        go.Scatter(
            x=fz_plot, y=y_fit,
            mode="lines", name=f"{resp} (modelo)",
            line=dict(color="blue", width=2),
            showlegend=False,
        ),
        row=row, col=col,
    )

    # IC 95%
    ic_sup = y_fit + modelo["ic_95"]
    ic_inf = y_fit - modelo["ic_95"]
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([fz_plot, fz_plot[::-1]]),
            y=np.concatenate([ic_sup, ic_inf[::-1]]),
            fill="toself", fillcolor="rgba(0,100,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 95%", showlegend=False,
        ),
        row=row, col=col,
    )

    fig.update_xaxes(title_text="fz (µm/dente)", row=row, col=col)
    fig.update_yaxes(title_text=resp, row=row, col=col)

fig.update_layout(height=350 * rows, title_text="Modelos de Regressão Ajustados")
st.plotly_chart(fig, width='stretch')

# =============================================================================
# 4. VALIDAÇÃO ESTATÍSTICA
# =============================================================================
st.header("✅ 4. Validação Estatística dos Modelos")

tab_anova, tab_residuos, tab_normalidade = st.tabs(
    ["ANOVA", "Análise de Resíduos", "Normalidade"]
)

with tab_anova:
    st.subheader("ANOVA One-Way (α = 0.05)")
    anova_results = []
    for resp in respostas_disponiveis:
        dados_resp = df_limpo[["fz", resp]].dropna()
        grupos = [g[resp].values for _, g in dados_resp.groupby("fz")]
        f_stat, p_valor = anova_one_way(grupos)
        if p_valor < 0.001:
            sig = "✅ Sim (p < 0.001)"
        elif p_valor < 0.01:
            sig = "✅ Sim (p < 0.01)"
        elif p_valor < 0.05:
            sig = "✅ Sim (p < 0.05)"
        else:
            sig = "❌ Não"
        anova_results.append({
            "Resposta": resp,
            "F-estatística": round(f_stat, 4),
            "p-valor": f"{p_valor:.6f}",
            "Significativo (α=0.05)?": sig,
        })
    st.dataframe(pd.DataFrame(anova_results), width='stretch', hide_index=True)

with tab_residuos:
    st.subheader("Análise de Resíduos")

    resp_selecionada = st.selectbox(
        "Selecionar resposta:", respostas_disponiveis, key="residuos_resp"
    )
    dados_resp = df_limpo[["fz", resp_selecionada]].dropna()
    modelo = modelos_ajustados[resp_selecionada]

    residuos_info = analise_residuos(
        dados_resp["fz"].values,
        dados_resp[resp_selecionada].values,
        modelo["coefs"],
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_res = px.scatter(
            x=residuos_info["y_pred"], y=residuos_info["residuos"],
            labels={"x": "Valores Ajustados", "y": "Resíduos"},
            title="Resíduos vs Valores Ajustados",
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, width='stretch')

    with col2:
        residuos_std = residuos_info["residuos_padronizados"]
        n_res = len(residuos_std)
        qq_teorico = scipy_stats.norm.ppf(
            (np.arange(1, n_res + 1) - 0.5) / n_res
        )
        residuos_ord = np.sort(residuos_std)

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=qq_teorico, y=residuos_ord,
            mode="markers", name="Resíduos",
        ))
        lim = max(abs(qq_teorico.min()), abs(qq_teorico.max()))
        fig_qq.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim],
            mode="lines", name="Referência",
            line=dict(dash="dash", color="red"),
        ))
        fig_qq.update_layout(
            title="QQ-Plot dos Resíduos",
            xaxis_title="Quantis Teóricos",
            yaxis_title="Quantis Amostrais",
        )
        st.plotly_chart(fig_qq, width='stretch')

    # Métricas de validação
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{modelo['r2']:.4f}")
    col2.metric("R² ajustado", f"{modelo['r2_adj']:.4f}")
    col3.metric("RMSE", f"{modelo['rmse']:.4f}")
    col4.metric("Durbin-Watson", f"{residuos_info['durbin_watson']:.4f}")

    st.caption(
        "Durbin-Watson ≈ 2.0 indica ausência de autocorrelação. "
        "Valores < 1.5 ou > 2.5 sugerem autocorrelação."
    )

with tab_normalidade:
    st.subheader("Teste de Normalidade dos Resíduos (Shapiro-Wilk, α = 0.05)")
    norm_results = []
    for resp in respostas_disponiveis:
        dados_resp = df_limpo[["fz", resp]].dropna()
        modelo = modelos_ajustados[resp]
        y_pred = np.polyval(modelo["coefs"], dados_resp["fz"].values)
        residuos = dados_resp[resp].values - y_pred
        w_stat, p_valor = teste_normalidade(residuos)
        norm_results.append({
            "Resposta": resp,
            "W (Shapiro-Wilk)": f"{w_stat:.4f}" if w_stat else "N/A",
            "p-valor": f"{p_valor:.6f}" if p_valor else "N/A",
            "Normal (α=0.05)?": "✅ Sim" if (p_valor and p_valor > 0.05) else "❌ Não",
        })
    st.dataframe(pd.DataFrame(norm_results), width='stretch', hide_index=True)

# =============================================================================
# 5. OTIMIZAÇÃO MULTIOBJETIVO
# =============================================================================
st.header("🎯 5. Otimização Multiobjetivo")

tab_pareto, tab_desirabilidade = st.tabs(
    ["Fronteira de Pareto (NSGA-II)", "Função Desirabilidade"]
)

with tab_pareto:
    st.subheader("Otimização NSGA-II - Fronteira de Pareto")

    st.markdown("**Selecione as respostas a otimizar e seus objetivos:**")

    respostas_para_otim = []
    objetivos = {}

    cols = st.columns(min(4, len(respostas_disponiveis)))
    for idx, resp in enumerate(respostas_disponiveis):
        with cols[idx % len(cols)]:
            incluir = st.checkbox(
                f"{resp}",
                value=(resp in ["rebarba", "Ra"]),
                key=f"obj_{resp}",
            )
            if incluir:
                obj = st.selectbox(
                    f"Objetivo",
                    ["Minimizar", "Maximizar"],
                    key=f"obj_tipo_{resp}",
                )
                objetivos[resp] = obj.lower()
                respostas_para_otim.append(resp)

    if len(respostas_para_otim) >= 2:
        if st.button("🚀 Executar NSGA-II", type="primary"):
            with st.spinner("Executando otimização NSGA-II..."):
                modelos_otim = {
                    r: modelos_ajustados[r]["coefs"] for r in respostas_para_otim
                }
                obj_tipos = {r: objetivos[r] for r in respostas_para_otim}

                resultado_nsga = otimizacao_nsga2(
                    modelos_otim,
                    obj_tipos,
                    fz_min=min(niveis_fz),
                    fz_max=max(niveis_fz),
                    n_geracoes=n_geracoes,
                    n_populacao=n_populacao,
                )

                st.session_state["resultado_nsga"] = resultado_nsga
                st.session_state["respostas_otim"] = respostas_para_otim

        if "resultado_nsga" in st.session_state:
            resultado_nsga = st.session_state["resultado_nsga"]
            respostas_otim_state = st.session_state["respostas_otim"]

            if len(respostas_otim_state) == 2:
                fig_pareto = go.Figure()
                pareto_front = resultado_nsga["pareto_front"]
                fig_pareto.add_trace(go.Scatter(
                    x=pareto_front[:, 0],
                    y=pareto_front[:, 1],
                    mode="markers+lines",
                    marker=dict(
                        size=10,
                        color=resultado_nsga["fz_otimos"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="fz (µm/dente)"),
                    ),
                    name="Fronteira de Pareto",
                ))
                fig_pareto.update_layout(
                    title="Fronteira de Pareto",
                    xaxis_title=respostas_otim_state[0],
                    yaxis_title=respostas_otim_state[1],
                    height=500,
                )
                st.plotly_chart(fig_pareto, width='stretch')

            elif len(respostas_otim_state) >= 3:
                pareto_df = pd.DataFrame(
                    resultado_nsga["pareto_front"],
                    columns=respostas_otim_state,
                )
                pareto_df["fz"] = resultado_nsga["fz_otimos"]
                fig_parallel = px.parallel_coordinates(
                    pareto_df,
                    dimensions=respostas_otim_state + ["fz"],
                    color="fz",
                    color_continuous_scale="Viridis",
                    title="Fronteira de Pareto (Coordenadas Paralelas)",
                )
                st.plotly_chart(fig_parallel, width='stretch')

            # Tabela de soluções
            st.subheader("Soluções da Fronteira de Pareto")
            pareto_df = pd.DataFrame(
                resultado_nsga["pareto_front"],
                columns=respostas_otim_state,
            )
            pareto_df.insert(0, "fz (µm/dente)", resultado_nsga["fz_otimos"])
            pareto_df = pareto_df.sort_values("fz (µm/dente)").round(4)
            st.dataframe(pareto_df, width='stretch', hide_index=True)

            # Solução de compromisso
            idx_compromisso = resultado_nsga.get("idx_compromisso", 0)
            fz_comp = resultado_nsga["fz_otimos"][idx_compromisso]
            st.success(f"**Solução de compromisso (TOPSIS): fz = {fz_comp:.2f} µm/dente**")

    else:
        st.warning("⚠️ Selecione pelo menos 2 respostas para otimização multiobjetivo.")

with tab_desirabilidade:
    st.subheader("Otimização por Função Desirabilidade (Derringer & Suich)")

    st.markdown("**Pesos das respostas (importância relativa):**")
    pesos = {}
    cols = st.columns(min(4, len(respostas_disponiveis)))
    for idx, resp in enumerate(respostas_disponiveis):
        with cols[idx % len(cols)]:
            pesos[resp] = st.slider(
                f"{resp}", 0.0, 2.0, 1.0, 0.1, key=f"peso_{resp}"
            )

    if st.button("Calcular Desirabilidade", type="primary"):
        fz_range = np.linspace(min(niveis_fz), max(niveis_fz), 500)
        d_values = []

        objetivos_desir = {r: "minimizar" for r in respostas_disponiveis}

        for fz_val in fz_range:
            d = desirabilidade_global(
                fz_val, modelos_ajustados, respostas_disponiveis,
                pesos, objetivos_desir,
            )
            d_values.append(d)

        d_values = np.array(d_values)
        idx_otimo = np.argmax(d_values)
        fz_otimo_d = fz_range[idx_otimo]
        d_otimo = d_values[idx_otimo]

        # Gráfico
        fig_desir = go.Figure()
        fig_desir.add_trace(go.Scatter(
            x=fz_range, y=d_values,
            mode="lines", line=dict(color="blue", width=2.5),
            name="Desirabilidade Global",
        ))
        fig_desir.add_vline(x=fz_otimo_d, line_dash="dash", line_color="red")
        fig_desir.add_annotation(
            x=fz_otimo_d, y=d_otimo,
            text=f"Ótimo: fz = {fz_otimo_d:.2f} µm/dente",
            showarrow=True, arrowhead=2, font=dict(size=13),
        )
        fig_desir.update_layout(
            title="Função Desirabilidade Global",
            xaxis_title="fz (µm/dente)",
            yaxis_title="Desirabilidade (D)",
            yaxis_range=[0, 1.05],
            height=450,
        )
        st.plotly_chart(fig_desir, width='stretch')

        # Resultados
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avanço Ótimo (fz)", f"{fz_otimo_d:.2f} µm/dente")
            st.metric("Desirabilidade Global", f"{d_otimo:.4f}")
        with col2:
            st.markdown("**Valores preditos no ponto ótimo:**")
            for resp in respostas_disponiveis:
                y_pred = np.polyval(modelos_ajustados[resp]["coefs"], fz_otimo_d)
                st.write(f"- {resp}: **{y_pred:.4f}**")

# =============================================================================
# 6. EXPORTAÇÃO DE RELATÓRIO
# =============================================================================
st.header("📄 6. Exportar Relatório")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Exportar Resultados (Excel)"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_limpo.to_excel(writer, sheet_name="Dados", index=False)

            desc_stats = df_limpo.groupby("fz")[respostas_disponiveis].agg(["mean", "std"])
            desc_stats.to_excel(writer, sheet_name="Estatísticas")

            df_modelos.to_excel(writer, sheet_name="Modelos", index=False)
            pd.DataFrame(anova_results).to_excel(writer, sheet_name="ANOVA", index=False)
            pd.DataFrame(norm_results).to_excel(writer, sheet_name="Normalidade", index=False)

            if "resultado_nsga" in st.session_state:
                pareto_export = pd.DataFrame(
                    st.session_state["resultado_nsga"]["pareto_front"],
                    columns=st.session_state["respostas_otim"],
                )
                pareto_export.insert(
                    0, "fz (µm/dente)",
                    st.session_state["resultado_nsga"]["fz_otimos"],
                )
                pareto_export.to_excel(writer, sheet_name="Pareto", index=False)

        st.download_button(
            "⬇️ Download Excel",
            data=output.getvalue(),
            file_name="relatorio_microusinagem.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with col2:
    st.markdown("""
    **O relatório Excel contém:**
    - Dados experimentais brutos
    - Estatísticas descritivas por condição
    - Equações dos modelos ajustados (R², RMSE)
    - Resultados da ANOVA
    - Teste de normalidade
    - Soluções da fronteira de Pareto (se calculada)
    """)

# =============================================================================
# RODAPÉ
# =============================================================================
st.divider()
st.caption(
    "Desenvolvido para estudo de otimização em microusinagem · "
    "Python + Streamlit + NSGA-II"
)
