# Classificação Supervisionada de Doença Cardíaca no Google Colab

Projeto didático de introdução à **classificação em Machine Learning** da Ocean por Me. Edilson Gabriel Veruz, desenvolvido a partir de um **prompt** executado no **Gemini** (Google Colab). O objetivo é que o aluno aplique a teoria vista no curso seguindo, passo a passo, um fluxo completo de análise: da leitura dos dados à predição de novos pacientes.

> ⚠️ **Aviso:** projeto estritamente educacional. Os modelos **não** devem ser usados para diagnóstico ou decisão clínica.

---

## 📌 Visão geral

| Item | Descrição |
|---|---|
| **Tipo de problema** | Classificação binária supervisionada |
| **Variável resposta** | `DoencaCard` (`Nao` = 0, `Sim` = 1) |
| **Classe positiva** | `Sim` (1), usada em todas as métricas |
| **Base de dados** | Arquivo Excel (`DadosCoracao.xlsx`) com 746 observações |
| **Ambiente** | Google Colab (Python 3) |
| **Bibliotecas** | pandas, numpy, matplotlib, seaborn, scikit-learn |
| **Formato** | 12 células lógicas, comentadas e executáveis em sequência |

---

## 🎓 Contexto do curso

Neste curso, os alunos aprendem a **aplicar a teoria de classificação usando engenharia de prompt**: o enunciado detalhado (o *prompt*) é enviado ao **Gemini**, que gera o código Python correspondente. O aluno então:

1. envia o prompt ao Gemini;
2. revisa o código gerado, célula por célula;
3. executa no Google Colab, enviando o arquivo `.xlsx` quando solicitado;
4. interpreta resultados, matrizes de confusão e curvas ROC.

O prompt funciona como a **especificação do projeto**: quanto mais claro e completo, mais fiel é o código gerado. Neste projeto, ele define variáveis, divisão dos dados, pré-processamento, algoritmos, métricas e gráficos.

---

## 🗂️ Dicionário de variáveis

| Variável | Tipo | Descrição | Valores / Observação |
|---|---|---|---|
| `Idade` | Numérica | Idade do paciente | — |
| `Sexo` | Categórica | Sexo do paciente | `Masculino`, `Feminino` |
| `DorPeito` | Categórica | Tipo de dor no peito | `Atipica`, `Sem_Dor`, `Assintomatico`, `Tipica` |
| `PSDescanso` | Numérica | Pressão sanguínea em repouso | — |
| `Colesterol` | Numérica | Nível de colesterol | — |
| `AcucarSangue` | Categórica | Açúcar no sangue | `Normal`, `Diabetes` |
| `ECGDescanso` | Categórica | Eletrocardiograma em repouso | `Normal`, `Anormal_ST`, `Hipertrofia_VE` |
| `BCMax` | Numérica | Batimento cardíaco máximo | — |
| `AnginaExercicio` | Categórica | Angina induzida por exercício | `Nao`, `Sim` |
| `DoencaCard` | **Resposta** | Presença de doença cardíaca | `Nao` → 0, `Sim` → 1 |

- **Numéricas:** `Idade`, `PSDescanso`, `Colesterol`, `BCMax`
- **Categóricas:** `Sexo`, `DorPeito`, `AcucarSangue`, `ECGDescanso`, `AnginaExercicio`

A variável resposta **não** recebe One-Hot Encoding: é apenas convertida para 0/1.

---

## 🧭 Estrutura do código (12 células)

| # | Célula | O que faz |
|---|---|---|
| 1 | Importação das bibliotecas | Carrega pandas, numpy, matplotlib, seaborn e scikit-learn |
| 2 | Carregamento da base | Upload do `.xlsx` via `files.upload()` e leitura com pandas |
| 3 | Inspeção inicial | 5 primeiras linhas, dimensões, tipos e valores ausentes |
| 4 | Definição das variáveis | Define `X` e `y`, converte `Nao`/`Sim` em 0/1 e garante os tipos corretos |
| 5 | Divisão treino/teste | 80% / 20%, `random_state=42`, `stratify=y`; mostra tamanhos e proporção das classes |
| 6 | Pré-processamento | `ColumnTransformer` com `StandardScaler` (numéricas) e `OneHotEncoder(handle_unknown='ignore')` (categóricas) |
| 7 | Escolha dos algoritmos | Menu com `input()` para escolher **dois** algoritmos, com validação |
| 8 | Treinamento | Uma `Pipeline` independente por modelo, treinada apenas com o conjunto de treino |
| 9 | Avaliação | Classes previstas, probabilidades de `Sim = 1` e tabela de métricas |
| 10 | Matriz de confusão | Uma matriz por modelo, com rótulos `Nao`/`Sim` e valores nas células |
| 11 | Curva ROC | Curvas dos dois modelos no mesmo gráfico, com AUC na legenda |
| 12 | Novas predições | Cinco pacientes fictícios classificados pelos dois modelos |

---

## 🤖 Algoritmos disponíveis

O usuário escolhe **exatamente dois** entre:

| Opção | Algoritmo | Classe do scikit-learn |
|:---:|---|---|
| 1 | Árvore de Decisão | `DecisionTreeClassifier` |
| 2 | Support Vector Machine | `SVC(probability=True)` |
| 3 | Naive Bayes | `GaussianNB` |
| 4 | k-Nearest Neighbors | `KNeighborsClassifier` |
| 5 | Random Forest | `RandomForestClassifier` |

- Parâmetros básicos, sem otimização de hiperparâmetros.
- `random_state=42` em todos os algoritmos que aceitam esse parâmetro.
- A entrada é validada: não aceita opções inexistentes nem o mesmo algoritmo duas vezes.

---

## ⚙️ Pré-processamento e prevenção de *data leakage*

Todo o pré-processamento faz parte da `Pipeline` de cada modelo:

```
Pipeline
├── ColumnTransformer
│   ├── Numéricas   → StandardScaler
│   └── Categóricas → OneHotEncoder(handle_unknown='ignore', saída densa)
└── Classificador escolhido
```

Como o encoder e o scaler são ajustados **somente com o conjunto de treino**, nenhuma informação do teste vaza para o treinamento. A padronização é importante principalmente para **SVM** e **kNN**, que são sensíveis à escala.

---

## 📊 Métricas e gráficos

**Métricas** (classe positiva `Sim = 1`, quatro casas decimais, em um único `DataFrame`):

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC |
|---|---:|---:|---:|---:|---:|

O código **não** declara automaticamente qual modelo é o melhor: a comparação e a interpretação ficam a cargo do aluno.

**Gráficos:**

- **Matriz de confusão** por modelo: eixos *Classe Real* × *Classe Predita*, rótulos `Nao`/`Sim`, valores em cada célula e título com o nome do algoritmo.
- **Curva ROC** com os dois modelos: eixo X = *False Positive Rate*, eixo Y = *True Positive Rate*, diagonal do classificador aleatório, legenda com a AUC e grade discreta. A AUC usa a probabilidade da classe `DoencaCard = 1`.

---

## 🩺 Demonstração com novos pacientes

A última célula cria o `DataFrame` `novos_pacientes` com **cinco observações fictícias**, com as mesmas variáveis explicativas da base (sem `DoencaCard`) e combinações diferentes de categorias.

A tabela final mostra, para **cada modelo escolhido**:

- classe prevista (0/1);
- classe convertida para `Nao`/`Sim`;
- probabilidade estimada de `DoencaCard = Sim`.

Os nomes das colunas se adaptam aos algoritmos selecionados, por exemplo `Predicao_Arvore`, `Probabilidade_Arvore`, `Predicao_SVM`, `Probabilidade_SVM`.

---

## ▶️ Como executar

1. Abra o [Google Colab](https://colab.research.google.com) e crie um novo notebook.
3. Execute a **Célula 2** e, quando solicitado, envie o arquivo `.xlsx` da base.
4. Execute as demais células em sequência.
5. Na **Célula 7**, digite o número do primeiro e do segundo algoritmo quando o `input()` pedir.

> Execute sempre em ordem, do início ao fim. Para trocar os algoritmos, rode novamente a partir da Célula 7.

### Requisitos

Nenhuma instalação é necessária no Colab. Bibliotecas usadas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` e `openpyxl` (leitura de `.xlsx`, já disponível no Colab).

---

## 🚫 Fora do escopo

Para manter o foco didático, **não** são abordados:

- seleção de variáveis;
- PCA;
- balanceamento de classes / SMOTE;
- validação cruzada;
- otimização de hiperparâmetros.

---

## 💡 Dicas para trabalhar com o prompt no Gemini

- Descreva o **contexto** e o **objetivo** antes de listar as etapas.
- Informe nomes exatos das variáveis, tipos e categorias válidas.
- Declare a **classe positiva** e como a resposta deve ser codificada.
- Especifique o que **não** deve ser feito (por exemplo, *não* fazer otimização de hiperparâmetros).
- Peça o código **dividido em células** e **comentado**.
- Sempre **revise e execute** o código gerado; não confie nele sem testar.

---

## 📁 Estrutura sugerida do repositório

```
.
├── README.md
├── prompt.md              # prompt completo enviado ao Gemini
├── notebook.ipynb         # notebook com as 12 células
└── dados/
    └── base_dados.xlsx    # base com 746 observações
```

---

## 👤 Autoria

- **Curso** Usando chats inteligentes para análise de dados e machine learning
- **Instituição:** OCEAN
