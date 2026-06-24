# Repositório de Código: Dados Sintéticos em Visão Computacional

Este é o repositório oficial de códigos do livro **"Dados Sintéticos em Visão Computacional: Do Treinamento à Explicabilidade com PyTorch e Unity"**, escrito por **Edilton Torres de Andrade**.

Aqui você encontrará todas as simulações matemáticas em Python, os scripts de treinamento de redes neurais convolucionais (CNNs) no PyTorch, os hooks de explicabilidade (XAI) e as rotinas em C# para captura e rotação na Unity 3D.

---

## Estrutura do Repositório

```text
├── LICENSE                      # Licença do repositório (MIT)
├── README.md                    # Documentação principal
├── notebooks_keras_original/    # Notebooks originais de pesquisa em Keras/TensorFlow (Tese de Mestrado, 2022)
│   ├── Engine.ipynb             # Seleção de classes, prep do dataset e baseline
│   ├── Engine_t1_real.ipynb     # Treinamento e avaliação no dataset real
│   ├── Engine_t1_sintetico.ipynb # Treinamento e avaliação no dataset sintético
│   ├── lime_ago20-Expr1.ipynb   # Explicabilidade LIME e cálculo de TP/FP/FN de superpixels
│   └── teste_mask.ipynb         # Testes de máscaras de interseção/sobreposição LIME vs Ground Truth
├── dados_auditoria_original/    # Planilhas originais de auditoria LIME vs Ground Truth (formato Excel)
│   ├── class_priori.xlsx        # Mapeamento de imagens para classes (dalmatian, german_shepherd, etc.)
│   ├── arrays6_calcs.xlsx       # Matrizes binárias de superpixels e cálculo de TP/FP/FN (precisão e recall)
│   ├── output_base1_t1_real_sample.xlsx # Logs de predições, probabilidades e métricas de auditoria XAI
│   └── resumo_ranking.xlsx      # Estatísticas consolidadas de Hit e acurácia por classe
└── src/                         # Códigos-fonte portados para PyTorch e C#
    ├── equations_solver.py      # Solucionador das equações teóricas (Convolução 1D, Neurônio, Softmax, etc.)
    ├── train.py                 # Loop de treinamento e ajuste fino (Fine-Tuning) do InceptionV3 no PyTorch
    ├── gradcam_eval.py          # Implementação e hooks do Grad-CAM para geração de mapas de ativação
    ├── lime_eval.py             # Script de auditoria explicável LIME contra Ground Truth (sensibilidade/precisão)
    └── unity_scripts/           # Scripts em C# para automação na Unity 3D
        ├── CameraRotator.cs     # Coroutine para rotação incremental uniforme em 360 graus
        └── ScreenshotHandler.cs # Captura de renders em RenderTexture de alta definição
```

---

## Notebooks Keras Originais e Dados de Auditoria (Pesquisa de Origem)

A pasta `notebooks_keras_original/` e a pasta `dados_auditoria_original/` preservam os experimentos originais e planilhas de validação desenvolvidas durante a dissertação de mestrado (UFABC, 2022). Esses artefatos históricos serviram como prova de conceito para as metodologias explicadas no livro e foram integralmente portados para scripts modulares em **PyTorch** contidos na pasta `src/`.

### Notebooks Originais:
* **Engine.ipynb**: Pipeline de ingestão, particionamento do dataset de cães do Stanford Dogs e treinamento de modelos de base.
* **Engine_t1_real.ipynb / Engine_t1_sintetico.ipynb**: Experimentos comparativos de fine-tuning utilizando imagens reais do mundo físico vs. imagens sintéticas renderizadas 3D com ruído.
* **lime_ago20-Expr1.ipynb**: Notebook onde foi validada a formulação matemática de interseção de superpixels do LIME contra o Ground Truth (definição de Verdadeiros Positivos na equação $VP = \sum_i \sum_j A_{i,j} \times B_{i,j}$).
* **teste_mask.ipynb**: Protótipo de validação local para extração de superpixels relevantes e geração das máscaras binárias.

### Dados de Auditoria (`dados_auditoria_original/`):
* **`class_priori.xlsx`**: Lista de mapeamento de cada imagem com sua classe real correspondente.
* **`arrays6_calcs.xlsx`**: Matriz 2D correspondente ao grid de superpixels da imagem avaliada (dimensões de entrada $299 \times 299$), demonstrando a marcação de Verdadeiros Positivos (TP), Falsos Positivos (FP) e Falsos Negativos (FN) em cada região para o cálculo de sensibilidade e precisão da explicação visual.
* **`output_base1_t1_real_sample.xlsx`**: Dados tabulados com probabilidades preditas, classe atribuída pela rede, classe explicada pelo LIME, e contagem de superpixels correspondentes à sobreposição XAI vs. Ground Truth.
* **`resumo_ranking.xlsx`**: Sumário executivo das métricas consolidadas (Hits de 1 a 6) por classe e conjunto de testes.

### Origem das Imagens e Máscaras de Ground Truth (GT):
No ambiente do desenvolvedor, a base de imagens original e as máscaras manuais de Ground Truth foram migradas para o drive `D:` no seguinte caminho:
`D:\audiodeep\image_downloader\image_downloader\images\`
* **Imagens Originais**: Localizadas nas pastas das respectivas classes (ex: `german-shepherd 2020_09_11_total/` ou `3dogs/teste/`).
* **Máscaras de Ground Truth**: Localizadas sob as subpastas `mask/` de cada diretório (ex: `german-shepherd 2020_09_11_total/mask/` e `3dogs/mask/`). Estas máscaras contêm imagens binárias de segmentação (silhueta em preto e branco) que servem como gabarito para aferir os Verdadeiros Positivos (TP) do LIME.

---

## Instalação e Requisitos

Os códigos em Python exigem o **Python 3.10 ou superior**. Recomendamos a criação de um ambiente virtual (`venv` ou `conda`).

Instale as dependências executando:

```bash
# Instalação do PyTorch (GPU com CUDA) - Ajuste a versão conforme necessário
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Instalação de bibliotecas adicionais
pip install numpy matplotlib seaborn pandas pillow opencv-python scikit-image lime
```

---

## Como Executar

### 1. Testar Simulação de Equações
Para verificar o comportamento matemático de convoluções 1D, neurônios artificiais, ativações Softmax/ReLU e filtros de Gabor:
```bash
python src/equations_solver.py
```

### 2. Treinamento no PyTorch
Para iniciar o treinamento e ajuste fino em duas fases da CNN InceptionV3:
```bash
python src/train.py --dataset_root path/to/dataset --classes path/to/classes.txt
```

### 3. Explicabilidade (XAI)
Para rodar a auditoria quantitativa das explicações locais contra as máscaras de Ground Truth:
```bash
# Grad-CAM
python src/gradcam_eval.py --image_path img.jpg --model_path model.pth

# LIME
python src/lime_eval.py --image_dir path/to/images --model_path model.pth --classes path/to/classes.txt
```

---

## Licenciamento e Avisos de Terceiros (Open Source compliance)

Este repositório é fornecido sob os termos da **Licença MIT** (consulte o arquivo [LICENSE](./LICENSE) para detalhes completos). Todos os scripts customizados desenvolvidos para o livro são de propriedade intelectual do autor e de uso livre comercial e acadêmico.

Para garantir conformidade legal com o ecossistema de software livre e integridade acadêmica, listamos abaixo as atribuições de autoria e licenças de todos os softwares de terceiros e publicações científicas utilizados como dependências nesta obra:

### Dependências de Software Livre

1. **PyTorch & Torchvision**
   * **Uso**: Framework principal de redes neurais profunda e modelos pré-treinados (InceptionV3).
   * **Licença**: [BSD 3-Clause License](https://github.com/pytorch/pytorch/blob/main/LICENSE). Copyright (c) 2016- Facebook, Inc (Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, etc.).
2. **LIME (Local Interpretable Model-agnostic Explanations)**
   * **Uso**: Geração de explicações locais baseadas em perturbação de superpixels.
   * **Licença**: [BSD 3-Clause License](https://github.com/marcotcr/lime/blob/master/LICENSE). Copyright (c) 2016, Marco Tulio Correia Ribeiro.
3. **NumPy**
   * **Uso**: Computação matricial e cálculo matemático estruturado.
   * **Licença**: [BSD 3-Clause License](https://github.com/numpy/numpy/blob/main/LICENSE.txt). Copyright (c) 2005-2022, NumPy Developers.
4. **Matplotlib**
   * **Uso**: Visualização gráfica e geração de figuras científicas.
   * **Licença**: [PSF License Agreement](https://matplotlib.org/stable/users/project/license.html). Copyright (c) 2002-2024 John D. Hunter, Michael Droettboom, Matplotlib Development Team.
5. **OpenCV-Python**
   * **Uso**: Processamento de imagens OpenCV e sobreposição de mapas de calor Grad-CAM.
   * **Licença**: [Apache License 2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE). Copyright (c) 2024, OpenCV Team.
6. **Pillow (PIL)**
   * **Uso**: Carregamento e manipulação de arquivos de imagem no Python.
   * **Licença**: [HPND License (MIT-like)](https://github.com/python-pillow/Pillow/blob/main/LICENSE). Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik Lundh.
7. **scikit-image**
   * **Uso**: Segmentação Quickshift de superpixels para a auditoria do LIME.
   * **Licença**: [BSD 3-Clause License](https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt). Copyright (c) 2009-2024, scikit-image Dev Team.
8. **Pandas**
   * **Uso**: Estruturação de dados tabulares das métricas quantitativas de auditoria.
   * **Licença**: [BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/main/LICENSE). Copyright (c) 2008-2011, AQR Capital Management, LLC, Lamppost Labs LLC and PyData Development Team.
9. **Unity Technologies Software**
   * **Uso**: Motor gráfico 3D para renderização e automação de capturas sintéticas.
   * **Licença**: Sujeito aos Termos de Serviço da Unity e licenças associadas aos assets do *Dog Pack* (Unity Asset Store Standard Unity EULA).

---

### Referências Científicas Citadas

Caso utilize estes algoritmos e metodologias em pesquisas acadêmicas, recomendamos citar as publicações originais de referência:

* **Grad-CAM**: Selvaraju, Ramprasaath R., et al. *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization."* Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.
* **LIME**: Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. *""Why Should I Trust You?": Explaining the Predictions of Any Classifier."* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.
* **Inception-v3**: Szegedy, Christian, et al. *"Rethinking the Inception Architecture for Computer Vision."* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
