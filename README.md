# Repositório de Código: Dados Sintéticos em Visão Computacional

Este é o repositório oficial de códigos do livro **"Dados Sintéticos em Visão Computacional: Do Treinamento à Explicabilidade com PyTorch e Unity"**, escrito por **Edilton Torres de Andrade**.

Aqui você encontrará todas as simulações matemáticas em Python, os scripts de treinamento de redes neurais convolucionais (CNNs) no PyTorch, os hooks de explicabilidade (XAI) e as rotinas em C# para captura e rotação na Unity 3D.

---

## Estrutura do Repositório

```text
├── LICENSE                      # Licença do repositório (MIT)
├── README.md                    # Documentação principal
└── src/                         # Códigos-fonte
    ├── equations_solver.py      # Solucionador das equações teóricas (Convolução 1D, Neurônio, Softmax, etc.)
    ├── train.py                 # Loop de treinamento e ajuste fino (Fine-Tuning) do InceptionV3 no PyTorch
    ├── gradcam_eval.py          # Implementação e hooks do Grad-CAM para geração de mapas de ativação
    ├── lime_eval.py             # Script de auditoria explicável LIME contra Ground Truth (sensibilidade/precisão)
    └── unity_scripts/           # Scripts em C# para automação na Unity 3D
        ├── CameraRotator.cs     # Coroutine para rotação incremental uniforme em 360 graus
        └── ScreenshotHandler.cs # Captura de renders em RenderTexture de alta definição
```

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
