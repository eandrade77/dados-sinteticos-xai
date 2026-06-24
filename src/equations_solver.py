import numpy as np
import matplotlib.pyplot as plt
import os

def print_header(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

# ==========================================
# 1. 1D Convolution Sum (Equation 1.1)
# y[i] = sum_{n} x[n] * h[i - n]
# ==========================================
def conv1d_solver(x, h):
    print_header("1. Equação 1.1: Soma de Convolução Unidimensional (1D)")
    print(f"Sinal de entrada x: {x}")
    print(f"Resposta ao impulso h: {h}")
    
    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h - 1
    y = np.zeros(len_y)
    
    print("\nCálculo passo a passo:")
    for i in range(len_y):
        steps = []
        val = 0.0
        for n in range(len_x):
            h_idx = i - n
            if h_idx >= 0 and h_idx < len_h:
                product = x[n] * h[h_idx]
                val += product
                steps.append(f"x[{n}]*h[{h_idx}] ({x[n]}*{h[h_idx]}={product:.2f})")
        
        y[i] = val
        step_str = " + ".join(steps)
        print(f"  y[{i}] = {step_str} = {val:.2f}")
        
    print(f"\nResultado final y = x * h: {y}")
    return y

# ==========================================
# 2. Artificial Neuron (Equation 1.3)
# y = g(sum_{i=1}^n w_i * x_i + beta)
# ==========================================
class ArtificialNeuron:
    def __init__(self, weights, bias, activation_type='relu'):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_type = activation_type.lower()
        
    def _activate(self, z):
        if self.activation_type == 'relu':
            return max(0.0, z), "ReLU(z) = max(0, z)"
        elif self.activation_type == 'sigmoid':
            sig = 1.0 / (1.0 + np.exp(-z))
            return sig, "Sigmoid(z) = 1 / (1 + e^-z)"
        else:
            return z, "Linear(z) = z"
            
    def solve(self, inputs):
        print_header("2. Equação 1.3: Modelo do Neurônio Artificial")
        inputs_arr = np.array(inputs)
        print(f"Entradas (x): {inputs_arr}")
        print(f"Pesos (w):    {self.weights}")
        print(f"Viés (beta):  {self.bias}")
        
        # Weighted sum: z = sum(w * x) + bias
        dot_product = np.dot(inputs_arr, self.weights)
        z = dot_product + self.bias
        
        # Activation
        output, act_formula = self._activate(z)
        
        print("\nCálculo passo a passo:")
        weight_sum_str = " + ".join([f"({w}*{x})" for w, x in zip(self.weights, inputs_arr)])
        print(f"  Soma Ponderada: z = {weight_sum_str} + ({self.bias})")
        print(f"  z = {dot_product:.2f} + {self.bias:.2f} = {z:.2f}")
        print(f"  Ativação: y = {act_formula} = {output:.4f}")
        return output

# ==========================================
# 3. Softmax Function (Equation 1.4)
# y_i = e^{z_i} / sum_{j=1}^C e^{z_j}
# ==========================================
def softmax_solver(logits):
    print_header("3. Equação 1.4: Função de Ativação Softmax")
    print(f"Logits de entrada (z): {logits}")
    
    # Stability trick: subtract max to prevent overflow
    max_logit = np.max(logits)
    shifted_logits = logits - max_logit
    
    exps = np.exp(shifted_logits)
    sum_exps = np.sum(exps)
    probabilities = exps / sum_exps
    
    print("\nCálculo passo a passo:")
    print(f"  1. Logit Máximo (estabilizador): {max_logit:.2f}")
    print(f"  2. Expoentes e^(z_i - max_z):")
    for idx, (z, e) in enumerate(zip(logits, exps)):
        print(f"     e^({z:.2f} - {max_logit:.2f}) = {e:.4f}")
    print(f"  3. Soma dos Exponentes (Denominador): {sum_exps:.4f}")
    print(f"  4. Probabilidades Finais (Divisão):")
    for idx, p in enumerate(probabilities):
        print(f"     Classe {idx}: {exps[idx]:.4f} / {sum_exps:.4f} = {p:.4f} ({p*100.0:.2f}%)")
        
    return probabilities

# ==========================================
# 4. Cross-Entropy Loss (Equation 1.5)
# Loss = - sum_{i=1}^k t_i * log(y_i)
# ==========================================
def cross_entropy_solver(predictions, targets):
    print_header("4. Equação 1.5: Função de Perda de Entropia Cruzada")
    print(f"Previsões do Modelo (y): {predictions}")
    print(f"Ground-Truth One-Hot (t): {targets}")
    
    loss = 0.0
    steps = []
    
    for idx, (y, t) in enumerate(zip(predictions, targets)):
        if t > 0:
            # Prevent log(0)
            eps = 1e-15
            y_clamped = max(y, eps)
            log_val = np.log(y_clamped)
            term = - t * log_val
            loss += term
            steps.append(f"- ({t} * log({y_clamped:.4f})) = {term:.4f}")
            
    step_str = " + ".join(steps)
    print("\nCálculo passo a passo:")
    print(f"  Loss = {step_str}")
    print(f"  Resultado da Perda (Loss): {loss:.4f}")
    return loss

# ==========================================
# 5. Gabor Filter Function
# g(x,y; lambda, theta, phi, gamma)
# ==========================================
def gabor_kernel_2d(size=31, theta=0.0, lambd=10.0, phi=0.0, gamma=0.5, sigma=4.0):
    print_header("5. Equação Filtro Gabor: Representação Bidimensional")
    print(f"Parâmetros: Tamanho={size}x{size}, Angulo (theta)={theta:.2f} rad, Comprimento de Onda (lambda)={lambd}")
    print(f"            Fase (phi)={phi}, Elipticidade (gamma)={gamma}, Sigma={sigma}")
    
    # Generate coordinates grid
    half_size = size // 2
    y_grid, x_grid = np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))
    
    # Rotate coordinates
    x_theta = x_grid * np.cos(theta) + y_grid * np.sin(theta)
    y_theta = -x_grid * np.sin(theta) + y_grid * np.cos(theta)
    
    # Gabor equation
    envelope = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * x_theta / lambd + phi)
    gabor = envelope * carrier
    
    # Plot and save visualization
    plt.figure(figsize=(5, 4.5), dpi=300)
    plt.imshow(gabor, cmap='gray')
    plt.title(f"Filtro Gabor (\\theta = {np.degrees(theta):.0f}^\\circ)")
    plt.colorbar(label="Amplitude do Filtro")
    plt.grid(False)
    
    # Save to both livro folders
    base_dir = r"C:\Users\andradee\Documents\antigravity\silly-brahmagupta"
    pt_img_dir = os.path.join(base_dir, "livro_pt", "imagens")
    en_img_dir = os.path.join(base_dir, "livro_en", "images")
    
    plt.savefig(os.path.join(pt_img_dir, "gabor_python.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(en_img_dir, "gabor_python.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    print("\nVisualização do Filtro Gabor gerada com sucesso e salva em: livro_pt/imagens/gabor_python.png")
    return gabor

# ==========================================
# Run Solvers
# ==========================================
if __name__ == '__main__':
    # 1. Conv 1D
    x_signal = np.array([1, 2, 3, 4])
    h_kernel = np.array([1, 0.5, 0.25])
    conv1d_solver(x_signal, h_kernel)
    
    # 2. Artificial Neuron
    neuron = ArtificialNeuron(weights=[0.8, -0.5, 1.2], bias=-0.2, activation_type='relu')
    neuron.solve(inputs=[1.0, 2.0, 0.5])
    
    # 3. Softmax
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    probs = softmax_solver(logits)
    
    # 4. Cross Entropy
    cross_entropy_solver(predictions=probs, targets=np.array([1.0, 0.0, 0.0, 0.0]))
    
    # 5. Gabor Filter
    gabor_kernel_2d(size=41, theta=np.pi/4, lambd=8.0, phi=0.0, gamma=0.5, sigma=5.0)
