import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# URL del repositorio de GitHub: https://github.com/smarc05/Laboratorio_VI.git[cite: 3]

# --- FUNCIONES MATEMÁTICAS GENERALES Y DE POLINOMIOS ---
def evaluar_poli(coefs, x):
    return sum(c * (x**(6-i)) for i, c in enumerate(coefs))

def evaluar_der_poli(coefs, x):
    der_coefs = [c * (6-i) for i, c in enumerate(coefs[:-1])]
    return sum(c * (x**(5-i)) for i, c in enumerate(der_coefs))

# --- PARTE 1: MÉTODOS DE RAÍCES ---
def biseccion(coefs, xl, xu, es, max_iter):
    if evaluar_poli(coefs, xl) * evaluar_poli(coefs, xu) >= 0:
        raise ValueError("El intervalo [xl, xu] no presenta cambio de signo.")
    xr = xl
    ea = 100.0
    for i in range(max_iter):
        xr_old = xr
        xr = (xl + xu) / 2.0
        if xr != 0:
            ea = abs((xr - xr_old) / xr) * 100
        test = evaluar_poli(coefs, xl) * evaluar_poli(coefs, xr)
        if test < 0:
            xu = xr
        elif test > 0:
            xl = xr
        else:
            ea = 0.0
        if i > 0 and ea < es:
            return xr, i + 1, ea, evaluar_poli(coefs, xr)
    raise ValueError("Máximo de iteraciones alcanzado sin converger.")

def newton_raphson(coefs, x0, es, max_iter):
    xr = x0
    ea = 100.0
    for i in range(max_iter):
        xr_old = xr
        f_xr = evaluar_poli(coefs, xr)
        df_xr = evaluar_der_poli(coefs, xr)
        if df_xr == 0:
            raise ValueError("Derivada nula. El método de Newton-Raphson falla.")
        xr = xr_old - f_xr / df_xr
        if xr != 0:
            ea = abs((xr - xr_old) / xr) * 100
        if i > 0 and ea < es:
            return xr, i + 1, ea, evaluar_poli(coefs, xr)
    raise ValueError("Máximo de iteraciones alcanzado sin converger.")

def secante(coefs, x_1, x0, es, max_iter):
    ea = 100.0
    for i in range(max_iter):
        f_x0 = evaluar_poli(coefs, x0)
        f_x_1 = evaluar_poli(coefs, x_1)
        if f_x0 - f_x_1 == 0:
            raise ValueError("División por cero. Diferencia de f(x) nula en Secante.")
        xr = x0 - (f_x0 * (x0 - x_1)) / (f_x0 - f_x_1)
        if xr != 0:
            ea = abs((xr - x0) / xr) * 100
        if i > 0 and ea < es:
            return xr, i + 1, ea, evaluar_poli(coefs, xr)
        x_1 = x0
        x0 = xr
    raise ValueError("Máximo de iteraciones alcanzado sin converger.")

# --- PARTE 2: DERIVADA NUMÉRICA ---
def derivada_centrada(coefs, x, h):
    if h == 0:
        raise ValueError("El paso 'h' (o 'dx') no puede ser cero.")
    f_x_mas_h = evaluar_poli(coefs, x + h)
    f_x_menos_h = evaluar_poli(coefs, x - h)
    return (f_x_mas_h - f_x_menos_h) / (2 * h)

# --- PARTE 3: SISTEMAS DE ECUACIONES (ELIMINACIÓN DE GAUSS) ---
def eliminacion_gauss(A, b, pivoteo=False):
    n = len(b)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    for k in range(n - 1):
        if pivoteo:
            max_idx = np.argmax(abs(A[k:n, k])) + k
            if A[max_idx, k] == 0:
                raise ValueError("El sistema no tiene solución única.")
            if max_idx != k:
                A[[k, max_idx]] = A[[max_idx, k]]
                b[[k, max_idx]] = b[[max_idx, k]]
        else:
            if A[k, k] == 0:
                raise ValueError(f"Pivote nulo detectado en la fila {k+1}. Método sin pivoteo falla.")
        
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]
            
    if A[n-1, n-1] == 0:
        raise ValueError("El sistema no tiene solución única.")
        
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# --- PARTE 4: INTEGRAL DEFINIDA (SIMPSON 1/3 COMPUESTA) ---
def simpson_13(coefs, a, b, n):
    if n % 2 != 0:
        raise ValueError("La cantidad de subintervalos 'n' debe ser par.")
    if n <= 0:
        raise ValueError("La cantidad de subintervalos 'n' debe ser mayor a cero.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = [evaluar_poli(coefs, xi) for xi in x]
    
    integral = y[0] + y[-1]
    for i in range(1, n):
        if i % 2 == 0:
            integral += 2 * y[i]
        else:
            integral += 4 * y[i]
    return integral * (h / 3)


# --- INTERFAZ GRÁFICA PRINCIPAL COMPLETA ---
class LaboVIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laboratorio VI.1 - Cálculo Numérico Completo")
        self.crear_widgets()
        self.cargar_valores_prueba()

    def crear_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- SECCIÓN POLINOMIO (COMPARTIDO) ---
        ttk.Label(main_frame, text="Coeficientes\ndel polinomio", justify="center").grid(row=0, column=0, rowspan=2, padx=10)
        self.entries_coefs = []
        labels_coefs = ['a6', 'a5', 'a4', 'a3', 'a2', 'a1', 'a0']
        for i, lbl in enumerate(labels_coefs):
            e = ttk.Entry(main_frame, width=6)
            e.grid(row=0, column=i+1, padx=2, pady=5)
            ttk.Label(main_frame, text=lbl).grid(row=1, column=i+1)
            self.entries_coefs.append(e)
            
        ttk.Button(main_frame, text="Borrar Todo", command=self.borrar_todo).grid(row=0, column=8, padx=15)
        ttk.Button(main_frame, text="Valores de prueba", command=self.cargar_valores_prueba).grid(row=1, column=8, padx=15)

        # --- SECCIÓN 1: RAÍCES ---
        lf_raices = ttk.LabelFrame(main_frame, text="1. Cálculo de una raíz (Bisección, Newton, Secante)", padding="12")
        lf_raices.grid(row=2, column=0, columnspan=9, sticky="ew", pady=10)

        self.var_metodo = tk.StringVar(value="biseccion")
        ttk.Radiobutton(lf_raices, text="Bisección", variable=self.var_metodo, value="biseccion", command=self.toggle_raices_inputs).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(lf_raices, text="Newton-Raphson", variable=self.var_metodo, value="newton", command=self.toggle_raices_inputs).grid(row=0, column=2, padx=5)
        ttk.Radiobutton(lf_raices, text="Secante", variable=self.var_metodo, value="secante", command=self.toggle_raices_inputs).grid(row=0, column=3, padx=5)

        ttk.Label(lf_raices, text="xl / x0 / x-1:").grid(row=1, column=0, sticky="e", pady=8)
        self.entry_raiz_x1 = ttk.Entry(lf_raices, width=10)
        self.entry_raiz_x1.grid(row=1, column=1, sticky="w")

        ttk.Label(lf_raices, text="xu / - / x0:").grid(row=1, column=2, sticky="e")
        self.entry_raiz_x2 = ttk.Entry(lf_raices, width=10)
        self.entry_raiz_x2.grid(row=1, column=3, sticky="w")

        self.var_tol = tk.StringVar(value="es")
        ttk.Radiobutton(lf_raices, text="es (%):", variable=self.var_tol, value="es").grid(row=2, column=0, sticky="e")
        self.entry_es = ttk.Entry(lf_raices, width=10)
        self.entry_es.grid(row=2, column=1, sticky="w")

        ttk.Radiobutton(lf_raices, text="Cifras signif.:", variable=self.var_tol, value="cifras").grid(row=2, column=2, sticky="e")
        self.entry_cifras = ttk.Entry(lf_raices, width=10)
        self.entry_cifras.grid(row=2, column=3, sticky="w")

        ttk.Label(lf_raices, text="Máx. iter:").grid(row=2, column=4, sticky="e")
        self.entry_max_iter = ttk.Entry(lf_raices, width=10)
        self.entry_max_iter.grid(row=2, column=5, sticky="w")

        ttk.Button(lf_raices, text="Calcular Raíz", command=self.calcular_raiz).grid(row=1, column=6, rowspan=2, padx=20)

        ttk.Label(lf_raices, text="Raíz:").grid(row=0, column=7, sticky="e")
        self.lbl_res_raiz = ttk.Label(lf_raices, text="---", font=("TkDefaultFont", 9, "bold"))
        self.lbl_res_raiz.grid(row=0, column=8, sticky="w")
        
        ttk.Label(lf_raices, text="Iteraciones:").grid(row=1, column=7, sticky="e")
        self.lbl_res_iter = ttk.Label(lf_raices, text="---")
        self.lbl_res_iter.grid(row=1, column=8, sticky="w")

        ttk.Label(lf_raices, text="ea (%):").grid(row=2, column=7, sticky="e")
        self.lbl_res_ea = ttk.Label(lf_raices, text="---")
        self.lbl_res_ea.grid(row=2, column=8, sticky="w")

        ttk.Label(lf_raices, text="f(raiz):").grid(row=3, column=7, sticky="e")
        self.lbl_res_fraiz = ttk.Label(lf_raices, text="---")
        self.lbl_res_fraiz.grid(row=3, column=8, sticky="w")

        # --- SECCIÓN 2: DERIVADA ---
        lf_der = ttk.LabelFrame(main_frame, text="2. Cálculo de la derivada primera (Diferencia centrada)", padding="12")
        lf_der.grid(row=3, column=0, columnspan=9, sticky="ew", pady=5)
        
        ttk.Label(lf_der, text="x0:").grid(row=0, column=0, padx=5)
        self.entry_der_x0 = ttk.Entry(lf_der, width=10)
        self.entry_der_x0.grid(row=0, column=1, padx=5)
        
        ttk.Label(lf_der, text="dx (paso h):").grid(row=0, column=2, padx=5)
        self.entry_der_dx = ttk.Entry(lf_der, width=10)
        self.entry_der_dx.grid(row=0, column=3, padx=5)
        
        ttk.Button(lf_der, text="Calcular Derivada", command=self.calcular_derivada).grid(row=0, column=4, padx=20)
        
        ttk.Label(lf_der, text="Derivada f'(x0):").grid(row=0, column=5, padx=5)
        self.lbl_res_der = ttk.Label(lf_der, text="---", font=("TkDefaultFont", 9, "bold"))
        self.lbl_res_der.grid(row=0, column=6, padx=5)

        # --- SECCIÓN 3: SISTEMA DE ECUACIONES (GAUSS) ---
        lf_sis = ttk.LabelFrame(main_frame, text="3. Sistema de 4 ecuaciones (Eliminación Gaussiana)", padding="12")
        lf_sis.grid(row=4, column=0, columnspan=9, sticky="ew", pady=5)
        
        self.entries_matriz = []
        self.entries_b = []
        
        for i in range(4):
            fila = []
            for j in range(4):
                e = ttk.Entry(lf_sis, width=6)
                e.grid(row=i+1, column=j, padx=2, pady=2)
                fila.append(e)
            self.entries_matriz.append(fila)
            
            eb = ttk.Entry(lf_sis, width=6)
            eb.grid(row=i+1, column=5, padx=10, pady=2)
            self.entries_b.append(eb)
            
        for j, txt in enumerate(['x1', 'x2', 'x3', 'x4']):
            ttk.Label(lf_sis, text=txt).grid(row=0, column=j)
        ttk.Label(lf_sis, text="b").grid(row=0, column=5)
        
        self.var_pivoteo = tk.BooleanVar(value=False)
        ttk.Radiobutton(lf_sis, text="Sin pivoteo", variable=self.var_pivoteo, value=False).grid(row=1, column=6, sticky="w", padx=10)
        ttk.Radiobutton(lf_sis, text="Con pivoteo parcial", variable=self.var_pivoteo, value=True).grid(row=2, column=6, sticky="w", padx=10)
        
        ttk.Button(lf_sis, text="Calcular Sistema", command=self.calcular_sistema).grid(row=3, column=6, padx=10)
        
        self.lbls_res_sis = []
        for i in range(4):
            lbl = ttk.Label(lf_sis, text=f"x{i+1} = ---", font=("TkDefaultFont", 9, "bold"))
            lbl.grid(row=i+1, column=7, sticky="w", padx=10)
            self.lbls_res_sis.append(lbl)

        # --- SECCIÓN 4: INTEGRAL (SIMPSON 1/3) ---
        lf_int = ttk.LabelFrame(main_frame, text="4. Integral definida (Simpson 1/3 compuesta)", padding="12")
        lf_int.grid(row=5, column=0, columnspan=9, sticky="ew", pady=5)
        
        ttk.Label(lf_int, text="Límite a:").grid(row=0, column=0, sticky="e")
        self.entry_int_a = ttk.Entry(lf_int, width=8)
        self.entry_int_a.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(lf_int, text="Límite b:").grid(row=0, column=2, sticky="e")
        self.entry_int_b = ttk.Entry(lf_int, width=8)
        self.entry_int_b.grid(row=0, column=3, sticky="w", padx=5)
        
        ttk.Label(lf_int, text="Subintervalos (n par):").grid(row=1, column=0, sticky="e")
        self.entry_int_n = ttk.Entry(lf_int, width=8)
        self.entry_int_n.grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Button(lf_int, text="Calcular Integral", command=self.calcular_integral).grid(row=0, column=4, rowspan=2, padx=15)
        
        ttk.Label(lf_int, text="Resultado Integral:").grid(row=0, column=5, sticky="e")
        self.lbl_res_int = ttk.Label(lf_int, text="---", font=("TkDefaultFont", 9, "bold"))
        self.lbl_res_int.grid(row=0, column=6, sticky="w")

    def toggle_raices_inputs(self):
        if self.var_metodo.get() == "newton":
            self.entry_raiz_x2.config(state="disabled")
        else:
            self.entry_raiz_x2.config(state="normal")

    def get_coefs(self):
        try:
            return [float(e.get()) for e in self.entries_coefs]
        except ValueError:
            raise ValueError("Por favor, complete todos los coeficientes del polinomio con números válidos.")

    def calcular_raiz(self):
        try:
            coefs = self.get_coefs()
            metodo = self.var_metodo.get()
            x1 = float(self.entry_raiz_x1.get())
            max_iter = int(self.entry_max_iter.get())
            
            cifras = None
            if self.var_tol.get() == "es":
                es = float(self.entry_es.get())
            else:
                cifras = int(self.entry_cifras.get())
                es = 0.5 * (10 ** (2 - cifras))
                
            if metodo == "biseccion":
                x2 = float(self.entry_raiz_x2.get())
                raiz, iters, ea, fraiz = biseccion(coefs, x1, x2, es, max_iter)
            elif metodo == "newton":
                raiz, iters, ea, fraiz = newton_raphson(coefs, x1, es, max_iter)
            elif metodo == "secante":
                x2 = float(self.entry_raiz_x2.get())
                raiz, iters, ea, fraiz = secante(coefs, x1, x2, es, max_iter)
                
            if cifras is not None:
                fmt = f"{{:.{cifras}g}}"
                self.lbl_res_raiz.config(text=fmt.format(raiz))
            else:
                self.lbl_res_raiz.config(text=f"{raiz:.6f}")
                
            self.lbl_res_iter.config(text=str(iters))
            self.lbl_res_ea.config(text=f"{ea:.6f}%")
            self.lbl_res_fraiz.config(text=f"{fraiz:.2e}")
            
        except ValueError as ve:
            messagebox.showerror("Error en Raíces", str(ve))
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {str(e)}")

    def calcular_derivada(self):
        try:
            coefs = self.get_coefs()
            x0 = float(self.entry_der_x0.get())
            dx = float(self.entry_der_dx.get())
            
            der = derivada_centrada(coefs, x0, dx)
            self.lbl_res_der.config(text=f"{der:.6f}")
            
        except ValueError as ve:
            messagebox.showerror("Error en Derivada", str(ve))
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {str(e)}")

    def calcular_sistema(self):
        try:
            A = [[float(e.get()) for e in fila] for fila in self.entries_matriz]
            b = [float(e.get()) for e in self.entries_b]
            pivoteo = self.var_pivoteo.get()
            
            x = eliminacion_gauss(A, b, pivoteo)
            for i, lbl in enumerate(self.lbls_res_sis):
                lbl.config(text=f"x{i+1} = {x[i]:.4f}")
        except ValueError as ve:
            messagebox.showerror("Error en Sistema Lineal", str(ve))
            for lbl in self.lbls_res_sis:
                lbl.config(text="---")

    def calcular_integral(self):
        try:
            coefs = self.get_coefs()
            a = float(self.entry_int_a.get())
            b = float(self.entry_int_b.get())
            n = int(self.entry_int_n.get())
            
            integral = simpson_13(coefs, a, b, n)
            self.lbl_res_int.config(text=f"{integral:.6f}")
        except ValueError as ve:
            messagebox.showerror("Error en Integral", str(ve))

    def cargar_valores_prueba(self):
        self.borrar_todo()
        vals_coefs = ['0', '1', '0', '-8', '0', '82', '-25']
        for e, v in zip(self.entries_coefs, vals_coefs): e.insert(0, v)
        
        self.var_metodo.set("biseccion")
        self.toggle_raices_inputs()
        
        self.entry_raiz_x1.insert(0, '0')
        self.entry_raiz_x2.insert(0, '1')
        self.var_tol.set("cifras")
        self.entry_es.insert(0, '0.5')
        self.entry_cifras.insert(0, '4')
        self.entry_max_iter.insert(0, '100')
        
        self.entry_der_x0.insert(0, '1')
        self.entry_der_dx.insert(0, '0.1')
        
        sis_A = [
            ['2', '2', '1', '-1'],
            ['2', '-2', '6', '-4'],
            ['2', '2', '1', '1'],
            ['1', '5', '-4', '-2']
        ]
        sis_b = ['5', '0', '13', '-9']
        for i in range(4):
            for j in range(4):
                self.entries_matriz[i][j].insert(0, sis_A[i][j])
            self.entries_b[i].insert(0, sis_b[i])
            
        self.entry_int_a.insert(0, '0')
        self.entry_int_b.insert(0, '2')
        self.entry_int_n.insert(0, '2')

    def borrar_todo(self):
        for e in self.entries_coefs: e.delete(0, tk.END)
        
        self.entry_raiz_x1.delete(0, tk.END)
        self.entry_raiz_x2.delete(0, tk.END)
        self.entry_es.delete(0, tk.END)
        self.entry_cifras.delete(0, tk.END)
        self.entry_max_iter.delete(0, tk.END)
        
        self.entry_der_x0.delete(0, tk.END)
        self.entry_der_dx.delete(0, tk.END)
        
        for fila in self.entries_matriz:
            for e in fila: e.delete(0, tk.END)
        for e in self.entries_b: e.delete(0, tk.END)
        
        self.entry_int_a.delete(0, tk.END)
        self.entry_int_b.delete(0, tk.END)
        self.entry_int_n.delete(0, tk.END)
        
        self.lbl_res_raiz.config(text="---")
        self.lbl_res_iter.config(text="---")
        self.lbl_res_ea.config(text="---")
        self.lbl_res_fraiz.config(text="---")
        self.lbl_res_der.config(text="---")
        self.lbl_res_int.config(text="---")
        for lbl in self.lbls_res_sis: lbl.config(text="---")

if __name__ == "__main__":
    root = tk.Tk()
    app = LaboVIApp(root)
    root.mainloop()