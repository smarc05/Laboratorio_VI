import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# URL del repositorio de GitHub: https://github.com/smarc05/Laboratorio_VI.git

def evaluar_poli(coefs, x):
    return sum(c * (x**(6-i)) for i, c in enumerate(coefs))

def evaluar_der_poli(coefs, x):
    der_coefs = [c * (6-i) for i, c in enumerate(coefs[:-1])]
    return sum(c * (x**(5-i)) for i, c in enumerate(der_coefs))

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

class LaboVIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Laboratorio VI.1 - Cálculo numérico")
        self.crear_widgets()
        self.cargar_valores_prueba()

    def crear_widgets(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Coeficientes\ndel polinomio", justify="center").grid(row=0, column=0, rowspan=2, padx=10)
        self.entries_coefs = []
        labels_coefs = ['a6', 'a5', 'a4', 'a3', 'a2', 'a1', 'a0']
        for i, lbl in enumerate(labels_coefs):
            e = ttk.Entry(main_frame, width=6)
            e.grid(row=0, column=i+1, padx=2, pady=5)
            ttk.Label(main_frame, text=lbl).grid(row=1, column=i+1)
            self.entries_coefs.append(e)
            
        ttk.Button(main_frame, text="Borrar", command=self.borrar_todo).grid(row=0, column=8, padx=15)
        ttk.Button(main_frame, text="Valores de prueba", command=self.cargar_valores_prueba).grid(row=1, column=8, padx=15)

        lf_raices = ttk.LabelFrame(main_frame, text="Cálculo de una raíz", padding="15")
        lf_raices.grid(row=2, column=0, columnspan=9, sticky="ew", pady=15)

        self.var_metodo = tk.StringVar(value="biseccion")
        ttk.Radiobutton(lf_raices, text="Bisección", variable=self.var_metodo, value="biseccion", command=self.toggle_raices_inputs).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(lf_raices, text="Newton-Raphson", variable=self.var_metodo, value="newton", command=self.toggle_raices_inputs).grid(row=0, column=2, padx=5)
        ttk.Radiobutton(lf_raices, text="Secante", variable=self.var_metodo, value="secante", command=self.toggle_raices_inputs).grid(row=0, column=3, padx=5)

        ttk.Label(lf_raices, text="xl / x0 / x-1:").grid(row=1, column=0, sticky="e", pady=10)
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

        ttk.Button(lf_raices, text="Calcular", command=self.calcular_raiz).grid(row=1, column=6, rowspan=2, padx=20)

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

    def toggle_raices_inputs(self):
        if self.var_metodo.get() == "newton":
            self.entry_raiz_x2.config(state="disabled")
        else:
            self.entry_raiz_x2.config(state="normal")

    def calcular_raiz(self):
        try:
            coefs = [float(e.get()) for e in self.entries_coefs]
            
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
            messagebox.showerror("Error de Entrada/Matemático", str(ve))
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {str(e)}")

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

    def borrar_todo(self):
        for e in self.entries_coefs: 
            e.delete(0, tk.END)
        self.entry_raiz_x1.delete(0, tk.END)
        self.entry_raiz_x2.delete(0, tk.END)
        self.entry_es.delete(0, tk.END)
        self.entry_cifras.delete(0, tk.END)
        self.entry_max_iter.delete(0, tk.END)
        
        self.lbl_res_raiz.config(text="---")
        self.lbl_res_iter.config(text="---")
        self.lbl_res_ea.config(text="---")
        self.lbl_res_fraiz.config(text="---")

if __name__ == "__main__":
    root = tk.Tk()
    app = LaboVIApp(root)
    root.mainloop()