import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===============================
# Load trained models
# ===============================
try:
    rf_model = joblib.load("random_forest_model.joblib")
    xgb_model = joblib.load("xgb_model.joblib")
except FileNotFoundError:
    messagebox.showerror("Error", "Trained model files not found. Run trainandeval.py first.")
    exit()

# Required features
required_features = [
    'Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3',
    'Wilderness_Area4'
] + [f'Soil_Type{i}' for i in range(1, 41)]

# ===============================
# Functions
# ===============================
def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("GZ Files", "*.gz")])
    if not file_path:
        return

    # Load CSV or GZ
    if file_path.endswith(".gz"):
        df = pd.read_csv(file_path, header=None, names=required_features+["Cover_Type"])
    else:
        df = pd.read_csv(file_path)

    # Check required features
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        messagebox.showerror("Missing Columns", f"Missing columns:\n{missing_cols}")
        return

    # Split X and y
    X = df[required_features]
    y_true = df["Cover_Type"] if "Cover_Type" in df.columns else None

    # Predict
    try:
        rf_preds = rf_model.predict(X)
        xgb_preds = xgb_model.predict(X)
        if y_true is not None:
            xgb_preds += 1  # revert XGBoost labels to 1-7
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))
        return

    # Clear previous plots
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    # Show results
    if y_true is not None:
        show_results("Random Forest", y_true, rf_preds)
        show_results("XGBoost", y_true, xgb_preds)
    else:
        messagebox.showinfo("Predictions Done", "CSV uploaded without actual labels.\nPredictions are ready.")

def show_results(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    # Text output
    text_box.insert(tk.END, f"\n{model_name} Results:\n")
    text_box.insert(tk.END, f"Accuracy: {acc:.4f}\n")
    text_box.insert(tk.END, report + "\n")
    text_box.see(tk.END)

    # Frame for this model's plots
    model_frame = tk.Frame(scrollable_frame, bd=2, relief=tk.GROOVE)
    model_frame.pack(pady=10, padx=10, fill="x")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = plt.Figure(figsize=(4,4))
    ax_cm = fig_cm.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title(f"{model_name} - Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    canvas_cm = FigureCanvasTkAgg(fig_cm, master=model_frame)
    canvas_cm.draw()
    canvas_cm.get_tk_widget().pack(side="left", padx=10, pady=5)

    # Predicted vs Actual counts
    df_compare = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    fig_count = plt.Figure(figsize=(4,4))
    ax_count = fig_count.add_subplot(111)
    sns.countplot(x="Actual", hue="Predicted", data=df_compare, palette="Set2", ax=ax_count)
    ax_count.set_title(f"{model_name} - Predicted vs Actual")
    canvas_count = FigureCanvasTkAgg(fig_count, master=model_frame)
    canvas_count.draw()
    canvas_count.get_tk_widget().pack(side="left", padx=10, pady=5)

# ===============================
# GUI Layout
# ===============================
root = tk.Tk()
root.title("Forest Cover Prediction GUI")
root.geometry("950x850")

btn_load = tk.Button(root, text="Upload CSV/GZ for Prediction", command=load_csv, bg="lightblue")
btn_load.pack(pady=10)

text_box = scrolledtext.ScrolledText(root, height=12, width=115)
text_box.pack(pady=10)

# Scrollable canvas for plots
plot_canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=plot_canvas.yview)
scrollable_frame = tk.Frame(plot_canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))
)

plot_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
plot_canvas.configure(yscrollcommand=scrollbar.set)

plot_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

root.mainloop()
