import tkinter as tk
from tkinter import ttk
import pystray
from PIL import Image, ImageTk
import subprocess
import os

'''
Welcome to the IDTA ( I DID THAT ALREADY) app launcher. 
This is a template that should be pretty easy to extend, its not perfect, its not necessarily
the best way to do this, but its contained, looks decent, is straightforward to update and 
isnt over engineered with 2343 classes. 

'''

class IDTA_Launcher:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("IDTA Launcher")
        self.window.geometry("1000x600")
        self.window.configure(bg="#2b2b2b")
        self.window.withdraw()

        self.current_tab = "Home"
        self.setup_ui()

        icon_path = 'AppLauncherIcon.png'  # Replace with your icon path
        image = Image.open(icon_path) if os.path.exists(icon_path) else Image.new('RGB', (16, 16), color=(73, 109, 137))
        menu = (pystray.MenuItem('Open', self.show_window),
                pystray.MenuItem('Exit', self.exit_app))
        self.icon = pystray.Icon("IDTA Launcher", image, "IDTA Launcher", menu)

    def setup_ui(self):
        # Left sidebar
        self.sidebar = tk.Frame(self.window, bg="#1e1e1e", width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(self.sidebar, text="IDTA LAUNCHER", foreground="white", background="#1e1e1e",
                  font=("Arial", 16, "bold")).pack(pady=20)

        buttons = ["Art", "Techart", "Engineers"]
        for btn_text in buttons:
            ttk.Button(self.sidebar, text=btn_text, command=lambda t=btn_text: self.change_tab(t)).pack(pady=5, padx=10,
                                                                                                        fill=tk.X)

        ttk.Label(self.sidebar, text="Docs", foreground="white", background="#1e1e1e",
                  font=("Arial", 12, "bold")).pack(pady=(20, 5))
        player_buttons = ["Jira", "Wiki", "Calendar", "Other?"]
        for btn_text in player_buttons:
            ttk.Button(self.sidebar, text=btn_text).pack(pady=2, padx=10, fill=tk.X)

        # Main content area
        self.main_area = tk.Frame(self.window, bg="#2b2b2b")
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.update_main_content()

    def update_main_content(self):
        for widget in self.main_area.winfo_children():
            widget.destroy()

        # Top bar
        top_bar = tk.Frame(self.main_area, bg="#2b2b2b")
        top_bar.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_bar, text=self.current_tab, foreground="white", background="#2b2b2b",
                  font=("Arial", 14, "bold")).pack(side=tk.LEFT)

        # Software grid
        software_frame = tk.Frame(self.main_area, bg="#2b2b2b")
        software_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

#todo: hard coded not great, but cant be bothered to fix this in a template.
        software = {
            "Art": [
                ("Krita", r"C:\Program Files\Krita (x64)\bin\krita.exe"),
                ("Blender", r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"),
                ("Houdini", r"C:\Program Files\Side Effects Software\Houdini 19.0.383\bin\houdini.exe"),
                ("ZBrush", r"C:\Program Files\Pixologic\ZBrush 2022\ZBrush.exe")
            ],
            "Techart": [
                ("Maya", "maya_path"),
                ("Substance Painter", "substance_painter_path"),
                ("Unreal Engine", "unreal_engine_path"),
                ("Rider", "rider_path")
            ],
            "Engineers": [
                ("Rider", "rider_path"),
                ("Unreal Engine", "unreal_path"),

            ]
        }

        current_software = software.get(self.current_tab, [])

        for i, (software_name, software_path) in enumerate(current_software):
            frame = tk.Frame(software_frame, bg="#2b2b2b")
            frame.grid(row=i // 4, column=i % 4, padx=5, pady=5)

            # Load image (replace with actual software images)
            img = Image.new('RGB', (150, 200), color=(73, 109, 137))
            img = ImageTk.PhotoImage(img)

            tk.Label(frame, image=img).pack()
            tk.Label(frame, text=software_name, bg="#2b2b2b", fg="white").pack()
            ttk.Button(frame, text="Launch", command=lambda path=software_path: self.launch_software(path)).pack(pady=5)
            frame.image = img  # Keep a reference

    def change_tab(self, tab_name):
        self.current_tab = tab_name
        self.update_main_content()

    def launch_software(self, software_path):
        try:
            subprocess.Popen(software_path)
            print(f"Launching {software_path}")
        except FileNotFoundError:
            print(f"Error: {software_path} not found.")
        except Exception as e:
            print(f"Error launching {software_path}: {e}")

    def run(self):
        self.icon.run_detached()
        self.window.mainloop()

    def show_window(self, *args):
        self.window.deiconify()

    def exit_app(self, *args):
        self.icon.stop()
        self.window.quit()


if __name__ == '__main__':
    app = IDTA_Launcher()
    app.run()