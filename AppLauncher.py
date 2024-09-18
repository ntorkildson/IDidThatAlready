import tkinter as tk
from tkinter import ttk, messagebox
import pystray

from PIL import Image, ImageTk

import os
import json
import shutil
import subprocess
from pathlib import Path

'''
Welcome to the IDTA ( I DID THAT ALREADY) app launcher. 
This is a template that should be pretty easy to extend, its not perfect, its not necessarily
the best way to do this, but its contained, looks decent, is straightforward to update and 
isnt over engineered with 2343 classes. 

'''


class StudioEnvironmentManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.studio_root = Path(self.config['studio_root'])
        self.environments_dir = self.studio_root / 'environments'
        self.shared_environments_dir = self.studio_root / 'shared_environments'
        self.environments_dir.mkdir(exist_ok=True)
        self.shared_environments_dir.mkdir(exist_ok=True)

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_python_version(self, app_name):
        return self.config['applications'][app_name]['python_version']

    def setup_shared_environment(self, python_version):
        env_dir = self.shared_environments_dir / f'python_{python_version}'

        if not env_dir.exists():
            self.create_shared_environment(python_version, env_dir)

        return env_dir

    def create_shared_environment(self, python_version, env_dir):
        # Create a new virtual environment
        subprocess.run(['python', '-m', 'venv', str(env_dir)])

        # Install shared dependencies
        pip = env_dir / 'bin' / 'pip'
        shared_deps = self.config.get('shared_dependencies', {}).get(python_version, {})
        for package, version in shared_deps.items():
            subprocess.run([str(pip), 'install', f'{package}=={version}'])

    def setup_app_environment(self, app_name, project=None):
        app_config = self.config['applications'][app_name]
        python_version = app_config['python_version']
        shared_env_dir = self.setup_shared_environment(python_version)

        app_env_dir = self.environments_dir / app_name / app_config['version']
        if project:
            app_env_dir = app_env_dir / project

        if not app_env_dir.exists():
            self.create_app_environment(app_name, app_env_dir, shared_env_dir, project)

        return app_env_dir, shared_env_dir

    def create_app_environment(self, app_name, app_env_dir, shared_env_dir, project=None):
        app_config = self.config['applications'][app_name]

        # Create a new virtual environment
        subprocess.run(['python', '-m', 'venv', str(app_env_dir)])

        # Install app-specific dependencies
        pip = app_env_dir / 'bin' / 'pip'
        for package, version in app_config['dependencies'].items():
            subprocess.run([str(pip), 'install', f'{package}=={version}'])

        # Install project-specific dependencies if any
        if project and project in self.config.get('projects', {}):
            project_deps = self.config['projects'][project].get('dependencies', {})
            for package, version in project_deps.items():
                subprocess.run([str(pip), 'install', f'{package}=={version}'])

        # Link shared libraries
        site_packages_dir = list(app_env_dir.glob('lib/python*/site-packages'))[0]
        shared_site_packages = list(shared_env_dir.glob('lib/python*/site-packages'))[0]

        with open(site_packages_dir / 'shared.pth', 'w') as f:
            f.write(str(shared_site_packages))

    def launch_application(self, app_name, project=None):
        app_config = self.config['applications'][app_name]
        app_env_dir, shared_env_dir = self.setup_app_environment(app_name, project)

        # Clean .pyc files
        self.clean_pyc_files(app_env_dir)
        self.clean_pyc_files(shared_env_dir)

        # Set environment variables
        os.environ.update(app_config.get('env_variables', {}))

        # Activate the app-specific environment
        activate_this = app_env_dir / 'bin' / 'activate_this.py'
        exec(open(str(activate_this)).read(), {'__file__': str(activate_this)})

        # Launch the application
        subprocess.Popen(app_config['launch_command'], shell=True)

    def add_shared_dependency(self, package, version, python_version):
        if 'shared_dependencies' not in self.config:
            self.config['shared_dependencies'] = {}
        if python_version not in self.config['shared_dependencies']:
            self.config['shared_dependencies'][python_version] = {}

        self.config['shared_dependencies'][python_version][package] = version
        self.save_config()

        # Reinstall shared environment
        shared_env_dir = self.shared_environments_dir / f'python_{python_version}'
        if shared_env_dir.exists():
            shutil.rmtree(shared_env_dir)
        self.setup_shared_environment(python_version)

    def clean_pyc_files(self, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.pyc'):
                    os.remove(os.path.join(root, file))

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)


class IDTA_Launcher:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("IDTA Launcher")
        self.window.geometry("1000x600")
        self.window.configure(bg="#2b2b2b")
        self.window.withdraw()

        self.current_tab = "Art"  # Set a default tab
        try:
            self.env_manager = StudioEnvironmentManager('studio_config.json')
        except Exception as e:
            messagebox.showerror("Configuration Error",
                                 f"Error loading configuration: {e}\nUsing default configuration.")
            self.env_manager = StudioEnvironmentManager('studio_config.json')

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

        software = self.load_software_config()

        current_software = software.get(self.current_tab, {})

        for i, (software_name, software_config) in enumerate(current_software.items()):
            frame = tk.Frame(software_frame, bg="#2b2b2b")
            frame.grid(row=i // 4, column=i % 4, padx=5, pady=5)

            # Load image (replace with actual software images)
            img = Image.new('RGB', (150, 200), color=(73, 109, 137))
            img = ImageTk.PhotoImage(img)

            tk.Label(frame, image=img).pack()
            tk.Label(frame, text=software_name, bg="#2b2b2b", fg="white").pack()
            ttk.Button(frame, text="Launch",
                       command=lambda name=software_name, config=software_config: self.launch_software(name,
                                                                                                       config)).pack(
                pady=5)
            frame.image = img  # Keep a reference

    def load_software_config(self):
        # Load software configuration from studio_config.json
        return self.env_manager.config.get('applications', {})

    def change_tab(self, tab_name):
        self.current_tab = tab_name
        self.update_main_content()

    def launch_software(self, software_name, software_config):
        try:
            # Get the launch command from the software configuration
            launch_command = software_config.get('launch_command')
            if not launch_command:
                raise ValueError(f"No launch command specified for {software_name}")

            # Launch the application
            subprocess.Popen(launch_command, shell=True)
            print(f"Launching {software_name}")
        except Exception as e:
            messagebox.showerror("Launch Error", f"Error launching {software_name}: {e}")

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