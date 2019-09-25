# -*- coding: utf-8 -*-
"""
screen2table is a simple Windows GUI app to generate geometric
data table (coordinates or level-width) by tracing shapes
displayed on the screen.

This module contains the GUI app (based on tkinter).

author: Maxime Liquet
"""
import webbrowser
import tkinter as tk
import configparser
import pathlib

# Adapted from https://pynput.readthedocs.io/en/latest/mouse.html#monitoring-the-mouse  # noqa
import pynput
# See https://matplotlib.org/tutorials/introductory/usage.html
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from screen2table import helpers

# Initiliaze some data from the configs.ini file.
configs = configparser.ConfigParser()
configs.read(pathlib.Path(__file__).parent / "configs.cfg")
PLOT_FONTSIZE = configs.getint("DISPLAY", "PLOT_FONTSIZE")
DIGITS_DISPLAYED_SUMMARY = configs.getint("DISPLAY", "DIGITS_DISPLAYED_SUMMARY")
RECORD_BUTTON = configs["CONTROL"]["RECORD_BUTTON"]
STOP_BUTTON = configs["CONTROL"]["STOP_BUTTON"]
DIGITS_CLIPBOARD = configs.getint("OUTPUT", "DIGITS_CLIPBOARD")


# Inspired from https://stackoverflow.com/questions/31440167/placing-plot-on-tkinter-main-window-in-python  # noqa
class App:

    def __init__(self, master):
        self.master = master
        self.master.title("screen2table")
        # From https://stackoverflow.com/questions/3295270/overriding-tkinter-x-button-control-the-button-that-close-the-window  # noqa
        self.master.protocol("WM_DELETE_WINDOW", self.toquit)
        # Used to keep the window always on top, useful when clicking
        # See https://stackoverflow.com/questions/1892339/how-to-make-a-tkinter-window-jump-to-the-front  # noqa
        self.master.attributes("-topmost", True)
        # Help that redirects to GitHub
        github_url = "https://github.com/maximlt/screen2table"
        self.help_label = tk.Label(
            self.master,
            text=github_url,
            fg="blue", cursor="hand2")
        self.help_label.bind(
            "<Button-1>",
            lambda e: webbrowser.open_new(github_url))
        # Trace button for the user to start tracing by left-clicking.
        self.trace_button = tk.Button(self.master, text="Trace")
        self.trace_button.bind(
            "<ButtonPress>",
            lambda e: self.hide_window()
        )
        self.trace_button.bind(
            "<ButtonRelease>",
            lambda e: self.trace()
        )

        # Two types of shape processed: culvert (closed-shape)
        # and cross-section (open-shape).
        self.mode = tk.IntVar()
        self.culvert_radiob = tk.Radiobutton(
            self.master,
            text='Culvert',
            variable=self.mode,
            value=0
        )
        self.xs_radiob = tk.Radiobutton(
            self.master,
            text='Cross-section',
            variable=self.mode,
            value=1
        )

        # Place the widgets.
        self.help_label.grid(row=0, column=0, columnspan=3)
        self.trace_button.grid(row=1, column=0, sticky="w")
        self.culvert_radiob.grid(row=1, column=1, sticky="w")
        self.culvert_radiob.select()
        self.xs_radiob.grid(row=1, column=2, sticky="w")

        # Make the window resizable.
        for col in range(3):
            tk.Grid.columnconfigure(self.master, col, weight=1)
        tk.Grid.rowconfigure(self.master, 0, weight=1)

    # From https://matplotlib.org/gallery/user_interfaces/embedding_in_tk_sgskip.html  # noqa
    def toquit(self):
        """Properly close the app.
        """
        # To stop the main loop.
        self.master.quit()
        # This is necessary on Windows to prevent:
        # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        self.master.destroy()

    def hide_window(self):
        """Hide the window.
        """
        self.master.withdraw()

    def add_user_entries(self, kind):
        """
        Add to the GUI the widgets required to allow the user to provide
        the shape dimensions to the code, an output summary, and a status bar.
        """
        # Number of characters for the entries
        entry_width = 10

        # Delete the widgets if there were already created.
        # There must be a better way.
        widgets = [
            'usergeom_label',
            'userx_label',
            'userminx_entry',
            'usermaxx_entry',
            'userz_label',
            'userminz_entry',
            'usermaxz_entry',
            'userparam_label',
            'userangle_label',
            'userangle_entry',
            'process_data',
            'summary_label',
            'summaryvalue_entry',
            'status_label',
            'status_entry',
        ]
        for widget in widgets:
            if hasattr(self, widget):
                getattr(self, widget).destroy()

        # Culvert geometry
        self.usergeom_label = tk.Label(self.master, text="Geometry extent")
        self.userx_label = tk.Label(self.master, text="Min X / Max X [m]:")
        self.userminx_entry = tk.Entry(self.master, width=entry_width)
        self.usermaxx_entry = tk.Entry(self.master, width=entry_width)
        self.userz_label = tk.Label(self.master, text="Min Z / Max Z [m]:")
        self.userminz_entry = tk.Entry(self.master, width=entry_width)
        self.usermaxz_entry = tk.Entry(self.master, width=entry_width)

        # Some more parameters.
        self.userparam_label = tk.Label(self.master, text="Parameters")
        self.userangle_label = tk.Label(self.master, text="Angle [°]:")
        self.userangle_entry = tk.Entry(self.master, width=entry_width)

        # Add a process button to trigger the calculation.
        self.process_data = tk.Button(
            self.master,
            text="Process",
            command=self.pick_process_mode
        )

        # Add a label and an entry for the displayed summary.
        summary_text = {
            'culvert': "Area [m\xb2]:",
            'xs': 'Length [m]',
        }
        self.summary_label = tk.Label(self.master, text=summary_text[kind])
        self.summary_string = tk.StringVar()
        self.summaryvalue_entry = tk.Entry(
            self.master,
            textvariable=self.summary_string,
            state="readonly",
            width=entry_width
        )

        # Add a simple status bar.
        self.status_label = tk.Label(self.master, text="Status:")
        self.status_string = tk.StringVar()
        self.status_entry = tk.Entry(
            self.master,
            textvariable=self.status_string,
            state="readonly"
        )

        # Origin of all the user params widgets on the widget grid.
        ro, co = 3, 0

        # Place the labels, entries and buttons.
        self.usergeom_label.grid(row=ro, column=co, columnspan=3)
        self.userx_label.grid(row=ro+1, column=co, sticky='e')
        self.userminx_entry.grid(row=ro+1, column=co+1)
        self.usermaxx_entry.grid(row=ro+1, column=co+2)
        self.userz_label.grid(row=ro+2, column=co, sticky='e')
        self.userminz_entry.grid(row=ro+2, column=co+1)
        self.usermaxz_entry.grid(row=ro+2, column=co+2)
        self.userparam_label.grid(row=ro+4, column=co, columnspan=3)
        self.userangle_label.grid(row=ro+6, column=co, sticky='e')
        self.userangle_entry.grid(row=ro+6, column=co+1)
        self.process_data.grid(row=ro+7, column=co, sticky='w')
        self.summary_label.grid(row=ro+7, column=co+1, sticky='e')
        self.summaryvalue_entry.grid(row=ro+7, column=co+2)
        self.status_label.grid(row=ro+8, column=co, sticky='w')
        self.status_entry.grid(row=ro+8,
                               column=co+1,
                               columnspan=2,
                               sticky='ew')

        # Allow the widgets to be resizable.
        for row in range(1, ro + 8 + 1):
            tk.Grid.rowconfigure(self.master, row, weight=1)
        for col in range(1, co + 3 + 1):
            tk.Grid.columnconfigure(self.master, col, weight=1)

        # Set default values (only for the angle parameter).
        self.userangle_entry.insert(0, 0)

    def plot(self, x1, y1, plot_type, x2=None, y2=None):
        """Plot the culvert shape or the level-width table.

        Inspired from:
        https://stackoverflow.com/questions/47602364/updating-canvas-in-tkinter
        """
        # Attempt to destroy the canvas created when plotting was already
        # done. Not sure if this is the right way to do it but it seems
        # to effectively destroy the widget.
        try:
            self.canvas.destroy()
        except AttributeError:
            pass

        self.fig = Figure(figsize=(3, 3), tight_layout=True)
        ax = self.fig.add_subplot(111)

        def set_title_labels(
            title,
            xlabel,
            ylabel,
            ax=ax,
            fontsize=PLOT_FONTSIZE
        ):
            """Set the title, xlabel et ylabel of an axes."""
            ax.set_title(title, fontsize=fontsize)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)

        if plot_type == "culvert":
            set_title_labels(
                "Culvert preview",
                "Horizontal Pixel",
                "Vertical* Pixel",
            )
            ax.plot(x1, y1, linestyle='-', marker='o')
            ax.plot(x2, y2,
                    linestyle='None', marker='.', markeredgecolor='red')

        elif plot_type == "table":
            set_title_labels(
                "Level-width table",
                "Width (m)",
                "Level (m)",
            )
            ax.plot(x1, y1, marker='.', color='black')

        elif plot_type == "xs_screen":
            set_title_labels(
                "Cross-Section preview",
                "Horizontal Pixel",
                "Vertical* Pixel",
            )
            ax.plot(x1, y1, marker='o')

        elif plot_type == "xs_real":
            set_title_labels(
                "Scaled cross-section",
                "Distance (m)",
                "Level (m)",
            )
            ax.plot(x1, y1, marker='o', color='black')

        ax.tick_params(labelsize=PLOT_FONTSIZE)

        # Create the object that will store the figure in a tk-like format.
        canvastkagg = FigureCanvasTkAgg(self.fig, master=self.master)
        # Get a tk canvas.
        self.canvas = canvastkagg.get_tk_widget()
        # Place it onto the grid.
        self.canvas.grid(
            row=2,
            columnspan=3,
            sticky="nesw"
        )

    def record_coordinates_from_screen(self):
        """Record XY screen coordinates from user clicks,
        and display the window again.
        """
        def store_xz_on_click(x, y, button, pressed):
            """Callback for when the mouse is clicked.

            Record the screen coordinates (in pixel) of the
            user's clicks and terminate the recording.
            """
            record_button = getattr(pynput.mouse.Button, RECORD_BUTTON)
            stop_button = getattr(pynput.mouse.Button, STOP_BUTTON)
            # Do something when the user clicks.
            if pressed:
                # Left button: Record xy screen coordinates.
                if button == record_button:
                    x_from_screen.append(x)
                    y_from_screen.append(y)
                # Right button: End the recording.
                if button == stop_button:
                    return False
        # The screen coordinates will be stored in these lists.
        x_from_screen, y_from_screen = [], []
        # Collect events until released
        with pynput.mouse.Listener(on_click=store_xz_on_click) as listener:
            listener.join()
        return (x_from_screen, y_from_screen)

    def trace(self):
        """Pick the record mode and display the window again."""
        self.pick_record_mode()
        self.master.deiconify()

    def pick_record_mode(self):
        """ Get the mode picked by the user (culvert or cross-section)
        and run the right function"""
        user_mode = self.mode.get()
        if user_mode == 0:
            self.record_culvert()
        else:
            self.record_xs()

    def pick_process_mode(self):
        """ Get the mode picked by the user (culvert or cross-section)
        and run the right function"""
        user_mode = self.mode.get()
        if user_mode == 0:
            self.process_culvert()
        else:
            self.process_xs()

    def status_to_error(self, kind):
        """
        Update the status entry with an error message.
        """
        txt_error = {
            'trace': "Tracing Error",
            'process': "Parameter Error",
        }
        if not hasattr(self, 'status_string'):
            self.status_string = tk.StringVar()
        self.status_string.set(txt_error[kind])

    def record_culvert(self):
        """
        Record the coordinates clicked by the user, process the data
        generated so that it can be plotted in the GUI and later processed to
        create a level-width table.
        """
        # Record the culvert x and y  coordinates
        x_from_screen, y_from_screen = self.record_coordinates_from_screen()
        # Custom object to store xz clicked points.
        self.screendata = helpers.ScreenData(
            x_from_screen,
            y_from_screen,
            'culvert'
        )
        # Call the main function that transforms the user xz coordinates
        # into an array that can be plotted and an array that can be
        # later processed to calculate a level-width table.
        self.screendata.process_screen_culvert()
        if not self.screendata.is_ok:
            # Change the status bar.
            self.status_to_error('trace')
            # Check which error was detected and prompt a message box.
            for val in self.screendata.dict_tkwarn.values():
                if val['is_error']:
                    tk.messagebox.showwarning(
                        val['tktxt'].title,
                        val['tktxt'].error_message
                    )
            return

        # Call the plot function to display the culvert.
        # It may not be exactly the one traced by the user, because
        # points on the same level are slightly modified. However,
        # just by a little, so the outcome is pretty close.
        # The location of the interpolated points are also plotted.
        self.plot(self.screendata.xz_to_plot[:, 0],
                  self.screendata.xz_to_plot[:, 1],
                  "culvert",
                  self.screendata.xzinterp[:, 0],
                  self.screendata.xzinterp[:, 1])

        # Add the user entries now for the processing step.
        self.add_user_entries('culvert')

        # Display the number of points clicked by the user
        txt_culv = f"Culvert mode: {self.screendata.nb_points} points."
        self.status_string.set(txt_culv)

    def record_xs(self):
        """
        Record the coordinates clicked by the user, process the data
        generated so that it can be plotted in the GUI and later processed to
        create a distance-level table for the drawn cross-section.
        """
        # Record the X and Y cross-section coordinates.
        x_from_screen, y_from_screen = self.record_coordinates_from_screen()
        # Custom object to store xz clicked points.
        self.screendata = helpers.ScreenData(
            x_from_screen,
            y_from_screen,
            'xs'
        )
        # Call the main function that transforms the user xz coordinates
        # into an array that can be plotted and an array that can be
        # later processes to calculate a level-width table.
        self.screendata.process_screen_xs()
        if not self.screendata.is_ok:
            # Change the status bar.
            self.status_to_error('trace')
            # Check which error was detected and prompt a message box.
            for val in self.screendata.dict_tkwarn.values():
                if val['is_error']:
                    tk.messagebox.showwarning(
                        val['tktxt'].title,
                        val['tktxt'].error_message
                    )
            return

        # Call the plot function to display the shape of
        # the cross-section drawn by the user.
        self.plot(
            self.screendata.xz_to_plot[:, 0],
            self.screendata.xz_to_plot[:, 1],
            "xs_screen"
        )

        # Add the user entries now.
        self.add_user_entries('xs')

        # Display the number of points clicked by the user.
        txt_xs = f"Cross-section mode: {self.screendata.nb_points} points."
        self.status_string.set(txt_xs)

    def validate_user_param(self):
        """Validate the user entered geometry extent and parameters.

        Returns
        -------
        dict
            User parameters converted to float, or None
            if an error was detected.
        """
        # Retrieve the parameters entered by the user.
        # They are str.
        userparam_dict = {
            'minx': self.userminx_entry.get(),
            'maxx': self.usermaxx_entry.get(),
            'minz': self.userminz_entry.get(),
            'maxz': self.usermaxz_entry.get(),
            'angle': self.userangle_entry.get()
        }
        # Create a custom object to conduct the validation.
        user_param_obj = helpers.UserParam(userparam_dict)
        # Check whether the data is OK for the subsequent calculation.
        rules_dict = user_param_obj.validate()
        for rule in rules_dict.values():
            if not rule.is_ok:
                # If False, pop a warning box with an error message
                # and stop the method.
                tk.messagebox.showwarning("Error", rule.err_message)
                return

        return user_param_obj.convert_dict

    def process_culvert(self):
        """Check the user data, calculate the Width-Level table,
        plot it, copy it to the clipboard, calculate the area.
        """
        # If for some reasons the output of the screen data processing
        # for culverts (interpolated xz) is not available,
        # throw an error message.
        if self.screendata.xzinterp is None:
            tk.messagebox.showwarning(
                'Error',
                'No culvert data, trace again.'
            )
            # Change the status bar.
            self.status_to_error('trace')
            return

        # Check the validity of the user entered parameters.
        user_param = self.validate_user_param()
        if user_param is None:
            # Change the status bar to report the error.
            self.status_to_error('process')
            return

        # Get the Level-Width table with the user dimensions
        # and an array used to plot the Level-Widh table as in Mike (DHI).
        zw_real, zw_plot = helpers.polygon_to_levelwidth_table(
            self.screendata.xzinterp,
            user_param,
        )

        # Plot the table.
        self.plot(x1=zw_plot[:, 1], y1=zw_plot[:, 0], plot_type="table")

        # Calculate and display the rounded area of the culvert.
        # The area is calculated based on the level-width table, from
        # which some points might have been removed if they were too close
        # to each other. The resulting area might differ from the original
        # geometry area (but just by a little in theory).
        culvert_area = helpers.calc_area(zw_real)
        culvert_area_rounded = round(culvert_area, DIGITS_DISPLAYED_SUMMARY)
        culvert_area_to_display = (
            f"{culvert_area_rounded:.{DIGITS_DISPLAYED_SUMMARY}f}"
        )
        self.summary_string.set(culvert_area_to_display)

        # Copy a rounded table to clipboard.
        helpers.to_clipboard_for_excel(zw_real, decimals=DIGITS_CLIPBOARD)

        # Display that processing is over.
        txt_success = f"LW table copied to clipboard."
        self.status_string.set(txt_success)

    def process_xs(self):
        """Check the user data, plot the real cross-section
        and copy it to the clipboard.
        """
        # If for some reasons the output of the screen data processing
        # for xs is not available, throw an error message.
        if self.screendata.xz is None:
            tk.messagebox.showwarning('Error',
                                      'No cross-section data, trace again.')
            # Change the status bar.
            self.status_to_error('trace')
            return

        # Check the validity of the user entered parameters.
        user_param = self.validate_user_param()
        if user_param is None:
            # Change the status bar to report the error.
            self.status_to_error('process')
            return

        # Transform the 'screen' cross-sections into a cross-section
        # based on the user defined dimensions.
        xs_xz = helpers.scale_to_realdim(self.screendata.xz, user_param)

        # Plot the cross-section.
        self.plot(x1=xs_xz[:, 0], y1=xs_xz[:, 1], plot_type='xs_real')

        # Calculate and display the rounded area of the culvert.
        xs_length = helpers.calc_length(xs_xz)
        xs_length_rounded = round(xs_length, DIGITS_DISPLAYED_SUMMARY)
        xs_length_to_display = (
            f"{xs_length_rounded:.{DIGITS_DISPLAYED_SUMMARY}f}"
        )
        self.summary_string.set(xs_length_to_display)

        # Copy the real, rounded cross-section coordinates to clipboard.
        helpers.to_clipboard_for_excel(xs_xz, DIGITS_CLIPBOARD)

        # Display that the processing is over
        txt_success = "Cross-section copied to clipboard."
        self.status_string.set(txt_success)


def main():
    """For declaring an entry point while packaging the app."""
    # Create a Tk object.
    root = tk.Tk()
    # Start the app with that object.
    App(root)
    # Keep it open and running in the mainloop.
    root.mainloop()


if __name__ == "__main__":
    main()
