# labeler_with_zoom_scroll.py
import sys,os
import random
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime

# -------------------------
# Configurable parameters
# -------------------------
PREVIEW_SIZE = (224, 224)  # internal working size for images & masks
LOG_FILENAME = "log.csv"   # saved inside labeled/ folder
# -------------------------
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pupil Presence Classification and Segmentation Tool")

        # Ask for main folder
        self.main_folder = filedialog.askdirectory(title="Select Main Folder")
        if not self.main_folder:
            messagebox.showerror("Error", "No folder selected, exiting.")
            root.destroy()
            return

        # prepare labeled folders inside main folder
        self.labeled_folder = os.path.join(self.main_folder, "labeled")
        self.true_folder = os.path.join(self.labeled_folder, "true")
        self.false_folder = os.path.join(self.labeled_folder, "false")
        self.masks_folder = os.path.join(self.labeled_folder, "masks")
        os.makedirs(self.true_folder, exist_ok=True)
        os.makedirs(self.false_folder, exist_ok=True)
        os.makedirs(self.masks_folder, exist_ok=True)

        # log file (keep in memory, rewrite on changes)
        self.log_path = os.path.join(self.labeled_folder, LOG_FILENAME)
        self.log_entries = []  # list of dicts
        self.processed_images = set()  # full paths of images that are classified true/false
        if os.path.exists(self.log_path):
            with open(self.log_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.log_entries.append(row)
                    if row.get("decision") in ("true", "false"):
                        self.processed_images.add(row.get("image_path"))

        # UI layout: left list, middle scrollable canvas, right buttons
        self.subfolders = [entry.path for entry in os.scandir(self.main_folder) if entry.is_dir() and entry.name != "labeled"]

        # ----- Left: subfolder list (resizable) -----
        self.left_frame = tk.Frame(root)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)  # center grows

        self.subfolder_list = tk.Listbox(self.left_frame, exportselection=False)
        self.subfolder_list.pack(side="left", fill="both", expand=True)
        self.subfolder_scroll_y = tk.Scrollbar(self.left_frame, orient="vertical", command=self.subfolder_list.yview)
        self.subfolder_scroll_y.pack(side="right", fill="y")
        self.subfolder_list.config(yscrollcommand=self.subfolder_scroll_y.set)

        # horizontal scrollbar for long names
        self.subfolder_scroll_x = tk.Scrollbar(root, orient="horizontal", command=self.subfolder_list.xview)
        self.subfolder_scroll_x.grid(row=1, column=0, sticky="ew")
        self.subfolder_list.config(xscrollcommand=self.subfolder_scroll_x.set)

        for sf in self.subfolders:
            self.subfolder_list.insert(tk.END, os.path.basename(sf))
        self.subfolder_list.bind("<<ListboxSelect>>", self.on_subfolder_select)

        # ----- Middle: scrollable canvas for image -----
        self.canvas_frame = tk.Frame(root, bg="black")
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(2, weight=0)

        # Canvas + scrollbars
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.hbar = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.vbar = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.hbar.pack(side="bottom", fill="x")
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind events
        # For zoom vs scroll behavior: hold Ctrl while using wheel to zoom.
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)    # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)      # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)      # Linux scroll down

        # For panning by dragging with middle mouse: use scan_mark/scan_dragto
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        # Left-click -> add point (mapped to image coordinates)
        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        # Right-click -> preview ellipse mask
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)

        # ----- Right: controls -----
        self.ctrl_frame = tk.Frame(root, width=160)
        self.ctrl_frame.grid(row=0, column=2, sticky="ns")
        self.ctrl_frame.grid_propagate(False)

        tk.Button(self.ctrl_frame, text="True (save)", command=self.mark_true).pack(fill="x", pady=2, padx=4)
        tk.Button(self.ctrl_frame, text="False (save)", command=self.mark_false).pack(fill="x", pady=2, padx=4)
        tk.Button(self.ctrl_frame, text="Skip (later)", command=self.skip_image).pack(fill="x", pady=2, padx=4)
        tk.Button(self.ctrl_frame, text="Undo last point", command=self.undo_point).pack(fill="x", pady=2, padx=4)
        tk.Button(self.ctrl_frame, text="Undo classification", command=self.undo_classification).pack(fill="x", pady=2, padx=4)
        tk.Button(self.ctrl_frame, text="Redo classification", command=self.redo_classification).pack(fill="x", pady=2, padx=4)

        # State
        self.current_folder = None
        self.available_images = []     # images (full paths) available to pick in current subfolder (not processed)
        self.current_image_path = None
        self.base_image = None         # PIL image at PREVIEW_SIZE (224x224)
        self.zoom_factor = 1.0
        self.points = []               # list of (x,y) in image space (0..PREVIEW_SIZE-1)
        self.history = []              # stack of applied classification entries for undo
        self.redo_stack = []           # undone entries for redo

        # counters for saved filenames (start from existing files to avoid overwriting)
        self.true_counter = self._get_next_counter(self.true_folder)
        self.false_counter = self._get_next_counter(self.false_folder)
        self.mask_counter = self._get_next_counter(self.masks_folder)

        # show initial empty text
        self.canvas.delete("all")
        self.canvas.create_text(200, 100, fill="white", text="Select a subfolder to start", anchor="nw")

    # ---------------------------
    # Utility helpers
    # ---------------------------
    def _get_next_counter(self, folder):
        nums = []
        for f in os.listdir(folder):
            name, ext = os.path.splitext(f)
            if name.isdigit():
                nums.append(int(name))
        return (max(nums) + 1) if nums else 1

    def write_log_file(self):
        # rewrite full CSV from self.log_entries
        fieldnames = ["timestamp", "image_path", "decision", "saved_image", "saved_mask"]
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.log_entries:
                writer.writerow(e)

    def append_log_entry(self, image_path, decision, saved_image="", saved_mask=""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "decision": decision,
            "saved_image": saved_image,
            "saved_mask": saved_mask
        }
        self.log_entries.append(entry)
        # append to file to keep it updated
        self.write_log_file()

    # ---------------------------
    # Subfolder selection & image loading
    # ---------------------------
    def on_subfolder_select(self, event=None):
        sel = self.subfolder_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self.current_folder = self.subfolders[idx]
        # collect all images in that folder, excluding ones already processed (true/false)
        all_images = [os.path.join(self.current_folder, f) for f in os.listdir(self.current_folder)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.available_images = [p for p in all_images if p not in self.processed_images]
        random.shuffle(self.available_images)
        self.load_next_image()

    def load_next_image(self):
        if not self.available_images:
            self.canvas.delete("all")
            self.canvas.create_text(10, 10, anchor="nw", fill="white", text="No more unprocessed images in this folder.")
            self.current_image_path = None
            return
        self.current_image_path = self.available_images.pop()
        self.points = []
        self.zoom_factor = 1.0
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.load_image_to_canvas(self.current_image_path)

    def load_image_to_canvas(self, path):
        # load image, resize to PREVIEW_SIZE and keep as base_image (PIL)
        pil = Image.open(path).convert("RGB").resize(PREVIEW_SIZE)
        self.base_image = pil
        self.redraw_canvas()

    # ---------------------------
    # Canvas draw / redraw
    # ---------------------------
    def redraw_canvas(self, overlay_image_pil=None):
        """
        Draw current base image (or overlay_image_pil if provided) scaled by zoom_factor,
        set scrollregion and draw the points (scaled).
        """
        if self.base_image is None:
            return
        img_to_show = overlay_image_pil if overlay_image_pil is not None else self.base_image
        w, h = img_to_show.size
        zoom_w = max(1, int(w * self.zoom_factor))
        zoom_h = max(1, int(h * self.zoom_factor))
        zoomed = img_to_show.resize((zoom_w, zoom_h), resample=Image.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(zoomed)

        self.canvas.delete("all")
        # place image at top-left of canvas
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        # keep reference so image doesn't get GC'd
        self.canvas.image = self.tk_img

        # set scroll region so scrollbars work
        self.canvas.config(scrollregion=(0, 0, zoom_w, zoom_h))

        # draw segmentation points (scaled)
        for (ix, iy) in self.points:
            zx = int(ix * self.zoom_factor)
            zy = int(iy * self.zoom_factor)
            r = 3
            self.canvas.create_oval(zx - r, zy - r, zx + r, zy + r, fill="red", outline="red")

    # ---------------------------
    # Mouse event handling
    # ---------------------------
    def on_mouse_wheel(self, event):
        """
        If CTRL is pressed -> zoom; else -> scroll canvas vertically (and horizontally with shift)
        Works on Windows (event.delta) and Linux (event.num==4/5).
        """
        ctrl = (event.state & 0x4) != 0  # Control key mask (works on many platforms)
        shift = (event.state & 0x1) != 0  # Shift likely has bit 0x1

        # Linux: event.num == 4 (up) or 5 (down)
        if hasattr(event, "num") and event.num in (4, 5) and not hasattr(event, "delta"):
            direction = 1 if event.num == 4 else -1
        else:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1

        if ctrl:
            # zoom around current view center
            if direction > 0:
                self.zoom_factor *= 1.15
            else:
                self.zoom_factor /= 1.15
            self.redraw_canvas()
        else:
            # scroll: vertical scroll by direction steps
            if shift:
                # horizontal scroll when Shift is held
                self.canvas.xview_scroll(-direction, "units")
            else:
                self.canvas.yview_scroll(-direction, "units")

    def on_canvas_left_click(self, event):
        """
        Map mouse click to image coordinates correctly even when zoomed and scrolled.
        Use canvas.canvasx/canvasy to get correct canvas coords including scroll offsets.
        """
        if self.base_image is None:
            return
        # canvas coordinates (accounting for scroll offset)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        # map to image coordinates (image space is PREVIEW_SIZE)
        img_x = int(canvas_x / self.zoom_factor)
        img_y = int(canvas_y / self.zoom_factor)
        img_x = max(0, min(PREVIEW_SIZE[0] - 1, img_x))
        img_y = max(0, min(PREVIEW_SIZE[1] - 1, img_y))
        self.points.append((img_x, img_y))
        self.redraw_canvas()

    def on_canvas_right_click(self, event):
        """Preview ellipse fit overlay on the image (no saving)."""
        if self.base_image is None:
            return
        if len(self.points) < 5:
            messagebox.showinfo("Need more points", "At least 5 points required for ellipse fitting.")
            return
        mask = np.zeros(PREVIEW_SIZE[::-1], dtype=np.uint8)  # (h,w)
        pts = np.array(self.points, dtype=np.int32)
        # fit ellipse if possible
        if len(pts) >= 5:
            ellipse = cv2.fitEllipse(pts)
            cv2.ellipse(mask, ellipse, 255, -1)
        else:
            cv2.fillPoly(mask, [pts], 255)
        # overlay onto base_image (numpy)
        base_arr = np.array(self.base_image)
        preview_arr = base_arr.copy()
        preview_arr[mask == 255] = (0, 255, 0)  # green overlay
        preview_pil = Image.fromarray(preview_arr)
        self.redraw_canvas(overlay_image_pil=preview_pil)

    # ---------------------------
    # Classification actions
    # ---------------------------
    def _save_mask_and_image(self, decision):
        """
        Save the current base_image and mask (if decision == 'true').
        Returns (saved_image_path, saved_mask_path_or_empty, mask_array_or_None)
        Also prepares a history entry (with image PIL and mask array) for undo/redo.
        """
        saved_img_path = ""
        saved_mask_path = ""
        saved_mask_arr = None

        # save image file
        if decision == "true":
            # build mask from points (ellipse fit if >=5)
            mask = np.zeros(PREVIEW_SIZE[::-1], dtype=np.uint8)  # shape (h,w)
            pts = np.array(self.points, dtype=np.int32)
            if len(pts) >= 5:
                ellipse = cv2.fitEllipse(pts)
                cv2.ellipse(mask, ellipse, 255, -1)
            else:
                cv2.fillPoly(mask, [pts], 255)

            # determine filenames using counters
            img_name = f"{self.true_counter:05d}.png"
            mask_name = f"{self.mask_counter:05d}.png"
            saved_img_path = os.path.join(self.true_folder, img_name)
            saved_mask_path = os.path.join(self.masks_folder, mask_name)

            # save image + mask
            self.base_image.save(saved_img_path)
            Image.fromarray(mask).save(saved_mask_path)

            saved_mask_arr = mask.copy()
            # increment counters
            self.true_counter += 1
            self.mask_counter += 1

        elif decision == "false":
            img_name = f"{self.false_counter:05d}.png"
            saved_img_path = os.path.join(self.false_folder, img_name)
            self.base_image.save(saved_img_path)
            self.false_counter += 1

        return saved_img_path, saved_mask_path, saved_mask_arr

    def mark_true(self):
        if self.base_image is None:
            return
        # require segmentation points
        if len(self.points) < 1:
            if not messagebox.askyesno("No segmentation", "No points drawn — save without mask?"):
                return
        saved_img, saved_mask, mask_arr = self._save_mask_and_image("true")
        # create log entry (and memory history entry)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "image_path": self.current_image_path,
            "decision": "true",
            "saved_image": saved_img,
            "saved_mask": saved_mask,
            # keep objects for undo/redo inside history only, not in log file:
            "_image_pil": self.base_image.copy(),
            "_mask_arr": mask_arr
        }
        self.log_entries.append({k: entry[k] for k in ("timestamp", "image_path", "decision", "saved_image", "saved_mask")})
        self.write_log_file()
        self.processed_images.add(self.current_image_path)
        # push to history for undo
        self.history.append(entry)
        # clear redo stack on new action
        self.redo_stack.clear()
        # next image
        self.load_next_image()

    def mark_false(self):
        if self.base_image is None:
            return
        saved_img, saved_mask, _ = self._save_mask_and_image("false")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "image_path": self.current_image_path,
            "decision": "false",
            "saved_image": saved_img,
            "saved_mask": saved_mask,
            "_image_pil": self.base_image.copy(),
            "_mask_arr": None
        }
        self.log_entries.append({k: entry[k] for k in ("timestamp", "image_path", "decision", "saved_image", "saved_mask")})
        self.write_log_file()
        self.processed_images.add(self.current_image_path)
        self.history.append(entry)
        self.redo_stack.clear()
        self.load_next_image()

    def skip_image(self):
        # log skip but do NOT mark as processed (so it can reappear later)
        if self.current_image_path:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "image_path": self.current_image_path,
                "decision": "skip",
                "saved_image": "",
                "saved_mask": "",
            }
            self.log_entries.append(entry)
            self.write_log_file()
        self.load_next_image()

    # ---------------------------
    # Undo / Redo classification
    # ---------------------------
    def undo_classification(self):
        """
        Undo last classification (true/false). We only undo actions that produced saved files.
        This removes saved files and removes the last log entry (rewriting the CSV).
        The undone item is pushed to redo_stack (including image and mask arrays) so redo can restore it.
        """
        if not self.history:
            messagebox.showinfo("Undo", "No classification action to undo (session-bound).")
            return
        last = self.history.pop()  # this contains internal keys _image_pil and _mask_arr
        # remove saved files if exist
        saved_img = last.get("saved_image", "")
        saved_mask = last.get("saved_mask", "")
        try:
            if saved_img and os.path.exists(saved_img):
                os.remove(saved_img)
            if saved_mask and os.path.exists(saved_mask):
                os.remove(saved_mask)
        except Exception as e:
            print("Warning removing saved files:", e)
        # remove the LAST matching log entry from log_entries (match by image_path & decision & saved_image)
        for i in range(len(self.log_entries) - 1, -1, -1):
            e = self.log_entries[i]
            if e.get("image_path") == last["image_path"] and e.get("decision") == last["decision"] and e.get("saved_image") == last.get("saved_image"):
                self.log_entries.pop(i)
                break
        # rewrite file
        self.write_log_file()
        # remove from processed list if it was true/false
        if last["decision"] in ("true", "false"):
            self.processed_images.discard(last["image_path"])
        # push to redo stack so we can restore
        self.redo_stack.append(last)
        # set current image back so user can re-check it
        self.current_image_path = last["image_path"]
        self.points = last.get("_points_copy", list(last.get("_mask_arr") if last.get("_mask_arr") is not None else [])) or []
        # load image into canvas
        self.load_image_to_canvas(self.current_image_path)
        messagebox.showinfo("Undo", f"Undid classification for {os.path.basename(self.current_image_path)}")

    def redo_classification(self):
        """
        Reapply the last undone classification (if any) by re-saving the files from saved objects kept in the entry.
        Then automatically advance to the next image.
        """
        if not self.redo_stack:
            messagebox.showinfo("Redo", "Nothing to redo.")
            return

        item = self.redo_stack.pop()
        decision = item["decision"]

        saved_img, saved_mask = "", ""

        if decision == "true":
            img_pil = item.get("_image_pil")
            mask_arr = item.get("_mask_arr")
            if img_pil is None:
                messagebox.showerror("Redo", "Image data not available to redo.")
                return
            img_name = f"{self.true_counter:05d}.png"
            mask_name = f"{self.mask_counter:05d}.png"
            saved_img = os.path.join(self.true_folder, img_name)
            saved_mask = os.path.join(self.masks_folder, mask_name)
            img_pil.save(saved_img)
            if mask_arr is not None:
                Image.fromarray(mask_arr).save(saved_mask)
            self.true_counter += 1
            self.mask_counter += 1

        else:  # false
            img_pil = item.get("_image_pil")
            if img_pil is None:
                messagebox.showerror("Redo", "Image data not available to redo.")
                return
            img_name = f"{self.false_counter:05d}.png"
            saved_img = os.path.join(self.false_folder, img_name)
            img_pil.save(saved_img)
            saved_mask = ""
            self.false_counter += 1

        # reconstruct a writable log entry
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_path": item["image_path"],
            "decision": decision,
            "saved_image": saved_img,
            "saved_mask": saved_mask,
            "_image_pil": item.get("_image_pil"),
            "_mask_arr": item.get("_mask_arr")
        }

        self.history.append(new_entry)
        self.log_entries.append({
            k: new_entry[k] for k in ("timestamp", "image_path", "decision", "saved_image", "saved_mask")
        })
        self.write_log_file()
        self.processed_images.add(new_entry["image_path"])

        # ✅ FIX: move to the next image automatically
        self.load_next_image()

        # Optional: still show confirmation
        messagebox.showinfo("Redo", f"Restored classification for {os.path.basename(new_entry['image_path'])}")

    # ---------------------------
    # Undo point
    # ---------------------------
    def undo_point(self):
        if self.points:
            self.points.pop()
            self.redraw_canvas()

    # ---------------------------
    # Save log helper (lower-level used above)
    # ---------------------------
    def write_log_file(self):
        # ensure the directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        fieldnames = ["timestamp", "image_path", "decision", "saved_image", "saved_mask"]
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.log_entries:
                # if e contains internal keys (with underscores) remove them
                row = {k: e.get(k, "") for k in fieldnames}
                writer.writerow(row)

    # ---------------------------
    # Exit handling: make sure log is saved
    # ---------------------------
    def on_close(self):
        # ensure we wrote the log
        self.write_log_file()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingApp(root)
    # bind closing to ensure log saved
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()