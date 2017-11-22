from tkinter import *
import numpy

MATRIX_SIZE = 28
CELL_SIZE = 15


class Matrix():
    def __init__(self, classifier):
        self.callback = classifier

        self.root = Tk()
        self.canvas = Canvas(self.root, width=MATRIX_SIZE * CELL_SIZE, height=MATRIX_SIZE * CELL_SIZE)
        self.canvas.configure(cursor="crosshair")
        self.canvas.pack()
        self.checkered()

        self.matrix = numpy.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype="float32")

        self.canvas.bind("<B1-Motion>", self.motion)
        self.root.bind("<space>", self.clear)
        self.root.bind("<Return>", self.recognize)

        self.root.mainloop()

    def checkered(self):
        # vertical lines at an interval of "line_distance" pixel
        for x in range(CELL_SIZE, MATRIX_SIZE * CELL_SIZE, CELL_SIZE):
            self.canvas.create_line(x, 0, x, MATRIX_SIZE * CELL_SIZE, fill="#476042")
        # horizontal lines at an interval of "line_distance" pixel
        for y in range(CELL_SIZE, MATRIX_SIZE * CELL_SIZE, CELL_SIZE):
            self.canvas.create_line(0, y, MATRIX_SIZE * CELL_SIZE, y, fill="#476042")

    def clear(self, event):
        self.matrix = numpy.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype="float32")
        self.canvas.delete("all")
        self.checkered()

    def recognize(self, event):
        self.callback(self.matrix)

    def add_matrix_value(self, x, y, value):
        if value <= 0 or x >= MATRIX_SIZE or y >= MATRIX_SIZE or y < 0 or x < 0:
            return
        self.matrix[x][y] += value
        if self.matrix[x][y] > 1:
            self.matrix[x][y] = 1

    def add_point(self, x, y, value):
        X = int(x)
        Y = int(y)
        self.add_matrix_value(Y, X, 0.7)
        self.add_matrix_value(Y, X + 1, 0.2)
        self.add_matrix_value(Y, X - 1, 0.2)
        self.add_matrix_value(Y + 1, X, 0.2)
        self.add_matrix_value(Y - 1, X, 0.2)

        self.canvas.create_oval(
            x * CELL_SIZE,
            y * CELL_SIZE,
            (x + 1) * CELL_SIZE,
            (y + 1) * CELL_SIZE,
            fill="#000000"
        )

    def motion(self, event):
        self.add_point(event.x / CELL_SIZE, event.y / CELL_SIZE, 0.7)

