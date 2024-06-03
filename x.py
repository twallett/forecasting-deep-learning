from manimlib import *
import pandas as pd

# manimgl x.py OOP

# class GraphExample(Scene):
#     def construct(self):
        
#         # Axes
#         axes = Axes((0, 10), 
#                     (0, 10),
#                     width = 10, 
#                     height = 6)
#         axes.add_coordinate_labels()
        
#         # Grid
#         grid = NumberPlane((0, 10), 
#                            (0, 10), 
#                            width = 10,
#                            height = 6).shift(RIGHT*0.03 + UP*0.03)
        
#         self.add(axes, grid)
        
#         # Function 
#         sin_graph = axes.get_graph(
#             lambda x: 2 * math.sin(x),
#             color=BLUE,
#         )
#         sin_label = axes.get_graph_label(sin_graph, "\\sin(x)")
#         self.play(
#             ShowCreation(sin_graph),
#             FadeIn(sin_label, RIGHT),
#         )
#         self.wait(2)
        
data = pd.read_csv("Unemployment.csv")

class TimeSeriesPlot(Scene):
    def construct(self):
        
        # Objects ------------------------------------------------------------------------------
        axes = Axes(
            x_range=[0, 915, 915/5],
            y_range=[0, 20, 20/5],
        )
        axes.add_coordinate_labels()
        
        axes_labels = axes.get_axis_labels(x_label_tex="Time", 
                                           y_label_tex="Unemployment Rate")
        
        for label in axes_labels:
            label.scale(0.8) 
            
        axes_labels[0]
        axes_labels[1].shift(0.5*UP)
        
        self.add(axes, 
                 axes_labels)
        
        ts1 = axes.get_graph(
            lambda x: data.loc[x, "Unemployment"], 
            x_range=[0, 800, 1], 
            color=BLUE
        )
        pred1 = axes.get_graph(
            lambda x: data.loc[x, "Unemployment"], 
            x_range=[800, 915, 1], 
            color=ORANGE
        )
        
        # Animation ------------------------------------------------------------------------------
        self.play(
            ShowCreation(ts1),
            run_time = 5
        )
        self.add(ts1)
        
        self.wait(1)
        
        self.play(
            ShowCreation(pred1),
            run_time = 5
        )
        self.add(pred1)