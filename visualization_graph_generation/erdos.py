from manim import *
import random
import itertools
import numpy as np


class ErdosRenyiAnimation(Scene):
    def construct(self):

        # Show a large title at the start, then fade it out before nodes appear
        title_big = Text(
            "Erdős–Rényi Random Graph G(n, p)",
            font="Serif",
            weight=BOLD,
        ).scale(1.2)
        # center the big title initially
        self.play(FadeIn(title_big))
        self.wait(1.0)
        # remove the big title before drawing nodes/params
        self.play(FadeOut(title_big))

        # Parameters (will be positioned next to the circle later)
        n = 7
        p = 0.35
        params = Text(f"n = {n},  p = {p}", font="Serif").scale(0.7)

        # Node placement
        nodes = []
        circle = Circle(radius=3)
        # place the parameters to the left of the node circle for clarity
        params.next_to(circle, LEFT, buff=0.8)
        # show params after positioning relative to the circle
        self.play(FadeIn(params))

        for i in range(n):
            angle = 2 * PI * i / n
            pt = circle.point_at_angle(angle)

            # Larger nodes
            dot = Dot(pt, color=BLUE, radius=0.12)

            # Place labels radially (away from center) to reduce overlap
            label = Text(str(i), font="Serif").scale(0.6).set_color(WHITE)
            # direction vector from center to point
            dir_vec = np.array(pt) - np.array(circle.get_center())
            # If the vector is degenerate, default to DOWN
            if np.linalg.norm(dir_vec) < 1e-6:
                label_dir = DOWN
            else:
                # Normalize and use as direction for next_to
                label_dir = np.array(dir_vec) / np.linalg.norm(dir_vec)

            label.next_to(dot, label_dir, buff=0.18)

            nodes.append(VGroup(dot, label))

        self.play(LaggedStart(*[FadeIn(node) for node in nodes], lag_ratio=0.1))
        self.wait(0.5)

        # Prepare a fixed right-side annotation position (symmetric to params on the left)
        # move it a bit further right for more clearance
        right_ann_pos = circle.get_right() + RIGHT * 2
        first_ann = True
        right_ann = None

        # Iterate over node pairs
        for i, j in itertools.combinations(range(n), 2):

            node_i = nodes[i][0]
            node_j = nodes[j][0]

            trial_line = Line(
                node_i.get_center(),
                node_j.get_center(),
                color=GRAY,
                stroke_opacity=0.35,
                stroke_width=2,
            )
            self.play(Create(trial_line), run_time=0.25)

            # midpoint of the line
            midpoint = (node_i.get_center() + node_j.get_center()) / 2

            # r display - show the entire annotation on the right side of the circle
            r = random.random()
            r_text = Text(f"r = {r:.2f}", font="Serif").scale(0.5)
            cond_text = Text("r < p ?", font="Serif").scale(0.5)

            annotation = VGroup(r_text, cond_text).arrange(RIGHT, buff=0.25)
            annotation.move_to(right_ann_pos)

            if first_ann:
                # first time: fade in the annotation at the right
                self.play(FadeIn(annotation), run_time=0.18)
                right_ann = annotation
                first_ann = False
            else:
                # transform the previous right annotation into the new value
                self.play(Transform(right_ann, annotation), run_time=0.12)

            if r < p:
                # Transform the trial (faint) line into a bright colored edge
                edge = Line(
                    node_i.get_center(), node_j.get_center(), color=YELLOW, stroke_width=4
                )

                # place the "Add edge" text to the right of the circle (symmetric to params)
                check = Text("Add edge", font="Serif").scale(0.45).set_color(GREEN)
                # position the check below the right annotation
                check.next_to(right_ann, DOWN, buff=0.2)

                self.play(FadeIn(check), run_time=0.12)
                # Transform the existing trial_line into the final edge in-place
                self.play(Transform(trial_line, edge), run_time=0.35)
                # keep the transformed line (no remove/add necessary)
                self.play(FadeOut(right_ann), FadeOut(check))
                # mark that next annotation will be treated as first (it will fade in)
                first_ann = True
                right_ann = None

            else:
                cross = Text("×", font="Serif").scale(0.6).set_color(RED)
                cross.next_to(right_ann if right_ann is not None else annotation, DOWN, buff=0.18)

                self.play(FadeIn(cross), run_time=0.12)
                # remove right annotation and the cross after showing
                self.play(FadeOut(right_ann if right_ann is not None else annotation), FadeOut(cross))
                self.play(FadeOut(trial_line), run_time=0.2)
                first_ann = True
                right_ann = None

        final_label = Text("Generated G(n, p) Graph", font="Serif").scale(0.7).to_edge(
            DOWN, buff=0.01)
        self.play(FadeIn(final_label))
        self.wait(2)
