from manim import *
import random
import numpy as np

class WattsStrogatzAnimation(Scene):
    def construct(self):
        # -------------------------
        #   PARAMETERS
        # -------------------------
        N = 12          # number of nodes
        K = 4           # even, number of neighbors
        beta = 0.3      # rewiring probability
        random.seed(2)

        # -------------------------
        #   TITLE + PARAMETERS
        # -------------------------
        # big opening title (centered) that fades out, then a small header remains
        title_big = Text("Watts–Strogatz Small-World Model", font="Serif", weight=BOLD).scale(1.1)
        self.play(FadeIn(title_big))
        self.wait(1.0)
        self.play(FadeOut(title_big))

        # title = Text("Watts–Strogatz Small-World Model", font="Serif", weight=BOLD).scale(0.9)
        # title.to_edge(UP, buff=0.4)
        # params will be positioned next to the circle for symmetry with the Erdos scene
        params = Text(f"N={N},   K={K},   β={beta}", font="Serif").scale(0.7)
        # self.play(FadeIn(title))

        # -------------------------
        #   PLACE NODES ON A CIRCLE
        # -------------------------
        circle = Circle(radius=3)
        # position the params to the left of the circle (symmetric to Erdos scene)
        # nudge params a bit further left of the circle and then shift slightly right
        # so the left-most character isn't clipped by the frame boundary.
        params.next_to(circle, LEFT, buff=0.3)
        params.set_color(WHITE)
        # show params after positioning relative to the circle
        self.play(FadeIn(params))
        nodes = []

        for i in range(N):
            angle = 2 * PI * i / N
            pt = circle.point_at_angle(angle)

            dot = Dot(pt, color=BLUE, radius=0.12)
            label = Text(str(i), font="Serif").scale(0.5)
            label.next_to(dot, DOWN, buff=0.20)

            nodes.append(VGroup(dot, label))

        self.play(LaggedStart(*[FadeIn(node) for node in nodes], lag_ratio=0.05))
        self.wait(0.3)

        # -------------------------
        #   STEP 1: CREATE RING LATTICE
        # -------------------------
        half_k = K // 2
        edges = []

        ring_text = Text("Step 1: Create Ring Lattice", font="Serif").scale(0.6)
        ring_text.to_edge(DOWN, buff=0.01)
        self.play(FadeIn(ring_text))

        for i in range(N):
            for j in range(1, half_k + 1):
                neighbor = (i + j) % N
                start = nodes[i][0].get_center()
                end = nodes[neighbor][0].get_center()
                edge = Line(start, end, color=GRAY)
                edges.append(edge)
                # slightly slower creation so the lattice builds visibly
                self.play(Create(edge), run_time=0.25)

        self.wait(0.6)

        # -------------------------
        #   STEP 2: REWIRING
        # -------------------------
        rewire_text = Text("Step 2: Rewire Edges with Probability β", font="Serif").scale(0.6)
        rewire_text.to_edge(DOWN, buff=0.01)
        self.play(Transform(ring_text, rewire_text))

        # Keep track of adjacency for valid rewiring
        adjacency = {i: set() for i in range(N)}
        # Prepare right-side annotation (similar to Erdos scene)
        right_ann_pos = circle.get_right() + RIGHT * 1.6
        first_ann = True
        right_ann = None

        for i in range(N):
            for j in range(1, half_k + 1):
                adjacency[i].add((i + j) % N)
                adjacency[(i + j) % N].add(i)

        # Actually animate rewiring like your ws.py
        for i in range(N):
            forward_neighbors = [(i + j) % N for j in range(1, half_k + 1)]

            for j in forward_neighbors:
                if j not in adjacency[i]:
                    continue  # may have been rewired

                # Show the random draw on the right side (r and condition)
                r = random.random()
                r_text = Text(f"r = {r:.2f}", font="Serif").scale(0.45)
                cond_text = Text("r < β ?", font="Serif").scale(0.45)
                annotation = VGroup(r_text, cond_text).arrange(RIGHT, buff=0.18)
                annotation.move_to(right_ann_pos)

                if first_ann:
                    self.play(FadeIn(annotation), run_time=0.3)
                    right_ann = annotation
                    first_ann = False
                else:
                    self.play(Transform(right_ann, annotation), run_time=0.12)

                self.wait(0.08)

                if r < beta:
                    # REWIRE EDGE: visually remove old and add a new one (slower)
                    old_line = Line(nodes[i][0].get_center(), nodes[j][0].get_center(), color=GRAY)
                    self.play(FadeOut(old_line), run_time=0.18)

                    adjacency[i].remove(j)
                    adjacency[j].remove(i)

                    # Choose a random candidate not already connected
                    candidate = None
                    for _ in range(N * 2):
                        k = random.randrange(N)
                        if k != i and k not in adjacency[i]:
                            candidate = k
                            break

                    if candidate is not None:
                        new_line = Line(nodes[i][0].get_center(), nodes[candidate][0].get_center(), color=YELLOW)
                        adjacency[i].add(candidate)
                        adjacency[candidate].add(i)

                        newtext = Text(f"Rewired → {candidate}", font="Serif").scale(0.45).set_color(YELLOW)
                        newtext.next_to(right_ann, DOWN, buff=0.15)

                        # show and create a bit more slowly so viewers can follow
                        self.play(FadeIn(newtext), Create(new_line), run_time=0.28)
                        self.wait(0.12)
                        self.play(FadeOut(right_ann), FadeOut(newtext), run_time=0.12)
                        first_ann = True
                        right_ann = None

                else:
                    # KEEP EDGE: annotate and keep the old connection (slower fade)
                    keeptext = Text("Keep edge", font="Serif").scale(0.45).set_color(GREEN)
                    keeptext.next_to(right_ann if right_ann is not None else annotation, DOWN, buff=0.15)

                    line = Line(nodes[i][0].get_center(), nodes[j][0].get_center(), color=GRAY)
                    self.play(FadeIn(keeptext), run_time=0.18)
                    self.wait(0.12)
                    self.play(FadeOut(right_ann if right_ann is not None else annotation), FadeOut(keeptext), run_time=0.12)
                    self.wait(0.04)
                    first_ann = True
                    right_ann = None

        # -------------------------
        # FINAL LABEL
        # -------------------------
        final_label = Text("Final Watts–Strogatz Graph", font="Serif").scale(0.7)
        final_label.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(final_label))
        self.wait(2)
