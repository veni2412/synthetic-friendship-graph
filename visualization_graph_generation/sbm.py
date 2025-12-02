from manim import *
import random
import numpy as np


class SBMAnimation(Scene):
    def construct(self):

        # ----------------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------------
        block_sizes = [6, 6, 6]   # number of nodes per block
        p_intra = 0.45            # within-block edge probability
        p_inter = 0.05            # across-block edge probability
        random.seed(3)

        K = len(block_sizes)
        N = sum(block_sizes)

        # ----------------------------------------------------------
        # TITLE SLIDE + PARAMETERS
        # ----------------------------------------------------------
        # big opening title that fades out
        title_big = Text("Stochastic Block Model (SBM)", font="Serif", weight=BOLD).scale(1.1)
        self.play(FadeIn(title_big))
        self.wait(1.0)
        self.play(FadeOut(title_big))

        # params placed in upper-left corner (no small header)
        p1 = Text(f"K = {K},   p_intra = {p_intra}", font="Serif").scale(0.62)
        p2 = Text(f"p_inter = {p_inter},   N = {N}", font="Serif").scale(0.62)
        params = VGroup(p1, p2).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        params.to_corner(UL, buff=0.1)

        self.play(FadeIn(params))

        # ----------------------------------------------------------
        # BLOCK COLORS
        # ----------------------------------------------------------
        block_colors = [BLUE, GREEN, RED, YELLOW, ORANGE]

        # ----------------------------------------------------------
        # PLACE CLUSTERS IN SEPARATE CIRCLES
        # ----------------------------------------------------------
        # move the entire cluster layout upward so bottom nodes are visible
        centers = [
            LEFT * 2.2 + UP * 1.4,
            RIGHT * 3 + UP * 1.4,
            DOWN * 2.0 + UP * 1.4,
        ]

        cluster_positions = []

        for b in range(K):
            cx = centers[b]
            r = 1.8

            positions = []
            size = block_sizes[b]

            for i in range(size):
                angle = 2 * PI * i / size
                pt = cx + r * np.array([np.cos(angle), np.sin(angle), 0])
                positions.append(pt)

            cluster_positions.append(positions)

        # ----------------------------------------------------------
        # CREATE NODES
        # ----------------------------------------------------------
        nodes = []
        block_of = []
        node_index = 0

        for b in range(K):
            for _ in range(block_sizes[b]):
                pos = cluster_positions[b].pop(0)
                dot = Dot(point=pos, radius=0.14, color=block_colors[b])
                label = Text(str(node_index), font="Serif").scale(0.5)
                label.next_to(dot, DOWN, buff=0.18)

                nodes.append(VGroup(dot, label))
                block_of.append(b)

                self.play(FadeIn(nodes[node_index], run_time=0.15))
                node_index += 1

        self.wait(0.3)

        # ----------------------------------------------------------
        # STEP LABEL
        # ----------------------------------------------------------
        step_label = Text(
            "Step 2: Add edges based on block probabilities",
            font="Serif"
        ).scale(0.55)
        step_label.to_edge(DOWN, buff=0.1)
        self.play(FadeIn(step_label))

        # ----------------------------------------------------------
        # ADD EDGES ACCORDING TO SBM
        # ----------------------------------------------------------
        adjacency = {i: set() for i in range(N)}

        # prepare a fixed right-side annotation position (moved further right)
        right_ann_pos = RIGHT * 5.5
        first_ann = True
        right_ann = None

        for i in range(N):
            bi = block_of[i]
            for j in range(i + 1, N):
                bj = block_of[j]

                p = p_intra if bi == bj else p_inter

                # Show the random draw on the right side (r and condition)
                r = random.random()
                r_text = Text(f"r = {r:.2f}", font="Serif").scale(0.45)
                cond_text = Text("r < p ?", font="Serif").scale(0.45)
                annotation = VGroup(r_text, cond_text).arrange(RIGHT, buff=0.18)
                annotation.move_to(right_ann_pos)

                if first_ann:
                    self.play(FadeIn(annotation), run_time=0.18)
                    right_ann = annotation
                    first_ann = False
                else:
                    self.play(Transform(right_ann, annotation), run_time=0.12)

                # small pause so viewers can read
                self.wait(0.06)

                if r < p:
                    # add edge
                    line_color = WHITE if bi != bj else block_colors[bi]
                    line = Line(nodes[i][0].get_center(), nodes[j][0].get_center(), color=line_color)
                    adjacency[i].add(j)
                    adjacency[j].add(i)

                    check = Text("Add edge", font="Serif").scale(0.45).set_color(GREEN)
                    check.next_to(right_ann, DOWN, buff=0.18)
                    self.play(FadeIn(check), run_time=0.12)
                    self.play(Create(line), run_time=0.15)
                    self.play(FadeOut(right_ann), FadeOut(check))
                    first_ann = True
                    right_ann = None
                else:
                    cross = Text("×", font="Serif").scale(0.6).set_color(RED)
                    cross.next_to(right_ann if right_ann is not None else annotation, DOWN, buff=0.18)
                    self.play(FadeIn(cross), run_time=0.12)
                    self.play(FadeOut(right_ann if right_ann is not None else annotation), FadeOut(cross))
                    # no edge was created in the "else" branch; just pause briefly
                    self.wait(0.02)
                    first_ann = True
                    right_ann = None

        # # ----------------------------------------------------------
        # # FINAL LABEL
        # # ----------------------------------------------------------
        # final_label = Text("Final Stochastic Block Model Graph", font="Serif").scale(0.7)
        # final_label.to_edge(DOWN, buff=0.01)
        # self.play(FadeIn(final_label))

        self.wait(2)