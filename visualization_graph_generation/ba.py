from manim import *
import random
import numpy as np

class BarabasiAlbertAnimation(Scene):
    def construct(self):

        # ----------------------------------------
        # PARAMETERS
        # ----------------------------------------
        n = 18        # total nodes
        m = 3         # edges per new node (initial core size m0 = m)
        random.seed(1)

        # ----------------------------------------
        # TITLE + PARAMETERS
        # ----------------------------------------
        # big opening title that fades out, then a small header appears
        title_big = Text("Barabási–Albert Scale-Free Model", font="Serif", weight=BOLD).scale(1.1)
        self.play(FadeIn(title_big))
        self.wait(1.0)
        self.play(FadeOut(title_big))

        # params will be positioned next to the circle (symmetric to other scenes)
        params = Text(f"n = {n},   m = {m}", font="Serif").scale(0.7)

        # ----------------------------------------
        # CREATE NODE POSITIONS IN A CIRCLE
        # (we will add nodes gradually)
        # ----------------------------------------
        circle = Circle(radius=3)

        positions = []
        for i in range(n):
            angle = 2 * PI * i / n
            positions.append(circle.point_at_angle(angle))

        # ----------------------------------------
        # INITIAL COMPLETE GRAPH on m nodes
        # ----------------------------------------
        # single persistent footer for bottom messages (prevents overlapping)
        footer = Text("Initial complete graph on m nodes", font="Serif").scale(0.55)
        footer.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(footer))

        nodes = []
        adjacency = {i: set() for i in range(n)}

        # position params to the left of the circle and show them
        params.next_to(circle, LEFT, buff=0.8)
        self.play(FadeIn(params))

        # Create initial m nodes (use radial labels to match style)
        for i in range(m):
            dot = Dot(positions[i], radius=0.13, color=BLUE)
            # radial label placement
            dir_vec = np.array(positions[i]) - np.array(circle.get_center())
            if np.linalg.norm(dir_vec) < 1e-6:
                label_dir = DOWN
            else:
                label_dir = dir_vec / np.linalg.norm(dir_vec)
            label = Text(str(i), font="Serif").scale(0.5)
            label.next_to(dot, label_dir, buff=0.18)
            node_group = VGroup(dot, label)
            nodes.append(node_group)
            self.play(FadeIn(node_group), run_time=0.25)

        # Draw complete graph among m nodes
        for i in range(m):
            for j in range(i + 1, m):
                line = Line(nodes[i][0].get_center(), nodes[j][0].get_center(), color=GRAY)
                adjacency[i].add(j)
                adjacency[j].add(i)
                self.play(Create(line), run_time=0.2)

        self.wait(0.5)

        # ----------------------------------------
        # BAG = Preferential attachment list
        # each node appears deg(node) times
        # ----------------------------------------
        bag = []
        for i in range(m):
            deg = len(adjacency[i])
            bag.extend([i] * deg)

    # ----------------------------------------
    # ITERATIVELY ADD ONE NODE AT A TIME
    # ----------------------------------------
    grow_text = Text("Growth: add one node at a time", font="Serif").scale(0.55)
    grow_text.to_edge(DOWN, buff=0.3)
    # transform the persistent footer into the growth message (smooth ease)
    self.play(Transform(footer, grow_text), run_time=0.7, rate_func=smooth)

    # prepare right-side annotation (similar to other scenes)
    # increase horizontal buffer so the right-side text isn't clipped or overlaps
    right_ann_pos = circle.get_right() + RIGHT * 2.0
    first_ann = True
    right_ann = None

    for new_node in range(m, n):

            # Create the new node (not yet connected)
            dot = Dot(positions[new_node], radius=0.14, color=YELLOW)
            label = Text(str(new_node), font="Serif").scale(0.5).next_to(dot, DOWN, buff=0.18)
            nodes.append(VGroup(dot, label))

            self.play(FadeIn(nodes[new_node]), run_time=0.3)

            # -----------------------------------
            # Preferential Attachment selection
            # -----------------------------------

            attach_text = Text(f"Select {m} target nodes ∝ degree", font="Serif").scale(0.45)
            attach_text.to_edge(DOWN, buff=0.25)
            # display attach_text using the persistent footer (avoids layering)
            self.play(Transform(footer, attach_text), run_time=0.7, rate_func=smooth)

            # show a right-side annotation about selection and update it with the chosen targets
            ann_text = Text("Selecting targets...", font="Serif").scale(0.45)
            ann = VGroup(ann_text)
            ann.move_to(right_ann_pos)
            if first_ann:
                self.play(FadeIn(ann), run_time=0.18)
                right_ann = ann
                first_ann = False
            else:
                self.play(Transform(right_ann, ann), run_time=0.12)

            # choose m distinct targets
            targets = set()
            while len(targets) < m:
                if bag:
                    t = random.choice(bag)
                else:
                    t = random.randrange(new_node)
                targets.add(t)

            # Highlight chosen targets
            highlights = []
            for t in targets:
                hl = Circle(radius=0.28, color=GREEN).move_to(nodes[t][0].get_center())
                highlights.append(hl)
                self.play(Create(hl), run_time=0.2)

            # -----------------------------------
            # Add edges
            # -----------------------------------
            # update right annotation to list the chosen targets by transforming
            # the existing right-side annotation (avoids overlaying two texts)
            targets_list_text = Text(f"Targets: {sorted(list(targets))}", font="Serif").scale(0.45)
            new_ann = VGroup(targets_list_text)
            new_ann.move_to(right_ann_pos)
            if right_ann is None:
                # no previous annotation, fade in the new one
                self.play(FadeIn(new_ann), run_time=0.18)
                right_ann = new_ann
            else:
                # transform the existing right annotation into the target list
                self.play(Transform(right_ann, new_ann), run_time=0.18)

            for t in targets:
                line = Line(nodes[new_node][0].get_center(), nodes[t][0].get_center(), color=YELLOW)
                self.play(Create(line), run_time=0.25)
                adjacency[new_node].add(t)
                adjacency[t].add(new_node)

            # remove the right annotation listing (so it can refresh next iteration)
            if right_ann is not None:
                self.play(FadeOut(right_ann), run_time=0.12)
                right_ann = None
                first_ann = True

            # restore the footer to the growth message (smooth ease)
            self.play(Transform(footer, grow_text), run_time=0.7, rate_func=smooth)

            # Remove highlights and the attach text
            self.play(*[FadeOut(h) for h in highlights], FadeOut(attach_text), run_time=0.3)

            # -----------------------------------
            # Update BAG = repeat each node according to degree
            # -----------------------------------
            for t in targets:
                bag.append(t)          # add 1 per edge
            bag.extend([new_node] * m)  # add m for new node

        # ----------------------------------------
        # FINAL LABEL
        # ----------------------------------------
    final_label = Text("Final Barabási–Albert Graph", font="Serif").scale(0.7)
    final_label.to_edge(DOWN, buff=0.01)
    self.play(FadeIn(final_label))
    self.wait(2)
