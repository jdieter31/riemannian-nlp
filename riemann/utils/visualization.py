"""
This module contains utilities to visualize points on various manifolds.
"""
from typing import List

import numpy as np
from . import svg3d
from numpy import sin, cos


# region: common parametric surfaces
from .svg3d import parametric_surface
from .. import RiemannianManifold, EuclideanManifold, ProductManifold


def sphere(u, v):
    x = sin(u) * cos(v)
    y = cos(u)
    z = -sin(u) * sin(v)
    return x, y, z


def torus(u, v):
    x = 3 * (1.5 + cos(v)) * cos(2*u)
    y = 3 * (1.5 + cos(v)) * sin(2*u)
    z = 3 * sin(v)
    return x, y, z


def plane(u, v):
    x = 3 * 2 * u - np.pi
    y = 3 * v - np.pi
    z = 3 * 0
    return x, y, z


def cylinder(u, v, r=1):
    x = 3 * r * sin(2 * u)
    y = 3 * r * cos(2 * u)
    z = 2 * v - np.pi
    return x, y, z


def manifold_scatter(manifold: RiemannianManifold, x: np.ndarray,
                     colors: List[str] = None, labels: List[str] = None) -> svg3d.Drawing:
    """
    Draw a scatter plot of points `x` on the manifold `manifold`.
    :param manifold:
    :param x:
    :param colors:
    :param labels:
    :return:
    """
    assert len(x.shape) == 2 and (x.shape[1] == 2 or x.shape[1] == 3)
    n, d = x.shape

    def face_shader(index, winding):
        ret = dict(
            fill="white",
            fill_opacity="0.50",
            stroke="black",
            stroke_linejoin="round",
            stroke_width="0.001",
        )
        if winding < 0:
            ret["stroke_dasharray"] = "0.01"
        return ret

    def point_shader(index, winding):
        return {
            "fill": colors[index] if index < len(colors) else "black",
            "fill_opacity": "0.95",
            "stroke": "black",
            "stroke_linejoin": "round",
            "stroke_width": "0.002",
            "radius": "0.005",
        }

    def point_annotator(index):
        if index >= len(labels):
            return None
        return {
            "text": str(labels[index]),
            "style": "font: 0.05px bold serif",
        }

    if isinstance(manifold, EuclideanManifold):
        assert d == 2
        points = np.hstack((x, np.zeros((n, 1)))).reshape((n, 1, 3))
        surface = svg3d.Mesh(3.0 * parametric_surface(8, 8, plane), face_shader)
        points = svg3d.Mesh(3.0 * points, point_shader, annotator=point_annotator)
        scene = svg3d.Scene([svg3d.Group([surface, points])])
        camera = svg3d.Camera.create(eye=(10, 0, 40))

    elif isinstance(manifold, ProductManifold):
        spec = str(manifold)
        if spec == "S1xS1":
            points = np.array([
                torus((u + 1) * np.pi/2, (v + 1) * np.pi) for u, v in x]).reshape(n, 1, 3)
            surface = svg3d.Mesh(3.0 * parametric_surface(44, 44, torus), face_shader)
            points = svg3d.Mesh(3.0 * points, point_shader, annotator=point_annotator)
            scene = svg3d.Scene([svg3d.Group([surface, points])])
            camera = svg3d.Camera.create(eye=(50, 120, 80))
        elif spec == "S1xE1":
            assert abs(x[:, 0]).max() <= 1.0
            # Normalize the euclidean dimension by its largest value.
            x[:, 1] /= x[:, 1].max()
            # Project them into cylindrical coordinates
            points = np.array([
                cylinder((u + 1) * np.pi/2, (v + 1) * np.pi) for u, v in x]).reshape(n, 1, 3)
            surface = svg3d.Mesh(3.0 * parametric_surface(24, 24, cylinder), face_shader)
            points = svg3d.Mesh(3.0 * points, point_shader, annotator=point_annotator)
            scene = svg3d.Scene([svg3d.Group([surface, points])])
            camera = svg3d.Camera.create(eye=(50, 120, 80))
        elif spec == "S1xE2":
            assert abs(x[:, 0]).max() <= 1.0
            # Normalize the euclidean dimensions by its largest value.
            x[:, 1] /= x[:, 1].max()
            x[:, 2] /= x[:, 2].max()
            # Project them into cylindrical coordinates
            points = np.array([
                cylinder((u + 1) * np.pi/2, (v + 1) * np.pi, r) for u, v, r in x]).reshape(n, 1, 3)
            surface = svg3d.Mesh(3.0 * parametric_surface(24, 24, cylinder), face_shader)
            points = svg3d.Mesh(3.0 * points, point_shader, annotator=point_annotator)
            scene = svg3d.Scene([svg3d.Group([surface, points])])
            camera = svg3d.Camera.create(eye=(50, 120, 80))
        else:
            raise NotImplemented
    else:
        raise NotImplemented

    view_box = "-0.5 -0.5 1.0 1.0"

    return svg3d.Engine([
        svg3d.View(camera, scene,
                   svg3d.Viewport.from_string(view_box))
    ]).render(size=(512, 512), view_box=view_box)


def test_manifold_scatter():
    # S1 x E1 x H1
    # S1 -> (x, y): |(x, y)| == 1 -> theta
    # S2 -> (x, y, z): |(x, y, z)| == 1 --> theta, phi
    # E1 -> (x in -inf, inf)
    # H2 -> (x, y) |x, y| < 1?
    # H3 -> (x, y, z) |x, y, z| < 1?
    # SxH2 -> (sin(θ), cos(θ), x, y) |x, y| < 3?
    drawing = manifold_scatter(ProductManifold.from_string("S1xE1"),
                               np.hstack((2 * np.random.rand(100, 1) - 1, np.random.rand(100, 1) * 3)),
                               sum(([color] * 20 for color in "red blue green black yellow".split()), []),
                               list(range(5)))
    drawing.saveas("S1xE1.svg")

    drawing = manifold_scatter(ProductManifold.from_string("S1xE2"),
                               np.hstack((2 * np.random.rand(100, 1) - 1, np.random.rand(100, 1) * 3, np.random.rand(100, 1) * 3)),
                               sum(([color] * 20 for color in "red blue green black yellow".split()), []),
                               list(range(5)))
    drawing.saveas("S1xE2.svg")

    drawing = manifold_scatter(ProductManifold.from_string("S1xS1"), 2 * np.random.rand(100, 2) - 1,
                               sum(([color] * 20 for color in "red blue green black yellow".split()), []),
                               list(range(5)))
    drawing.saveas("S1xS1.svg")

    drawing = manifold_scatter(EuclideanManifold(), np.random.randn(100, 2),
                               sum(([color] * 20 for color in "red blue green black yellow".split()), []),
                               list(range(5)))
    drawing.saveas("E2.svg")
