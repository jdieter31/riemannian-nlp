# Single-file Python library for generating 3D wireframes in SVG format.
# Changed by Arun Chaganty
# Built on top of `svg3d`; see details below:
# svg3d :: https://prideout.net/blog/svg_wireframes/
# Copyright (c) 2019 Philip Rideout
# Distributed under the MIT License, see bottom of file.


import numpy as np
import pyrr         # type:ignore

from svgwrite import Drawing
from typing import NamedTuple, Callable, List, Tuple, Optional


class Viewport(NamedTuple):
    minx: float = -0.5
    miny: float = -0.5
    width: float = 1.0
    height: float = 1.0

    @classmethod
    def from_aspect(cls, aspect_ratio: float):
        return cls(-aspect_ratio / 2.0, -0.5, aspect_ratio, 1.0)

    @classmethod
    def from_string(cls, string_to_parse):
        """Parses a viewport from the format 'min-x min-y width height' format."""
        args = [float(f) for f in string_to_parse.split()]
        return cls(*args)


class Camera(NamedTuple):
    view: np.ndarray
    projection: np.ndarray

    @classmethod
    def create(cls,
               eye=(50, 120, 50),
               target=(0, 0, 0),
               up=(0, 0, 1),
               fovy=28,
               aspect=1,
               near=1,
               far=200):

        return cls(pyrr.matrix44.create_look_at(eye=eye, target=target, up=up),
                   pyrr.matrix44.create_perspective_projection(fovy=fovy, aspect=aspect,
                                                               near=near, far=far))


class Mesh(NamedTuple):
    #: A N x V x 3 matrix; N is the number of faces, V is the number of vertices
    #  per face. If V=1, we will treat this as a list of points.
    faces: np.ndarray
    #: A shading function from each  each 
    #: A function returning the style to use for the given face index and
    #  winding.
    shader: Optional[Callable[[int, float], dict]] = None
    #: A function returning text style to render at the centroid of the provided face index.
    #  The text returned should be a dictionary with format {'text': <text>,
    #  'fill': ...}
    annotator: Optional[Callable[[int], Optional[dict]]] = None


class Group(NamedTuple):
    meshes: List[Mesh]
    #: Default style for this group
    style: Optional[dict] = None


class Scene(NamedTuple):
    groups: List[Group]


class View(NamedTuple):
    camera: Camera
    scene: Scene
    viewport: Viewport = Viewport()


class Engine:
    def __init__(self, views: List[View], precision: float = 5):
        self.views = views
        self.precision = precision

    def render(self, size: Tuple[int, int] = (512, 512),
               view_box: str = "-0.5 -0.5 1.0 1.0", **extra) -> Drawing:
        drawing = Drawing("", size, viewBox=view_box, **extra)

        for view in self.views:
            projection = np.dot(view.camera.view, view.camera.projection)

            clip_path = drawing.defs.add(drawing.clipPath())
            clip_min = view.viewport.minx, view.viewport.miny
            clip_size = view.viewport.width, view.viewport.height
            clip_path.add(drawing.rect(clip_min, clip_size))

            # TODO: Handle Z-index with meshes
            for group in view.scene.groups:
                for g in self._create_group(drawing, projection, view.viewport, group):
                    g["clip-path"] = clip_path.get_funciri()
                    drawing.add(g)
        return drawing

    def _create_group(self, drawing: Drawing, projection: np.ndarray,
                      viewport: Viewport, group: Group):
        """
        Render all the meshes contained in this group.

        The main consideration here is that we will consider the z-index of
        every object in this group.
        """
        default_style = group.style or {}
        shaders = [mesh.shader or (lambda face_index, winding: {}) for mesh in group.meshes]
        annotators = [mesh.annotator or (lambda face_index: None) for mesh in group.meshes]

        # A combination of mesh and face indes
        mesh_faces: List[np.ndarray] = []

        for i, mesh in enumerate(group.meshes):
            faces = mesh.faces

            # Extend each point to a vec4, then transform to clip space.
            faces = np.dstack([faces, np.ones(faces.shape[:2])])
            faces = np.dot(faces, projection)

            # Reject trivially clipped polygons.
            xyz, w = faces[:, :, :3], faces[:, :, 3:]
            accepted = np.logical_and(np.greater(xyz, -w), np.less(xyz, +w))
            accepted = np.all(accepted, 2)  # vert is accepted if xyz are all inside
            accepted = np.any(accepted, 1)  # face is accepted if any vert is inside
            degenerate = np.less_equal(w, 0)[:, :, 0]  # vert is bad if its w <= 0
            degenerate = np.any(degenerate, 1)  # face is bad if any of its verts are bad
            accepted = np.logical_and(accepted, np.logical_not(degenerate))
            faces = np.compress(accepted, faces, axis=0)

            # Apply perspective transformation.
            xyz, w = faces[:, :, :3], faces[:, :, 3:]
            faces = xyz / w
            mesh_faces.append(faces)

        # Sort faces from back to front.
        mesh_face_indices = self._sort_back_to_front(mesh_faces)

        # Apply viewport transform to X and Y.
        for faces in mesh_faces:
            faces[:, :, 0:1] = (1.0 + faces[:, :, 0:1]) * viewport.width / 2
            faces[:, :, 1:2] = (1.0 - faces[:, :, 1:2]) * viewport.height / 2
            faces[:, :, 0:1] += viewport.minx
            faces[:, :, 1:2] += viewport.miny

        # Compute the winding direction of each polygon.
        mesh_windings: List[np.ndarray] = []
        for faces in mesh_faces:
            windings = np.zeros(faces.shape[0])
            if faces.shape[1] >= 3:
                p0, p1, p2 = faces[:, 0, :], faces[:, 1, :], faces[:, 2, :]
                normals = np.cross(p2 - p0, p1 - p0)
                np.copyto(windings, normals[:, 2])
            mesh_windings.append(windings)

        group_ = drawing.g(**default_style)
        text_group_ = drawing.g(**default_style)

        # Finally draw the group
        for mesh_index, face_index in mesh_face_indices:
            face = mesh_faces[mesh_index][face_index]
            style = shaders[mesh_index](face_index, mesh_windings[mesh_index][face_index])
            if style is None:
                continue
            face = np.around(face[:, :2], self.precision)

            if len(face) == 1:
                group_.add(drawing.circle(face[0], style.pop("radius", 0.005), **style))
            if len(face) == 2:
                group_.add(drawing.line(face[0], face[1], **style))
            else:
                group_.add(drawing.polygon(face, **style))

            annotation = annotators[mesh_index](face_index)
            if annotation is not None:
                centroid = face.mean(axis=0)
                text_group_.add(drawing.text(insert=centroid, **annotation))


        return [group_, text_group_]

    def _sort_back_to_front(self, mesh_faces: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Sorts the faces in mesh_faces from back to front.

        Args:
            mesh_faces: Each element of this list is a matrix of N x V x 3;
                        N is the number of faces represented, V is the number
                        of vertices in that list.
        Returns:
            A list of tuples of (mesh_idx, face_idx)
        """
        ixs = [(i, j) for i, faces in enumerate(mesh_faces) for j, _ in enumerate(faces)]
        z_centroids = -np.hstack([faces[:, :, 2].mean(axis=1) for faces in mesh_faces])
        return [ixs[i] for i in np.argsort(z_centroids)]


def parametric_surface(slices: int,
                       stacks: int,
                       func: Callable[[float, float], Tuple[float, float, float]]) -> np.ndarray:
    """
    Create a surface using a parametric function.

    The function receives a (theta, phi) and must return (x, y, z) coordinates.
    Theta ranges from (0, pi), phi from (0, 2*pi); each is subdivided into segments of slice and
    stack elements respectively.

    Args:
        slices: The number of slices to use
        stacks: The number of stacks to use
        func: The parametric function

    Returns:
        An numpy array of faces.
    """
    verts = []
    for theta in np.array(np.linspace(0, np.pi, slices + 1)):
        for phi in np.array(np.linspace(0, 2*np.pi, slices)):
            verts.append(func(theta, phi))
    verts = np.float32(verts)

    faces = []
    v = 0
    for i in range(slices):
        for j in range(stacks):
            next_ = (j + 1) % stacks
            faces.append((v + j, v + j + stacks, v + next_ + stacks, v + next_))
        v = v + stacks
    faces = np.int32(faces)
    return verts[faces]


# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
