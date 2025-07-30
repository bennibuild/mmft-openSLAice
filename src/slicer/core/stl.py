import numpy as np
import trimesh
from affine import Affine  #type: ignore
from PIL import Image
from rasterio.features import rasterize  #type: ignore
from shapely.geometry import mapping #type: ignore
import math
from scipy.spatial import cKDTree # type: ignore
from joblib import Parallel, delayed  # type: ignore
import concurrent.futures
from collections import defaultdict
from collections import Counter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.slicer.core.slicer import Slicer 
from src.slicer.core.section import Section, AngleSection, AngleSectionFragment
from src.slicer.core.helper_functions import round_TOL
from src.slicer.core.constants import *



class Stl:
    """
    A class to hold a single stl file and the functions for the slicing process.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        The trimesh object representing the stl file.
    angle_sections : dict
        A dictionary containing the angle sections of the mesh. The keys are the z levels and the values are dictionaries
        containing the dominant angles and active angles at that z level.
    layers_path_2d : list[trimesh.path.Path2D]
        A list of 2D paths representing the sliced layers of the mesh. Each path is a trimesh.path.Path2D object.
    layers_image : list[PIL.Image]
        A list of images representing the rasterized layers of the mesh. Each image is a PIL.Image object.
    
        
    Properties
    ----------
    angle_sections : dict
        A dictionary containing the angle sections of the mesh. The keys are the z levels and the values are dictionaries
        containing the dominant angles and active angles at that z level.
    """
    def __init__(self, file_path: str,  slicer_object: 'Slicer'):
        self._mesh: trimesh.Trimesh = self._load_mesh(file_path)
        self._slicer: 'Slicer' = slicer_object
        self._opposing_faces: list[int] | None = self._find_opposing_face_indecies(MF_STRUCTURE_WIDTH_MAX, MF_ANGLE_MAX)
        self._mf_plane: list[float] = self._find_mf_plane()
        self._stl_position: tuple[int, int] = (0, 0)  # positions of the stls in the xy plane of the print area (unit: pixels)
        self._angle_sections: list[AngleSection] | None = None  # To store the angle sections of the mesh
        self._angle_sections_future: concurrent.futures.Future[list[AngleSection]] | None = None  # To store the future object for non-blocking computation
        self._rotation_in_xy_plane: float = 0.0 # abs angle in xy plane (0-360) -> to enable reset and ony transform if necesarry
        self.mf_section: Section = self._compute_mf_section()   # list of min, max z-cord of the microfluidic sections
        self.layers_path_2d: list["trimesh.path.Path2D"] = []
        self.layers_image: list[Image.Image] = []


    @property
    def angle_sections(self) -> list[AngleSection]:
        """
        Getter for angle_sections. If the computation is still running, it waits for it to complete.
        Returns a dictionary containing the angle sections of the mesh. The keys are the z levels and the values are dictionaries
        containing the dominant angles and active angles at that z level.
        """
        if self._angle_sections_future:
            print("Waiting for angle sections computation to complete...")
            self._angle_sections = self._angle_sections_future.result()
            self._angle_sections_future = None
            print("Angle sections computation completed.")

        elif self._angle_sections == None:
            self._angle_sections = _compute_angle_sections(self._mesh, self._slicer.printer.settings.z_resolution, self._opposing_faces)

        return self._angle_sections

    # class internal methods

    def _load_mesh(self, file_path: str) -> trimesh.Trimesh:
        print(f"Loading mesh from {file_path}")
        mesh = trimesh.load_mesh(file_path)
        if mesh.is_empty:
            raise ValueError(f"Mesh is empty or could not be loaded from {file_path}")
        if not mesh.is_watertight:
            print(f"Warning: Mesh is not watertight. This may cause issues during slicing.")
        # set mesh to (0,0,0) position -> easier for slicing as each stl is sliced individually and position in the print area is handled separately
        x,y,z = mesh.bounds[0]
        if np.allclose([0,0,0], [x,y,z], atol=0.01):
            print("already at the origin")
        print("Setting xy position to origin (0,0,0)...")
        translation_matrix = trimesh.transformations.translation_matrix([-x, -y, -z])
        mesh.apply_transform(translation_matrix)
        return mesh
    

    def _find_opposing_face_indecies(self, max_dist: float, angular_tol: float, n_jobs: int=-1) -> list[int] | None:
            """
            Find pairs of opposing faces in the mesh using parallel processing.

            Parameters
            ----------
            mesh : trimesh.Trimesh
                The trimesh object representing the mesh.
            max_dist : float
                The maximum distance between the centroids of the faces to be considered as opposing.
            angular_tol : float
                The angular tolerance in degrees for the normals of the faces to be considered as opposing.
            n_jobs : int
                The number of parallel jobs to run. -1 means using all processors.

            Returns
            -------
            list[int]
                A list of indices of the opposing faces in the mesh.
            """
            #print(f"Finding opposing faces with max_dist={max_dist} and angular_tol={angular_tol} degrees...")
            centroids = self._mesh.triangles_center
            normals = self._mesh.face_normals
            tree = cKDTree(centroids)
            cos_tol = np.cos(np.deg2rad(angular_tol))

            def process(i):
                local_pairs = []
                for j in tree.query_ball_point(centroids[i], r=max_dist):
                    if i < j and np.dot(normals[i], normals[j]) < -cos_tol:
                        # Check if faces are facing each other and are not back to back
                        # v from centroid of face i to centroid of face j
                        v_ij = centroids[j] - centroids[i]
                        # cond 1: Face i looks towards centroid_j
                        cond1 = np.dot(normals[i], v_ij) > 0
                        # cond 2: Face j looks towards centroid_i (using -v_ij which is centroids[i] - centroids[j])
                        cond2 = np.dot(normals[j], -v_ij) > 0
                        
                        if cond1 and cond2:
                            local_pairs.append((i, j))
                return local_pairs

            results = Parallel(n_jobs=n_jobs)(delayed(process)(i) for i in range(len(centroids)))
            pairs = [pair for sublist in results for pair in sublist]
            opposing_face_indices = np.unique(np.array(pairs).flatten())
            print(f"Found {len(opposing_face_indices)} opposing faces.")
            return opposing_face_indices if len(opposing_face_indices) > 0 else None


    def _find_mf_plane(self) -> list[float]:
        def fit_plane_ransac(mesh: trimesh.Trimesh, face_indices: list[int] | None, threshold: float = 1e-3, max_trials: int = 1000, random_state: int = 42):
            """
            Deterministic RANSAC plane fit returning only the plane normal.

            Parameters
            ----------
            mesh : trimesh.Trimesh
                The trimesh object representing the mesh.
            face_indices : np.ndarray
                The indices of the faces to be considered for the RANSAC plane fit.
            threshold : float
                The distance threshold for inliers.
            max_trials : int
                The maximum number of RANSAC trials.
            random_state : int
                The random state for reproducibility.
            """

            # get center-points from the mesh
            if face_indices is None:
                print("No faces provided for RANSAC plane fit, returning default normal.")
                return np.array([0, 0, 1])  # default normal if no faces are provided
            points = mesh.triangles_center[face_indices]
            best_inliers_count = 0
            best_normal = None
            n_points = len(points)

            rng = np.random.RandomState(random_state)
            for _ in range(max_trials):
                # deterministic 3-point sample
                idx = rng.choice(n_points, 3, replace=False)
                p0, p1, p2 = points[idx]
                # compute candidate normal
                normal = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(normal)
                if norm == 0:
                    continue
                normal /= norm
                # count inliers
                dists = np.abs((points - p0) @ normal)
                inlier_count = np.sum(dists < threshold)
                if inlier_count > best_inliers_count:
                    best_inliers_count = inlier_count
                    best_normal = normal
                    # early stop
                    if inlier_count > 0.9 * n_points:
                        break

            if best_normal is None:
                raise ValueError("RANSAC failed to find a fitting plane")

            # refine normal via SVD on inliers for consistency
            inlier_mask = np.abs((points - p0) @ best_normal) < threshold
            pts_in = points[inlier_mask]
            centroid = pts_in.mean(axis=0)
            _, _, vh = np.linalg.svd(pts_in - centroid, full_matrices=False)
            refined = vh[-1]
            return refined / np.linalg.norm(refined)
        
        # find the plane the mf structure is spreading in
        normal_ransac = fit_plane_ransac(self._mesh, self._opposing_faces)
        normal_ransac = np.round(normal_ransac, 2) # round to 2 decimal places for better readability
        print(f"RANSAC plane normal {normal_ransac}")
        
        # flip normal vector if necessary 
        flipped_normal_ransac = normal_ransac * -1
        count_normal = np.sum(np.all(np.isclose(self._mesh.face_normals, normal_ransac, atol=ANGLE_TOL), axis=1))
        count_flipped_normal = np.sum(np.all(np.isclose(self._mesh.face_normals, flipped_normal_ransac, atol=ANGLE_TOL), axis=1))
        if count_flipped_normal > count_normal:
            print(f"Flipped RANSAC plane normal {flipped_normal_ransac}")
            normal_ransac = flipped_normal_ransac
        
        return normal_ransac


    def _compute_mf_section(self) -> Section:
        """
        Compute the microfluidic section based on the opposing faces of the mesh.
        The section is defined by the minimum and maximum z values of the opposing faces.

        Returns
        -------
        Section
            The microfluidic section defined by the minimum and maximum z values.
        """
        z_res = self._slicer.printer.settings.z_resolution
        if self._opposing_faces is None:
            print("No opposing faces found, returning empty section at (0, 0)")
            return Section(0, 0)
        vertex_indices_of_faces = self._mesh.triangles[self._opposing_faces]
        z_vals: np.ndarray = vertex_indices_of_faces[:, :, 2]
        mf_section = Section(round_TOL(z_vals.min(), z_res), round_TOL(z_vals.max(), z_res))
        print(f"Microfluidic section: {mf_section}")
        return mf_section


    def _recalc_dependend_parameter(self):
        """
        Recalculate all parameters that depend on the mesh orientation, position, or slicing.
        """
        self.layers_path_2d = None
        self.layers_image = None
        self._angle_sections = None
        self._angle_sections_future = None
        # Run __compute_angle_sections in a separate process
        with concurrent.futures.ProcessPoolExecutor() as executor:
            self._angle_sections_future = executor.submit(_compute_angle_sections, self._mesh, self._slicer.printer.settings.z_resolution, self._opposing_faces)
        self.mf_section = self._compute_mf_section()


    def _find_mf_layers_with_angle(self, angle) -> set[float]:
        z_res = self._slicer.printer.settings.z_resolution
        angle_layers = [round_TOL(section.start, z_res) for section in self.angle_sections if np.isclose(section.angle, angle, atol=ANGLE_TOL)]
        bounds = [self.get_z_min(), self.get_z_max()]
        filtered_angle_layers = []
        for z_val in angle_layers:
            if z_val in bounds: # if layer is at the stl bounds -> skip
                continue
            if self.mf_section.includes_point(z_val): # only add if inside the mf_section
                filtered_angle_layers.append(z_val)
        return set(filtered_angle_layers)


    # getter & setter methods

    def set_orientation(self, orientation: list[float]) -> bool:
        """
        Set the orientation of the mesh to the given vector. The mesh is rotated so that the given vector
        aligns with the z-axis. The mesh is then translated to the origin.

        Parameters
        ----------
        orientation : list[float]
            The orientation vector to align with the z-axis.
        """
        if np.allclose(orientation, [0, 0, 1], atol=0.01):
            print("orientation is already aligned with z-axis")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                self._angle_sections_future = executor.submit(_compute_angle_sections, self._mesh, self._slicer.printer.settings.z_resolution, self._opposing_faces)
            return False
        print("Setting orientation to:", orientation)
        rotation_matrix = trimesh.geometry.align_vectors(orientation, [0, 0, 1])
        self._mesh.apply_transform(rotation_matrix)
        # translate the mesh to the origin
        min_z = np.min(self._mesh.vertices[:, 2])
        if min_z != 0:
            print("Moving the mesh to z = 0...")
            translation_matrix = trimesh.transformations.translation_matrix([0, 0, -min_z])
            self._mesh.apply_transform(translation_matrix)

        self._recalc_dependend_parameter()
        return True

    def set_roation_in_xy_plane(self, angle: float) -> bool:
        """
        Rotate the mesh in the xy plane to the given angle. The mesh is rotated so that the given angle
        aligns with the x-axis.

        Parameters
        ----------
        angle : float
            The angle to rotate the mesh to. The angle is in degrees (0-360).
        """
        if np.isclose(angle, self._rotation_in_xy_plane, atol=0.01):
            print("angle is already set to:", angle)
            return False
        print("Rotating mesh in xy plane to:", angle)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            math.radians(self._rotation_in_xy_plane - angle), [0, 0, 1]
        )
        self._rotation_in_xy_plane = angle
        self._mesh.apply_transform(rotation_matrix)
        self._recalc_dependend_parameter()
        return True

    def set_position(self, x: int, y: int) -> bool:
        self._stl_position = (x, y)
        return True

    def set_position_mm(self, x: float, y: float) -> bool:
        """
        Set the position of the mesh in the xy plane in mm. The position is rounded to the nearest pixel.
        """
        pixel_size = self._slicer.printer.settings.pixel_size
        x_px = round(x / pixel_size)
        y_px = round(y / pixel_size)
        return self.set_position(x_px, y_px)

    def get_position(self) -> tuple[int, int]:
        return self._stl_position

    def get_z_feature_levels(self) -> set[float]:
        z_res = self._slicer.printer.settings.z_resolution
        # 1) Build a sorted set of unique z values from both z_start and extended_z_end (plus mesh bounds)
        z_keys = set()
        for section in self.angle_sections:
            z_keys.add(round_TOL(section.start, z_res))
            z_keys.add(round_TOL(section.end, z_res))
        z_keys.add(self.get_z_min())
        z_keys.add(self.get_z_max())
        return z_keys

    def get_z_max(self) -> float:
        """
        Get the maximum z value of the mesh. The z value is rounded to the nearest multiple of the z resolution.
        """
        z_res = self._slicer.printer.settings.z_resolution
        return round_TOL(self._mesh.bounds[1][2], z_res)
    
    def get_z_min(self) -> float:
        """
        Get the minimum z value of the mesh. The z value is rounded to the nearest multiple of the z resolution.
        """
        z_res = self._slicer.printer.settings.z_resolution
        return round_TOL(self._mesh.bounds[0][2], z_res)

    def get_volume(self) -> float:
        """
        Get the volume of the mesh in ml. The volume is calculated using the trimesh library and converted from mm^3 to ml.
        """
        return self._mesh.volume / 1000 # mm3 convert to ml

    def get_top_layer_pos(self) -> set[float]:
        """
        Finds all z levels inside the microfluidic section where a section is facing down parralel to the build platform (angle of 180°).
        """
        return self._find_mf_layers_with_angle(180)
    
    def get_bottom_layer_pos(self) -> set[float]:
        """
        Finds all z levels inside the microfluidic section where a section is facing up parralel to the build platform (angle of 0).
        """
        return self._find_mf_layers_with_angle(0)

    def get_rasterized_layer_i(self, i: int) -> Image.Image | None:
        if i < 0 or i >= len(self.layers_image):
            return None
        return self.layers_image[i]


    # pub class methods

    def find_dominant_angle(self, section: Section) -> float:
        """
        Find the dominant angle for a given section of the mesh. 

        Arguments
        ----------
        section : Section
            The section of the mesh to find the dominant angle for.

        Returns
        -------
        float
            The dominant angle for the section.
        """
        z_res = self._slicer.printer.settings.z_resolution
        active_angles: list[AngleSection] = []
        active_angles_mf: list[AngleSection] = []
        # active sections: two conditions:
        # 1) beginning before the current section ends -> z_start < z_keys[i+1]
        # 2) not ended before the current section starts -> z_end > z 
        for angle_section in self.angle_sections:
            if round_TOL(angle_section.start, z_res) < section.end and round_TOL(angle_section.end, z_res) > section.start:
                active_angles.append(angle_section)
                if angle_section.part_of_mf:
                    active_angles_mf.append(angle_section)

        if not active_angles:
            return 0 # no considered for layer height calculation
        # ranks angles by their frequency and not by area the planes cover (idea: micro fluid strucures very small but complex structure -> many small faces)
        if active_angles_mf: 
            angle_counts = Counter([iv.angle for iv in active_angles_mf])
        else:
            angle_counts = Counter([iv.angle for iv in active_angles])
        dominant_angles = angle_counts.most_common(5) # so only angles with are more common can constrain the layer height
        new_dominant_angles = [round(abs(angle - 90.0)) for angle, _ in dominant_angles]
        return max(new_dominant_angles) # max -> angle with the biggest constrains on the layer heights


    def prepare_arrange(self) -> tuple[int, int]:
        """
        Prepare the mesh for arranging in the xy plane. The mesh is rotated so that the width is greater than the depth.
        Returns the width and depth of the mesh in pixels.

        Returns
        -------
        tuple[int, int]
            The width and depth of the mesh in pixels.
        """
        bounds = self._mesh.bounds
        width = math.ceil((bounds[1][0] - bounds[0][0]) / self._slicer.printer.settings.pixel_size) + 2
        depth = math.ceil((bounds[1][1] - bounds[0][1]) / self._slicer.printer.settings.pixel_size) + 2
        if width < depth:
            self.set_roation_in_xy_plane(90)
            width, depth = depth, width
        return width, depth


    def auto_orientation(self) -> bool:
        """
        Automatically orient the mesh based on the dominant normal vector of the mesh. It may flip the dominant normal vector
        if the majority of the face normals are aligned with the flipped vector so if more details are on the bottom side of the mesh.

        Returns
        -------
        bool
            True if the orientation was successful, False otherwise.
        """
        print("Auto-orienting the mesh...")
        # set the orientation of the mesh to the normal vector
        self.set_orientation(self._mf_plane)
        return True


    def slice(self, z_levels: list[float]) -> bool:
        """
        Slice the mesh at the given z levels. The mesh is sliced into 2D paths at the specified z levels.
        The z levels are clamped to the bounds of the mesh.

        Parameters
        ----------
        z_levels : list[float]
            The z levels at which to slice the mesh.

        Returns
        -------
        bool
            True if the slicing was successful, False otherwise.
        """
        OFFSET = 0.0001 # to avoid top layers from being printed
        z_levels = [z - OFFSET for z in z_levels if self.get_z_min() < z <= self.get_z_max()] # clamp z levels to bounds excluding z_min (first layer at z_min + layer_height)
        path_2d = self._mesh.section_multiplane(plane_origin=self._mesh.bounds[0], plane_normal=[0,0,1], heights=z_levels)
        self.layers_path_2d = [p for p in path_2d if p is not None]
        return True

    
    def rasterize(self, min_aa: int) -> bool:
        """
        Rasterize each 2D path layer into a mask image where the polygon area is white (255)
        and the background is black (0), with optional anti-alias thresholding.

        Parameters
        ----------
        min_aa : int
            The minimum anti-aliasing value to use.

        Returns
        -------
        bool
            True if the rasterization was successful, False otherwise.
        """
        def _rasterize_layer(path_2d, pixel_size: float, min_aa: int, span_xy: tuple[float, float]) -> Image.Image:
            # Compute grid resolution in pixels
            width = math.ceil(span_xy[0] / pixel_size) + 2
            height = math.ceil(span_xy[1] / pixel_size) + 2
            out_shape = (height, width)

            # Affine transform mapping pixel (col,row) to world coords; origin at lower-left
            transform = Affine(pixel_size, 0, 0,
                               0, -pixel_size, span_xy[1])

            # Burn value: full white where polygon exists
            burn_value = 255

            # Collect shapes: each Polygon in polygons_full includes holes
            shapes = [(mapping(poly), burn_value) for poly in path_2d.polygons_full]

            # Rasterize all shapes in one call
            mask = rasterize(
                shapes,
                out_shape=out_shape,
                transform=transform,
                fill=0,
                all_touched=False,
                dtype='uint8'
            )

            # If using anti-alias threshold <255, map 255->min_aa
            if 0 < min_aa < 255:
                mask = np.where(
                    (mask < min_aa) & (mask < min_aa / 2), 0,
                    np.where((mask < min_aa) & (mask >= min_aa / 2), min_aa, mask)
                ).astype(np.uint8)

            # The polygon areas are now white (min_aa or 255), background is black
            return Image.fromarray(mask, mode='L')

        # Compute X/Y span from mesh bounds
        span_xy = np.ptp(self._mesh.bounds, axis=0)[:2]
        self.layers_image = []

        # Rasterize each path_2d layer
        for i, path_2d in enumerate(self.layers_path_2d):
            if path_2d is None:
                print(f"[Layer {i}] No path found!!")
                return False
            try:
                img = _rasterize_layer(
                    path_2d,
                    pixel_size=self._slicer.printer.settings.pixel_size,
                    min_aa=min_aa,
                    span_xy=span_xy
                )
            except Exception as e:
                print(f"[Layer {i}] Rasterization error: {e}")
                return False
            self.layers_image.append(img)

        return True


def _compute_angle_sections(mesh: trimesh.Trimesh, z_res: float, mf_face_ids: list[int] | None) -> list[AngleSection]:
    """
    Compute angle sections for faces in the mesh. Each section is defined by a starting z value,
    an angle, and the maximum z value of the face. The sections are extended by performing a DFS
    to find connected components of faces with similar normals.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to compute angle sections for.
    z_res : float
        The z resolution to use for snapping z values.
    mf_face_ids : list[int] | None
        The face indices of the microfluidic structure, if any.

    Returns
    -------
    list[AngleSection]
        The computed angle sections.
    """
    # helper functions
    def _build_face_neighbors(mesh: trimesh.Trimesh) -> dict:
        """Build a dictionary mapping each face index to its adjacent face indices."""
        face_neighbors = defaultdict(list)
        # mesh.face_adjacency is an array of shape (N, 2) with each row [face_i, face_j]
        for pair in mesh.face_adjacency:
            i, j = pair
            face_neighbors[i].append(j)
            face_neighbors[j].append(i)
        return face_neighbors

    def _dfs_extended_component(face_idx: int, face_neighbors: dict, mesh: trimesh.Trimesh) -> set:
        """
        Perform a DFS starting at face_idx. Only traverse to adjacent faces whose normals are
        similar to face_idx normal (within ANGLE_TOL). Returns the set of face indices in this
        connected component.
        """
        stack = [face_idx]
        comp = set()
        while stack:
            current = stack.pop()
            if current in comp:
                continue
            comp.add(current)
            for nb in face_neighbors[current]:
                if nb not in comp and np.allclose(mesh.face_normals[nb], mesh.face_normals[face_idx], atol=ANGLE_TOL):
                    stack.append(nb)
        return comp

    def _compute_component_max_z(component, mesh: trimesh.Trimesh) -> float:
        """Given a set of face indices, return the maximum z value among all their vertices."""
        max_z = -np.inf
        for f in component:
            face_vertex_indices = mesh.faces[f]
            face_vertices = mesh.vertices[face_vertex_indices]
            max_z = max(max_z, face_vertices[:, 2].max())
        return max_z

    # 0) Check if input is valid
    if z_res <= 0:
        raise ValueError(f"Invalid z resolution: {z_res}. Must be greater than 0.")
    if mf_face_ids is None:
        print("Warning: No microfluidic structure faces provided. Angle sections will not be marked as microfluidic structures.")
        mf_face_ids = []

    # 1) building angle_section looping over all vertices (vi = vertex index; vertex = vertex coordinates)
    angle_section_fragments: set[AngleSectionFragment] = set()
    for vi, vertex in enumerate(mesh.vertices):
        # 1.1) Get valid face indices for this vertex
        face_indices = [fi for fi in mesh.vertex_faces[vi] if fi != -1]
        if not face_indices:
            continue
        # 1.2) Get the normals of the faces that share this vertex, and check if there are at least two unique normals -> if not no change in angle -> skip
        normals = []
        unique_normals = False
        for face_idx in face_indices:
            normal = mesh.face_normals[face_idx]
            normals.append(normal)
            if not np.allclose(normal, normals[0], atol=ANGLE_TOL):
                unique_normals = True
        if not unique_normals:
            continue
        # 1.3) For each face, check if it extends upward (or is parallel to xy plane) -> if so, compute the angle and add to sections
        for face_idx, normal in zip(face_indices, normals):
            # get max z value of the face
            face_vert_indices = mesh.faces[face_idx]
            face_verts = mesh.vertices[face_vert_indices]
            max_z = face_verts[:, 2].max()
            # only consider face that extends upward
            z_val = vertex[2]
            if max_z >= z_val:  
                z_val = round_TOL(z_val, z_res)
                max_z = round_TOL(max_z, z_res)
                dot_val = np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)
                angle_deg = math.degrees(math.acos(dot_val))
                angle_deg = round_TOL(angle_deg, ANGLE_TOL)
                # If the face is flat (all vertices at the same height) and not a downward face (180°), skip
                if max_z == z_val:
                    if angle_deg == 180.0 or angle_deg == 0.0:
                        angle_section_fragments.add(AngleSectionFragment(start=z_val, end=z_val, angle=angle_deg, face_index=-1)) # duplicates are automatically ignored
                else:
                    angle_section_fragments.add(AngleSectionFragment(start=z_val, end=max_z, angle=angle_deg, face_index=face_idx)) # duplicates are automatically ignored
    #print(f"angle section fragments sorted by start: {sorted(angle_section_fragments, key=lambda x: x.start)}")

    # 2) Extend the z_end for each section by performing a DFS along adjacent faces with similar normals
    extended_sections: set[AngleSection] = set()
    dfs_cache: dict[int, tuple[float, bool]] = {}  # Cache: maps face index -> extended_z_end for the connected component
    face_neighbors = _build_face_neighbors(mesh)
    for section_fragment in angle_section_fragments:
        if section_fragment.face_index == -1:
            extended_sections.add(AngleSection(section_fragment.start, section_fragment.end, section_fragment.angle, False))
            continue
        if section_fragment.face_index in dfs_cache:
            extended_z_end, part_of_mf = dfs_cache[section_fragment.face_index]
        else:
            comp = _dfs_extended_component(section_fragment.face_index, face_neighbors, mesh)
            part_of_mf = True if set(mf_face_ids) - set(comp) else False
            extended_z_end = round_TOL(_compute_component_max_z(comp, mesh), TOL)
            for f in comp:
                dfs_cache[f] = (extended_z_end, part_of_mf)
        new_section = AngleSection(section_fragment.start, extended_z_end, section_fragment.angle, part_of_mf)
        extended_sections.add(new_section)  # duplicates are automatically ignored
    
    print (f"angle sections sorted by start: {sorted(extended_sections, key=lambda x: x.start)}")
    return sorted(extended_sections, key=lambda x: x.start)



