#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import drjit as dr
import mitsuba as mi
import numpy as np
from pyproj import Transformer
from scipy.spatial import KDTree

from ..constants import DEFAULT_MIN_CELL_SIZE, DEFAULT_MAX_CELL_SIZE
from ..base_stations import BaseStationDB
from .logging_utils import setup_logging


class QuadTree:
    def __init__(
        self,
        utm_zone,
        bbox,
        heightmap,
        min_cell=DEFAULT_MIN_CELL_SIZE,
        max_cell=DEFAULT_MAX_CELL_SIZE,
        parallel=False,
    ):
        """
        Initialize a quadtree given a bounding box of coordinates.

        Params
        ------

        bbox : WGS84 bounding box of the region of interest.
        heightmap: HeightMap object to query heights from.
        min_cell : minimum size of a cell in meters.
        max_cell : maximum size of a cell in meters.
        """

        self.logger = setup_logging(parallel)
        self.min_lat, self.min_lon, self.max_lat, self.max_lon = bbox
        self.heightmap = heightmap
        self.to_utm = Transformer.from_crs("EPSG:4326", utm_zone, always_xy=True)
        self.to_wgs = Transformer.from_crs(utm_zone, "EPSG:4326", always_xy=True)

        # Build a kd-tree with all the base stations, in WGS84
        tx_db = BaseStationDB.from_file()
        region_db = tx_db.get_region(
            (self.min_lat, self.min_lon), (self.max_lat, self.max_lon)
        )
        # Build a kd-tree with the transmitters in the zone of interest, in UTM
        tx_wgs84 = region_db.latlon()
        tx_utm = np.vstack(self.to_utm.transform(tx_wgs84[:, 1], tx_wgs84[:, 0])).T
        self.tx_tree = KDTree(tx_utm)

        # Convert to Point3f for Mitsuba
        self.bl = mi.Point3f(
            *self.to_utm.transform(self.min_lon, self.min_lat), 0.0
        )  # Bottom left
        self.tl = mi.Point3f(
            *self.to_utm.transform(self.min_lon, self.max_lat), 0.0
        )  # Top left
        self.tr = mi.Point3f(
            *self.to_utm.transform(self.max_lon, self.max_lat), 0.0
        )  # Top right
        self.br = mi.Point3f(
            *self.to_utm.transform(self.max_lon, self.min_lat), 0.0
        )  # Bottom right
        dr.make_opaque(self.bl, self.tl, self.tr, self.br)
        dr.eval(self.bl, self.tl, self.tr, self.br)

        # Evaluate the box size in UTM to determine minimum and maximum depth
        height = dr.maximum(dr.norm(self.tl - self.bl), dr.norm(self.tr - self.br))
        width = dr.maximum(dr.norm(self.tr - self.tl), dr.norm(self.br - self.bl))
        max_distance = dr.maximum(width, height)
        self.diagonal = dr.sqrt(width**2 + height**2)
        dr.make_opaque(self.diagonal)
        self.logger.info(
            f"Estimated ground polygon size: width={width[0]}m, height={height[0]}m"
        )

        self.min_depth = int(np.ceil(np.log2(max_distance / max_cell)))
        self.max_depth = min(int(np.ceil(np.log2(max_distance / min_cell))), 15)
        dr.make_opaque(self.min_depth, self.max_depth)
        self.logger.debug(
            f"QuadTree initialized with min depth {self.min_depth} and max depth {self.max_depth}."
        )
        self.logger.debug(
            f"Maximum cell size: {max_distance[0] / (2**self.min_depth):.2f}m"
        )
        self.logger.debug(
            f"Minimum cell size: {max_distance[0] / (2**self.max_depth):.2f}m"
        )

        # First tile in WGS84
        self.grid_res = 2**self.max_depth

        # All nodes at all levels
        self.is_leaf_node = dr.zeros(mi.Bool, (4 ** (self.max_depth + 1) - 1) // 3)
        self.visited = dr.zeros(mi.Bool, dr.width(self.is_leaf_node))

        self.build_tree()

    def build_tree(self):
        self.logger.info("Building quadtree...")
        depth = mi.UInt32(0)
        active_idx = mi.UInt32(0)
        dr.make_opaque(depth, active_idx)

        while True:
            dr.eval(depth, active_idx)
            node_offset = mi.UInt32(4**depth - 1) // 3

            # Mark active nodes as visited
            dr.scatter(self.visited, True, node_offset + active_idx)

            # Compute UTM coordinates of the center of the cell
            cell_id = mi.math.morton_decode2(active_idx)
            uv = (cell_id + 0.5) / 2**depth
            # Bilinear interpolation from region corners
            p1 = self.bl + uv.x * (self.br - self.bl)
            p2 = self.tl + uv.x * (self.tr - self.tl)
            p = p1 + uv.y * (p2 - p1)
            p_np = p.numpy()
            # Query kdtree for the nearest base stations
            dist = mi.Float(self.tx_tree.query(p_np.T[:, :2], k=1)[0])
            cell_diagonal = self.diagonal / 2**depth
            tx_check = dist > 20 * cell_diagonal

            leaf_node = (depth == self.max_depth) | (
                (depth >= self.min_depth) & tx_check
            )

            dr.scatter(self.is_leaf_node, True, node_offset + active_idx, leaf_node)

            if dr.all(leaf_node):
                break

            depth += 1
            nonleaf_node_idx = dr.gather(mi.UInt32, active_idx, dr.compress(~leaf_node))
            active_idx_next = dr.zeros(mi.UInt32, 4 * dr.width(nonleaf_node_idx))
            idx = dr.arange(mi.UInt32, dr.width(nonleaf_node_idx))
            for i in range(4):
                dr.scatter(active_idx_next, 4 * nonleaf_node_idx + i, 4 * idx + i)

            active_idx = active_idx_next

    def make_balanced(self):
        self.logger.info("Making quadtree balanced...")
        while True:
            leaf_nodes = dr.compress(self.is_leaf_node)
            leaf_depths = mi.UInt32(dr.log(3.0 * leaf_nodes + 1) / dr.log(4))
            leaf_offsets = mi.UInt32(4**leaf_depths - 1) // 3
            next_depth_offsets = mi.UInt32(4 ** (leaf_depths + 1) - 1) // 3
            leaf_coord = mi.math.morton_decode2(leaf_nodes - leaf_offsets)

            depth_res = 2**leaf_depths

            offsets = [
                mi.Vector2i(1, 0),
                mi.Vector2i(-1, 0),
                mi.Vector2i(0, 1),
                mi.Vector2i(0, -1),
            ]
            # indices of child nodes of the neighbor that share an edge with the node
            children_indices = [
                (0, 2),  # Right neighbor shares left edge
                (1, 3),  # Left neighbor shares right edge
                (0, 1),  # Top neighbor shares bottom edge
                (2, 3),  # Bottom neighbor shares top edge
            ]
            needs_subdiv = dr.zeros(mi.Bool, dr.width(leaf_nodes))
            for offset, children_offset in zip(offsets, children_indices):
                has_neighbor = dr.all(
                    (leaf_coord + offset >= 0) & (leaf_coord + offset < depth_res)
                )
                neighbor_node = leaf_offsets + mi.math.morton_encode2(
                    leaf_coord + offset
                )
                # First check if the neighbor is a leaf, or if the leaf is lower in the hierarchy
                neighbor_is_leaf = dr.gather(
                    mi.Bool, self.is_leaf_node, neighbor_node, has_neighbor
                )
                neighbor_visited = dr.gather(
                    mi.Bool, self.visited, neighbor_node, has_neighbor
                )
                # If it's lower, check if it's one level deeper
                check_children = has_neighbor & neighbor_visited & ~neighbor_is_leaf
                for c in children_offset:
                    child_node = (
                        next_depth_offsets + 4 * (neighbor_node - leaf_offsets) + c
                    )
                    child_is_leaf = dr.gather(
                        mi.Bool, self.is_leaf_node, child_node, check_children
                    )
                    # If the neighbor is more than one level deeper, we need to subdivide
                    needs_subdiv |= check_children & ~child_is_leaf

            # If no nodes need subdivision, we are done
            if dr.none(needs_subdiv):
                break

            # Mark leaf nodes that will be subdivided as no longer leaf nodes
            dr.scatter(self.is_leaf_node, False, leaf_nodes, needs_subdiv)
            new_leaf_idx = next_depth_offsets + 4 * (leaf_nodes - leaf_offsets)
            # Mark children as leaf and visited
            for i in range(4):
                dr.scatter(self.is_leaf_node, True, new_leaf_idx + i, needs_subdiv)
                dr.scatter(self.visited, True, new_leaf_idx + i, needs_subdiv)

    def to_mesh(self):
        # First ensure the tree is balanced, i.e. adjacent leaves differ by at most one level
        self.make_balanced()
        # Get the vertex positions of the leaf nodes only
        leaf_nodes = dr.compress(self.is_leaf_node)
        # Recompute leaf depth based on index
        # This could be computed directly with log,
        # but we run into precision issues for large indices
        leaf_depths = dr.zeros(mi.UInt32, dr.width(leaf_nodes))
        for d in range(self.min_depth, self.max_depth + 1):
            node_offset = mi.UInt32(4**d - 1) // 3
            next_node_offset = mi.UInt32(4 ** (d + 1) - 1) // 3
            mask = (leaf_nodes >= node_offset) & (leaf_nodes < next_node_offset)
            dr.scatter(leaf_depths, d, dr.arange(mi.UInt32, dr.width(leaf_nodes)), mask)

        assert dr.all(leaf_depths >= self.min_depth) and dr.all(
            leaf_depths <= self.max_depth
        )

        depth_res = 2**leaf_depths
        leaf_offsets = mi.UInt32(4**leaf_depths - 1) // 3
        leaf_idx = mi.math.morton_decode2(leaf_nodes - leaf_offsets)

        cell_size = mi.UInt32(2 ** (self.max_depth - leaf_depths))
        dr.make_opaque(cell_size)
        bottom_left = (
            leaf_idx.y * cell_size * (self.grid_res + 1) + leaf_idx.x * cell_size
        )
        bottom_right = bottom_left + cell_size
        top_left = bottom_left + cell_size * (self.grid_res + 1)
        top_right = top_left + cell_size
        middle = bottom_left + cell_size * (self.grid_res + 1) // 2 + cell_size // 2
        middle_left = bottom_left + cell_size * (self.grid_res + 1) // 2
        middle_right = middle_left + cell_size
        middle_bottom = bottom_left + cell_size // 2
        middle_top = middle_bottom + cell_size * (self.grid_res + 1)

        offsets = [
            mi.Vector2i(1, 0),
            mi.Vector2i(-1, 0),
            mi.Vector2i(0, 1),
            mi.Vector2i(0, -1),
        ]
        edge_vertices = [
            (bottom_right, middle_right, top_right),  # Right edge
            (top_left, middle_left, bottom_left),  # Left edge
            (top_right, middle_top, top_left),  # Top edge
            (bottom_left, middle_bottom, bottom_right),  # Bottom edge
        ]

        split_any = dr.zeros(mi.Bool, dr.width(leaf_nodes))
        faces = dr.zeros(mi.Vector3i, 8 * dr.width(leaf_nodes))
        idx = dr.arange(mi.UInt32, dr.width(leaf_nodes))
        valid_face = dr.zeros(mi.Bool, dr.width(faces))
        split_any = dr.zeros(mi.Bool, dr.width(leaf_nodes))
        split_edges = []
        # First look for edges that need to be split
        for i, (offset, verts) in enumerate(zip(offsets, edge_vertices)):
            has_neighbor = dr.all(
                (leaf_idx + offset >= 0) & (leaf_idx + offset < depth_res)
            )
            neighbor_node = leaf_offsets + mi.math.morton_encode2(leaf_idx + offset)
            # First check if the neighbor is a leaf, or if the leaf is lower in the hierarchy
            neighbor_is_leaf = dr.gather(
                mi.Bool, self.is_leaf_node, neighbor_node, has_neighbor
            )
            neighbor_visited = dr.gather(
                mi.Bool, self.visited, neighbor_node, has_neighbor
            )
            neighbor_deeper = has_neighbor & neighbor_visited & ~neighbor_is_leaf
            v1, v2, v3 = verts
            split_edges.append(neighbor_deeper)
            split_any |= neighbor_deeper

        # Nodes that don't need a split can be handled with 2 faces
        dr.scatter(
            faces,
            mi.Vector3i(bottom_left, bottom_right, top_right),
            8 * idx,
            ~split_any,
        )
        dr.scatter(valid_face, True, 8 * idx, ~split_any)
        dr.scatter(
            faces,
            mi.Vector3i(bottom_left, top_right, top_left),
            8 * idx + 1,
            ~split_any,
        )
        dr.scatter(valid_face, True, 8 * idx + 1, ~split_any)

        # Otherwise, add a middle vertex and 2 triangles per edge that needs splitting
        for i, (split, verts) in enumerate(zip(split_edges, edge_vertices)):
            v1, v2, v3 = verts
            # If the neighboring leaf is deeper, we need to add 2 triangles for this edge
            dr.scatter(faces, mi.Vector3i(v1, v2, middle), 8 * idx + 2 * i, split)
            dr.scatter(valid_face, True, 8 * idx + 2 * i, split)
            dr.scatter(faces, mi.Vector3i(v2, v3, middle), 8 * idx + 2 * i + 1, split)
            dr.scatter(valid_face, True, 8 * idx + 2 * i + 1, split)
            # Otherwise, just one is enough
            dr.scatter(
                faces, mi.Vector3i(v1, v3, middle), 8 * idx + 2 * i, split_any & ~split
            )
            dr.scatter(valid_face, True, 8 * idx + 2 * i, split_any & ~split)

        active_faces = dr.gather(mi.Vector3i, faces, dr.compress(valid_face))
        # Get unique vertex indices
        verts_idx, faces = np.unique(
            dr.ravel(active_faces).numpy(), return_inverse=True
        )
        vertex_count = len(verts_idx)

        cell_size = 1 / self.grid_res
        u = cell_size * (verts_idx % (self.grid_res + 1))
        v = cell_size * (verts_idx // (self.grid_res + 1))
        # Convert to WGS84
        lon = self.min_lon + u * (self.max_lon - self.min_lon)
        lat = self.min_lat + v * (self.max_lat - self.min_lat)
        # Convert to UTM
        x_utm, y_utm = self.to_utm.transform(lon, lat)
        # Get the heights from the heightmap
        h = self.heightmap.height_from_wgs84(lon, lat)
        # Create the active vertices in UTM
        active_verts = mi.Point3f(x_utm, y_utm, h) - self.bl

        mesh = mi.Mesh("DrQuadTreeMesh", vertex_count, dr.width(active_faces))
        params = mi.traverse(mesh)
        params["vertex_positions"] = dr.ravel(active_verts)
        params["faces"] = mi.UInt32(faces)
        params.update()
        return mesh
